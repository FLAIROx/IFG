"""Labels keyword-guided news article generations using a judge model."""

import dataclasses
import json
import os
from multiprocessing import dummy as mp
import functools
import tqdm

import gllm
import torch
import transformers
import tyro
from prompting import comment_generation_utils
import dotenv
import matplotlib.pyplot as plt
import numpy as np

dotenv.load_dotenv()
print(torch.cuda.is_available())
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclasses.dataclass
class Config:
    json_dir: str = "data/bbc/reddit_data/no_context"
    test_json: str = "data/bbc/reddit_data/jsonl/no_context/oxford_ai_supervised_no_contex_test.jsonl"
    train_json: str = "data/bbc/reddit_data/jsonl/no_context/oxford_ai_supervised_no_contex_train.jsonl"
    valid_json: str = "data/bbc/reddit_data/jsonl/no_context/oxford_ai_supervised_no_contex_valid.jsonl"
    labelling_json_path: str = (
        "notebooks/prompts/few_shot_keyword_comment_labeling.json"
    )
    comments_only: bool = False
    label_target: str = "valid"
    count_tokens: bool = False
    num_articles: int = 30
    num_comments: int = 20
    model_semantic_entropy: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_comments: str = "Qwen/Qwen2.5-7B-Instruct"
    output_dir: str = "data/bbc/reddit_data/kws"
    separator: str = "\n" + "#" * 20 + "\n"
    engine: str = "hf"
    mode: str = "direct"
    diversity_penalty: float = 0.0
    num_beam_groups: int = 1
    server_address: str = "http://127.0.0.1:9833"
    load_model: bool = True
    n_threads: int = 50

    def __post_init__(self):
        if self.engine not in ["hf", "vllm"]:
            raise ValueError("Invalid engine")
        if self.mode not in ["kw", "direct"]:
            raise ValueError("Invalid mode")
        # if os.path.exists(self.output_dir) and (os.path.isfile(self.output_dir) or os.listdir(self.output_dir)):
        #     raise ValueError("Output directory path already exists and is not empty/is a file")


def process_record(record, prompt, model_name, model):
    try:
        article_ = (
            record["prompt"].split("News article:\n")[1].split("\nUser comment:")[0]
        )
        comment_ = record["response"]
        messages = [
            {"role": "system", "content": prompt[0]["content"]},
            {
                "role": "user",
                "content": prompt[1]["content"].format(
                    article=article_, comment=comment_
                ),
            },
        ]

        result = model.get_chat_completion(
            model_name, messages, 100, 0.5, return_mode="primitives"
        )
        keywords_ = result[0]["content"]
        record["keywords"] = keywords_
        return record
    except Exception as e:
        print(f"Error processing record: {str(e)}")
        return None


def process_record_comment_only(record, prompt, model_name, model):
    try:
        article_ = (
            record["prompt"].split("News article:\n")[1].split("\nUser comment:")[0]
        )
        comment_ = record["response"]
        messages = [
            {"role": "system", "content": prompt[0]["content"]},
            {"role": "user", "content": prompt[1]["content"].format(comment=comment_)},
        ]

        result = model.get_chat_completion(
            model_name, messages, 100, 0.5, return_mode="primitives"
        )
        keywords_ = result[0]["content"]
        record["keywords"] = keywords_
        return record
    except Exception as e:
        print(f"Error processing record: {str(e)}")
        return None


def count_context_length(record, tokenizer):
    """Count the total number of tokens in the article and comment."""
    article = record["prompt"].split("News article:\n")[1].split("\nUser comment:")[0]
    comment = record["response"]

    # Combine article and comment
    full_context = f"{article}\n{comment}"

    # Count tokens
    tokens = tokenizer(full_context)["input_ids"]
    return len(tokens)


def main(cfg: Config):
    # Initialize model and tokenizer
    model = gllm.DistributionServerInterface(cfg.server_address)

    if cfg.load_model:
        model.load_model(cfg.model_comments)

    if cfg.label_target == "train":
        label_file_name = cfg.train_json
    elif cfg.label_target == "valid":
        label_file_name = cfg.valid_json
    else:
        label_file_name = cfg.test_json

    # Load data and prompt
    records = comment_generation_utils.read_jsonl(label_file_name)
    print("len records", len(records))
    # print(records[-1][:300])

    # Count and print context lengths
    if cfg.count_tokens:
        tokenizer = transformers.AutoTokenizer.from_pretrained(cfg.model_comments)
        context_lengths = [
            count_context_length(record, tokenizer) for record in records
        ]
        print(
            f"Average context length: {sum(context_lengths)/len(context_lengths):.2f} tokens"
        )
        print(f"Max context length: {max(context_lengths)} tokens")
        print(f"Min context length: {min(context_lengths)} tokens")

        # Create histogram
        plt.figure(figsize=(10, 6))
        plt.hist(context_lengths, bins=50, edgecolor="black")
        plt.title("Distribution of Context Lengths")
        plt.xlabel("Number of Tokens")
        plt.ylabel("Frequency")
        plt.axvline(
            x=np.mean(context_lengths),
            color="r",
            linestyle="dashed",
            label=f"Mean ({np.mean(context_lengths):.0f})",
        )
        plt.axvline(
            x=np.median(context_lengths),
            color="g",
            linestyle="dashed",
            label=f"Median ({np.median(context_lengths):.0f})",
        )
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save the plot
        histogram_path = os.path.join(
            cfg.output_dir, f"context_length_histogram_{cfg.label_target}_limited.png"
        )
        plt.savefig(histogram_path, dpi=300, bbox_inches="tight")
        plt.close()

    # print(f"Histogram saved to {histogram_path}")

    # breakpoint()

    model.wait_for_health()

    prompt_path = cfg.labelling_json_path

    if cfg.comments_only:
        name, ext = os.path.splitext(cfg.labelling_json_path)
        prompt_path = f"{name}_ONLY{ext}"

    with open(prompt_path) as f:
        prompt = json.load(f)

    # Create partial function with fixed arguments
    f = functools.partial(
        process_record, prompt=prompt, model_name=cfg.model_comments, model=model
    )
    if cfg.comments_only:
        f = functools.partial(
            process_record_comment_only,
            prompt=prompt,
            model_name=cfg.model_comments,
            model=model,
        )

    # Process records in parallel
    results = []
    with mp.Pool(cfg.n_threads) as pool:
        for i, result in enumerate(tqdm.tqdm(pool.imap_unordered(f, records))):
            if result:
                results.append(result)

    # Create output directory if it doesn't exist
    output_dir = cfg.output_dir
    if cfg.comments_only:
        output_dir = os.path.join(output_dir, "comments_only")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save results with one dictionary per line
    output_path = os.path.join(output_dir, label_file_name.split("/")[-1])
    # If file doesn't exist, create it and write results
    if not os.path.exists(output_path):
        with open(output_path, "w") as f:
            for result in results:
                json.dump(result, f)
                f.write("\n")
        print(
            f"Created new file. Processed {len(results)} records. Results saved to {output_path}"
        )

    # If file exists, append results
    else:
        with open(output_path, "a") as f:
            for result in results:
                json.dump(result, f)
                f.write("\n")
        print(f"Appended {len(results)} new records to {output_path}")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
