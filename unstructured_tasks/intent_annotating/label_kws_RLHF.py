"""Labels keyword-guided generations for RLHF training using a judge model."""

import dataclasses
import gc
import json
import os
import sys
from multiprocessing import dummy as mp
import functools
import tqdm

import gllm
import torch
import transformers
import tyro
import vllm
from data import create_views
from prompting_kw import comment_generation_utils
import dotenv
import matplotlib.pyplot as plt
import numpy as np

dotenv.load_dotenv()
print(torch.cuda.is_available())
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclasses.dataclass
class Config:
    json_dir: str = "."
    test_json: str = "prompting_kw/data/test.json"
    train_json: str = "prompting_kw/data/train.json"
    valid_json: str = None
    labelling_json_path: str = "notebooks/prompts/few_shot_keyword_labeling_RLHF.json"
    comments_only: bool = False
    label_target: str = "test"
    count_tokens: bool = False
    num_articles: int = 30
    num_comments: int = 20
    model_semantic_entropy: str = None
    model_comments: str = "qwen/qwen-2.5-7b-instruct"
    output_dir: str = "prompting_kw/data/kws"
    separator: str = "\n" + "#" * 20 + "\n"
    engine: str = "hf"
    mode: str = "direct"
    diversity_penalty: float = 0.0
    num_beam_groups: int = 1
    server_address: str = "https://openrouter.ai/api/"
    api_key: str = (
        "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"
    )
    load_model: bool = False
    n_threads: int = 50

    def __post_init__(self):
        if self.engine not in ["hf", "vllm"]:
            raise ValueError("Invalid engine")
        if self.mode not in ["kw", "direct"]:
            raise ValueError("Invalid mode")


def process_record(record, prompt, model_name, model):
    try:
        # Process chosen response
        chosen_parts = record["chosen"].split("\n\nAssistant: ")
        rejected_parts = record["rejected"].split("\n\nAssistant: ")
        len_parts = len(chosen_parts)

        # Get initial keywords for first response
        human_query = chosen_parts[0].split("\n\nHuman: ")[1].strip()
        chosen_response = chosen_parts[1].strip()
        formatted_user_content = prompt[1]["content"].format(
            human=human_query, assistant=chosen_response
        )
        messages = [
            {"role": "system", "content": prompt[0]["content"]},
            {"role": "user", "content": formatted_user_content},
        ]

        if len_parts == 2:
            result = model.get_chat_completion(
                model_name, messages, 100, 0.5, return_mode="primitives", stop="\n\n"
            )
            chosen_keywords = result[0]["content"]
            new_chosen = (
                chosen_parts[0]
                + "\n\nAssistant: "
                + chosen_keywords
                + " ### "
                + chosen_response
            )
            # get separate keywrod for rejected
            human_query = rejected_parts[0].split("\n\nHuman: ")[1].strip()
            rejected_response = rejected_parts[1].strip()
            formatted_user_content = prompt[1]["content"].format(
                human=human_query, assistant=rejected_response
            )
            messages = [
                {"role": "system", "content": prompt[0]["content"]},
                {"role": "user", "content": formatted_user_content},
            ]

            result = model.get_chat_completion(
                model_name, messages, 100, 0.5, return_mode="primitives", stop="\n\n"
            )
            rejected_keywords = result[0]["content"]
            new_rejected = (
                rejected_parts[0]
                + "\n\nAssistant: "
                + rejected_keywords
                + " ### "
                + rejected_response
            )
            record["rejected"] = new_rejected
            record["chosen"] = new_chosen
            return record

        # Process middle pairs with shared keywords
        new_chosen = ""
        new_rejected = ""
        for i in range(0, len_parts // 2):  # Stop before last pair
            # Process chosen
            human_query = chosen_parts[i].split("\n\nHuman: ")[1].strip()
            chosen_response = chosen_parts[i + 1].split("\n\nHuman: ")[0].strip()

            formatted_user_content = prompt[1]["content"].format(
                human=human_query, assistant=chosen_response
            )
            messages = [
                {"role": "system", "content": prompt[0]["content"]},
                {"role": "user", "content": formatted_user_content},
            ]

            result = model.get_chat_completion(
                model_name, messages, 100, 0.5, return_mode="primitives", stop="\n\n"
            )
            shared_keywords = result[0]["content"]

            new_chosen += (
                "\n\nHuman: "
                + human_query
                + "\n\nAssistant: "
                + shared_keywords
                + " ### "
                + chosen_response
            )

            # Process rejected with same keywords
            rejected_response = rejected_parts[i + 1].split("\n\nHuman: ")[0].strip()
            new_rejected += (
                "\n\nHuman: "
                + human_query
                + "\n\nAssistant: "
                + shared_keywords
                + " ### "
                + rejected_response
            )

        # Last chosen pair
        last_human_query = chosen_parts[-2].split("\n\nHuman: ")[-1].strip()
        last_chosen_response = chosen_parts[-1].split("\n\nHuman: ")[0].strip()

        formatted_user_content = prompt[1]["content"].format(
            human=last_human_query, assistant=last_chosen_response
        )
        messages = [
            {"role": "system", "content": prompt[0]["content"]},
            {"role": "user", "content": formatted_user_content},
        ]

        result = model.get_chat_completion(
            model_name, messages, 100, 0.5, return_mode="primitives", stop="\n\n"
        )
        chosen_keywords = result[0]["content"]
        new_chosen += (
            "\n\nHuman: "
            + last_human_query
            + "\n\nAssistant: "
            + chosen_keywords
            + " ### "
            + last_chosen_response
        )

        # Last rejected pair with its own keywords
        last_rejected_response = rejected_parts[-1].split("\n\nHuman: ")[0].strip()
        formatted_user_content = prompt[1]["content"].format(
            human=last_human_query, assistant=last_rejected_response
        )
        messages = [
            {"role": "system", "content": prompt[0]["content"]},
            {"role": "user", "content": formatted_user_content},
        ]

        result = model.get_chat_completion(
            model_name, messages, 100, 0.5, return_mode="primitives", stop="\n\n"
        )
        rejected_keywords = result[0]["content"]
        new_rejected += (
            "\n\nHuman: "
            + last_human_query
            + "\n\nAssistant: "
            + rejected_keywords
            + " ### "
            + last_rejected_response
        )

        record["chosen"] = new_chosen
        record["rejected"] = new_rejected

        return record
    except Exception as e:
        print(f"Error processing record: {str(e)}")
        return None


def main(cfg: Config):
    # Initialize model and tokenizer
    model = gllm.GLLM(cfg.server_address, cfg.api_key)

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

    # Process records in parallel
    results = []
    with mp.Pool(cfg.n_threads) as pool:
        for i, result in enumerate(tqdm.tqdm(pool.imap_unordered(f, records))):
            if result:
                results.append(result)

    # Create output directory if it doesn't exist
    output_dir = cfg.output_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save results with one dictionary per line
    output_path = os.path.join(output_dir, label_file_name.split("/")[-1])
    with open(output_path, "a") as f:
        for i, result in enumerate(results):
            if result:
                json.dump(result, f, indent=2)
                f.write("\n")


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
