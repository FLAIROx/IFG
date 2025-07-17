"""Evaluates generated comments using a reward model and saves the scores."""

import dataclasses
import gc
import json
import os
from typing import List, Dict

import numpy as np
import torch
import transformers
import tyro
import vllm
from tqdm import tqdm

from data import create_views
from prompting_kw import comment_generation_utils_RLHF as comment_generation_utils
import dotenv
import torch.distributed as dist

dotenv.load_dotenv()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()


@dataclasses.dataclass
class Config:
    test_file_path: str = "training/data/test_formatted.json"
    reward_model: str = "RLHF/reward_model/Qwen2.5-7B-Reward/checkpoint-443"
    test_dataset: str = "training/data/test_formatted.json"
    output_dir: str = "output_ablations_rewarded"
    num_articles: int = 100


@torch.no_grad()
def evaluate_comments(
    model, tokenizer, comments: List[str], device="cuda"
) -> List[float]:
    """Evaluate a list of comments using the reward model"""
    rewards = []
    model = model.to(device)

    # Process comments in batches to avoid OOM
    batch_size = 2
    for i in range(0, len(comments), batch_size):
        batch = comments[i : i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
        ).to(device)

        outputs = model(**inputs)
        batch_rewards = outputs.logits.squeeze(-1).cpu().tolist()

        rewards.extend(
            batch_rewards if isinstance(batch_rewards, list) else [batch_rewards]
        )

    return rewards


def process_json_file(filepath: str, model, tokenizer, cfg: Config) -> None:
    """Process a single JSON file and update it with rewards"""
    print(f"Processing {filepath}")

    with open(filepath, "r") as f:
        data = json.load(f)

        # Read test_formatted.json
    with open(cfg.test_file_path, "r") as f:
        test_set = json.load(f)

    test_set_reduced = test_set[: cfg.num_articles]
    prompt_template = "{article}"
    articles = [i["prompt"] for i in test_set_reduced]

    # Check which section exists and process it
    if "direct" in data and "comments" in data["direct"]:
        section = "direct"
    elif "keywords" in data and "comments" in data["keywords"]:
        section = "keywords"
    else:
        print(f"No valid comments section found in {filepath}")
        return

    print(f"Processing {section} comments")
    comments = data[section]["comments"]

    # Calculate rewards for each set of comments
    rewards = []
    for i, comment_set in enumerate(comments):
        prompts = [
            prompt_template.format(article=articles[i]) for _ in range(len(comment_set))
        ]
        conversations = [
            prompt + comment for prompt, comment in zip(prompts, comment_set)
        ]
        set_rewards = evaluate_comments(model, tokenizer, conversations)
        rewards.append(set_rewards)

    # Add rewards to the JSON
    data[section]["rewards"] = rewards

    # Calculate mean reward
    all_rewards = [r for sublist in rewards for r in sublist]
    data[section]["mean_reward"] = float(np.mean(all_rewards))

    print(f"Mean reward for {section}: {data[section]['mean_reward']}")

    # Save updated JSON
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)


def main(cfg: Config):
    # Load model and tokenizer
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        cfg.reward_model,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        cfg.reward_model, trust_remote_code=True, return_tensors="pt", padding=True
    )

    # Find all JSON files first
    json_files = []
    for root, dirs, files in os.walk(cfg.output_dir):
        for file in files:
            if file.endswith("comments.json"):
                json_files.append(os.path.join(root, file))

    # Process files with tqdm progress bar
    for filepath in tqdm(json_files, desc="Processing comment files"):
        try:
            process_json_file(filepath, model, tokenizer, cfg)
        except Exception as e:
            print(f"Error processing {filepath}: {str(e)}")
            continue

    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
