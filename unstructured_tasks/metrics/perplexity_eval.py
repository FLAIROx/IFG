"""Evaluates and saves perplexity scores for generated comments."""

import numpy as np
import dataclasses
import os
import glob
import tyro
import json
import re
import metrics.ppl_recurrent_lm as lmppl


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@dataclasses.dataclass
class Config:
    direct_comments_dir: str = "aws/outputs-new5/7B"
    kw_comments_dir: str = "output_ablations_rewarded"
    test_json: str = "training/data/test_formatted.json"
    output_dir: str = "output_perplexity"
    win_rate_prompt_path: str = "semantic_entropy/prompts/win_rate_prompt.txt"
    comments_only: bool = False
    label_target: str = "test"
    count_tokens: bool = False
    num_articles: int = 30
    num_comments: int = 20
    model_judge: str = "openai/gpt-4o"
    model_perplexity: str = "meta-llama/Llama-3.1-8B"
    separator: str = "\n" + "#" * 20 + "\n"
    engine: str = "hf"
    diversity_penalty: float = 0.0
    num_beam_groups: int = 1
    server_address: str = "https://openrouter.ai/api/"
    api_key: str = (
        "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"
    )
    load_model: bool = False
    n_threads: int = 50


def get_comments_paths(directory):
    """Get relative paths of all comments.json files in the given directory."""
    # Get all comments.json files recursively
    pattern = os.path.join(directory, "*", "comments.json")
    paths = glob.glob(pattern, recursive=True)
    return paths


def extract_key_from_dirname(dirname):
    """
    Extract key from directory name based on the pattern.
    For directories with 'ft_X_Y', returns tuple (X, Y)
    For directories with 'ft_X', returns tuple (X,)
    For other directories, returns the directory name
    """
    # Check if it's a directory with ft_X_Y pattern
    match = re.search(r"ft_(\d+\.?\d*)_(\d+\.?\d*)", dirname)
    if match:
        return (float(match.group(1)), float(match.group(2)))

    # Check if it's a directory with ft_X pattern
    match = re.search(r"ft_(\d+\.?\d*)", dirname)
    if match:
        return (float(match.group(1)),)

    # If no numbers found, return the full directory name
    return dirname


def score_perplexity_to_json(scorer, filepaths, cfg):
    for filepath in filepaths:
        print(f"Processing {filepath}")
        with open(filepath, "r") as f:
            data = json.load(f)

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

        perplexity_scores = []
        for i, comment_set in enumerate(comments):
            scores = scorer.get_perplexity(comment_set, batch_size=5)
            perplexity_scores.append(scores)

        data[section]["perplexity_scores"] = perplexity_scores
        data[section]["perplexity_scores_mean"] = np.mean(perplexity_scores)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=4)


def main(cfg: Config):
    import pprint

    # Get paths and scores for pairing
    ifg_paths = get_comments_paths(cfg.kw_comments_dir)

    scorer = lmppl.LM(cfg.model_perplexity, use_auth_token=True, use_tqdm=False)

    score_perplexity_to_json(scorer, ifg_paths, cfg)

    # save config
    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
