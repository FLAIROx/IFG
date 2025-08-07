"""Generates responses from prompts using direct and keyword-guided approaches."""

import dataclasses
import gc
import json
import os

import numpy as np
import torch
import tyro
import vllm

import unstructured_tasks.inference.rlhf_response_generation_utils as comment_generation_utils
from unstructured_tasks.metrics import semantic_entropy
import dotenv
import torch.distributed as dist

dotenv.load_dotenv()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

torch.cuda.empty_cache()


@dataclasses.dataclass
class Config:
    prompt_keywords_path: str = "unstructured_tasks/prompts/few_shot_keyword_comment.txt"
    prompt_direct_path: str = "unstructured_tasks/prompts/few_shot_comment.txt"
    semantic_entropy_prompt_path: str = "unstructured_tasks/metrics/semantic_entropy/prompts/semantic_clusters.txt"
    test_file_path: str = (
        "data/reddit_comments/test.jsonl"
    )
    load_from_test: bool = True
    num_articles: int = 1
    num_comments: int = 10
    model_semantic_entropy: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_comments: str = "Qwen/Qwen2.5-7B"
    output_dir: str = "data/generated_responses/reddit_comments/ifg"
    separator: str = "###"
    engine: str = "vllm"
    mode: str = "kw"
    diversity_penalty: float = 0.0
    keyword_temp: float = 0.8
    comments_temp: float = 0.5
    num_beam_groups: int = 1

    def __post_init__(self):
        if self.engine not in ["hf", "vllm"]:
            raise ValueError("Invalid engine")
        if self.mode not in ["kw", "direct"]:
            raise ValueError("Invalid mode")

        if os.path.exists(self.output_dir) and (
            os.path.isfile(self.output_dir) or os.listdir(self.output_dir)
        ):
            raise ValueError(
                "Output directory path already exists and is not empty/is a file"
            )


def main(
    cfg: Config,
):
    test_set = comment_generation_utils.read_jsonl(cfg.test_file_path)
    all_articles = [
        record["prompt"].split("News article:\n")[1].split("\nUser comment:")[0]
        for record in test_set
    ]
    articles = list(dict.fromkeys(all_articles))[: cfg.num_articles]


    with open(cfg.prompt_keywords_path, "r") as f:
        prompt_keywords = f.read()
    with open(cfg.prompt_direct_path, "r") as f:
        prompt_direct = f.read()
    with open(cfg.semantic_entropy_prompt_path, "r") as f:
        semantic_entropy_prompt = f.read()

    if cfg.engine == "hf":
        comments_model = comment_generation_utils.FixedQwen.from_pretrained(
            cfg.model_comments, device_map="cuda"
        )
        assert cfg.mode != "kw", "Not implemented yet"

    else:
        comments_model = vllm.LLM(
            model=cfg.model_comments, max_model_len=8192, gpu_memory_utilization=0.6
        )

    data = {}
    if cfg.mode == "kw":
        comments_kw, keywords = comment_generation_utils.generate_comments_through_kw(
            articles=articles,
            prompt_keywords=prompt_keywords,
            model=comments_model,
            n_comments=cfg.num_comments,
            keyword_temp=cfg.keyword_temp,
            comment_temp=cfg.comments_temp,
            comment_stop_str=cfg.separator,
        )
        data["keywords"] = {"comments": comments_kw, "keywords": keywords}

    elif cfg.mode == "direct":
        comments_direct = comment_generation_utils.generate_comments_directly(
            articles=articles,
            prompt=prompt_direct,
            model=comments_model,
            n_comments=cfg.num_comments,
            num_beam_groups=cfg.num_beam_groups,
            diversity_penalty=cfg.diversity_penalty,
            temperature=cfg.comments_temp,
            comment_stop_token=None,
        )
        data["direct"] = {"comments": comments_direct}

    if cfg.model_comments == cfg.model_semantic_entropy:
        semantic_entropy_model = comments_model
    else:
        del comments_model
        gc.collect()
        torch.cuda.empty_cache()
        semantic_entropy_model = vllm.LLM(
            model=cfg.model_semantic_entropy,
            max_model_len=8192,
            gpu_memory_utilization=0.6,
        )

    def compute_entropy_over_articles(comments_data):
        """Compute semantic entropy for the commments on each article."""
        entropy = []

        for comment_group in comments_data["comments"]:
            entropy.append(
                semantic_entropy.interface.semantic_entropy_from_comments(
                    comment_group, semantic_entropy_prompt, semantic_entropy_model
                )
            )
        return entropy

    comment_data = list(data.values())[0]
    entropies = compute_entropy_over_articles(comment_data)
    comment_data["entropies"] = entropies
    comment_data["mean_entropy"] = np.mean(entropies)

    print(f"Entropy for generated comments: {np.mean(entropies)}")

    # raise Exception("Stop here")
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    # breakpoint()
    with open(os.path.join(cfg.output_dir, "comments.json"), "w") as f:
        json.dump(data, f, indent=4)

    with open(os.path.join(cfg.output_dir, "articles.json"), "w") as f:
        json.dump(articles, f, indent=4)

    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    import logging
    import os
    from datetime import datetime

    # Create logs directory if it doesn't exist
    log_dir = "data/logs"
    os.makedirs(log_dir, exist_ok=True)

    # Configure logging to file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"generate_comments_{timestamp}.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also log to console
        ]
    )

    cfg = tyro.cli(Config)
    logging.info("Config:")
    for key, value in dataclasses.asdict(cfg).items():
        logging.info(f"{key}: {value}")
    logging.info(f"Starting job with keyword_temp = {cfg.keyword_temp}, comments_temp = {cfg.comments_temp}")
    main(cfg)
