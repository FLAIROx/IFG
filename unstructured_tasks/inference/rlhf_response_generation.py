import dataclasses
import gc
import json
import os

import numpy as np
import torch
import transformers
import tyro
import vllm


import unstructured_tasks.inference.rlhf_response_generation_utils as comment_generation_utils
from unstructured_tasks.metrics import semantic_entropy
import dotenv
import torch.distributed as dist

dotenv.load_dotenv()

os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'

torch.cuda.empty_cache()


@dataclasses.dataclass
class Config:
    test_file_path: str = "data/hh_annotated/test_formatted.json"
    num_articles: int = 100
    num_comments: int = 10
    semantic_entropy_prompt_path: str = (
        "unstructured_tasks/metrics/semantic_entropy/prompts/semantic_clusters_RLHF.txt"
    )
    model_semantic_entropy: str = "meta-llama/Llama-3.1-8B-Instruct"
    model_dpo: str = "data/checkpoints/rlhf_experiments/dpo_stage/Qwen2.5-7B/final/"
    output_dir: str = "data/generated_responses/kw-7B-dpo_ft_0.5_0.5"
    separator: str = ""
    engine: str = "vllm"
    mode: str = "kw"
    diversity_penalty: float = 0.0
    keyword_temp: float = 0.5
    comments_temp: float = 0.5
    num_beam_groups: int = 1

    def __post_init__(self):
        if self.engine not in ["hf", "vllm"]:
            raise ValueError("Invalid engine")
        if self.mode not in ["kw", "direct"]:
            raise ValueError("Invalid mode")

        if os.path.exists(self.output_dir):
            if os.path.isfile(self.output_dir):
                raise ValueError("Output directory path already exists and is a file")
            elif os.path.isdir(self.output_dir) and os.listdir(self.output_dir):
                raise ValueError("Output directory path already exists and is not empty")


def main(
    cfg: Config,
):
    # Read test_formatted.json
    with open(cfg.test_file_path, 'r') as f:
        test_set = json.load(f)
    
    # Read semantic_entropy_prompt_path
    with open(cfg.semantic_entropy_prompt_path, "r") as f:
        semantic_entropy_prompt = f.read()
    
    test_set_reduced = test_set[:cfg.num_articles]
    prompts = "{article}"
    articles = [i["prompt"] for i in test_set_reduced]


    if cfg.engine == "hf":
        comments_model = comment_generation_utils.FixedQwen.from_pretrained(
            cfg.model_dpo,
            device_map="cuda"
        )
        assert cfg.mode != 'kw', "Not implemented yet"
        
    else:
        comments_model = vllm.LLM(model=cfg.model_dpo, max_model_len=8192, gpu_memory_utilization=0.8)

    data = {}
    if cfg.mode == "kw":
        comments_kw, keywords = comment_generation_utils.generate_comments_through_kw(
            articles=articles,
            prompt_keywords=prompts,
            model=comments_model,
            n_comments=cfg.num_comments,
            keyword_temp=cfg.keyword_temp,
            comment_temp=cfg.comments_temp,
        )
        data["keywords"] = {"comments": comments_kw, "keywords": keywords}

    elif cfg.mode == "direct" :
        comments_direct = comment_generation_utils.generate_comments_directly(
            articles=articles,
            prompt=prompts,
            temperature=cfg.comments_temp,
            model=comments_model,
            n_comments=cfg.num_comments,
            num_beam_groups=cfg.num_beam_groups,
            diversity_penalty=cfg.diversity_penalty,
            comment_stop_token = None
        )
        data["direct"] = {"comments": comments_direct}

    if cfg.model_dpo == cfg.model_semantic_entropy:
        semantic_entropy_model = comments_model
    else:
        del comments_model
        gc.collect()
        torch.cuda.empty_cache()
        semantic_entropy_model = vllm.LLM(
            model=cfg.model_semantic_entropy, max_model_len=8192, gpu_memory_utilization=0.9
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



    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)


    with open(os.path.join(cfg.output_dir, "comments.json"), "w") as f:
        json.dump(data, f, indent=4)

    # with open(os.path.join(cfg.output_dir, "articles.json"), "w") as f:
    #     json.dump(articles, f, indent=4)

    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)
        
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)