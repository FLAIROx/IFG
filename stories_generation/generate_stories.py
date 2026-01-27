"""Generate short stories from prompts using vLLM."""

import dataclasses
import json
import os
from typing import Any

import tyro
import vllm
import torch


@dataclasses.dataclass
class Config:
    prompts_path: str = "stories_generation/prompts/story_prompts.json"
    template_path: str = "stories_generation/prompts/story_template.txt"
    model: str = "Qwen/Qwen3-32B"
    output_dir: str = "stories_generation/outputs/qwen3-32b"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    outline_temp: float = 0.7
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 1000
    outlines_per_style: int = 15
    enable_thinking: bool = False
    outline_stop_str: str = "Story:"
    outline_start_str: str = "### Outline:"
    story_start_str: str = "### Story"
    max_outline_tokens: int = 100

    def __post_init__(self) -> None:
        if self.tensor_parallel_size < 1 or self.data_parallel_size < 1:
            raise ValueError("Parallel sizes must be >= 1")
        if os.path.exists(self.output_dir):
            if os.path.isfile(self.output_dir):
                raise ValueError("Output directory path already exists and is a file")
            if os.path.isdir(self.output_dir) and os.listdir(self.output_dir):
                raise ValueError("Output directory path already exists and is not empty")


def _load_prompt_data(prompts_path: str) -> dict[str, Any]:
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "intents" not in data or "prompts" not in data:
        raise ValueError("Prompt file must include 'intents' and 'prompts' arrays.")
    return data


def _build_requests(
    prompt_data: dict[str, Any], template: str
) -> tuple[list[dict[str, Any]], list[str]]:
    outline_styles = prompt_data["intents"]
    prompts = prompt_data["prompts"]

    if len(outline_styles) != 3:
        raise ValueError("Expected exactly 3 outline styles in prompts file.")
    if len(prompts) != 10:
        raise ValueError("Expected exactly 10 prompts in prompts file.")

    requests: list[dict[str, Any]] = []
    rendered_prompts: list[str] = []

    for prompt in prompts:
        for style in outline_styles:
            requests.append(
                {
                    "prompt_id": prompt["id"],
                    "prompt": prompt["prompt"],
                    "style_description": style["description"],
                }
            )
            rendered_prompts.append(
                template.format(
                    story_prompt=prompt["prompt"],
                    style_description=style["description"],
                    outline="",
                )
            )

    return requests, rendered_prompts


def _render_chat_prompts(
    llm: vllm.LLM,
    rendered_prompts: list[str],
    enable_thinking: bool,
) -> list[str]:
    tokenizer = llm.get_tokenizer()
    chat_prompts: list[str] = []
    for prompt in rendered_prompts:
        messages = [{"role": "user", "content": prompt}]
        chat_prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        )
    return chat_prompts


def _extract_outline(text: str, stop_str: str, start_str: str) -> str:
    text = text.replace(stop_str, "")
    text = text.replace(start_str, "")
    return text.strip()


def _extract_story(text: str, start_str: str) -> str:
    if start_str in text:
        text = text.split(start_str)[1]
    return text


def main(cfg: Config) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Request a GPU node (e.g., --gres=gpu:1) "
            "or run with srun on a GPU partition."
        )
    prompt_data = _load_prompt_data(cfg.prompts_path)

    with open(cfg.template_path, "r", encoding="utf-8") as f:
        template = f.read()

    requests, rendered_prompts = _build_requests(prompt_data, template)

    llm = vllm.LLM(
        model=cfg.model,
        max_model_len=cfg.max_model_len,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        enforce_eager=True,  # Disable CUDA graphs (PTX compatibility issue on GH200)
        tensor_parallel_size=cfg.tensor_parallel_size,
        data_parallel_size=cfg.data_parallel_size,
    )
    outline_sampling_params = vllm.SamplingParams(
        temperature=cfg.outline_temp,
        top_p=cfg.top_p,
        max_tokens=cfg.max_outline_tokens,
        n=cfg.outlines_per_style,
        stop=cfg.outline_stop_str,
        include_stop_str_in_output=True,
    )

    # First pass: generate outlines
    outline_prompts = _render_chat_prompts(llm, rendered_prompts, cfg.enable_thinking)
    outline_results = llm.generate(
        outline_prompts, sampling_params=outline_sampling_params
    )
    outlines_per_request = [
        [_extract_outline(output.text, cfg.outline_start_str, cfg.story_start_str) for output in result.outputs]
        for result in outline_results
    ]
    
    # breakpoint()

    # Second pass: generate stories using the outlines
    story_requests: list[dict[str, Any]] = []
    story_prompts_raw: list[str] = []
    for request, outlines in zip(requests, outlines_per_request, strict=True):
        for outline in outlines:
            story_requests.append(
                {
                    "prompt_id": request["prompt_id"],
                    "prompt": request["prompt"],
                    "style_description": request["style_description"],
                    "outline_text": outline,
                }
            )
            # Build the full prompt with outline filled in, ending at "### Story:"
            story_prompts_raw.append(
                template.format(
                    story_prompt=request["prompt"],
                    style_description=request["style_description"],
                    outline=outline,
                )
            )

    story_sampling_params = vllm.SamplingParams(
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        max_tokens=cfg.max_tokens,
        n=1,
        include_stop_str_in_output=False,
    )

    story_prompts = _render_chat_prompts(llm, story_prompts_raw, cfg.enable_thinking)
    story_results = llm.generate(story_prompts, sampling_params=story_sampling_params)
    stories = [_extract_story(result.outputs[0].text, cfg.outline_stop_str) for result in story_results]
    # breakpoint()
    output_by_prompt: dict[str, dict[str, Any]] = {}
    for request, story in zip(story_requests, stories, strict=True):
        prompt_id = request["prompt_id"]
        if prompt_id not in output_by_prompt:
            output_by_prompt[prompt_id] = {
                "prompt_id": prompt_id,
                "prompt": request["prompt"],
                "outline_styles": prompt_data["intents"],
                "stories": [],
            }
        output_by_prompt[prompt_id]["stories"].append(
            {
                "outline": {
                    "style_description": request["style_description"],
                    "text": request["outline_text"],
                },
                "story": story,
            }
        )

    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)

    for prompt_id, payload in output_by_prompt.items():
        output_path = os.path.join(cfg.output_dir, f"{prompt_id}.json")
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=4)

    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
