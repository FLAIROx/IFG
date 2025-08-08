import shlex
import os
import subprocess
from pathlib import Path


models = [
    "google/gemma-3-12b-pt",
    "Qwen/Qwen3-8B-Base",
]

files = [
    "data/generated_responses/reddit_comments/Qwen2.5-7B/ifg_0.5_0.3/comments.json",
    "data/generated_responses/reddit_comments/Qwen2.5-7B/ifg_0.7_0.5/comments.json",
    "data/generated_responses/reddit_comments/Qwen2.5-7B/ifg_1.0_0.7/comments.json",
    "data/generated_responses/reddit_comments/Qwen2.5-7B/ifg_1.2_0.7/comments.json",
]

# Command template for reference:
# python -m rebuttals.rse_llm_study.measure_rse \
#   --input-comments-path data/generated_responses/reddit_comments/Qwen2.5-7B/ifg_0.5_0.3/comments.json \
#   --model-semantic-entropy Qwen/Qwen3-4B-Base \
#   --output-path data/generated_responses/reddit_comments/re_evals/qwen3-4b/Qwen2.5-7B/ifg_0.5_0.3/comments_rse.json


def _sanitize_model_dir(model_name: str) -> str:
    name = model_name.split("/")[-1]
    sanitized = "".join(c if c.isalnum() or c in ["-", "_"] else "-" for c in name)
    return sanitized.lower()


def _build_output_path(input_path: str, model_name: str) -> Path:
    input_p = Path(input_path)

    # Try to keep path structure after 'reddit_comments'
    tail_parts = None
    if "reddit_comments" in input_p.parts:
        idx = input_p.parts.index("reddit_comments")
        tail_parts = input_p.parts[idx + 1 : -1]
    elif "generated_responses" in input_p.parts:
        idx = input_p.parts.index("generated_responses")
        tail_parts = input_p.parts[idx + 1 : -1]
    else:
        # fallback to last two folders
        tail_parts = input_p.parts[-3:-1]

    model_dir = _sanitize_model_dir(model_name)
    out_dir = Path(
        "data/generated_responses/reddit_comments/re_evals"
    ) / model_dir / Path(*tail_parts)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = input_p.name
    if filename.endswith("comments.json"):
        out_filename = filename.replace("comments.json", "comments_rse.json")
    else:
        out_filename = f"{input_p.stem}_rse{input_p.suffix}"

    return out_dir / out_filename


def main() -> None:
    for model in models:
        for i, input_file in enumerate(files):
            output_path = _build_output_path(input_file, model)
            cmd = [
                "python",
                "-m",
                "rebuttals.rse_llm_study.measure_rse",
                "--input-comments-path",
                input_file,
                "--model-semantic-entropy",
                model,
                "--output-path",
                str(output_path),
            ]

            print("Running:", " ".join(shlex.quote(part) for part in cmd))

            assert 0 <= i <= 7, "We only have 8 GPUs"
            try:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(i)
                subprocess.run(cmd, check=True, env=env)
            except subprocess.CalledProcessError as exc:
                print(f"Command failed with exit code {exc.returncode}")


if __name__ == "__main__":
    main()