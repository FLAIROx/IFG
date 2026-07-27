"""Subsample a fixed random subset of questions from a maths dataset.

Writes the subset to disk in the layout `datasets.load_dataset` expects, so
it can be dropped straight into any existing config by pointing `dataset:`
at `output_dir` and `split:` at `split_name` -- no changes needed to the
scripts that consume it (ifg_infer_and_score.py, paraphrase_questions.py,
star.py, etc.).
"""

import dataclasses
import logging
import os

import datasets
import tyro


@dataclasses.dataclass(kw_only=True)
class SubsampleConfig:
    output_dir: str
    source_dataset: str = "JeremiahZ/hendrycks_math_merged"
    source_split: str = "test"
    split_name: str = "test"
    """Name of the split written to output_dir. Scripts that load the
    subsampled dataset should set their `split` config field to this value.
    """
    num_problems: int = 500
    seed: int = 42


def main(cfg: SubsampleConfig):
    logging.info("Loading %s split=%s", cfg.source_dataset, cfg.source_split)
    dataset = datasets.load_dataset(
        cfg.source_dataset, split=cfg.source_split, trust_remote_code=True
    )
    assert dataset is not None
    assert cfg.num_problems <= len(dataset), (
        f"Requested {cfg.num_problems} problems but the source split only has "
        f"{len(dataset)}."
    )

    dataset = dataset.shuffle(seed=cfg.seed).select(range(cfg.num_problems))

    data_dir = os.path.join(cfg.output_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, f"{cfg.split_name}-00000-of-00001.parquet")
    dataset.to_parquet(output_path)

    logging.info(
        "Wrote %d problems to %s (split=%s)", len(dataset), output_path, cfg.split_name
    )
    logging.info(
        "Point configs at this subset with: dataset: %s, split: %s",
        cfg.output_dir,
        cfg.split_name,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    cfg = tyro.cli(SubsampleConfig)
    main(cfg)
