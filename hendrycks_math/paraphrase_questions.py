"""Generate N paraphrases of each question in a maths dataset.

This is the first half of the "paraphrase pass@N" pipeline: instead of
sampling a single question N times (as ifg_infer_and_score.py does to
compute Pass@K), we reword each question N times. The resulting
paraphrases are consumed by paraphrase_infer_and_score.py, which samples
a single response per paraphrase and reports Pass@N over the paraphrases.
"""

import dataclasses
import datetime
import json
import logging
import os
import queue
import sys
from typing import Optional

import datasets
import dotenv
import tqdm
import transformers
import vllm

import gllm
from hendrycks_math.ifg_infer_and_score import aggregate_solution_attempts
from hendrycks_math.utils import config_utils
from hendrycks_math.utils.consts import Filenames, GenJsonKeys
from hendrycks_math.utils.math_types import (
    AllExternalHosts,
    ModelTypes,
    ProblemQueueEntry,
)
from hendrycks_math.threaded_worker_lib import MultiThreadedIFGSampler

dotenv.load_dotenv()


@dataclasses.dataclass(kw_only=True)
class ParaphraseConfig:
    output_dir: Optional[str] = None
    prompt_keywords_path: str
    num_problems: Optional[int] = None
    num_paraphrases: int
    model: Optional[str] = None
    model_type: str
    temperature: float
    dataset: Optional[str] = None
    split: Optional[str] = None
    separator: str
    solution_end: str
    seed: int = 42
    max_tokens: int
    engine: str
    gllm_host: Optional[str] = None
    gllm_load_model: bool = False
    num_workers: int
    log_every: int
    pre_existing_files: list[str] = dataclasses.field(default_factory=list)
    """Files that can already exist in the output directory and
    should not be ignored when checking if the output directory is empty.
    """

    def __post_init__(self):
        assert self.model_type in [
            ModelTypes.BASE.value,
            ModelTypes.CHAT.value,
            ModelTypes.MATHSTRAL.value,
        ]


def write_results_to_disk(
    problems: list[dict], cfg: ParaphraseConfig
):
    """Write the generated paraphrases and run metadata to disk."""
    assert cfg.output_dir is not None
    with open(os.path.join(cfg.output_dir, Filenames.PARAPHRASES), "w") as f:
        json.dump(problems, f, indent=4)

    with open(os.path.join(cfg.output_dir, Filenames.GEN_CONFIG), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)

    with open(os.path.join(cfg.output_dir, "command.sh"), "w") as f:
        command = " ".join([sys.executable] + sys.argv)
        f.write("#!/bin/bash\n")
        f.write(f"{command}\n")


def main(cfg: ParaphraseConfig, dataset: Optional[datasets.Dataset] = None):
    logging.info("Starting paraphrase generation")
    logging.info("Config: %s", cfg)
    assert cfg.output_dir is not None
    assert cfg.model is not None
    if cfg.engine == "vllm":
        logging.info("Using VLLM for paraphrase model")
        logging.info(f"Model: {cfg.model}")
        model = vllm.LLM(cfg.model)
    elif cfg.engine == "gllm":
        logging.info("Using GLLM for paraphrase model")
        logging.info(f"Model: {cfg.model}")
        logging.info(f"Host: {cfg.gllm_host}")
        api_key = os.getenv("OPENAI_API_KEY", None)
        assert cfg.gllm_host is not None
        model = gllm.GLLM(cfg.gllm_host, api_key=api_key)
        if cfg.gllm_load_model:
            model.load_model(cfg.model)
    else:
        raise ValueError()

    model_tokenizer = None
    if cfg.model_type == ModelTypes.MATHSTRAL:
        model_tokenizer = transformers.AutoTokenizer.from_pretrained(
            cfg.model, trust_remote_code=True
        )

    if cfg.engine == "gllm":
        assert isinstance(model, gllm.GLLM)
        if cfg.gllm_host not in AllExternalHosts:
            model.wait_for_health()
        logging.info("Model is healthy")

    if dataset is None:
        assert cfg.dataset is not None
        assert cfg.split is not None
        logging.info("Using dataset %s", cfg.dataset)
        logging.info("Using split %s", cfg.split)
        dataset = datasets.load_dataset(  # type: ignore
            cfg.dataset, split=cfg.split, trust_remote_code=True
        )
        assert dataset is not None
        dataset = dataset.shuffle(seed=cfg.seed)

        if cfg.num_problems is None:
            logging.info("Using all problems in the dataset")
        else:
            logging.info("Using %d problems from the dataset", cfg.num_problems)
            dataset = dataset.select(range(cfg.num_problems))
    else:
        logging.info("Using provided dataset, ignoring supplied num_problems and path")

    logging.info("Loaded dataset %s", cfg.dataset)
    logging.info("Dataset size %d", len(dataset))
    problems = list(dataset["problem"])
    solutions = list(dataset["solution"])

    with open(cfg.prompt_keywords_path) as f:
        if cfg.prompt_keywords_path.endswith(".json"):
            prompt_template = json.load(f)
        else:
            assert cfg.prompt_keywords_path.endswith(".txt")
            prompt_template = f.read()

    # Place each problem in the queue num_paraphrases times so that
    # num_paraphrases independently sampled rewordings are produced for it.
    problem_queue = queue.Queue()
    for i, (problem, solution) in tqdm.tqdm(
        enumerate(zip(problems, solutions)), total=len(problems), desc="Enqueuing problems"
    ):
        for _ in range(cfg.num_paraphrases):
            problem_queue.put(
                ProblemQueueEntry(
                    id=i,
                    problem=problem,
                    reference_answer=solution,
                    reference_solution="Not Implemented",
                )
            )

    # Paraphrasing has no notion of correctness, every generation is accepted.
    always_accept = lambda **kwargs: True

    multi_threaded_solver = MultiThreadedIFGSampler(
        n_workers=cfg.num_workers,
        problem_queue=problem_queue,
        model=model,
        model_name=cfg.model,
        model_type=cfg.model_type,
        model_tokenizer=model_tokenizer,
        prompt_template=prompt_template,
        max_tokens_per_step=cfg.max_tokens,
        temperature_even_index=cfg.temperature,
        temperature_odd_index=None,
        max_n_steps=1,
        solution_end=cfg.solution_end,
        step_separator=cfg.separator,
        evaluation_fn=always_accept,
    )

    multi_threaded_solver.start()

    try:
        problems_with_all_paraphrases = aggregate_solution_attempts(
            multi_threaded_solver,
            problems,  # type: ignore
            num_attempts=cfg.num_paraphrases,
            log_every=cfg.log_every,
        )
    finally:
        multi_threaded_solver.close()

    output_records = [
        {
            GenJsonKeys.PROBLEM: evaluated_problem.problem,
            GenJsonKeys.REFERENCE_ANSWER: evaluated_problem.reference_answer,
            GenJsonKeys.PARAPHRASES: evaluated_problem.attempts,
        }
        for evaluated_problem in problems_with_all_paraphrases
    ]

    logging.info("Writing %d paraphrased problems to %s", len(output_records), cfg.output_dir)
    write_results_to_disk(output_records, cfg)


if __name__ == "__main__":
    dotenv.load_dotenv()
    cfg = config_utils.tyro_cli_with_yaml_support(ParaphraseConfig)
    config_utils.prepare_output_dir(cfg.output_dir, cfg.pre_existing_files)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(cfg.output_dir, f"run-paraphrase-{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Config: %s", cfg)
    main(cfg)
