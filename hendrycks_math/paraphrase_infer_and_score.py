"""Compute Pass@N over paraphrases of maths questions.

Second half of the "paraphrase pass@N" pipeline. Where ifg_infer_and_score.py
computes Pass@N by sampling the *same* question N times, this script samples
a single response for each of the N paraphrases produced by
paraphrase_questions.py, groups the responses back by their original
question, and reports Pass@N once.
"""

import dataclasses
import datetime
import json
import logging
import os
import queue
import sys
from typing import Optional

import dotenv
import tqdm
import transformers
import vllm

import gllm
from hendrycks_math import solve_and_eval_lib
from hendrycks_math.ifg_infer_and_score import (
    aggregate_solution_attempts,
    compute_performance_metrics,
    write_results_to_disk,
)
from hendrycks_math.utils import config_utils
from hendrycks_math.utils.consts import GenJsonKeys
from hendrycks_math.utils.math_types import (
    AllExternalHosts,
    ModelTypes,
    ProblemQueueEntry,
)
from hendrycks_math.threaded_worker_lib import MultiThreadedIFGSampler

dotenv.load_dotenv()


@dataclasses.dataclass(kw_only=True)
class ParaphraseEvalConfig:
    output_dir: Optional[str] = None
    paraphrases_path: str
    """Path to the paraphrases.json file written by paraphrase_questions.py."""
    num_paraphrases_to_use: Optional[int] = None
    """If set, only the first num_paraphrases_to_use paraphrases of each
    question are used, so Pass@N is reported for N=num_paraphrases_to_use
    instead of N=the full number of paraphrases generated.
    """
    prompt_keywords_path: str
    model: Optional[str] = None
    model_type: str
    temperature_even_index: Optional[float] = None
    temperature_odd_index: Optional[float] = None
    global_temperature: Optional[float] = None
    # If global_temperature is set, it will override the
    # temperature_even_index and temperature_odd_index
    separator: str
    solution_end: str
    max_steps: int
    max_tokens_per_step: int
    engine: str
    gllm_host: Optional[str] = None
    gllm_load_model: bool = False
    evaluator: str
    num_workers: int
    log_every: int
    pre_existing_files: list[str] = dataclasses.field(default_factory=list)
    """Files that can already exist in the output directory and
    should not be ignored when checking if the output directory is empty.
    """

    def __post_init__(self):
        if self.global_temperature is not None:
            self.temperature_even_index = self.global_temperature
            self.temperature_odd_index = self.global_temperature

        assert self.evaluator in [
            "math",
        ], "Evaluator must be math."
        assert self.model_type in [
            ModelTypes.BASE.value,
            ModelTypes.CHAT.value,
            ModelTypes.MATHSTRAL.value,
        ]


def main(cfg: ParaphraseEvalConfig):
    logging.info("Starting paraphrase Pass@N evaluation")
    logging.info("Config: %s", cfg)
    assert cfg.output_dir is not None
    assert cfg.model is not None
    if cfg.engine == "vllm":
        logging.info("Using VLLM for policy")
        logging.info(f"Model: {cfg.model}")
        model = vllm.LLM(cfg.model)
    elif cfg.engine == "gllm":
        logging.info("Using GLLM for policy")
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

    if cfg.evaluator == "math":
        logging.info("Using MATH for evaluation")
        evaluation_fn = solve_and_eval_lib.is_correct_math
    else:
        raise ValueError()

    logging.info("Loading paraphrases from %s", cfg.paraphrases_path)
    with open(cfg.paraphrases_path) as f:
        paraphrased_problems = json.load(f)
    assert paraphrased_problems, "No paraphrased problems found."

    num_paraphrases = len(paraphrased_problems[0][GenJsonKeys.PARAPHRASES])
    for entry in paraphrased_problems:
        assert len(entry[GenJsonKeys.PARAPHRASES]) == num_paraphrases, (
            "Every problem must have the same number of paraphrases, found "
            f"{len(entry[GenJsonKeys.PARAPHRASES])} and {num_paraphrases}."
        )
    logging.info(
        "Loaded %d problems with %d paraphrases each",
        len(paraphrased_problems),
        num_paraphrases,
    )

    if cfg.num_paraphrases_to_use is not None:
        assert cfg.num_paraphrases_to_use <= num_paraphrases, (
            f"Requested {cfg.num_paraphrases_to_use} paraphrases per question but "
            f"only {num_paraphrases} were generated."
        )
        num_paraphrases = cfg.num_paraphrases_to_use
        logging.info("Using only the first %d paraphrases of each question", num_paraphrases)
    solutions = [entry[GenJsonKeys.REFERENCE_ANSWER] for entry in paraphrased_problems]
    problems = {i: entry[GenJsonKeys.PROBLEM] for i, entry in enumerate(paraphrased_problems)}

    with open(cfg.prompt_keywords_path) as f:
        if cfg.prompt_keywords_path.endswith(".json"):
            prompt_template = json.load(f)
        else:
            assert cfg.prompt_keywords_path.endswith(".txt")
            prompt_template = f.read()

    # Each paraphrase is sampled exactly once. All paraphrases of a given
    # problem share that problem's id so that aggregate_solution_attempts
    # groups them together for the Pass@N computation.
    problem_queue = queue.Queue()
    for i, entry in tqdm.tqdm(
        enumerate(paraphrased_problems),
        total=len(paraphrased_problems),
        desc="Enqueuing paraphrases",
    ):
        for paraphrase in entry[GenJsonKeys.PARAPHRASES][:num_paraphrases]:
            problem_queue.put(
                ProblemQueueEntry(
                    id=i,
                    problem=paraphrase,
                    reference_answer=entry[GenJsonKeys.REFERENCE_ANSWER],
                    reference_solution="Not Implemented",
                )
            )

    assert cfg.temperature_even_index is not None
    multi_threaded_solver = MultiThreadedIFGSampler(
        n_workers=cfg.num_workers,
        problem_queue=problem_queue,
        model=model,
        model_name=cfg.model,
        model_type=cfg.model_type,
        model_tokenizer=model_tokenizer,
        prompt_template=prompt_template,
        max_tokens_per_step=cfg.max_tokens_per_step,
        temperature_even_index=cfg.temperature_even_index,
        temperature_odd_index=cfg.temperature_odd_index,
        max_n_steps=cfg.max_steps,
        solution_end=cfg.solution_end,
        step_separator=cfg.separator,
        evaluation_fn=evaluation_fn,
    )

    multi_threaded_solver.start()

    try:
        problems_with_all_attempts = aggregate_solution_attempts(
            multi_threaded_solver,
            problems,
            num_attempts=num_paraphrases,
            log_every=cfg.log_every,
        )
    finally:
        multi_threaded_solver.close()

    passed_at_k, passed_at_1_maj = compute_performance_metrics(
        problems_with_all_attempts, num_attempts=num_paraphrases, verbose=True
    )

    logging.info("Writing results to disk to %s", cfg.output_dir)
    # write_results_to_disk only relies on cfg.output_dir and treats cfg as
    # an opaque dataclass to dump alongside the results, so it is reused
    # as-is from ifg_infer_and_score.py.
    write_results_to_disk(
        problems_with_all_attempts=problems_with_all_attempts,
        solutions=solutions,
        passed_at_k=passed_at_k,
        passed_at_1_maj=passed_at_1_maj,
        cfg=cfg,  # type: ignore[arg-type]
    )


if __name__ == "__main__":
    dotenv.load_dotenv()
    cfg = config_utils.tyro_cli_with_yaml_support(ParaphraseEvalConfig)
    config_utils.prepare_output_dir(cfg.output_dir, cfg.pre_existing_files)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(cfg.output_dir, f"run-paraphrase-eval-{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Config: %s", cfg)
    main(cfg)
