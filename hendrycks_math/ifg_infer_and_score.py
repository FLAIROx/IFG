"""Eval Pass@K for math problems with and without KW entropy injection."""

import collections
import dataclasses
import datetime
import functools
import json
import logging
import os
import queue
import sys
import typing
from typing import Optional

import datasets
import dotenv
import tqdm
import transformers
import vllm

import gllm
from hendrycks_math.utils import config_utils
from hendrycks_math import solve_and_eval_lib
from hendrycks_math.utils.consts import Filenames, GenJsonKeys
from hendrycks_math.utils.math_types import (
    AllExternalHosts,
    GradedSolutionAttempt,
    ModelTypes,
    ProblemQueueEntry,
    ModelTypes,
)
from hendrycks_math.threaded_worker_lib import MultiThreadedIFGSampler

dotenv.load_dotenv()


@dataclasses.dataclass(kw_only=True)
class IfgEvalConfig:
    output_dir: Optional[str] = None
    prompt_keywords_path: str
    num_problems: Optional[int] = None
    num_attempts: int
    model: Optional[str] = None
    model_type: str
    temperature_even_index: Optional[float] = None
    temperature_odd_index: Optional[float] = None
    global_temperature: Optional[float] = None
    # If global_temperature is set, it will override the
    # temperature_even_index and temperature_odd_index
    separator: str
    solution_end: str
    dataset: Optional[str] = None
    split: Optional[str] = None
    seed: int = 42
    max_steps: int
    max_tokens_per_step: int
    engine: str
    gllm_host: Optional[str] = None
    gllm_load_model: bool = False
    judge_gllm_host: Optional[str] = None
    judge_gllm_load_model: bool = False
    judge_gllm_model_identifier: Optional[str] = None
    evaluator: str
    num_workers: int
    log_every: int
    pre_existing_files: list[str] = dataclasses.field(default_factory=list)
    """Files that can already exist in the output directory and 
    should not be ignored when checking if the output directory is empty.
    """

    def __post_init__(self):
        # Note due to config_utils.tyro_cli_with_yaml_support works
        # we cannot put assertions in the __post_init__ method
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


EvaluatedProblem = typing.NamedTuple(
    "EvaluatedProblem",
    [
        ("problem", str),
        ("attempts", list[str]),
        ("is_correct", list[bool]),
        ("prompt", str),
        ("reference_answer", str),
    ],
)


def compute_performance_metrics(
    attempted_problems: list[EvaluatedProblem], num_attempts: int, verbose: bool = True
) -> tuple[int, int]:
    """Compute the Pass@K and Pass@1 majority metrics."""
    passed_at_k = sum(
        any(evaluated_problem.is_correct) for evaluated_problem in attempted_problems
    )
    passed_at_1_maj = sum(
        sum(evaluated_problem.is_correct) >= num_attempts / 2
        for evaluated_problem in attempted_problems
    )
    problems_attempted = len(attempted_problems)
    if verbose:
        logging.info(f"Pass@{num_attempts}: {passed_at_k} / {problems_attempted}")
        logging.info(f"Pass@1 majority: {passed_at_1_maj} / {problems_attempted}")
    return passed_at_k, passed_at_1_maj


def aggregate_solution_attempts(
    solutions_iterable: typing.Iterable[GradedSolutionAttempt],
    problems: typing.Mapping[int, str],
    num_attempts: int,
    log_every: int = 1,
) -> list[EvaluatedProblem]:
    """Group together solution attempts for each problem.

    Args:
        solutions_iterable: An iterable of GradedSolutionAttempt objects. We
            expect that each problem has num_attempts solution attempts.
        problems: A mapping from problem ID to problem text.
        num_attempts: The number of solution attempts per problem.
        log_every: Log progress every log_every problems. Set to 0 to disable.
    Returns:
        A list of EvaluatedProblem objects, each of which contains all the
        solution attempts for a single problem.
    """
    problems_with_all_attempts = []
    attempts_for_problem_id = collections.defaultdict(list)

    cur_pbar_size = 0
    n_solved = 0

    last_logged_at = -1
    pbar = tqdm.tqdm(total=len(problems))
    for graded_attempt in solutions_iterable:
        attempts_for_problem_id[graded_attempt.problem_id].append(graded_attempt)

        to_delete = []
        for problem_id, graded_solutions in attempts_for_problem_id.items():
            if len(graded_solutions) == num_attempts:
                evaluated_problem = EvaluatedProblem(
                    problem=problems[problem_id],
                    attempts=[
                        gs.solution_attempt.solution_attempt for gs in graded_solutions
                    ],
                    is_correct=[gs.is_correct for gs in graded_solutions],
                    prompt=graded_solutions[0].solution_attempt.prompt,
                    reference_answer=graded_solutions[0].reference_answer,
                )
                problems_with_all_attempts.append(evaluated_problem)
                to_delete.append(problem_id)

        for key in to_delete:
            del attempts_for_problem_id[key]

        # Update progress bar
        n_solved = len(problems_with_all_attempts)
        if cur_pbar_size != n_solved:
            pbar.update(n_solved - cur_pbar_size)
            cur_pbar_size = n_solved

        if n_solved - last_logged_at >= log_every and log_every > 0:
            compute_performance_metrics(
                problems_with_all_attempts, num_attempts=num_attempts, verbose=True
            )
            last_logged_at = n_solved

    pbar.close()
    return problems_with_all_attempts


def write_results_to_disk(
    problems_with_all_attempts: list[EvaluatedProblem],
    solutions: list[str],
    passed_at_k: int,
    passed_at_1_maj: int,
    cfg: IfgEvalConfig,
):
    """Write the results of the evaluation to disk.

    Args:
        problems_with_all_attempts: A list of EvaluatedProblem objects, each of
            which contains all the solution attempts for a single problem.
        solutions: A list of reference solutions.
        passed_at_k: The number of problems for which at least one solution
            attempt was correct.
        passed_at_1_maj: The number of problems for which the majority of
            solution attempts were correct.
        cfg: The configuration object used for the evaluation. Used to determine
            the output directory. It is also logged to disk.
    """
    assert cfg.output_dir is not None
    with open(os.path.join(cfg.output_dir, Filenames.SOLUTIONS), "w") as f:
        solution_attempts = [
            {
                GenJsonKeys.PROBLEM: evaluated_problem.problem,
                GenJsonKeys.REFERENCE_ANSWER: evaluated_problem.reference_answer,
                GenJsonKeys.ATTEMPTS: evaluated_problem.attempts,
                GenJsonKeys.IS_CORRECT: evaluated_problem.is_correct,
            }
            for evaluated_problem in problems_with_all_attempts
        ]
        json.dump(solution_attempts, f, indent=4)

    with open(os.path.join(cfg.output_dir, Filenames.GEN_CONFIG), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)

    with open(os.path.join(cfg.output_dir, Filenames.PROMPTS), "w") as f:
        json.dump(
            [problem.prompt for problem in problems_with_all_attempts],
            f,
            indent=4,
        )
    with open(os.path.join(cfg.output_dir, Filenames.RESULTS), "w") as f:
        json.dump(
            {
                "Pass@k": passed_at_k,
                "Pass@1_majority": passed_at_1_maj,
                "total_problems": len(problems_with_all_attempts),
            },
            f,
            indent=4,
        )
    with open(os.path.join(cfg.output_dir, Filenames.REFERENCE_SOLUTIONS), "w") as f:
        json.dump(solutions, f, indent=4)

    with open(os.path.join(cfg.output_dir, "command.sh"), "w") as f:
        command = " ".join([sys.executable] + sys.argv)
        f.write("#!/bin/bash\n")
        f.write(f"{command}\n")


def main(cfg: IfgEvalConfig, dataset: Optional[datasets.Dataset] = None):
    logging.info("Starting evaluation")
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

    judge_model = judge_tokenizer = None

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

    if cfg.evaluator == "math":
        logging.info("Using MATH for evaluation")
        solutions = dataset["solution"]
        evaluation_fn = solve_and_eval_lib.is_correct_math

    elif cfg.evaluator == "true":
        logging.info(
            "Using true for evaluation."
            "All generated solutions are assumed to be correct."
        )
        solutions = dataset["solution"]
        evaluation_fn = lambda *args, **kwargs: True
    else:
        raise ValueError()

    logging.info("Loaded dataset %s", cfg.dataset)
    logging.info("Dataset %s", dataset)
    logging.info("Dataset size %d", len(dataset))
    logging.info("Dataset columns %s", dataset.column_names)
    problems = dataset["problem"]

    with open(cfg.prompt_keywords_path) as f:
        if cfg.prompt_keywords_path.endswith(".json"):
            prompt_template = json.load(f)
        else:
            assert cfg.prompt_keywords_path.endswith(".txt")
            prompt_template = f.read()

    # Place each problem in the queue for the workers to solve.
    # Each problem is placed in the queue num_attempts times.
    problem_queue = queue.Queue()
    for i, (problem, solution) in enumerate(zip(problems, solutions)):
        for _ in range(cfg.num_attempts):
            problem_queue.put(
                ProblemQueueEntry(
                    id=i,
                    problem=problem,
                    reference_answer=solution,
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
            problems,  # type: ignore
            num_attempts=cfg.num_attempts,
            log_every=cfg.log_every,
        )
    finally:
        multi_threaded_solver.close()

    passed_at_k, passed_at_1_maj = compute_performance_metrics(
        problems_with_all_attempts, num_attempts=cfg.num_attempts, verbose=True
    )

    # Write results to disk.
    logging.info("Writing results to disk to %s", cfg.output_dir)
    write_results_to_disk(
        problems_with_all_attempts=problems_with_all_attempts,
        solutions=solutions,
        passed_at_k=passed_at_k,
        passed_at_1_maj=passed_at_1_maj,
        cfg=cfg,
    )


if __name__ == "__main__":
    dotenv.load_dotenv()
    cfg = config_utils.tyro_cli_with_yaml_support(IfgEvalConfig)
    config_utils.prepare_output_dir(cfg.output_dir, cfg.pre_existing_files)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(cfg.output_dir, f"run-math-eval-{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )
    logging.info("Config: %s", cfg)
    main(cfg)
