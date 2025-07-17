import copy
import dataclasses
import datetime
import json
import logging
import os
import subprocess
from typing import Optional

import datasets
import dotenv
import wandb
import yaml

import gllm
from hendrycks_math.utils import config_utils
from hendrycks_math import ifg_infer_and_score
from hendrycks_math.utils.consts import Filenames, GenJsonKeys, TrainingJsonKeys
from training import finetune_completion


@dataclasses.dataclass(kw_only=True)
class StarConfig:
    # General configuration
    output_dir: str
    train_dataset_path: str
    train_dataset_split: str
    n_problems_per_train_iter: Optional[int] = (
        None  # Set to None to use the entire dataset
    )

    eval_dataset_path: Optional[str] = None
    eval_dataset_split: Optional[str] = None
    num_eval_problems: Optional[int] = None  # Set to None to use the entire dataset

    dataset_name: Optional[str] = (
        None  # The name argument for huggingface datasets.load_dataset
    )
    base_model: str

    seed: int
    num_iterations: int

    # Generation Args
    generation_config: ifg_infer_and_score.IfgEvalConfig

    # Training Args
    training_config: finetune_completion.TrainingConfig
    train_accelerate_conf_path: Optional[str] = None

    # Evaluation Args
    eval_config: ifg_infer_and_score.IfgEvalConfig

    # Misc Args
    wandb_project: str
    wandb_run_name: Optional[str] = None
    terminate_on_empty_data: bool = False


def generate_subpaths(base_dir: str, iteration: int) -> dict[str, str]:
    """Generate paths for relavant subdirs for the current iteration."""
    iter_base_path = os.path.join(base_dir, f"iteration_{iteration}/")
    paths = {}
    paths["iter_base"] = iter_base_path
    paths["generations"] = os.path.join(iter_base_path, "generations/")
    paths["filtered"] = os.path.join(iter_base_path, "filtered/")
    paths["aggregate"] = os.path.join(iter_base_path, "aggregate/")
    paths["checkpoints"] = os.path.join(iter_base_path, "checkpoints/")
    paths["final_checkpoint"] = os.path.join(paths["checkpoints"], "final/")
    paths["eval"] = os.path.join(iter_base_path, "eval/")
    paths["eval_on_start"] = os.path.join(iter_base_path, "eval_on_start/")

    return paths


def run_generation_step(
    output_dir: str,
    eval_config: ifg_infer_and_score.IfgEvalConfig,
    dataset: datasets.Dataset,
    iteration: int = 0,
) -> str:
    """Run a generation step on a model checkpoint.
    Args:
        output_dir (str): Path to the base output directory (which contains all iterations).
            Overrides the output_dir specified in the eval_config if present.
        eval_config (run_maths_eval.IfgEvalConfig): Evaluation configuration.
        dataset (datasets.Dataset): Training dataset.
        iteration (int): Index of the current iteration.
    Returns:
        str: Path to the generated solutions.
    """
    logging.info("Commencing generation step.")
    eval_config = copy.deepcopy(eval_config)

    generations_dir = generate_subpaths(output_dir, iteration)["generations"]

    config_utils.prepare_output_dir(generations_dir)
    eval_config.output_dir = generations_dir

    logging.info(f"Running generation step with model {eval_config.model}")
    logging.info(f"Output directory: {generations_dir}")

    ifg_infer_and_score.main(eval_config, dataset)
    return generations_dir


def filter_generations(generations_dir: str, output_dir: str) -> tuple[int, int, int]:
    """Filter generations keeping only correct solutions.

    Args:
        generations_dir (str): Path to the directory containing generations.
        output_dir (str): Directory to save the filtered generations.
    Returns:
        tuple[int, int, int]: Number of individual correct solutions, number of problems with
            correct solutions, and total number of problems.
    """
    prompt_filepath = os.path.join(generations_dir, Filenames.PROMPTS)
    generation_filepath = os.path.join(generations_dir, Filenames.SOLUTIONS)
    with open(prompt_filepath, "r") as f:
        prompts = json.load(f)
    with open(generation_filepath, "r") as f:
        generations = json.load(f)
    correct_solutions = []
    prompts_with_correct_solutions = []

    for prompt, generation in zip(prompts, generations):
        # Initialize new entry with fixed fields and no solutio attempts.
        new_entry = {
            GenJsonKeys.PROBLEM: generation[GenJsonKeys.PROBLEM],
            GenJsonKeys.REFERENCE_ANSWER: generation[GenJsonKeys.REFERENCE_ANSWER],
            GenJsonKeys.ATTEMPTS: [],
            GenJsonKeys.IS_CORRECT: [],
        }

        # Iterate over individual solution attempts and keep only correct ones.
        for attempt, is_correct in zip(
            generation[GenJsonKeys.ATTEMPTS],
            generation[GenJsonKeys.IS_CORRECT],
        ):
            if is_correct:
                new_entry[GenJsonKeys.ATTEMPTS].append(attempt)
                new_entry[GenJsonKeys.IS_CORRECT].append(True)

        if new_entry[GenJsonKeys.ATTEMPTS]:
            correct_solutions.append(new_entry)
            prompts_with_correct_solutions.append(prompt)
    logging.info(
        f"Found correct solution for %d problems out of %d.",
        len(correct_solutions),
        len(prompts),
    )
    num_correct_solutions = sum(
        len(entry[GenJsonKeys.ATTEMPTS]) for entry in correct_solutions
    )
    logging.info(f"Found %s individual correct solutions.", num_correct_solutions)

    correct_solutions_path = os.path.join(output_dir, Filenames.SOLUTIONS)
    prompts_with_correct_solutions_path = os.path.join(output_dir, Filenames.PROMPTS)

    logging.info(f"Writing correct solutions to {correct_solutions_path}")
    with open(correct_solutions_path, "w") as f:
        json.dump(correct_solutions, f)

    logging.info(
        f"Writing prompts with correct solutions to {prompts_with_correct_solutions_path}"
    )
    with open(prompts_with_correct_solutions_path, "w") as f:
        json.dump(prompts_with_correct_solutions, f)

    return num_correct_solutions, len(correct_solutions), len(prompts)


def prepare_training_data(prompts: list[str], solutions: list[dict]) -> list[dict]:
    """Format training data as list of training examples.

    This involves prepending corresponding the prompt to each solution attempt.
    Args:
        prompts (list[str]): List of prompts.
        solutions (list[dict]): List of solutions. Each solution might consist of
            multiple attempts.
    """
    training_data = []
    for prompt, attempts in zip(prompts, solutions):
        for attempt in attempts[GenJsonKeys.ATTEMPTS]:
            training_data.append(
                {
                    TrainingJsonKeys.PROMPT: prompt,
                    TrainingJsonKeys.RESPONSE: attempt,
                }
            )
    return training_data


def aggregate_training_data(
    base_output_dir: str, iteration_index: int
) -> tuple[str, int]:
    """Aggregate the correct answers from all generations and format as training file.

    Args:
        base_output_dir (str): Path to the base output directory.
        iteration_index (int): Index of the current iteration.
    Returns:
        path (str) : Path to the aggregated training data. It is a jsonl file.
        num_entries (int): Number of entries in the aggregated training data.
    """
    filtered_dirs = [
        generate_subpaths(base_output_dir, i)["filtered"]
        for i in range(iteration_index + 1)
    ]
    prompts = []
    solutions = []

    # Load and concatenate correct prompts across all iterations.
    for filtered_dir in filtered_dirs:
        prompts_path = os.path.join(filtered_dir, Filenames.PROMPTS)
        solutions_path = os.path.join(filtered_dir, Filenames.SOLUTIONS)
        with open(prompts_path, "r") as f:
            prompts.extend(json.load(f))
        with open(solutions_path, "r") as f:
            solutions.extend(json.load(f))

    # Write the aggregated data to a jsonl file.
    training_data = prepare_training_data(prompts, solutions)

    aggregate_path = generate_subpaths(base_output_dir, iteration_index)["aggregate"]
    os.makedirs(aggregate_path, exist_ok=True)
    filepath = os.path.join(aggregate_path, Filenames.AGGREGATE)
    logging.info(
        "Writing aggregated training data consisting of %d datapoints to %s",
        len(training_data),
        filepath,
    )
    with open(filepath, "w") as f:
        for record in training_data:
            f.write(json.dumps(record) + "\n")

    return filepath, len(training_data)


def run_training_step(
    output_dir: str,
    iteration: int,
    train_cfg: finetune_completion.TrainingConfig,
    training_file: str,
    accelerate_conf_path: Optional[str],
) -> str:
    """Run a training_step on the aggregated training data.
    Args:
        output_dir (str): Path to the base output directory (which contains all iterations)
        iteration (int): Index of the current iteration.
        train_cfg (finetune_completion.TrainingConfig): Training configuration.
        training_file (str): Path to the aggregated training data.
        accelerate_conf_path (Optional[str]): Path to the accelerate config file.
    Returns:
        str: Path to the trained model.
    """
    train_output_dir = generate_subpaths(output_dir, iteration)["checkpoints"]

    os.makedirs(train_output_dir, exist_ok=True)

    train_cfg = copy.deepcopy(train_cfg)
    train_cfg.output_dir = train_output_dir
    train_cfg.pre_existing_files = [Filenames.TRAIN_CONFIG.value]
    train_cfg.train_dataset = training_file
    train_cfg.valid_dataset = ""

    train_cfg.seed = train_cfg.seed + iteration

    assert train_cfg.run_name is not None
    train_cfg.run_name += f"_iter_{iteration}"

    train_cfg_path = os.path.join(train_output_dir, Filenames.TRAIN_CONFIG)

    # Write the training config to a yaml file.
    with open(train_cfg_path, "w") as f:
        yaml.dump(dataclasses.asdict(train_cfg), f)

    command = [
        "accelerate",
        "launch",
        finetune_completion.__file__,
        "--yaml",
        train_cfg_path,
    ]
    if accelerate_conf_path is not None:
        # Insert the accelerate config arg after launch
        # and before the script path.
        command[2:2] = ["--config_file", accelerate_conf_path]

    logging.info("Commencing training step.")
    logging.info(f"Running command: {' '.join(command)}")

    # We set check=False to allow the program to continue
    # if the training script crashes. Training currently
    # completes successfully, but crashes on cleanup/exit
    # Throws a Cuda out of Memory on torch.distributed.destroy_process_group()
    # and this does not actually negatively affect our ability to resume
    # the job. If training didn't finished then the STaR loop will crash
    # further down the line when it tries to do an eval on a non-existent
    # checkpoint.
    # Gotta love CUDA (sigh)
    subprocess.run(command, check=False)
    logging.info("Training step completed.")

    checkpoint_path = generate_subpaths(output_dir, iteration)["final_checkpoint"]
    assert os.path.exists(checkpoint_path), f"Checkpoint not found at {checkpoint_path}"
    assert os.listdir(
        checkpoint_path
    ), f"Checkpoint directory is empty: {checkpoint_path}"
    return checkpoint_path


def release_gllm_gpus(cfg: ifg_infer_and_score.IfgEvalConfig) -> None:
    """Release GPUs held by generation and judging GLLM server."""
    gllm_handles = []
    if cfg.gllm_host is not None:
        gllm_handles.append(gllm.GLLM(cfg.gllm_host))
    if cfg.judge_gllm_host is not None:
        gllm_handles.append(gllm.GLLM(cfg.judge_gllm_host))
    for handle in gllm_handles:
        logging.info("Released GPUs from %s." % handle.server_address)
        handle.release_gpus()


def run_star_iteration(
    cfg: StarConfig, dataset: datasets.Dataset, iteration: int
) -> Optional[str]:
    """Run a single iteration of the STAR algorithm.

    Args:
        cfg (StarConfig): Configuration for the current iteration.
        dataset (datasets.Dataset): Training dataset.
        iteration (int): Index of the current iteration.
    Returns:
        str: Path to the trained model. None if no training data remains
            after filtering incorrect solutions.
    """
    assert cfg.training_config.model_path is not None
    generations_dir = run_generation_step(
        output_dir=cfg.output_dir,
        dataset=dataset,
        eval_config=cfg.generation_config,
        iteration=iteration,
    )

    filtered_dir = generate_subpaths(cfg.output_dir, iteration)["filtered"]
    os.makedirs(filtered_dir, exist_ok=True)

    solutions_found, solved_problems, problems_attempted = filter_generations(
        generations_dir=generations_dir, output_dir=filtered_dir
    )
    training_file, dataset_size = aggregate_training_data(
        base_output_dir=cfg.output_dir, iteration_index=iteration
    )
    metrics = {
        "n_solutions_found": solutions_found,
        "solved_problems": solved_problems,
        "problems_attempted": problems_attempted,
        "total_dataset_size": dataset_size,
    }

    wandb.log(
        metrics,
        step=iteration + 1,
    )
    logging.info("Generation Step Metrics: %s", metrics)
    logging.info("Releasing Generation GPUs.")

    release_gllm_gpus(cfg.generation_config)

    if solved_problems == 0:
        logging.info("No new training data found. Skipping training step.")
        return
    logging.info("Commencing training.")

    assert training_file is not None
    checkpoint_path = run_training_step(
        output_dir=cfg.output_dir,
        iteration=iteration,
        train_cfg=cfg.training_config,
        training_file=training_file,
        accelerate_conf_path=cfg.train_accelerate_conf_path,
    )
    return checkpoint_path


def fetch_subset(
    dataset: datasets.Dataset, num_problems: int, i: int
) -> datasets.Dataset:
    """Fetch a i'th slice of num_problems of the dataset.

    When the dataset is exhausted, the slice wraps around to
    the beginning.
    Args:
        dataset (datasets.Dataset): The dataset to fetch from.
        num_problems (int): Number of problems to fetch.
        i (int): Index of the slice.
    Returns:
        datasets.Dataset: A subset of the dataset.
    """
    # We wrap indices with modulus to support multiple epochs.
    assert num_problems > 0, "Number of problems per iteration must be greater than 0."
    start_index = (i * num_problems) % len(dataset)
    end_index = ((i + 1) * num_problems) % len(dataset)

    if start_index < end_index:
        return dataset.select(range(start_index, end_index))
    # Handle wrap around case when the slice goes beyond the
    # end of the dataset. We wrap aound to the beginning.
    else:
        tail_half = dataset.select(range(start_index, len(dataset)))
        head_half = dataset.select(range(0, end_index))
        return datasets.concatenate_datasets([tail_half, head_half])


def run_evaluation(
    model_to_eval: str,
    dataset: datasets.Dataset,
    eval_config: ifg_infer_and_score.IfgEvalConfig,
    output_dir: str,
) -> dict[str, int | float]:
    """Run an evaluation of the model on a dataset.

    Args:
        model (str): Path to the model to evaluate.
        dataset (datasets.Dataset): Dataset to evaluate on.
        eval_config (run_maths_eval.IfgEvalConfig): Evaluation configuration.
        output_dir (str): Path to the output directory to save generations
        and results.
    Returns:
        dict[str, int | float]: Results of the evaluation.
    """
    eval_config = copy.deepcopy(eval_config)
    eval_config.output_dir = output_dir  # Override output_dir in eval_config
    eval_config.model = model_to_eval

    ifg_infer_and_score.main(eval_config, dataset)
    results_file = os.path.join(output_dir, Filenames.RESULTS)
    with open(results_file, "r") as f:
        results = json.load(f)
    return results


def run_star_loop(
    cfg: StarConfig,
    train_dataset: datasets.Dataset,
    eval_dataset: Optional[datasets.Dataset],
    num_iterations: int,
    start_iteration: int = 0,
):
    assert cfg.generation_config.model is not None, "Model path must be provided."
    model_to_gen_from = cfg.generation_config.model

    # Run evaluation on the base model before starting STaR training.
    run_eval(
        cfg,
        model_to_gen_from,
        eval_dataset,
        start_iteration - 1,
        dir_type="eval_on_start",
    )

    # We start with the base model for the first iteration.
    for i in range(start_iteration, num_iterations):
        logging.info(f"Starting start iteration {i}")
        if cfg.n_problems_per_train_iter is not None:
            iteration_dataset = fetch_subset(
                train_dataset, cfg.n_problems_per_train_iter, i
            )
            logging.info("Fetching %d'th subset of size %d", i, len(iteration_dataset))
        else:
            iteration_dataset = train_dataset

        iteration_dataset = iteration_dataset.shuffle(seed=cfg.seed + i)

        logging.info(f"StaR iteration Dataset size: {len(iteration_dataset)}")
        new_checkpoint_path = run_star_iteration(cfg, iteration_dataset, i)

        if cfg.terminate_on_empty_data and new_checkpoint_path is None:
            logging.info("Terminating early due to no new training data.")
            return

        if new_checkpoint_path is not None:
            model_to_gen_from = new_checkpoint_path
        else:
            logging.info(
                "No new training data found. "
                "Continuing with the current model <%s>." % model_to_gen_from
            )

        cfg.generation_config.model = model_to_gen_from
        logging.info(f"Model to generate from for next iter: {model_to_gen_from}")
        logging.info(f"Completed iteration {i}")

        run_eval(cfg, model_to_gen_from, eval_dataset, i)

    logging.info("Training complete. Final checkpoint at %s" % model_to_gen_from)


def run_eval(
    cfg: StarConfig,
    model_to_gen_from: str,
    eval_dataset: Optional[datasets.Dataset],
    i: int,
    dir_type: str = "eval",
) -> None:
    """Run evaluation on the model."""
    if eval_dataset is None:
        logging.info("No evaluation dataset provided. Skipping evaluation.")
        return
    eval_output_dir = generate_subpaths(cfg.output_dir, i)[dir_type]

    os.makedirs(eval_output_dir, exist_ok=True)

    logging.info(f"Evaluation output directory: {eval_output_dir}")
    logging.info("Running evaluation on the model.")

    eval_results = run_evaluation(
        model_to_gen_from, eval_dataset, cfg.eval_config, eval_output_dir
    )
    logging.info(f"Eval results: {eval_results}")
    eval_results = {"eval/" + k: v for k, v in eval_results.items()}
    wandb.log(eval_results, step=i + 1)
    logging.info("Evaluation complete. %s", eval_results)
    logging.info("Released Evaluation GPUs.")


def count_previous_iterations(
    output_dir: str,
) -> tuple[int, Optional[str]]:
    """Count the number of previous iterations and return the last model path.

    Args:
        output_dir (str): Path to the base output directory.
    Returns:
        tuple[int, str]: Number of previous iterations and path to the last model.
    """
    iteration = 0
    checkpoint_path = None

    while True:
        new_checkpoint_path = generate_subpaths(output_dir, iteration)[
            "final_checkpoint"
        ]

        if not os.path.exists(new_checkpoint_path) or not os.listdir(
            new_checkpoint_path
        ):
            break
        checkpoint_path = new_checkpoint_path
        iteration += 1

    return iteration - 1, checkpoint_path


if __name__ == "__main__":
    dotenv.load_dotenv()

    cfg: StarConfig = config_utils.tyro_cli_with_yaml_support(StarConfig)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # Initalize logging.
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = os.path.join(cfg.output_dir, "star-%s.log" % timestamp)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    # Check if we are resuming from a previous run.
    directory_contents = [
        file for file in os.listdir(cfg.output_dir) if not file.endswith(".log")
    ]
    if directory_contents:
        logging.info("Found pre-existing files in output directory.")
        logging.info("Assuming that this is a resume run.")
        logging.info("Resuming from the last iteration.")
        iteration, last_model = count_previous_iterations(cfg.output_dir)
        start_iter = iteration + 1

        if start_iter == cfg.num_iterations:
            logging.info("Training was already complete")
            logging.info("Terminating star")
            quit()

        if start_iter == 0:
            logging.info("Previous did not complete first iter, starting from scratch.")
            last_model = cfg.base_model

        logging.info("Overriding model path with last model.")
        cfg.generation_config.model = last_model
        cfg.training_config.model_path = last_model

        logging.info(
            "Resuming from iteration %d. Last model path: %s",
            start_iter,
            last_model,
        )

        # Delete data from potentially partially completed steps.
        incomplete_iteration_path = generate_subpaths(cfg.output_dir, start_iter)[
            "iter_base"
        ]
        if os.path.exists(incomplete_iteration_path):
            logging.info(
                "Deleting incomplete iteration directory %s",
                incomplete_iteration_path,
            )
            subprocess.run(["rm", "-rf", incomplete_iteration_path])

        # Delete the eval_on_start directory if it exists.
        # -1 is because we start be evaluating the last complete iteration.
        eval_on_start_path = generate_subpaths(cfg.output_dir, start_iter - 1)[
            "eval_on_start"
        ]
        if os.path.exists(eval_on_start_path):
            logging.info(
                "Deleting eval_on_start directory %s",
                eval_on_start_path,
            )
            subprocess.run(["rm", "-rf", eval_on_start_path])

    # We are starting a new run.
    else:
        logging.info("No pre-existing files found in output directory.")
        logging.info("Assuming that this is a new run.")
        start_iter = 0

        # The first iteration starts generating with base model.
        # We start training every iteration with the base model.
        cfg.generation_config.model = cfg.base_model
        cfg.training_config.model_path = cfg.base_model

    logging.info(f"Config: {cfg}")
    wandb.require("core")

    time = datetime.datetime.now().strftime("%m/%d-%H:%M")
    run_name = f"{cfg.wandb_run_name} - {time}" if cfg.wandb_run_name else None
    wandb_run = wandb.init(
        project=cfg.wandb_project, name=run_name, config=config_utils.flatten_config(cfg)
    )

    logging.info("Wandb run name is %s", wandb_run.name)

    cfg.training_config.run_name = wandb_run.name

    # Load the training dataset.
    logging.info(
        "Loading train dataset. Dataset = %s, Split = %s"
        % (cfg.train_dataset_path, cfg.train_dataset_split)
    )
    dataset: datasets.Dataset = datasets.load_dataset(  # type: ignore
        cfg.train_dataset_path,
        split=cfg.train_dataset_split,
        trust_remote_code=True,
        name=cfg.dataset_name,
    )

    # Load eval dataset if provided.
    eval_dataset: Optional[datasets.Dataset] = None  # type: ignore
    if cfg.eval_dataset_path is not None:
        assert cfg.eval_dataset_split is not None
        logging.info(
            "Loading evaluation dataset. Dataset = %s, Split = %s"
            % (cfg.eval_dataset_path, cfg.eval_dataset_split)
        )
        eval_dataset: datasets.Dataset = datasets.load_dataset(  # type: ignore
            cfg.eval_dataset_path,
            split=cfg.eval_dataset_split,
            trust_remote_code=True,
            name=cfg.dataset_name,
        )

        # Subsample the eval dataset if num_eval_problems if specified in config.
        if cfg.num_eval_problems is not None:
            eval_dataset = eval_dataset.shuffle(seed=cfg.seed)
            eval_dataset = eval_dataset.select(range(cfg.num_eval_problems))
            logging.info(
                "Evaluation dataset subsampled to %d problems.", len(eval_dataset)
            )

    # Start training.
    logging.info("Starting STAR loop for %d iterations." % cfg.num_iterations)
    run_star_loop(
        cfg,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        num_iterations=cfg.num_iterations,
        start_iteration=start_iter,
    )

    logging.info("STaR Training complete.")
