import copy
import dataclasses
import datetime
import json
import logging
import os
import subprocess
import sys
from typing import Optional

import dotenv
from hendrycks_math.utils import config_utils
from hendrycks_math.utils import consts
from hendrycks_math.ifg_infer_and_score import IfgEvalConfig
from hendrycks_math.utils import config_utils
import wandb

dotenv.load_dotenv()


@dataclasses.dataclass(kw_only=True)
class SweepTempForMultipleK:
    sweep_file: str
    eval_config: IfgEvalConfig
    wandb_run_name: str
    num_tuning_problems: Optional[int] = 2048 # If None, use all problems.
    num_test_problems: Optional[int] = None # If None, use all problems.

    k_values: list[int] = dataclasses.field(default_factory=lambda: [1, 4, 16])
    wandb_project: str = "k-sweeps"

    hparam_sweep_script: str = (
        "hendrycks_math/experiment_pipelines/hparam_sweeper.py"
    )
    math_eval_script: str = (
        "hendrycks_math/ifg_infer_and_score.py"
    )


def get_best_args_from_sweep_path(report_path: str) -> str:
    """Get the best args from the result of the sweep."""

    assert os.path.exists(report_path), f"Report file {report_path} does not exist."

    # Load the report file
    with open(report_path) as f:
        report = json.load(f)

    best_args = report["best"]["args"]

    temperature_args = " ".join([f"--{k}={v}" for k, v in best_args.items() if v is not None])
    return temperature_args


def extract_results_from_test_run_path(report_path: str) -> dict[str, int | float]:
    """Extract the pass@1 accuracy from the test run path."""

    assert os.path.exists(report_path), f"Report file {report_path} does not exist."

    # Load the report file
    with open(report_path) as f:
        report = json.load(f)

    return report

def args_to_flags(args: dict) -> list[str]:
    """Convert a dictionary of args to a list of flags."""
    flags = []
    for k, v in args.items():
        # We drop these are we override them as we sweep/
        if k == "num_attempts" or k == "num_problems" or k == "split":
            continue
        if v is None:
            continue
        if isinstance(v, bool) and v:
            flags.append(f"--{k}")
        elif isinstance(v, bool) and not v:
            this_flag = k.split(".")
            this_flag[-1] = "no_" + this_flag[-1]
            this_flag = ".".join(this_flag)
            flags.append(f"--{this_flag}")
        else:
            flags.append(f"--{k}={v}")
    return flags

def run_sweep_for_k(
    k: int,
    output_dir_for_k: str,
    base_sweep_config: dict,
    sweep_cfg_file_path: str,
    sweep_output_dir_for_k: str,
    global_config: SweepTempForMultipleK,
):
    """Run the sweep for a given k value."""

    os.makedirs(output_dir_for_k, exist_ok=True)

    if os.path.exists(sweep_cfg_file_path):
        logging.info("Sweep config already exists for k=%d", k)
        with open(sweep_cfg_file_path, "r") as f:
            sweep_config_for_k = json.load(f)
    else:
        logging.info("Creating sweep config for k=%d", k)

        sweep_config_for_k = copy.deepcopy(base_sweep_config)

        sweep_config_for_k["output_dir"] = sweep_output_dir_for_k
        sweep_config_for_k["fixed_args"] += " --num_attempts=%d" % k
        sweep_config_for_k["fixed_args"] += (
            " --num_problems=%d" % global_config.num_tuning_problems
        )
        sweep_config_for_k["fixed_args"] += " --split=train"

        with open(sweep_cfg_file_path, "w") as f:
            json.dump(sweep_config_for_k, f, indent=4)

    logging.info(
        "Launching sweep with config for k=%d:\n %s",
        k,
        json.dumps(sweep_config_for_k, indent=2),
    )

    sweep_command = [
        sys.executable,
        global_config.hparam_sweep_script,
        f"--sweep_file={sweep_cfg_file_path}",
        f"--script={global_config.math_eval_script}",
    ]


    # Run the sweep
    logging.info("Running command:\n %s", " ".join(sweep_command))
    subprocess.run(
        sweep_command,
        check=True,
    )

def save_current_results(
    best_accuracies: list[dict[str, int | float]],
    results_path: str,
) -> None:
    """Save results thus far current results to a file.
    
    Appends to the file if it exists.
    """
    if os.path.exists(results_path):
        logging.info("Loading existing results from %s", results_path)
        with open(results_path, "r") as f:
            existing_results = json.load(f)
            logging.info("Existing results: %s", existing_results)

            # Remove duplicates
            for accuracy in best_accuracies:
                if accuracy['k'] not in [x['k'] for x in existing_results]:
                    existing_results.append(accuracy)
            # Sort by k
            existing_results.sort(key=lambda x: x["k"])
            best_accuracies = existing_results

    logging.info("Best accuracies: %s", best_accuracies)

    logging.info("Saving results to %s", results_path)
    with open(results_path, "w") as f:
        json.dump(best_accuracies, f, indent=4)

def main():
    cfg = config_utils.tyro_cli_with_yaml_support(SweepTempForMultipleK)

    with open(cfg.sweep_file, "r") as f:
        sweep_config = json.load(f)

    output_dir = sweep_config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    london_tz = datetime.timezone(datetime.timedelta(hours=1))
    timestamp = datetime.datetime.now(tz=london_tz).strftime("%Y-%m-%d_%H-%M-%S")

    log_path = os.path.join(output_dir, "k_vs_pas_at_k_%s.log" % timestamp)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )

    logging.info("Starting sweep with config: %s", cfg)
    logging.info("Sweep base config: %s", sweep_config)
    logging.info("Output dir: %s", output_dir)

    wandb.init(
        project=cfg.wandb_project,
        name=f"{cfg.wandb_run_name}_{timestamp}",
        config=config_utils.flatten_config(cfg),
    )

    sweep_file_log_path = os.path.join(output_dir, f"base_sweep_{timestamp}.json")
    k_sweep_cfg_log_path = os.path.join(output_dir, f"k_sweep_cfg_{timestamp}.json")

    # Log configuration files to output directory
    logging.info("Sweep config path: %s", sweep_file_log_path)
    with open(sweep_file_log_path, "w") as f:
        json.dump(sweep_config, f, indent=4)

    with open(k_sweep_cfg_log_path, "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=4)

    logging.info("Config: %s", cfg)
    logging.info("Base sweep config: %s", sweep_config)

    eval_args = args_to_flags(dataclasses.asdict(cfg.eval_config))
    sweep_config["fixed_args"] += " " + " ".join(eval_args)

    best_accuracies = []

    for k in cfg.k_values:

        output_dir_for_k = os.path.join(output_dir, f"k_{k}")

        sweep_output_dir_for_k = os.path.join(output_dir_for_k, "sweep")
        sweep_cfg_file_path = os.path.join(output_dir_for_k, f"sweep.json")
        sweep_report_for_k_path = os.path.join(
            sweep_output_dir_for_k, "report.json"
        )

        test_run_dir_for_k = os.path.join(output_dir_for_k, "test_run")
        test_run_report_k = os.path.join(test_run_dir_for_k, consts.Filenames.RESULTS)

        if not os.path.exists(sweep_report_for_k_path):
            logging.info("Sweep not run yet for k=%d", k)
            logging.info("Running sweep for k=%d", k)
            run_sweep_for_k(
                k=k,
                output_dir_for_k=output_dir_for_k,
                base_sweep_config=sweep_config,
                sweep_cfg_file_path=sweep_cfg_file_path,
                sweep_output_dir_for_k=sweep_output_dir_for_k,
                global_config=cfg,
            )
        else:
            logging.info("Sweep already run for k=%d", k)
            logging.info("Loading existing sweep config for k=%d", k)

        logging.info("Loading best args from %s", sweep_report_for_k_path)
        best_args = get_best_args_from_sweep_path(sweep_report_for_k_path)
        logging.info("Best args for k=%d: %s", k, best_args)

        os.makedirs(test_run_dir_for_k, exist_ok=True)
        # Run the test run
        eval_args = args_to_flags(dataclasses.asdict(cfg.eval_config))

        test_run_command = [
            sys.executable,
            cfg.math_eval_script,
            *eval_args,
            *best_args.split(" "),
            f"--output_dir={test_run_dir_for_k}",
            f"--split=test",
            f"--num_problems={cfg.num_test_problems}",
            f"--num_attempts={k}",
        ]
        if os.path.exists(test_run_report_k):
            logging.info("Test run already exists for k=%d", k)
            logging.info("Loading existing test run report from %s", test_run_report_k)
        else:
            logging.info("Test run not run yet for k=%d", k)
            logging.info("Running test run for k=%d", k)
            logging.info("Running command for test run:\n %s", " ".join(test_run_command))
            subprocess.run(
                test_run_command,
                check=True,
            )   

        logging.info("Loading results from %s", test_run_dir_for_k)
        test_results = extract_results_from_test_run_path(test_run_report_k)
        logging.info("Results for k=%d: %s", k, test_results)

        test_results["k"] = k

        best_accuracies.append(test_results)

        wandb.log(
            test_results,
        )

        results_path = os.path.join(output_dir, "results.json")
        logging.info("Results from this run so far %s", best_accuracies)
        save_current_results(
            best_accuracies=best_accuracies,
            results_path=results_path,
        )

    # load best accuracies and append to it if it exists
   

    logging.info("Finished sweep.")
    wandb.finish()
    logging.info("Wandb finished.")
    logging.info("All done.")

if __name__ == "__main__":
    main()