"""Run a hparam sweep for temperatures then run STaR training."""
import json
import logging
import os
import subprocess
import sys
import time

import tyro


@tyro.cli
def main(
    temp_sweep_file: str,
    star_config_file: str,
    sweeper_script: str = "hendrycks_math/experiment_pipelines/hparam_sweeper.py",
    temp_eval_script: str = "hendrycks_math/ifg_infer_and_score.py",
    star_script: str = "hendrycks_math/star.py",
) -> None:
    """Sweep for temperatures then run star.

    Args:
        temp_sweep_file: The sweep file to use for the temperature sweep.
        star_config_file: The config file to use for the star run, temperature, if set
            will be overridden by the sweep results.
        sweeper_script: The script to sweep hyperparameters.
        temp_eval_script: The script to evaluate the sweep results.
        star_script: The script to run the star evaluation
    """

    with open(temp_sweep_file, "r") as f:
        sweep_job = json.load(f)

    output_dir = sweep_job["output_dir"]

    # Log to a timestamped log file
    timestamp = str(int(time.time()))
    # The following line is way to get the parent directory
    # of a path of a dir in python. os.path.dirname has weird behavior
    # on directories.
    logdir = os.path.abspath(os.path.join(output_dir, os.pardir))
    log_path = os.path.join(logdir, f"sweep-then-star-{timestamp}.log")
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )
    report_path = os.path.join(output_dir, "report.json")
    if os.path.exists(report_path):
        logging.info(f"Report already exists at {report_path}. Skipping sweep.")
    else:
        logging.info(f"Starting sweep exp with {output_dir=}")
        cmd = f"{sys.executable} {sweeper_script} --sweep-file={temp_sweep_file} --script={temp_eval_script}"
        logging.info(f"Running command:\n {cmd}")
        subprocess.run(cmd.split(), check=True)

    with open(report_path) as f:
        report = json.load(f)

    best_args = report["best"]["args"]

    temperature_args = " ".join(
        [f"--generation_config.{k}={v}" for k, v in best_args.items()]
    )

    cmd = f"{sys.executable} {star_script} --yaml={star_config_file} {temperature_args}"
    logging.info(f"Running command:\n {cmd}")
    subprocess.run(cmd.split(), check=True)
