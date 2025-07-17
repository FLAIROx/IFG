import dataclasses
import logging
import os
import tempfile

import accelerate
import torch
import shutil
import tyro

from training import finetune_completion


@dataclasses.dataclass(kw_only=True)
class BatchSizeTuneConfig:
    """Config for tuning batch size.

    We fundamentally only depend on model, sequence length and
    gradient checkpointing.
    """

    model_path: str
    gradient_checkpointing: bool = True
    max_seq_length: int = 2048

    start_batch_size: int = 1
    max_batch_size: int = 64

    # We must match whatever value is in the acccelerate config
    # file.
    gradient_accumulation_steps: int = 4


def create_dataset_of_size_batch_size(
    input_dataset_path: str,
    output_dataset_path: str,
    batch_size: int,
):

    with open(input_dataset_path, "r") as f:
        lines = f.readlines()
    entry = lines[0]

    with open(output_dataset_path, "w") as f:
        for _ in range(batch_size):
            f.write(entry + "\n")

    return output_dataset_path


def batch_size_fits_in_memory(
    batch_size: int,
    base_data_filepath: str,
    base_output_dir: str,
    tune_cfg: BatchSizeTuneConfig,
    accelerator: accelerate.Accelerator,
):
    output_dir = os.path.join(
        base_output_dir,
        f"batch_size_{batch_size}",
    )
    os.makedirs(output_dir, exist_ok=True)

    train_filename = "train.jsonl"
    dataset_path = os.path.join(output_dir, train_filename)

    dataset_size = batch_size * tune_cfg.gradient_accumulation_steps * accelerator.num_processes

    create_dataset_of_size_batch_size(
        base_data_filepath,
        dataset_path,
        dataset_size,
    )

    cfg = finetune_completion.TrainingConfig(  # type: ignore
        output_dir=output_dir,
        model_path=tune_cfg.model_path,
        gradient_checkpointing=tune_cfg.gradient_checkpointing,
        max_seq_length=tune_cfg.max_seq_length,
        num_epochs=1,
        batch_size=batch_size,
        train_dataset=dataset_path,
        gradient_accumulation_steps=tune_cfg.gradient_accumulation_steps,
        pre_existing_files=[train_filename],
    )



    try:
        finetune_completion.training_loop(cfg, accelerator)
    except torch.cuda.OutOfMemoryError:
        print(f"Out of memory for batch size {batch_size}")
        return False

    return True


def main(output_dir):
    os.environ["WANDB_MODE"] = "offline"

    tune_cfg = tyro.cli(BatchSizeTuneConfig)
    accelerator = accelerate.Accelerator()

    base_data_filepath = os.path.join(os.path.dirname(__file__), "dummy_train.jsonl")

    # Suppress logs from accelerate and transformers
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("finetune_completion").setLevel(logging.ERROR)

    batch_size = tune_cfg.start_batch_size
    initial_batch_size_fits = batch_size_fits_in_memory(
        batch_size,
        base_data_filepath,
        output_dir,
        tune_cfg,
        accelerator,
    )

    if initial_batch_size_fits:
        print(f"Initial batch size {batch_size} fits in memory.")

    else:
        print(f"Initial batch size {batch_size} does not fit in memory.")
        print("Aborting...")
        exit(1)

    batch_size *= 2
    while batch_size <= tune_cfg.max_batch_size:
        if batch_size_fits_in_memory(
            batch_size,
            base_data_filepath,
            output_dir,
            tune_cfg,
            accelerator,
        ):
            print(f"Batch size {batch_size} fits in memory.")

        else:
            print(f"Batch size {batch_size} does not fit in memory.")

            print(f"Batch size {batch_size // 2} fits is max.")
            return
        batch_size *= 2


if __name__ == "__main__":
    import sys

    print(sys.executable)
    output_dir = tempfile.mkdtemp()
    try:
        main(output_dir)
    finally:
        # Cleanup the temporary directory
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"Temporary directory {output_dir} removed.")
