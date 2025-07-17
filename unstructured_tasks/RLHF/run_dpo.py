"""Trains models using Direct Preference Optimization with pairwise preference data."""

import dataclasses
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, set_seed
import tyro
from typing import Optional
from typing import Any, Dict, List
import logging
import datasets
import json

from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE


@dataclasses.dataclass
class Config:
    # Model and dataset configuration
    seed: int = 1
    model_name_or_path: str = "data/checkpoints/rlhf_experiments/sft_stage/Qwen2.5-7B/final"
    dataset_name: str = "Unified-Language-Model-Alignment/Anthropic_HH_Golden"
    train_dataset: Optional[str] = "data/hh_annotated/preference_train.json"
    test_dataset: Optional[str] = "data/hh_annotated/preference_test.json"
    valid_dataset: Optional[str] = "data/hh_annotated/preference_test.json"
    output_dir: str = "data/checkpoints/rlhf_experiments/dpo_stage/Qwen2.5-7B"

    # Weights & Biases configuration
    wandb_project: str = "DPO"
    wandb_entity: str = "dpo-training"
    wandb_run_name: Optional[str] = None
    # Training parameters
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    gradient_checkpointing: bool = True
    learning_rate: float = 1e-6
    logging_steps: int = 25
    eval_strategy: str = "steps"
    eval_steps: int = 50
    max_length: int = 2048
    use_peft: bool = False
    lora_task_type: str = "SEQ_CLS"
    remove_unused_columns: bool = False  # equivalent to no_remove_unused_columns

    def __post_init__(self):
        if os.path.exists(self.output_dir) and (
            os.path.isfile(self.output_dir) or os.listdir(self.output_dir)
        ):
            raise ValueError(
                "Output directory path already exists and is not empty/is a file"
            )


def main(cfg: Config):
    # Convert to HF dataclasses
    parser = HfArgumentParser((ScriptArguments, DPOConfig, ModelConfig))

    # Define which arguments should be excluded (not supported by HF parser)
    excluded_args = {
        "train_dataset",
        "test_dataset",
        "valid_dataset",
        "method",
        "wandb_project",
        "wandb_entity",
        "wandb_run_name",
    }

    # Convert dict to list of args, filtering out excluded ones
    cfg_dict = vars(cfg)
    args_list = []
    for k, v in cfg_dict.items():
        if (
            v is not None and k not in excluded_args
        ):  # Only add supported non-None values
            args_list.extend([f"--{k.replace('_', '-')}", str(v)])

    script_args, training_args, model_args = parser.parse_args_into_dataclasses(
        args_list
    )
    training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
    training_args.bf16 = True
    training_args.bf16_full_eval = True

    # Create output directory if it doesn't exist
    os.makedirs(training_args.output_dir, exist_ok=True)

    set_seed(training_args.seed)
    print(f"Training args: {training_args}")

    ################
    # Model & Tokenizer
    ###################

    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch.bfloat16,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            **model_kwargs,
        )
    else:
        ref_model = None

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    logging.info(f"Config: {cfg}")

    # Load training dataset
    train_dataset: datasets.Dataset = datasets.load_dataset(  # type: ignore
        "json", data_files=cfg.train_dataset, split="train"
    )
    train_dataset = train_dataset.shuffle(seed=cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name_or_path, add_eos_token=True
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

    if not cfg.valid_dataset:
        logging.info("No validation dataset provided")
        valid_dataset = None
    else:
        valid_dataset: Optional[datasets.Dataset] = datasets.load_dataset(  # type: ignore
            "json",
            data_files=cfg.valid_dataset,
            split="train",  # Using "train" split since it's a separate file
        )
        assert valid_dataset is not None

    ##########
    # Training
    ################
    # Set up wandb configuration
    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    os.environ["WANDB_ENTITY"] = cfg.wandb_entity

    # If you have a run name
    if cfg.wandb_run_name:
        os.environ["WANDB_RUN_NAME"] = cfg.wandb_run_name + "-" + str(cfg.seed)

    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f)
    final_model_path = os.path.join(cfg.output_dir, "final")
    trainer.save_model(final_model_path)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
