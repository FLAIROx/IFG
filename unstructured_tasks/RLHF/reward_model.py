"""Trains a reward model for RLHF using preference data from human feedback."""

import warnings
import dataclasses
import os

import torch
from typing import Optional
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)
import tyro
import json

from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    setup_chat_format,
)


@dataclasses.dataclass
class Config:
    # Model and dataset configuration
    seed: int = 42
    model_name_or_path: str = "Qwen/Qwen2.5-7B"
    dataset_name: str = "Unified-Language-Model-Alignment/Anthropic_HH_Golden"
    output_dir: str = "RLHF/reward_model/Qwen2.5-7B-Reward"
    # Weights & Biases configuration
    wandb_project: str = "Reward-Model"
    wandb_run_name: Optional[str] = None
    # Training parameters
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_train_epochs: int = 1
    gradient_checkpointing: bool = True
    learning_rate: float = 1.0e-5
    logging_steps: int = 25
    eval_strategy: str = "steps"
    eval_steps: int = 50
    max_length: int = 2048
    use_peft: bool = False
    lora_task_type: str = "SEQ_CLS"

    wandb_entity: str = "reward-model"

    def __post_init__(self):
        if os.path.exists(self.output_dir) and (
            os.path.isfile(self.output_dir) or os.listdir(self.output_dir)
        ):
            raise ValueError(
                "Output directory path already exists and is not empty/is a file"
            )


def main(cfg: Config):
    # Convert to HF dataclasses
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))

    # Define which arguments should be excluded (not supported by HF parser)
    excluded_args = {"wandb_project", "wandb_entity", "wandb_run_name"}

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

    set_seed(training_args.seed)

    ################
    # Model & Tokenizer
    ################
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        use_cache=False if training_args.gradient_checkpointing else True,
        torch_dtype=torch_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding=True,
        trust_remote_code=model_args.trust_remote_code,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        trust_remote_code=model_args.trust_remote_code,
        **model_kwargs,
    )
    # Align padding tokens between tokenizer and model
    model.config.pad_token_id = tokenizer.pad_token_id

    # If post-training a base model, use ChatML as the default template
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer)

    if model_args.use_peft and model_args.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.",
            UserWarning,
        )

    ##############
    # Load dataset
    ##############
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    ##########
    # Training
    ##########
    os.environ["WANDB_PROJECT"] = cfg.wandb_project
    os.environ["WANDB_ENTITY"] = cfg.wandb_entity

    # If you have a run name
    if cfg.wandb_run_name:
        os.environ["WANDB_RUN_NAME"] = cfg.wandb_run_name + "-" + str(cfg.seed)

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()

    ############################
    # Save model and push to Hub
    ############################
    trainer.save_model(training_args.output_dir)

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and push to hub
    trainer.save_model(training_args.output_dir)

    with open(os.path.join(cfg.output_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f)


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
