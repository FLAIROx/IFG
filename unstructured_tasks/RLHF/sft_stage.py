"""Fine-tunes LLMs on general text completion tasks using supervised learning."""

import dataclasses
import datetime
import functools
import gc
import json
import logging
import os
import pprint
import wandb
import time
from typing import Optional
from typing import Any, Dict, List

import accelerate
import datasets
import dotenv
import torch
import transformers

import ifg_utils


@dataclasses.dataclass(kw_only=True)
class TrainingConfig:
    output_dir: Optional[str] = "data/checkpoints/rlhf_experiments/sft_stage/Qwen2.5-7B"
    model_path: Optional[str] = "Qwen/Qwen2.5-7B"
    batch_size: int = 2
    num_epochs: int = 1
    project_name: Optional[str] = "SFT"
    run_name: Optional[str] = None
    dataset_name: str = "Unified-Language-Model-Alignment/Anthropic_HH_Golden"
    train_dataset: Optional[str] = "data/hh_annotated/train_formatted.json"
    test_dataset: Optional[str] = "data/hh_annotated/test_formatted.json"
    valid_dataset: Optional[str] = "data/hh_annotated/test_formatted.json"

    learning_rate: float = 1e-5
    save_steps: float = 100
    max_seq_length: int = 1024 * 3
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 8

    logging_steps: int = 10
    eval_steps: int = 10
    add_explicit_eos_token: bool = False

    seed: int = 11

    prompt_response_separator: str = ""
    pre_existing_files: list = dataclasses.field(default_factory=list)
    """Files that already exist in the output directory before the run starts."""
    save_only_model: bool = True
    """Save only the model without the optimizer and scheduler state."""

    wandb_entity: str = "HH_Golden_SFT"


class CustomCollator(transformers.DataCollatorWithPadding):
    def __init__(self, *args, max_seq_length=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seq_length = max_seq_length

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        collated = {}

        for key in features[0].keys():
            if key == "keywords":  # Skip non-tensor fields
                continue

            values = []
            for feature in features:
                # Ensure the feature is a list of integers
                if isinstance(feature[key], list):
                    tensor = torch.tensor(feature[key], dtype=torch.long)
                else:
                    continue
                values.append(tensor)

            if values:  # Only process if we have valid tensors
                assert isinstance(self.tokenizer.pad_token_id, int)
                collated[key] = torch.nn.utils.rnn.pad_sequence(
                    values,
                    batch_first=True,
                    padding_value=(
                        self.tokenizer.pad_token_id if key == "input_ids" else 0
                    ),
                )

        if self.max_seq_length is not None:
            collated = self.pad_or_truncate(collated)

        return collated

    def pad_or_truncate(self, collated, pad=False, truncate=True):
        """Pad or truncate the tensors in the collated dictionary."""
        # Truncate if longer
        if truncate:
            collated = {
                key: value[:, : self.max_seq_length] for key, value in collated.items()
            }

        if not pad:
            return collated

        # Pad if shorter
        for key, value in collated.items():
            if value.shape[1] < self.max_seq_length:
                assert isinstance(self.tokenizer.pad_token_id, int)
                padding_value = self.tokenizer.pad_token_id if key == "input_ids" else 0
                padding = torch.full(  # type: ignore
                    (value.shape[0], self.max_seq_length - value.shape[1]),
                    padding_value,
                    dtype=value.dtype,
                    device=value.device,
                )
                collated[key] = torch.cat([value, padding], dim=1)

        return collated


class CustomTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        if wandb.run is not None:
            run_name = wandb.run.name
        else:
            run_name = "<wandb not active>"
        logging.info("Wandb run name %s", run_name)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs
    ):
        inputs_for_model = {k: v for k, v in inputs.items() if k != "loss_mask"}
        outputs = model(**inputs_for_model)
        logits = outputs.logits[:, :-1]  # remove logits for the next token
        labels = inputs["input_ids"][:, 1:]  # remove the start token

        loss = self.loss_fn(logits.permute(0, 2, 1), labels)
        loss = loss * inputs["loss_mask"][:, 1:]
        loss = loss.sum() / inputs["loss_mask"][:, 1:].sum()

        return (loss, outputs) if return_outputs else loss


def read_prompt_file(file_path):
    with open(file_path, "r") as file:
        return file.read().strip()


def preprocess_sample(
    sample: Dict[str, str],
    tokenizer: transformers.AutoTokenizer,
    separator: str,
    add_explicit_eos_token: bool = False,
) -> dict[str, list[int]]:
    """Unified preprocessing function that handles both generic and news article cases"""

    # Generic preprocessing
    prompt = sample["prompt"]
    response = sample["response"]
    full_string = prompt + response

    # Tokenize the full string first
    tokenized = tokenizer(full_string, add_special_tokens=True)

    # Now tokenize just the prompt to find its length in tokens
    prompt_tokens = tokenizer(prompt, add_special_tokens=True)
    prompt_length = len(prompt_tokens["input_ids"])

    # Create loss mask - 0 for prompt tokens, 1 for response tokens
    n_tokens = len(tokenized["input_ids"])

    prompt_length = prompt_length - 4
    prompt_length = max(prompt_length, 0)

    loss_mask = [0] * prompt_length + [1] * (n_tokens - prompt_length)

    input_ids = tokenized["input_ids"]
    attn_mask = tokenized["attention_mask"]

    if add_explicit_eos_token:
        input_ids = input_ids + [tokenizer.eos_token_id]
        attn_mask = attn_mask + [1]
        loss_mask = loss_mask + [1]

    # Ensure all sequences are the same length
    assert (
        len(input_ids) == len(attn_mask) == len(loss_mask)
    ), f"Lengths don't match: {len(input_ids)} vs {len(attn_mask)} vs {len(loss_mask)}"

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "loss_mask": loss_mask,
    }


def training_loop(config: TrainingConfig, accelerator: accelerate.Accelerator):
    assert config.output_dir is not None, "Output directory must be specified"
    assert config.model_path is not None, "Model path must be specified"
    if accelerator.is_local_main_process:
        ifg_utils.prepare_output_dir(
            config.output_dir, pre_existing_files=config.pre_existing_files
        )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # device_map=accelerator.device,
        use_cache=False,
    )

    # load jsonl dataset
    logging.info(f"Training loop started {accelerator}")
    logging.info(f"Config: {config}")

    # Load training dataset
    train_dataset: datasets.Dataset = datasets.load_dataset(  # type: ignore
        "json", data_files=config.train_dataset, split="train"
    )
    train_dataset = train_dataset.shuffle(seed=config.seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model_path, add_eos_token=True
    )

    tokenizer.pad_token_id = tokenizer.eos_token_id

    preprocess_function = functools.partial(
        preprocess_sample,
        tokenizer=tokenizer,
        separator=config.prompt_response_separator,
    )
    # Process training dataset
    processed_train = train_dataset.map(
        preprocess_function,
        remove_columns=train_dataset.column_names,  # type: ignore
    )

    if not config.valid_dataset:
        logging.info("No validation dataset provided")
        valid_dataset = None
        processed_valid = None
    else:
        valid_dataset: Optional[datasets.Dataset] = datasets.load_dataset(  # type: ignore
            "json",
            data_files=config.valid_dataset,
            split="train",  # Using "train" split since it's a separate file
        )
        assert valid_dataset is not None
        # Process validation dataset
        processed_valid = valid_dataset.map(
            preprocess_function,
            remove_columns=valid_dataset.column_names,  # type: ignore
        )

    eval_strategy = "steps" if valid_dataset is not None else "no"
    eval_steps = config.eval_steps if valid_dataset is not None else None

    training_args = transformers.TrainingArguments(  # type: ignore
        output_dir=config.output_dir,
        seed=config.seed,
        per_device_train_batch_size=config.batch_size,
        bf16=True,
        bf16_full_eval=True,
        gradient_checkpointing=config.gradient_checkpointing,
        save_strategy="steps",
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=1,
        remove_unused_columns=False,
        learning_rate=config.learning_rate,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_epochs,
        ddp_find_unused_parameters=False,
        report_to=["wandb"],
        logging_first_step=True,
        run_name=config.run_name,
        save_only_model=config.save_only_model,
    )

    collator = CustomCollator(tokenizer=tokenizer, max_seq_length=config.max_seq_length)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_valid,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    logging.info("Training complete")
    final_save_path = os.path.join(config.output_dir, "final")
    trainer.save_model(final_save_path)
    logging.info("Model saved to %s", final_save_path)

    if accelerator.is_local_main_process:
        with open(os.path.join(config.output_dir, "config.json"), "w") as f:
            json.dump(dataclasses.asdict(config), f)

    logging.info("done")


if __name__ == "__main__":
    try:
        accelerator = accelerate.Accelerator()
        dotenv.load_dotenv()
        config = ifg_utils.tyro_cli_with_yaml_support(TrainingConfig)
        pprint.pprint(config)

        if accelerator.is_local_main_process:
            ifg_utils.prepare_output_dir(
                config.output_dir, pre_existing_files=config.pre_existing_files
            )
        accelerator.wait_for_everyone()

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = os.path.join(
            config.output_dir,
            "finetune-proc-%d-%s.log" % (accelerator.process_index, timestamp),
        )
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d - %(message)s",
            handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
        )
        accelerator.wait_for_everyone()
        # Add the log file to the list of pre-existing files. This signifies
        # even though the log file exists, the directory is otherwise empty.
        # and we will not overwrite a previous run.
        # Add log file to pre-existing files list to prevent overwriting previous runs
        config.pre_existing_files.append("*.log")

        os.environ["WANDB_PROJECT"] = config.project_name
        os.environ["WANDB_ENTITY"] = config.wandb_entity
        training_loop(config, accelerator=accelerator)
        logging.info("Training complete")
    finally:
        gc.collect()
        time.sleep(1)
        torch.cuda.empty_cache()
        time.sleep(1)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
