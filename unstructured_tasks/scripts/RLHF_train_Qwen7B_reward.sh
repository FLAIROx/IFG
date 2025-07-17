#!/bin/bash

# Launch distributed training using accelerate with DeepSpeed config
accelerate launch \
    --config_file=unstructured_tasks/configs/zero_stage_3.yaml \
    unstructured_tasks/RLHF/reward_model.py \
    --output_dir=RLHF/reward_model/Qwen2.5-7B-Reward \
    --model_name_or_path=Qwen/Qwen2.5-7B
