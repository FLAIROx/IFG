#!/bin/bash

# Check if GPU ID argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <starting_gpu_id>"
    echo "Example: $0 0"
    exit 1
fi

# Get the starting GPU ID from command line argument
gpu_id=$1

# Define the range of keyword temperatures to test
# You can modify these values as needed
keyword_temps=(1.88 1.93 1.93)
comments_temps=(1.46 1.45 1.3)
log_dir="data/logs"

# Use index to iterate through both arrays simultaneously
for i in "${!keyword_temps[@]}" 
do
    kw_temp="${keyword_temps[$i]}"
    comm_temp="${comments_temps[$i]}"
    
    echo "Starting job with keyword_temp = $kw_temp, comments_temp = $comm_temp on GPU $gpu_id"
    
    # Create output directory name with both temperatures
    output_dir="data/generated_responses/kw-7B-dpo_ft_${kw_temp}_${comm_temp}"
    
    # Set up logging
    mkdir -p $log_dir
    # Get current timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    # Create log filename with timestamp
    log_file="$log_dir/kw-7B-temp_${kw_temp}_gpu${gpu_id}_${timestamp}.log"
    model_path="data/checkpoints/rlhf_experiments/dpo_stage/Qwen2.5-7B/final/"
    
    CUDA_VISIBLE_DEVICES=$gpu_id python -m unstructured_tasks.inference.rlhf_response_generation \
        --model_dpo $model_path \
        --keyword-temp $kw_temp \
        --comments-temp $comm_temp \
        --output-dir "$output_dir" \
        --mode "kw" \
        > "$log_file" 2>&1 &
    
    echo "Job started in background. Check $log_file for output"
    echo "----------------------------------------"
    
    # Increment GPU counter
    ((gpu_id++))
done 

echo "Jobs launched. Use 'tail -f data/logs/kw-7B-temp_*.log' to monitor progress" 
