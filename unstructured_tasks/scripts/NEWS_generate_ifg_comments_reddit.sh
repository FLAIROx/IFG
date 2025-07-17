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
comments_temps=0.5
keyword_temps=(0.3 0.57 0.64 0.71 0.79 0.86 0.93 1.0)

# Use index to iterate through both arrays simultaneously
for i in "${!keyword_temps[@]}" 
do
    kw_temp="${keyword_temps[$i]}"
    comm_temp=$comments_temps
    
    echo "Starting job with keyword_temp = $kw_temp, comments_temp = $comm_temp on GPU $gpu_id"
    
    # Create output directory name with both temperatures
    output_dir="data/generated_responses/reddit_comments/Qwen2.5-7B/ifg_${kw_temp}_${comm_temp}"
    
    # Create log filename
    log_dir="data/logs"
    timestamp=$(date +"%Y%m%d_%H%M%S")

    log_file="$log_dir/Qwen2.5-7B-ifg_${kw_temp}_${comm_temp}_gpu${gpu_id}_${timestamp}.log"
    
    # Ensure logs directory exists
    mkdir -p $log_dir
    
    CUDA_VISIBLE_DEVICES=$gpu_id python3 -m unstructured_tasks.inference.generate_comments_reddit \
        --model-comments "Qwen/Qwen2.5-7B" \
        --num-articles 100 \
        --num-comments 15 \
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

echo "Use 'tail -f data/logs/kw_news-7B-temp_*.log' to monitor progress" 
