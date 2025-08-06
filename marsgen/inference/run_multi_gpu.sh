#!/bin/bash
#./run_multi_gpu.sh 1400 2000 -200
#./run_multi_gpu.sh 1400 2000 200

# Default parameters

CKPT_BASE_PATH=""
VIDEO_ROOT_DIR=''
ANNOTATION_TXT=''

# Checkpoint range parameters (can be overridden via command line)
START_CKPT=${1:-4000}
END_CKPT=${2:-4000}
SKIP_CKPT=${3:-200}
PREFIX=${4:-""} 
# Validate and adjust loop direction based on SKIP_CKPT
if [ $SKIP_CKPT -eq 0 ]; then
    echo "Error: SKIP_CKPT cannot be 0"
    exit 1
fi

# Determine processing order based on SKIP_CKPT sign
if [ $SKIP_CKPT -gt 0 ]; then
    # For positive step, start from smaller number
    PROCESS_START=$START_CKPT
    PROCESS_END=$END_CKPT
    should_continue() { [ $ckpt -le $PROCESS_END ]; }
else
    # For negative step, start from larger number
    PROCESS_START=$START_CKPT  # Changed from END_CKPT
    PROCESS_END=$END_CKPT      # Changed from START_CKPT
    should_continue() { [ $ckpt -ge $PROCESS_END ]; }
fi

echo "Processing checkpoints from $PROCESS_START to $PROCESS_END with step $SKIP_CKPT"

# Get the number of available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=gpu_name --format=csv,noheader | wc -l)
echo "Found $NUM_GPUS GPUs"

# Loop through checkpoints
ckpt=$PROCESS_START
while should_continue; do
    echo "Processing checkpoint-$ckpt"
    CONTROLNET_MODEL_PATH="$CKPT_BASE_PATH/checkpoint-$ckpt.pt"
    OUTPUT_PATH="$CKPT_BASE_PATH/validation/${PREFIX}_step$ckpt"
    
    # Skip if checkpoint doesn't exist
    if [ ! -f "$CONTROLNET_MODEL_PATH" ]; then
        echo "Checkpoint $CONTROLNET_MODEL_PATH not found, skipping..."
        ckpt=$((ckpt + SKIP_CKPT))
        continue
    fi

    # Launch processes for each GPU
    for ((gpu_id=0; gpu_id<$NUM_GPUS; gpu_id++))
    do
        echo "Starting process on GPU $gpu_id for checkpoint $ckpt"
        python cli_demo_i2v_multi.py \
            --gpu_id $gpu_id \
            --num_gpus $NUM_GPUS \
            --video_root_dir "$VIDEO_ROOT_DIR" \
            --controlnet_model_path "$CONTROLNET_MODEL_PATH" \
            --annotation_txt "$ANNOTATION_TXT" \
            --output_path "$OUTPUT_PATH" \
            --controlnet_transformer_out_proj_dim_zero_init \
            --save_ref &
        
        # Store the PID of the background process
        pids[${gpu_id}]=$!
    done

    # Wait for all GPUs to complete current checkpoint
    echo "Waiting for all processes to complete checkpoint $ckpt..."
    for pid in ${pids[*]}; do
        wait $pid
    done
    echo "Checkpoint $ckpt completed"
    
    ckpt=$((ckpt + SKIP_CKPT))
done

echo "All checkpoints completed"
