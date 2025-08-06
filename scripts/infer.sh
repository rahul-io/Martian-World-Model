#!/bin/bash

# Change the absolute path first!
DATA_ROOT_DIR="<absolute path>"
DATASETS=(assets)
MODEL_NAME="assets"
temp_var="depth"

GPU=$1
TRAJ="zoom_in_out"
SCENES=(demo)

N_VIEWS=(2)

gs_train_iter=2000

# Function to get the id of an available GPU
get_available_gpu() {
    local mem_threshold=5000
    local target_gpu="$GPU"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -v gpu_id="$target_gpu" -F', ' '
    $1 == gpu_id && $2 < threshold { print $1; exit }
    '
}
# Function: Run task on specified GPU
run_on_gpu() {
    local GPU_ID=$1
    local DATASET=$2
    local SCENE=$3
    local N_VIEW=$4
    local gs_train_iter=$5
    SOURCE_PATH=${DATA_ROOT_DIR}/${DATASET}/${SCENE}/
    MODEL_PATH=${DATA_ROOT_DIR}/${MODEL_NAME}/${SCENE}
    mkdir -p ${MODEL_PATH}

    echo "======================================================="
    echo "Starting process: ${DATASET}/${SCENE} (${N_VIEW} views/${gs_train_iter} iters) on GPU ${GPU_ID}"
    echo "======================================================="

    # (1) Co-visible Global Geometry Initialization
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting Co-visible Global Geometry Initialization..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python -W ignore ./init_geo.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    --n_views ${N_VIEW} \
    --co_vis_dsp \
    > ${MODEL_PATH}/01_init_geo.log 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Co-visible Global Geometry Initialization completed. Log saved in ${MODEL_PATH}/01_init_geo.log"

    # (2) Train: jointly optimize pose
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting training..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./train.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    -r 1 \
    --n_views ${N_VIEW} \
    --iterations ${gs_train_iter} \
    --pp_optimizer \
    --optim_pose \
    -d ${temp_var} \
    --use_bilateral_grid \
    > ${MODEL_PATH}/02_train.log 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training completed. Log saved in ${MODEL_PATH}/02_train.log"

    # (3) Render-Video
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting rendering training views..."
    CUDA_VISIBLE_DEVICES=${GPU_ID} python ./render.py \
    -s ${SOURCE_PATH} \
    -m ${MODEL_PATH} \
    -r 1 \
    --n_views ${N_VIEW} \
    --iterations ${gs_train_iter} \
    --infer_video \
    --traj ${TRAJ} \
    > ${MODEL_PATH}/03_render_${TRAJ}.log 2>&1
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Rendering completed. Log saved in ${MODEL_PATH}/03_render_${TRAJ}.log"


    echo "======================================================="
    echo "Task completed: ${DATASET}/${SCENE} on GPU ${GPU_ID}"
    echo "======================================================="
}

# Main loop
total_tasks=$((${#DATASETS[@]} * ${#SCENES[@]} * ${#N_VIEWS[@]} * ${#gs_train_iter[@]}))
current_task=0

for DATASET in "${DATASETS[@]}"; do
    for SCENE in "${SCENES[@]}"; do
        for N_VIEW in "${N_VIEWS[@]}"; do
            for gs_train_iter in "${gs_train_iter[@]}"; do
                current_task=$((current_task + 1))
                echo "Processing task $current_task / $total_tasks"

                # Get available GPU
                GPU_ID=$(get_available_gpu)

                # If no GPU is available, wait for a while and retry
                while [ -z "$GPU_ID" ]; do
                    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No GPU available, waiting 60 seconds before retrying..."
                    sleep 40
                    GPU_ID=$(get_available_gpu)
                done

                # Run the task in the background
                (run_on_gpu $GPU_ID "$DATASET" "$SCENE" "$N_VIEW" "$gs_train_iter") &

                # Wait for 20 seconds before trying to start the next task
                sleep 50
            done
        done
    done
done

# Wait for all background tasks to complete
wait

echo "======================================================="
echo "All tasks completed! Processed $total_tasks tasks in total."
echo "======================================================="
