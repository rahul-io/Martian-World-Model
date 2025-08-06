export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$(echo $PATH | tr ':' '\n' | grep -v "conda" | tr '\n' ':')
export NCCL_TIMEOUT=2400

#  Log INFO: wandb
export WANDB_API_KEY=""
export WANDB_NAME=""
export WANDB_NOTES=""
export WANDB_ENTITY=""
export WANDB_PROJECT=""
export WANDB_DIR=""

dir=`pwd`
# data & output path
video_root_dir="../marsvideodata/"
file_name_txt="annotations/train.json"
file_test_name_txt="annotations/validation.json"
output_dir=${dir}/out/${WANDB_NAME}
# model path
export MODEL_PATH="../THUDM/CogVideoX-5b-I2V"
controlnet_path="../ckpt/checkpoint-14000.pt"

cd training

source /opt/conda/bin/activate ac3d
python -m accelerate.commands.launch --config_file accelerate_config_machine_single.yaml  \
  train_controlnet.py \
  --tracker_name $WANDB_PROJECT \
  --gradient_checkpointing \
  --pretrained_model_name_or_path $MODEL_PATH \
  --enable_tiling \
  --enable_slicing \
  --num_inference_steps 28 \
  --seed 42 \
  --mixed_precision bf16 \
  --output_dir $output_dir \
  --report_to wandb \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 49 \
  --video_root_dir $video_root_dir \
  --file_name_txt $file_name_txt \
  --file_test_name_txt $file_test_name_txt \
  --stride_min 1 \
  --stride_max 1 \
  --hflip_p 0.0 \
  --controlnet_transformer_num_layers 8 \
  --controlnet_input_channels 6 \
  --downscale_coef 8 \
  --controlnet_weights 1.0 \
  --train_batch_size 1 \
  --dataloader_num_workers 16 \
  --num_train_epochs 50 \
  --checkpointing_steps 1000 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --lr_scheduler cosine_with_restarts \
  --lr_warmup_steps 250 \
  --lr_num_cycles 1 \
  --enable_slicing \
  --enable_tiling \
  --gradient_checkpointing \
  --optimizer AdamW \
  --adam_beta1 0.9 \
  --adam_beta2 0.95 \
  --max_grad_norm 1.0 \
  --allow_tf32 \
  --enable_time_sampling \
  --time_sampling_type truncated_normal \
  --time_sampling_mean 0.95 \
  --time_sampling_std 0.1 \
  --controlnet_guidance_start 0.0 \
  --controlnet_guidance_end 0.4 \
  --controlnet_transformer_num_attn_heads 4 \
  --controlnet_transformer_attention_head_dim 64 \
  --controlnet_transformer_out_proj_dim_factor 64 \
  --controlnet_transformer_out_proj_dim_zero_init \
  --validation_steps 300 \
  --pretrained_controlnet_path $controlnet_path \
  --use_text_prompt