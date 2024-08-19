#!/bin/bash
#SBATCH --job-name=MB
#SBATCH --partition=DGX
##SBATCH --exclude=hgx-hyperplane[02]
#SBATCH --account=research
#SBATCH --qos=lv2
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --output=./slurm_logs/train-mb-llava-7b.out
#SBATCH --error=./slurm_logs/train-mb-llava-7b.error.out

# --master_port 18983 
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
    # --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b/mm_projector.bin \
deepspeed --master_port 24612 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/llava-v1.5-7b \
    --version v1 \
    --data_path ./playground/data/llava_videochat2_filter.json \
    --x_folder ./playground/data \
    --video_tower ./checkpoints/LanguageBind_Video_merge \
    --image_tower ./checkpoints/LanguageBind_Image \
    --freeze_image_tower True \
    --freeze_video_tower True \
    --num_frames 16 \
    --X "VIDEO" "IMAGE" \
    --mm_projector_type rmt_r_transformer1x \
    --mm_vision_select_layer -2 \
    --mm_use_x_start_end False \
    --mm_use_x_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/videollamb-llava-7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard