#!/bin/bash
# MSTMHA-MSTE训练脚本 - 使用MSTE多尺度Mamba时序建模
# 使用方法: bash train_msrvtt.sh

CUDA_VISIBLE_DEVICES=4,5,6,7 \
python -m torch.distributed.launch \
--master_port 2515 \
--nproc_per_node=4 \
--use_env \
main_retrieval.py \
--do_train 1 \
--workers 8 \
--n_display 50 \
--epochs 5 \
--lr 1e-3 \
--coef_lr 1e-4 \
--batch_size 128 \
--batch_size_val 64 \
--anno_path /home/mkyvkbwh/sun/MSTMHA-MSTE/datasets/MSR-VTT/anns \
--video_path /home/mkyvkbwh/sun/MSTMHA-MSTE/datasets/MSR-VTT/MSRVTT_Videos \
--datatype msrvtt \
--max_words 32 \
--max_frames 12 \
--video_framerate 1 \
--output_dir /home/mkyvkbwh/sun/MSTMHA-MSTE-3/outputs/ckpt/msrvtt_mste_hca_ssr \
--center 1 \
--temp 3 \
--hca_alpha 0.5 \
--ssr_beta 0.07 \
--ssr_delta 0.75 \
--query_number 8 \
--base_encoder ViT-B/32 \
--agg_module MSTE \
--num_hidden_layers 4 \
--cross_att_layer 3 \
--query_share 1 \
--cross_att_share 1 \
--loss2_weight 0.5
