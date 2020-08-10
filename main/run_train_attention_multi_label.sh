#!/usr/bin/env bash

resume_path=/home/hby/Documents/Exps/top_view_xray_cls/attention/best.ckpt
model_save_path=/home/hby/Documents/Exps/top_view_xray_cls/attention_cls

python3 train_attention_multi_label.py \
--resume_path ${resume_path} \
--model_save_path ${model_save_path} \
