#!/usr/bin/env bash

model_save_path=/home/hby/Documents/Exps/top_view_xray_cls/448_attention_112

python3 train_bbox_insight.py \
--model_save_path ${model_save_path} \
