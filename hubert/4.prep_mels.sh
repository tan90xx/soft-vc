#!/bin/bash

#db_dir="/data/ssd0/tianyi.tan/data/wTM/wavs"
#target_dir="/data/ssd0/tianyi.tan/data/wTM/mels"
db_dir="/data/database/wTM/wavs"
target_dir="/data/database/wTM/mels"

CUDA_VISIBLE_DEVICES=0 python ../acoustic-model/mel-single.py "${db_dir}/train/normal" "${target_dir}/train"
CUDA_VISIBLE_DEVICES=0 python ../acoustic-model/mel-single.py "${db_dir}/eval/normal" "${target_dir}/eval"
CUDA_VISIBLE_DEVICES=0 python ../acoustic-model/mel-single.py "${db_dir}/dev/normal" "${target_dir}/dev"
