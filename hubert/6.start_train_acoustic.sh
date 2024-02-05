db_dir="/data/database/wTM"

CUDA_VISIBLE_DEVICES=0,1 python ../acoustic-model/train.py ${db_dir} /data/hdd0/tianyi.tan/ckpt-acoustic-model
