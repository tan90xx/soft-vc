#!/bin/bash

#db_dir="/data/ssd0/tianyi.tan/data/wTM/wavs" #g5
#target_dir="/data/ssd0/tianyi.tan/data/wTM/soft" #g5
db_dir="/data/database/wTM/wavs"
target_dir="/data/database/wTM/soft"
ckpt_path="/data/hdd0/tianyi.tan/hubert/model-best.pt"

#python encode.py soft /data/database/LJSpeech-1.1/003/wavs/train /data/database/LJSpeech-1.1/003/soft/train --extension .wav
#python encode.py soft /data/database/LJSpeech-1.1/003/wavs/dev /data/database/LJSpeech-1.1/003/soft/dev --extension .wav
python encode.py soft "${db_dir}/train/normal" "${target_dir}/train" --extension .WAV --checkpoint ${ckpt_path}
python encode.py soft "${db_dir}/dev/normal" "${target_dir}/dev" --extension .WAV --checkpoint ${ckpt_path}
python encode.py soft "${db_dir}/eval/normal" "${target_dir}/eval" --extension .WAV --checkpoint ${ckpt_path}
