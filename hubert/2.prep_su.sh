#!/bin/bash

#db_dir="/data/ssd0/tianyi.tan/data/wTM/wavs" #g5
#target_dir="/data/ssd0/tianyi.tan/data/wTM/discrete" #g5
db_dir="/data/database/wTM/wavs"
target_dir="/data/database/wTM/discrete"

python encode.py discrete ${db_dir} ${target_dir} --extension '.WAV'
