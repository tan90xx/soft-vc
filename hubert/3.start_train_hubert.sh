hubert_checkpoint_path="/home/nis/tianyi.tan/.cache/torch/hub/checkpoints/model-layer12-450000.pt"
CUDA_VISIBLE_DEVICES=2,3 python train.py /data/ssd0/tianyi.tan/data/wTM /data/ssd0/tianyi.tan/ckpt-re --warmstart --mask --resume ${hubert_checkpoint_path}
