import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio
from torchaudio.functional import resample
from hubert.model import HubertSoft

from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

def load_hubert(checkpoint_path=None, rank=0, device='cuda'):
    print("### load_hubert", checkpoint_path, device)
    assert checkpoint_path is not None
    print("### loading checkpoint from: ", checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    hubert = HubertSoft().to(device) if device!='cuda' else HubertSoft().to(rank)

    checkpoint = checkpoint['hubert'] if checkpoint['hubert'] is not None else checkpoint
    consume_prefix_in_state_dict_if_present(checkpoint, "module.")

    hubert.load_state_dict(checkpoint, strict=True)
    hubert.eval().to(device)
    return hubert

def encode_dataset(args):
    print(f"Loading hubert checkpoint")
    #hubert = torch.hub.load(
    #    "bshall/hubert:main",
    #    f"hubert_{args.model}",
    #    trust_repo=True,
    #).cuda()
    hubert = load_hubert(args.checkpoint)
    print(f"Encoding dataset at {args.in_dir}")
    for in_path in tqdm(list(args.in_dir.rglob(f"*{args.extension}"))):
        wav, sr = torchaudio.load(in_path)
        wav = resample(wav, sr, 16000)
        wav = wav.unsqueeze(0).cuda()

        with torch.inference_mode():
            units = hubert.units(wav)

        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), units.squeeze().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "model",
        help="available models (HuBERT-Soft or HuBERT-Discrete)",
        choices=["soft", "discrete"],
    )
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .flac).",
        default=".flac",
        type=str,
    )
    parser.add_argument(
        "--checkpoint",
        help="path to the checkpoint to resume from.",
        default="/home/nis/tianyi.tan/.cache/torch/hub/checkpoints/model-layer12-450000.pt",
        type=str,
    )
    args = parser.parse_args()
    encode_dataset(args)
