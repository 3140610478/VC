import os
import sys
import argparse
import pickle
import glob
import random
import numpy as np
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import torchaudio
import torchaudio.transforms as T


base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    import config
    from Networks import Vocoder


def normalize_mel(wavspath, threshold=1/256):
    wav_files = glob.glob(
        os.path.join(wavspath, '**', '*.wav'), recursive=True
    )

    mel = list()
    for wavpath in tqdm(wav_files, desc='Preprocess wav to mel'):
        wave, sample_rate = torchaudio.load(wavpath)
        vocoder = Vocoder()
        m = vocoder(wave, sample_rate)

        if m.shape[-1] >= config.temporal_stride:
            mel.append(m[0])

    if len(mel) > 1:
        mel = torch.cat(mel, dim=1)
    else:
        mel = mel[0]

    mel_sum = mel.sum(dim=0)
    filter = mel_sum > (mel_sum.mean()*threshold - 1e-9)

    filter = torch.logical_not(filter)
    cnt = 1 if filter[0] else 0
    lst = []
    for i in range(1, len(filter)):
        if filter[i]:
            cnt += 1
        elif filter[i-1]:
            lst.append(cnt)
            cnt = 0
    if cnt:
        lst.append(cnt)
    if lst:
        smooth = int(torch.tensor(lst, dtype=torch.float32).mean())
        smooth = min(smooth, filter.shape[0])
        filter = torch.logical_not(filter)

        filter = torch.cat((filter, filter[:smooth]))
        filter = torch.tensor(
            [torch.any(filter[i:i+smooth]) for i in range(len(filter) - smooth)]
        )
        mel = (mel.T[filter]).T

    mel_mean = mel.mean(dim=1, keepdim=True)
    mel_std = mel.std(dim=1, keepdim=True) + 1e-9
    mel = ((mel - mel_mean) / mel_std)

    mel_normalized = []
    n_frames = mel.shape[1]
    for i in range(1, n_frames-config.temporal_stride, config.temporal_stride):
        mel_normalized.append(mel[:, i:i+config.temporal_stride])
    mel_normalized = torch.stack(mel_normalized)

    return mel_normalized.cpu(), mel_mean.cpu(), mel_std.cpu()


def preprocess_dataset(data_path, speaker_id, preprocessed_path, threshold=1/256):
    print(f"Preprocessing data for speaker: {speaker_id}.")

    mel_normalized, mel_mean, mel_std = normalize_mel(data_path, threshold)

    if not os.path.exists(os.path.join(preprocessed_path, speaker_id)):
        os.makedirs(os.path.join(preprocessed_path, speaker_id))

    data = {
        "spec": mel_normalized,
        "mean": mel_mean,
        "std": mel_std,
    }

    torch.save(
        data,
        os.path.join(preprocessed_path, speaker_id, f"{speaker_id}.data"),
    )

    print(f"Preprocessed and saved data for speaker: {speaker_id}.")


if __name__ == '__main__':
    os.chdir(base_folder)

    for speaker_id in config.speaker_ids:
        print("train")
        preprocess_dataset(
            os.path.join(config.original_train_data, speaker_id),
            speaker_id=speaker_id,
            preprocessed_path=config.preprocessed_train_data,
        )
        print("val")
        preprocess_dataset(
            os.path.join(config.original_eval_data, speaker_id),
            speaker_id=speaker_id,
            preprocessed_path=config.preprocessed_eval_data,
            threshold=0,
        )
