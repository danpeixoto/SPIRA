#!/usr/bin/env python3
from utils.audio_processor import AudioProcessor
import random
import librosa
from sklearn.preprocessing import minmax_scale
import models.Gradcam as gcam
from models.spiraconv import SpiraConvV1, SpiraConvV2
from utils.dataset import test_dataloader
from utils.tensorboard import TensorboardWriter
from utils.generic_utils import save_best_checkpoint
from utils.generic_utils import NoamLR, binary_acc
from utils.generic_utils import set_init_dict
from utils.generic_utils import load_config, save_config_file
import torchaudio
import soundfile as sf
import argparse
import cv2
import numpy as np
from PIL import Image
import time
import pandas as pd
import traceback
import torch.nn as nn
import torch
import math
import os
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.append('./')


# set random seed
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


CONFIG_PATH = "../checkpoints/grad-cam/config.json"
CHECKPOINT_PATH = "../checkpoints/grad-cam/best_checkpoint.pt"
TEST_CSV = "../Dataset/SPIRA_Dataset_V2/grad_cam.csv"
ROOT_DIR = "../Dataset/SPIRA_Dataset_V2/"
CUDA = False
INSERT_NOISE = False
BATCH_SIZE = 1
NUM_WORKERS = 10
NUM_NOISE_CONTROL = 4
NUM_NOISE_PATIENT = 3
HOP_LENGTH = 320
WIN_LENGTH = 1024
PATH_AUDIO_PARA_TESTAR = "../Dataset/SPIRA_Dataset_V2/controle/be100edb-73f5-4920-82fa-a07092ca28b8_1.wav"
MIN_DB_LVL = -80
REF_DB_LVL = 40
N_FFT = 1200
NUM_FREQ = 601
SAMPLE_RATE = 16000
NUM_MELS = 64


def mel_to_spec(mel):
    audio_class = torchaudio.transforms.InverseMelScale(
        n_stft=NUM_FREQ, n_mels=NUM_MELS, sample_rate=SAMPLE_RATE)
    mel = torch.from_numpy(mel)
    return audio_class(mel)


def torch_spec2wav(spectrogram, phase=None):
    # spectrogram = spectrogram.transpose(2,1)
    # phase = phase.transpose(2,1)
    # denormalise spectrogram
    # print(spectrogram.shape)
    S = (torch.clamp(spectrogram, min=0.0, max=1.0))  # * - MIN_DB_LVL
    #S = S + REF_DB_LVL
    # db_to_amp
    stft_matrix = torch.pow(10.0, S * 0.05)
    # invert phase
    phase = torch.stack([phase.cos(), phase.sin()], dim=-
                        1).to(dtype=stft_matrix.dtype, device=stft_matrix.device)
    stft_matrix = stft_matrix.unsqueeze(-1).expand_as(phase)
    return torch.istft(stft_matrix * torch.exp(phase), N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=torch.hamming_window(WIN_LENGTH, periodic=False, alpha=0.5, beta=0.5).to(device=stft_matrix.device), center=True, normalized=False, onesided=True, length=None)


def amp_to_db(x):
    return 20.0 * np.log10(np.maximum(1e-5, x))


def db_to_amp(x):
    return np.power(10.0, x * 0.05)


def denormalize(S):
    return (np.clip(S, 0.0, 1.0) - 1.0) * - MIN_DB_LVL


def istft_phase(mag, phase, win_length=WIN_LENGTH):
    # print(phase.shape)
    stft_matrix = mag * np.exp(1j*phase)
    return librosa.istft(stft_matrix, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)


def gerar_audio_from_cam(audio, phase, save_path, nome, normalize=False):
    # print(type(audio), type(phase))
    spec_from_mel = mel_to_spec(audio)
    # print(spec_from_mel.shape)
    result = istft_phase(spec_from_mel.numpy(), phase.numpy())
    # result = torch_spec2wav(spec_from_mel)
    # result = librosa.griffinlim(
    # spec_from_mel.numpy(), n_iter=400, hop_length=HOP_LENGTH, win_length=WIN_LENGTH)

    if normalize:
        result = librosa.util.normalize(result, axis=0)

    # print(np.max(result), np.min(result))

    sf.write("{}/{}".format(save_path, nome), result, SAMPLE_RATE)
