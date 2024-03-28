import warnings
import pyworld as pw
import random
import librosa
from sklearn.preprocessing import minmax_scale
import torchaudio
import soundfile as sf
import numpy as np
from PIL import Image
import time
import pandas as pd
import traceback
import torch
import math
import os
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def extrair_f0(wav, sr):
    # print(type(wav))
    wav = wav.astype(np.double)

    # 160 = hop length
    f0, t = pw.dio(wav, fs=sr, frame_period=1000*160/sr)
    f0 = f0[f0 != 0.0]
    # f0,pf0,ppf0 = librosa.pyin(wav,sr=sr,fmin=50,fmax=600)
    return f0[~np.isnan(f0)]
    # return np.nan_to_num(f0, copy=True, nan=0.0)


def gerar_imagem(wav, sr, mel_spec, idade, genero):

    # shape certo é [120,401]
    imagem = torch.zeros([120, 401])

    mel_spec = torch.flip(torch.squeeze(
        mel_spec, dim=0).transpose(0, 1), [1, 0])
    f0 = torch.from_numpy(extrair_f0(torch.squeeze(wav, dim=0).numpy(), sr))

    if(f0.shape[0] > 401):
        f0 = f0[:401]

    imagem[:80, :] = mel_spec
    imagem[80:100, :133] = idade
    imagem[80:100, 133:268] = torch.std(f0)
    imagem[80:100, 268:401] = 1 if genero == "F" else 0
    imagem[100:120, :f0.shape[0]] = f0

    # salvar_imagem(imagem)
    imagem[torch.isnan(imagem)] = 0.0

    return imagem.transpose(0, 1).unsqueeze(0)


def salvar_imagem(imagem):
    im = plt.imshow(imagem)
    plt.savefig("prototipo.png")
