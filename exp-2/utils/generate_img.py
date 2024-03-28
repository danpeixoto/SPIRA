import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import os
import math
import torch
import traceback
import pandas as pd
import time
from PIL import Image
import numpy as np
import soundfile as sf
import torchaudio
from sklearn.preprocessing import minmax_scale
import librosa
import random
import pyworld as pw
import cv2
import imutils

def extrair_f0(wav,sr):
                # print(type(wav))
                wav = wav.astype(np.double)

                #160 = hop length
                f0, t = pw.dio(wav,fs=sr, frame_period=1000*160/sr)
                #f0 = f0[f0!=0.0]
                # f0,pf0,ppf0 = librosa.pyin(wav,sr=sr,fmin=50,fmax=600)
                return f0[~np.isnan(f0)]
                #return np.nan_to_num(f0, copy=True, nan=0.0)


def gerar_imagem(wav,sr,mel_spec,idade,genero):
                
                #shape certo é [120,401]        
                imagem = torch.zeros([40,401])

                #mel_spec= torch.flip(torch.squeeze(mel_spec,dim=0).transpose(0,1),[1,0])
                #f0 = torch.from_numpy(extrair_f0(torch.squeeze(wav,dim=0).numpy(),sr))
                f0 = extrair_f0(torch.squeeze(wav,dim=0).numpy(),sr)

                #if(f0.shape[0] > 401):
                #    f0 = f0[:401]
                #print(f0.min(),f0.max())
                retangulo_f0 = torch.zeros([20,f0.shape[0]]).numpy()
                retangulo_f0[:,:] = f0
                #print(retangulo_f0.shape[0],retangulo_f0.shape[1])
                f0 = cv2.resize(retangulo_f0, (401,20), interpolation= cv2.INTER_CUBIC ) 
                #print(f0.shape)
                #print(f0[400])
                #print(np.resize(f0,(401,1)).max())
                #f0 = imutils.resize(f0.reshape(-1,1),height=401)

                #f0 = np.resize(f0,(401,1))
                f0 = torch.from_numpy(f0)
                #f0 = f0.squeeze(1)

                #imagem[:80,:] = mel_spec
                imagem[0:20,:133] = idade
                imagem[0:20,133:268] = torch.std(f0)
                imagem[0:20,268:401] = 1 if genero == "F" else 0
                imagem[20:40,:f0.shape[1]] = f0

                #salvar_imagem(imagem)
                imagem[torch.isnan(imagem)] = 0.0
                return imagem.transpose(0,1).unsqueeze(0)

def salvar_imagem(imagem):
                im = plt.imshow(imagem)
                plt.savefig("prototipo.png")




