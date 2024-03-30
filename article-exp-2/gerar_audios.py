#!/usr/bin/env python3
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('./')
import matplotlib.pyplot as plt

import os
import math
import torch
import torch.nn as nn
import traceback
import pandas as pd
import shap
import time
from PIL import Image
import numpy as np
import cv2
import argparse
import soundfile as sf
import torchaudio
from lime import lime_image

from utils.generic_utils import load_config, save_config_file
from utils.generic_utils import set_init_dict

from utils.generic_utils import NoamLR, binary_acc

from utils.generic_utils import save_best_checkpoint

from utils.tensorboard import TensorboardWriter

from utils.dataset import test_dataloader

from Models.spiraconv import SpiraConvV1, SpiraConvV2
import Models.Gradcam as gcam
from utils.audio_processor import AudioProcessor 

from sklearn.preprocessing import minmax_scale
import librosa
import random
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
NUM_NOISE_CONTROL = 1
NUM_NOISE_PATIENT = 0
HOP_LENGTH= 160
WIN_LENGTH= 400
PATH_AUDIO_PARA_TESTAR = "../Dataset/SPIRA_Dataset_V2/controle/be100edb-73f5-4920-82fa-a07092ca28b8_1.wav"
MIN_DB_LVL= -80
REF_DB_LVL = 40
N_FFT=1200
NUM_FREQ = 601
SAMPLE_RATE = 16000
NUM_MELS = 80


def mel_to_spec(mel):
	audio_class = torchaudio.transforms.InverseMelScale(n_stft=NUM_FREQ,n_mels=NUM_MELS,sample_rate=SAMPLE_RATE)
	return audio_class(mel)


def torch_spec2wav(spectrogram, phase=None):
		# spectrogram = spectrogram.transpose(2,1)
		# phase = phase.transpose(2,1)
		# denormalise spectrogram
		print(spectrogram.shape)
		S =  (torch.clamp(spectrogram, min=0.0, max=1.0)) #* - MIN_DB_LVL
		#S = S + REF_DB_LVL
		# db_to_amp
		stft_matrix = torch.pow(10.0, S * 0.05)
		# invert phase
		phase = torch.stack([phase.cos(), phase.sin()], dim=-1).to(dtype=stft_matrix.dtype, device=stft_matrix.device)
		stft_matrix = stft_matrix.unsqueeze(-1).expand_as(phase)
		return torch.istft(stft_matrix * torch.exp(phase), N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH, window=torch.hamming_window(WIN_LENGTH, periodic=False, alpha=0.5, beta=0.5).to(device=stft_matrix.device), center=True, normalized=False, onesided=True, length=None)



def amp_to_db(x):
	return 20.0 * np.log10(np.maximum(1e-5, x))

def db_to_amp(x):
	return np.power(10.0, x * 0.05)

def denormalize(S):
	return (np.clip(S, 0.0, 1.0) - 1.0) * - MIN_DB_LVL


def istft_phase(mag, phase,win_length=WIN_LENGTH):
	stft_matrix = mag * np.exp(1j*phase)
	return librosa.istft(stft_matrix,hop_length=HOP_LENGTH,win_length=WIN_LENGTH)

def gerar_audio_inicial(ap):
	ap.feature = "spectrogram"

	wav = ap.load_wav(PATH_AUDIO_PARA_TESTAR)

	spec = ap.get_feature_from_audio(wav)
	magnitude, phase = librosa.magphase(np.float32(spec))
	print(spec.shape,phase.shape)
	torch_spec2wav_result = np.float32(torch_spec2wav(spec,torch.from_numpy(phase)).squeeze(0))

	inverse_stft = librosa.istft(magnitude*phase)

	griffinlim_magXphase = librosa.griffinlim(magnitude*phase)
	griffinlim = librosa.griffinlim(np.float32(spec.squeeze(0)))

	sf.write("./audios_gerados/griffinlim_magXphase.wav",griffinlim_magXphase, 16000)
	sf.write("./audios_gerados/griffinlim.wav",griffinlim, 16000)
	sf.write("./audios_gerados/inverse_stft.wav",inverse_stft, 16000)
	sf.write("./audios_gerados/torch_spec2wav_result.wav",torch_spec2wav_result, 16000)
	# #####código do edresson
	spectrogram, phase = np.float32(spec.T), phase.T
	print(spec.T.shape)
	# S = db_to_amp(denormalize(spectrogram) + REF_DB_LVL)
	S = spectrogram + REF_DB_LVL
	result = istft_phase(spectrogram, phase)

	sf.write("./audios_gerados/mozilla.wav",inverse_stft, 16000)

def gerar_audio_librosa_apenas():
	wav,sr = librosa.core.load(PATH_AUDIO_PARA_TESTAR,16000)
	print(wav.shape[0]/sr)

	spectrogram = librosa.stft(wav, n_fft= N_FFT,hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
	print(spectrogram.shape, spectrogram.max())

	magnitude, phase = librosa.magphase(spectrogram)
	print(magnitude.shape,phase.shape)
	print(phase.max())
	inverse_stft = librosa.istft(magnitude*phase)

	griffinlim_magXphase = librosa.griffinlim(magnitude*phase)
	griffinlim = librosa.griffinlim(spectrogram)

	
	
	result = istft_phase(spectrogram, phase)
	print(result.max())
	sf.write("./audios_gerados/griffinlim_magXphase.wav",griffinlim_magXphase, 16000)
	sf.write("./audios_gerados/griffinlim.wav",griffinlim, 16000)
	sf.write("./audios_gerados/inverse_stft.wav",inverse_stft, 16000)
	sf.write("./audios_gerados/mozilla.wav",result, 16000)

def gerar_audio(ap):

	# \| librosa

	wav,sr = librosa.core.load(PATH_AUDIO_PARA_TESTAR,SAMPLE_RATE)
	spectrogram = librosa.stft(wav, n_fft= N_FFT,hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
	magnitude, phase = librosa.magphase(spectrogram)

	print("Spec.shape:{}, magnitude.shape:{}, phase.shape:{}".format(spectrogram.shape,magnitude.shape,phase.shape))
	print("Spec type:{}, magnitude type:{}, phase type:{}".format(type(spectrogram),type(magnitude),type(phase)))
	
	result_librosa = istft_phase(spectrogram, phase)

	sf.write("./audios_gerados/audio-reconstruido-librosa.wav",result_librosa, SAMPLE_RATE)

	
	# \| torch
	ap.feature = "spectrogram"

	wav = ap.load_wav(PATH_AUDIO_PARA_TESTAR)
	
	spectrogram_torch = ap.wav2feature(wav)
	
	print("Spec.shape:{}, magnitude.shape:{}, phase.shape:{}".format(spectrogram_torch.shape,magnitude.shape,phase.shape))
	print("Spec type:{}, magnitude type:{}, phase type:{}".format(type(spectrogram_torch),type(magnitude),type(phase)))
	

	result_torch = istft_phase(spectrogram_torch.numpy()[0],phase)
	
	sf.write("./audios_gerados/audio-reconstruido-torch.wav",result_torch, SAMPLE_RATE)

	
	# \| torch griffinlim

	griffinlim = librosa.griffinlim(spectrogram_torch.squeeze(0).numpy(), hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
	sf.write("./audios_gerados/audio-reconstruido-torch-griffinlim.wav",griffinlim, SAMPLE_RATE)




def gerar_audio_from_mel(ap):
	# \| librosa

	wav,sr = librosa.core.load(PATH_AUDIO_PARA_TESTAR,SAMPLE_RATE)
	spectrogram = librosa.stft(wav, n_fft= N_FFT,hop_length=HOP_LENGTH, win_length=WIN_LENGTH)
	magnitude, phase = librosa.magphase(spectrogram)

	print("Spec.shape:{}, magnitude.shape:{}, phase.shape:{}".format(spectrogram.shape,magnitude.shape,phase.shape))
	print("Spec type:{}, magnitude type:{}, phase type:{}".format(type(spectrogram),type(magnitude),type(phase)))

	result_librosa = istft_phase(spectrogram, phase)

	sf.write("./audios_gerados/testes_mel/audio-reconstruido-librosa.wav",result_librosa, SAMPLE_RATE)

	ap.feature = "melspectrogram"

	wav = ap.load_wav(PATH_AUDIO_PARA_TESTAR)
	
	mel_torch = ap.wav2feature(wav)
	print("Mel.shape:{}".format(mel_torch.shape))

	spec_from_mel= mel_to_spec(mel_torch)
	print("Spec_from_mel.shape:{}".format(spec_from_mel.shape))
	result_torch = istft_phase(spec_from_mel.numpy()[0],phase)
	print(result_torch.shape)
	sf.write("./audios_gerados/testes_mel/audio-reconstruido-torch.wav",result_torch, SAMPLE_RATE)
	


def gerar_audio_from_cam(audio,phase,save_path,nome, normalize = False):
	spec_from_mel = mel_to_spec(audio)

	result = istft_phase(spec_from_mel.numpy(),phase)

	if normalize:
		result = librosa.util.normalize(result, axis=0)

	print(np.max(result),np.min(result))

	sf.write("{}/{}".format(save_path,nome),result, SAMPLE_RATE)

if __name__ == "__main__":
	c = load_config(CONFIG_PATH)


	ap = AudioProcessor(**c.audio)	
	
	# print(ap.__dict__)
	TYPE = ap.feature

	if not INSERT_NOISE:
		c.data_aumentation['insert_noise'] = True
	else:
		c.data_aumentation['insert_noise'] = False

	# set values for noisy insertion in test
	c.data_aumentation["num_noise_control"] = NUM_NOISE_CONTROL
	c.data_aumentation["num_noise_patient"] = NUM_NOISE_PATIENT

	print("Insert noise ?", c.data_aumentation['insert_noise'])

	c.dataset['test_csv'] = TEST_CSV
	c.dataset['test_data_root_path'] = ROOT_DIR


	c.test_config['batch_size'] = BATCH_SIZE
	c.test_config['num_workers'] = NUM_WORKERS
	max_seq_len = c.dataset['max_seq_len'] 
	
	# gerar_audio_librosa_apenas()
	# gerar_audio(ap)
	gerar_audio_from_mel(ap)
