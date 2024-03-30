
#!/usr/bin/env python3
import sys
import matplotlib.pyplot as plt
import os
import math
import torch
import torch.nn as nn
import traceback
import pandas as pd
import time
import numpy as np
import cv2
import argparse
import soundfile as sf
import models.Gradcam as gcam
import librosa
import random
import warnings

from PIL import Image
from gerar_audios import gerar_audio_from_cam
from sklearn.preprocessing import minmax_scale
from utils.audio_processor import AudioProcessor
from utils.models import return_model
from utils.generic_utils import load_config, save_config_file
from utils.generic_utils import set_init_dict
from utils.generic_utils import NoamLR, binary_acc
from utils.generic_utils import save_best_checkpoint
from utils.tensorboard import TensorboardWriter
from utils.dataset import test_dataloader
from models.spiraconv import SpiraConvV1, SpiraConvV2

warnings.filterwarnings("ignore")
sys.path.append('./')


# set random seed
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


CONFIG_PATH = "./checkpoint/exp-pann-5-config.json"
CHECKPOINT_PATH = "./checkpoint/exp-pann-5-checkpoint.pt"
TEST_CSV = "../Dataset/SPIRA_Dataset_V2/metadata_test.csv"
ROOT_DIR = "../Dataset/SPIRA_Dataset_V2/"
CUDA = True
INSERT_NOISE = True
BATCH_SIZE = 1
NUM_WORKERS = 10
NUM_NOISE_CONTROL = 4
NUM_NOISE_PATIENT = 3
RESULTS_PATH = "./resultados"

########################################################################################################


def save_spectrogram(spec, path, title="", ylabel='y', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    # axs.set_title(title + " (db)")

    axs.set_ylabel("mel")
    axs.set_xlabel('tempo (ms)')
    im = axs.imshow(spec,
                    origin='lower', aspect=aspect, cmap="inferno")
    # im = axs.imshow(spec, origin='lower', aspect=aspect,cmap="hot")
    if xmax:
        axs.set_xlim((0, xmax))

    fig.colorbar(im, ax=axs)
    plt.show(block=False)
    plt.savefig(path)


def use_grad_cam(model, testloader):
    # print(model)
    grad_cam = gcam.GradCam(model=model, feature_module=model.base,
                            target_layer_names=["conv_block6"], use_cuda=False)
    audio_counter = 0
    for batch in testloader:
        # break
        if(audio_counter >= 0):
            print(
                "---------------------------------------------------------------------------")
            print("Áudio", audio_counter+1)
            audio_counter += 1
            images, targets, _, _, audio_path, phases = batch
            window_counter = 0
            # print(model(images))
            # break
            for image, target, phase in zip(images, targets, phases):

                phase = torch.from_numpy(np.array(phase))

                print("\tWindow ", window_counter)
                # print(image.shape)
                # print(model(image.unsqueeze(0)))
                # break
                grayscale_cam, output = grad_cam(
                    image.unsqueeze(0), int(target.item()))
                # print(output)
                # break
                # continue
                grayscale_cam = torch.from_numpy(grayscale_cam)
                # image = librosa.db_to_amplitude(image)
                # image_max_amp = image.max()

                cam = torch.mul(grayscale_cam, image)

                # image = librosa.amplitude_to_db(image)
                # cam = librosa.amplitude_to_db(cam)

                # print(cam.min(), cam.max())

                # break
                folder_type = "paciente" if int(
                    target.item()) == 1 else "controle"
                classified_as = "paciente" if int(
                    output.item()) == 1 else "controle"
                folder = "{}/{}/{}-{}/window-{}/".format(RESULTS_PATH,
                                                         folder_type, folder_type, audio_counter, window_counter)

                os.makedirs(folder) if not os.path.isdir(folder) else None

                heatmap = cv2.flip(cv2.rotate(
                    grayscale_cam.numpy(), cv2.ROTATE_90_CLOCKWISE), 1)
                original = cv2.flip(cv2.rotate(
                    image.numpy(), cv2.ROTATE_90_CLOCKWISE), 1)
                product = cv2.flip(cv2.rotate(
                    cam.numpy(), cv2.ROTATE_90_CLOCKWISE), 1)

                # phase = torch.rot90(torch.fliplr(
                # phase), 1, [0, 1]).numpy()

                save_spectrogram(
                    heatmap, '{}heatmap-{}-{}.jpg'.format(folder, folder_type, classified_as))
                save_spectrogram(
                    original, '{}original-{}-{}.jpg'.format(folder, folder_type, classified_as))
                save_spectrogram(
                    product, '{}produto-{}-{}.jpg'.format(folder, folder_type, classified_as))

                print(original.shape, "original")
                # gerar_audio_from_cam(
                #     original, phase, folder, "original.wav")
                # gerar_audio_from_cam(
                #     product, phase, folder, "product.wav")
                # window_counter += 1
                # break
            #break
    print("fim cam")


def normalize_tensor(tensor):
    min_value = torch.min(tensor)
    range_value = torch.max(tensor) - min_value

    if(range_value > 0):
        normalized = (tensor - min_value) / range_value
    else:
        normalized = torch.zeros(tensor.shape)

    return normalized


def run_test(testloader, c, model_name, ap):

    # define loss function
    criterion = nn.BCELoss(reduction='sum')

    model = return_model(c, None)
    # print(model.base.conv_block6)
    # print(model)
    # sys.exit()
    if c.train_config['optimizer'] == 'adam':
        print("optimizer")
        optimizer = torch.optim.Adam(model.parameters(
        ), lr=c.train_config['learning_rate'], weight_decay=c.train_config['weight_decay'])
    else:
        raise Exception("The %s  not is a optimizer supported" %
                        c.train['optimizer'])

    step = 0
    if CHECKPOINT_PATH is not None:
        print("Loading checkpoint: %s" % CHECKPOINT_PATH)
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
            # print(checkpoint["model"].keys())
            # if("module." in list(checkpoint["model"].keys())[0]):
            #    model = nn.DataParallel(model)
            # model.load_state_dict(checkpoint["model"])
            model.load_state_dict(checkpoint['model'])
            print("Model Sucessful Load !")
        except Exception as e:
            raise ValueError(
                "You need pass a valid checkpoint, may be you need check your config.json because de the of this checkpoint cause the error: {} ".format(e))
        step = checkpoint['step']
    else:
        raise ValueError("You need pass a checkpoint_path")
    model.train(False)
    use_grad_cam(model, testloader)


def normalize_tensor(tensor):
    min_value = torch.min(tensor)
    range_value = torch.max(tensor) - min_value

    if(range_value > 0):
        normalized = (tensor - min_value) / range_value
    else:
        normalized = torch.zeros(tensor.shape)

    return normalized


if __name__ == "__main__":
    c = load_config(CONFIG_PATH)

    ap = AudioProcessor(**c.audio)

    # print(ap.__dict__)
    TYPE = ap.feature

    if INSERT_NOISE:
        c.data_aumentation['insert_spira_noise'] = True
    else:
        c.data_aumentation['insert_spira_noise'] = False
    # c.data_aumentation['insert_noise'] = False
    # set values for noisy insertion in test
    c.data_aumentation["num_noise_control"] = NUM_NOISE_CONTROL
    c.data_aumentation["num_noise_patient"] = NUM_NOISE_PATIENT

    print("Insert spira_noise noise ?",
          c.data_aumentation['insert_spira_noise'])

    c.dataset['test_csv'] = TEST_CSV
    c.dataset['test_data_root_path'] = ROOT_DIR

    c.test_config['batch_size'] = BATCH_SIZE
    c.test_config['num_workers'] = NUM_WORKERS
    max_seq_len = c.dataset['max_seq_len']

    test_dataloader = test_dataloader(
        c, ap, insert_noise=c.data_aumentation['insert_noise'], max_seq_len=max_seq_len, num_additive_noise=3)
    run_test(test_dataloader, c, c.model_name, ap)
