#!/usr/bin/env python3
import sys
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
from lime import lime_image
from utils.generic_utils import load_config, save_config_file
from utils.generic_utils import set_init_dict
from utils.generic_utils import NoamLR, binary_acc
from utils.generic_utils import save_best_checkpoint
from utils.tensorboard import TensorboardWriter
from utils.dataset import test_dataloader
from Models.spiraconv import SpiraConvV1, SpiraConvV2
import Models.Gradcam as gcam
from sklearn.preprocessing import minmax_scale
from gerar_audios import gerar_audio_from_cam
import librosa
import random
from utils.audio_processor import AudioProcessor
import warnings
warnings.filterwarnings("ignore")

warnings.filterwarnings("ignore")
sys.path.append('./')


# set random seed
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


CONFIG_PATH = "./checkpoint/config.json"
CHECKPOINT_PATH = "./checkpoint/best_checkpoint.pt"
TEST_CSV = "../Dataset/SPIRA_Dataset_V2/metadata_test.csv"
ROOT_DIR = "../Dataset/SPIRA_Dataset_V2/"
CUDA = False
INSERT_NOISE = True
BATCH_SIZE = 1
NUM_WORKERS = 10
NUM_NOISE_CONTROL = 3
NUM_NOISE_PATIENT = 3
RESULTS_PATH = "./resultados"

########################################################################################################


def save_spectrogram(spec, path, title="", ylabel='y', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    # axs.set_title(title + " (db)")

    # axs.set_ylabel("frequências (Hz)")
    # axs.set_xlabel('tempo (ms)')
    im = axs.imshow(spec,
                    origin='lower', aspect=aspect, cmap="inferno")
    # im = axs.imshow(spec, origin='lower', aspect=aspect,cmap="hot")
    if xmax:
        axs.set_xlim((0, xmax))

    fig.colorbar(im, ax=axs)
    plt.show(block=False)
    plt.savefig(path)


def use_grad_cam(model, testloader):
    # print(model[0])
    grad_cam = gcam.GradCam(model=model, feature_module=model.conv,
                            target_layer_names=["15"], use_cuda=False)
    audio_counter = 0
    for batch in testloader:
        if(audio_counter >= 0):
            print(
                "---------------------------------------------------------------------------")
            print("Áudio", audio_counter+1)
            audio_counter += 1
            images, targets, _, _ = batch
            window_counter = 0
            for image, target in zip(images, targets):
                # print(image.shape, target)
                print("\tWindow ", window_counter)
                grayscale_cam, output = grad_cam(
                    image.unsqueeze(0).unsqueeze(0), int(target.item()))

                grayscale_cam = torch.from_numpy(grayscale_cam)

                # isso é feito para converter o intervalo do sexo de 0 - 1 para 0-100
                image[268:401, 80:100] *= 100.0
                image[:, :80] *= 5.0
                # print(image.shape)
                cam = image.mul(grayscale_cam)

                folder_type = "paciente" if int(
                    target.item()) == 1 else "controle"
                classified_as = "paciente" if int(
                    output.item()) == 1 else "controle"
                folder = "{}/{}/{}-{}/window-{}/".format(RESULTS_PATH,
                                                         folder_type, folder_type, audio_counter, window_counter)

                os.makedirs(folder) if not os.path.isdir(folder) else None

                # print("product:", cam.transpose(0, 1).numpy().shape)
                # print("heatmap:", grayscale_cam.transpose(0, 1).numpy().shape)
                # print("image:", image.transpose(0, 1).numpy().shape)

                # original = image.transpose(0, 1)
                # original_spec = torch.fliplr(original[:80, :])
                # original_other = original[80:, :]
                # original = torch.cat((original_spec, original_other), 0)

                # heatmap = grayscale_cam.transpose(0, 1)
                # heatmap_spec = torch.fliplr(heatmap[:80, :])
                # heatmap_other = heatmap[80:, :]
                # heatmap = torch.cat((heatmap_spec, heatmap_other), 0)

                # product = cam.transpose(0, 1)
                # product_spec = torch.fliplr(product[:80, :])
                # product_other = product[80:, :]
                # product = torch.cat((product_spec, product_other), 0)

                original = image
                heatmap = grayscale_cam
                product = cam

                original_spec = torch.flipud(original[:, :80])
                original_other = original[:, 80:]
                original = torch.cat((original_spec, original_other), 1)

                heatmap_spec = torch.flipud(heatmap[:, :80])
                heatmap_other = heatmap[:, 80:]
                heatmap = torch.cat((heatmap_spec, heatmap_other), 1)

                product_spec = torch.flipud(product[:, :80])
                product_other = product[:, 80:]
                product = torch.cat((product_spec, product_other), 1)

                heatmap = torch.rot90(heatmap, 1, [0, 1])
                original = torch.rot90(original, 1, [0, 1])
                product = torch.rot90(product, 1, [0, 1])
                # heatmap = cv2.rotate(
                #     grayscale_cam.numpy()*255, cv2.ROTATE_90_CLOCKWISE)
                # original = cv2.rotate(
                #     image.numpy(), cv2.ROTATE_90_CLOCKWISE)
                # product = cv2.rotate(
                #     cam.numpy(), cv2.ROTATE_90_CLOCKWISE)

                # cv2.imwrite('{}heatmap-{}-{}-2.jpg'.format(folder, folder_type, classified_as),
                #             heatmap)

                # cv2.imwrite('{}original-{}-{}-2.jpg'.format(folder, folder_type, classified_as),
                #             original)

                # cv2.imwrite('{}produto-{}-{}-2.jpg'.format(
                #     folder, folder_type, classified_as), product*3)

                save_spectrogram(
                    heatmap, '{}heatmap-{}-{}.jpg'.format(folder, folder_type, classified_as))
                save_spectrogram(
                    original, '{}original-{}-{}.jpg'.format(folder, folder_type, classified_as))
                save_spectrogram(
                    product, '{}produto-{}-{}.jpg'.format(folder, folder_type, classified_as))
                window_counter += 1

                # break
            # break
    print("fim cam")


def run_test(testloader, c, model_name, ap):

    # define loss function
    criterion = nn.BCELoss(reduction='sum')

    padding_with_max_lenght = c.dataset['padding_with_max_lenght']
    if(model_name == 'spiraconv_v1'):
        model = SpiraConvV1(c)
    elif (model_name == 'spiraconv_v2'):
        model = SpiraConvV2(c)
    # elif(model_name == 'voicesplit'):
    else:
        raise Exception(" The model '"+model_name+"' is not suported")

    if c.train_config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=c.train_config['learning_rate'])
    else:
        raise Exception("The %s  not is a optimizer supported" %
                        c.train['optimizer'])

    step = 0
    if CHECKPOINT_PATH is not None:
        print("Loading checkpoint: %s" % CHECKPOINT_PATH)
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
            # print(checkpoint["model"].keys())
            if("module." in list(checkpoint["model"].keys())[0]):
                model = nn.DataParallel(model)
                # model.load_state_dict(checkpoint["model"])
            model.load_state_dict(checkpoint['model'])
            print("Model Sucessful Load !")
        except Exception as e:
            raise ValueError(
                "You need pass a valid checkpoint, may be you need check your config.json because de the of this checkpoint cause the error: {} ".format(e))
        step = checkpoint['step']
    else:
        raise ValueError("You need pass a checkpoint_path")

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

    test_dataloader = test_dataloader(c, ap, max_seq_len=max_seq_len)
    run_test(test_dataloader, c, c.model_name, ap)
