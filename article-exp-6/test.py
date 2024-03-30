#!/usr/bin/env python3
from utils.audio_processor import AudioProcessor
import random
import Models.Gradcam as gcam
from Models.spiraconv import SpiraConvV1, SpiraConvV2
from utils.dataset import test_dataloader
from utils.tensorboard import TensorboardWriter
from utils.generic_utils import save_best_checkpoint
from utils.generic_utils import NoamLR, binary_acc
from utils.generic_utils import set_init_dict
from utils.generic_utils import load_config, save_config_file
from lime import lime_image
import argparse
import cv2
import numpy as np
from PIL import Image
import time
import shap
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


def test(criterion, ap, model, c, testloader, step,  cuda, confusion_matrix=False):
    padding_with_max_lenght = c.dataset['padding_with_max_lenght']
    losses = []
    accs = []
    # print(model.conv)
    model.zero_grad()
    model.eval()
    loss = 0
    acc = 0
    preds = []
    targets = []
    with torch.no_grad():
        for feature, target, slices, targets_org in testloader:
            # try:
            if cuda:
                feature = feature.cuda()
                target = target.cuda()
            output = model(feature).float()

            # Calculate loss
            if not padding_with_max_lenght and not c.dataset['split_wav_using_overlapping']:
                target = target[:, :output.shape[1], :target.shape[2]]

            if c.dataset['split_wav_using_overlapping']:
                # unpack overlapping for calculation loss and accuracy
                if slices is not None and targets_org is not None:
                    idx = 0
                    new_output = []
                    new_target = []
                    for i in range(slices.size(0)):
                        num_samples = int(slices[i].cpu().numpy())

                        samples_output = output[idx:idx+num_samples]
                        output_mean = samples_output.mean()
                        samples_target = target[idx:idx+num_samples]
                        target_mean = samples_target.mean()

                        new_target.append(target_mean)
                        new_output.append(output_mean)
                        idx += num_samples

                    target = torch.stack(new_target, dim=0)
                    output = torch.stack(new_output, dim=0)
                    #print(target, targets_org)
                    if cuda:
                        output = output.cuda()
                        target = target.cuda()
                        targets_org = targets_org.cuda()
                    if not torch.equal(targets_org, target):
                        raise RuntimeError(
                            "Integrity problem during the unpack of the overlay for the calculation of accuracy and loss. Check the dataloader !!")

            loss += criterion(output, target).item()

            # calculate binnary accuracy
            y_pred_tag = torch.round(output)
            acc += (y_pred_tag == target).float().sum().item()
            preds += y_pred_tag.reshape(-1).int().cpu().numpy().tolist()
            targets += target.reshape(-1).int().cpu().numpy().tolist()
        if confusion_matrix:
            print("======== Confusion Matrix ==========")
            y_target = pd.Series(targets, name='Target')
            y_pred = pd.Series(preds, name='Predicted')
            df_confusion = pd.crosstab(y_target, y_pred, rownames=[
                                       'Target'], colnames=['Predicted'], margins=True)
            print(df_confusion)

        mean_acc = acc / len(testloader.dataset)
        mean_loss = loss / len(testloader.dataset)
    print("Test\n Loss:", mean_loss, "Acurracy: ", mean_acc)
    # target = [1,1]
    # print(model)
    # batch = next(iter(testloader))
    # images , target, _ , _ = batch
    # e = shap.GradientExplainer(model, images[5:15])
    # shap_values = e.shap_values(images[:5])
    # print(images.shape)
    # print(target[0])
    #test_numpy = images[0].unsqueeze(0).numpy()
    # im = plt.imshow(images[0].numpy(), cmap='hot')
    # print(images[0].numpy())
    # plt.savefig('my.png')
    # fig = shap.image_plot(shap_values, images[:5].numpy())
    # plt.figimage(fig,cmap="Greys")
    # plt.savefig('teste_controle.png')
    # data = Image.fromarray(images[0].numpy()).convert('LA')
    # print(target[0])
    # data.save("teste.png")
    return mean_acc


def use_grad_cam(model, testloader):

    grad_cam = gcam.GradCam(model=model, feature_module=model.conv,
                            target_layer_names=["15"], use_cuda=False)
    batch = next(iter(testloader))
    images, target, _, _ = batch
    # torch.set_printoptions(profile="full")
    # print(images[21].unsqueeze(0))
    # print(model(images[:20]))

    grayscale_cam = grad_cam(images[21].unsqueeze(0).unsqueeze(0), None)
    img = images[21].unsqueeze(1).transpose(
        0, 1).contiguous().transpose(0, 2).contiguous()
    # print("greyscale_cam")
    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    cam = gcam.show_cam_on_image(img, grayscale_cam)

    gb_model = gcam.GuidedBackpropReLUModel(model=model, use_cuda=False)
    gb = gb_model(images[21].unsqueeze(0), target_category=None)
    gb = gb.transpose((2, 1, 0))
    # print("gb_model")
    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = gcam.deprocess_image(cam_mask*gb)

    gb = gcam.deprocess_image(gb)
    # print(img.shape)
    # img = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2BGR)

    cv2.imwrite('cam.jpg', cam)
    cv2.imwrite('cam255.jpg', cam*255)

    cv2.imwrite('audio_original.jpg', np.float32(images[21]))
    cv2.imwrite('audio_original2.jpg', np.float32(images[21])*255)
    # im = plt.imshow(cam, cmap='hot')
    # plt.savefig('cam.jpg')
    cv2.imwrite('gb.jpg', gb)
    cv2.imwrite('cam_gb.jpg', cam_gb)
    # print("fim do cam")


def run_test(args, checkpoint_path, testloader, c, model_name, ap, cuda=True):

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
    if checkpoint_path is not None:
        print("Loading checkpoint: %s" % checkpoint_path)
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
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

    # use_grad_cam(model,testloader)
    # convert model from cuda
    if cuda:
        model = model.cuda()

    model.train(False)
    test_acc = test(criterion, ap, model, c, testloader,
                    step, cuda=cuda, confusion_matrix=True)


if __name__ == '__main__':
    # python test.py --test_csv ../SPIRA_Dataset_V1/metadata_test.csv -r ../SPIRA_Dataset_V1/ --checkpoint_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1/spiraconv/checkpoint_1068.pt --config_path ../checkpoints/spiraconv-trained-with-SPIRA_Dataset_V1/spiraconv/config.json  --batch_size 5 --num_workers 2 --no_insert_noise True

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_csv', type=str, required=True,
                        help="test csv example: ../SPIRA_Dataset_V1/metadata_test.csv")
    parser.add_argument('-r', '--test_root_dir', type=str, required=True,
                        help="Test root dir example: ../SPIRA_Dataset_V1/")
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="json file with configurations get in checkpoint path")
    parser.add_argument('--checkpoint_path', type=str, default=None, required=True,
                        help="path of checkpoint pt file, for continue training")
    parser.add_argument('--batch_size', type=int, default=20,
                        help="Batch size for test")
    parser.add_argument('--num_workers', type=int, default=10,
                        help="Number of Workers for test data load")
    parser.add_argument('--no_insert_noise', type=bool, default=False,
                        help=" No insert noise in test ?")
    parser.add_argument('--num_noise_control', type=int, default=1,
                        help="Number of Noise for insert in control")
    parser.add_argument('--num_noise_patient', type=int, default=0,
                        help="Number of Noise for insert in patient")

    args = parser.parse_args()

    c = load_config(args.config_path)
    ap = AudioProcessor(**c.audio)

    if not args.no_insert_noise:
        c.data_aumentation['insert_noise'] = True
    else:
        c.data_aumentation['insert_noise'] = False

    # ste values for noisy insertion in test
    c.data_aumentation["num_noise_control"] = args.num_noise_control
    c.data_aumentation["num_noise_patient"] = args.num_noise_patient

    print("Insert noise ?", c.data_aumentation['insert_noise'])

    c.dataset['test_csv'] = args.test_csv
    c.dataset['test_data_root_path'] = args.test_root_dir

    c.test_config['batch_size'] = args.batch_size
    c.test_config['num_workers'] = args.num_workers
    max_seq_len = c.dataset['max_seq_len']

    test_dataloader = test_dataloader(c, ap, max_seq_len=max_seq_len)

    run_test(args, args.checkpoint_path, test_dataloader,
             c, c.model_name, ap, cuda=False)
