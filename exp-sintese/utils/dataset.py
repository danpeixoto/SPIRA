import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch import stack
import numpy as np
import pandas as pd
import random
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import torchaudio

import librosa


class Dataset(Dataset):
    """
    Class for load a train and test from dataset generate by import_librispeech.py and others
    """

    def __init__(self, c, ap, train=True, max_seq_len=None, test=False):
        # set random seed
        random.seed(c['seed'])
        torch.manual_seed(c['seed'])
        torch.cuda.manual_seed(c['seed'])
        np.random.seed(c['seed'])
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        self.c = c
        self.ap = ap
        self.train = train
        self.test = test
        self.dataset_csv = c.dataset['train_csv'] if train else c.dataset['eval_csv']
        self.dataset_root = c.dataset['train_data_root_path'] if train else c.dataset['eval_data_root_path']
        if test:
            self.dataset_csv = c.dataset['test_csv']
            self.dataset_root = c.dataset['test_data_root_path']
        self.noise_csv = c.dataset['noise_csv']
        self.noise_root = c.dataset['noise_data_root_path']
        assert os.path.isfile(
            self.dataset_csv), "Test or Train CSV file don't exists! Fix it in config.json"
        assert os.path.isfile(
            self.noise_csv), "Noise CSV file don't exists! Fix it in config.json"
        assert (not self.c.dataset['padding_with_max_lenght'] and self.c.dataset['split_wav_using_overlapping']) or (self.c.dataset['padding_with_max_lenght']
                                                                                                                     and not self.c.dataset['split_wav_using_overlapping']), "You cannot use the padding_with_max_length option in conjunction with the split_wav_using_overlapping option, disable one of them !!"

        # read csvs
        self.dataset_list = pd.read_csv(self.dataset_csv, sep=',').values
        self.noise_list = pd.read_csv(self.noise_csv, sep=',').values
        # noise config
        self.num_noise_files = len(self.noise_list)-1
        self.control_class = c.dataset['control_class']
        self.patient_class = c.dataset['patient_class']
        print(self.control_class)
        # get max seq lenght for padding
        if self.c.dataset['padding_with_max_lenght'] and train and not self.c.dataset['max_seq_len'] and not self.c.dataset['split_wav_using_overlapping']:
            self.max_seq_len = 0
            min_seq = float('inf')
            for idx in range(len(self.dataset_list)):
                wav = self.ap.load_wav(os.path.join(
                    self.dataset_root, self.dataset_list[idx][0]))
                # calculate time step dim using hop lenght
                seq_len = int((wav.shape[1]/c.audio['hop_length'])+1)
                if seq_len > self.max_seq_len:
                    self.max_seq_len = seq_len
                if seq_len < min_seq:
                    min_seq = seq_len
                    print(self.dataset_list[idx][0])
            print("The Max Time dim Lenght is: {} (+- {} seconds)".format(self.max_seq_len,
                  (self.max_seq_len*self.c.audio['hop_length'])/self.ap.sample_rate))
            print("The Min Time dim Lenght is: {} (+- {} seconds)".format(min_seq,
                  (min_seq*self.c.audio['hop_length'])/self.ap.sample_rate))

        elif self.c.dataset['split_wav_using_overlapping']:
            # set max len for window_len seconds multiply by sample_rate and divide by hop_lenght
            self.max_seq_len = int(
                ((self.c.dataset['window_len']*self.ap.sample_rate)/c.audio['hop_length'])+1)
            print("The Max Time dim Lenght is: ", self.max_seq_len, "It's use overlapping technique, window:",
                  self.c.dataset['window_len'], "step:", self.c.dataset['step'])
        else:  # for eval set max_seq_len in train mode
            if self.c.dataset['max_seq_len']:
                self.max_seq_len = self.c.dataset['max_seq_len']
            else:
                self.max_seq_len = max_seq_len

    def get_max_seq_lenght(self):
        return self.max_seq_len

    def __getitem__(self, idx):
        wav = self.ap.load_wav(os.path.join(
            self.dataset_root, self.dataset_list[idx][0]))
        #print("FILE:", os.path.join(self.dataset_root, self.dataset_list[idx][0]), wav.shape)
        class_name = self.dataset_list[idx][1]  # TROCAR numero original 1

        # print(self.dataset_list[idx][0])
        # its assume that noise file is biggest than wav file !!
        if self.c.data_aumentation['insert_noise']:
            # print("inserting noise")
            # Experiments do different things within the dataloader. So depending on the experiment we will have a different random state here in get item. To reproduce the tests always using the same noise we need to set the seed again, ensuring that all experiments see the same noise in the test !!
            if self.test:
                # multiply by idx, because if not we generate same some for all files !
                random.seed(self.c['seed']*idx)
                torch.manual_seed(self.c['seed']*idx)
                torch.cuda.manual_seed(self.c['seed']*idx)
                np.random.seed(self.c['seed']*idx)
            if self.control_class == class_name:  # if sample is a control sample
                # torchaudio.save('antes_control.wav', wav, self.ap.sample_rate)
                for _ in range(self.c.data_aumentation['num_noise_control']):
                    # choise random noise file
                    valid_noise = False  # sometimes the noise is smaller than the wav it causes some errors
                    while not valid_noise:
                        noise_wav = self.ap.load_wav(os.path.join(
                            self.noise_root, self.noise_list[random.randint(0, self.num_noise_files)][0]))
                        noise_wav_len = noise_wav.shape[1]
                        valid_noise = True if noise_wav_len > wav.shape[1] else False

                    wav_len = wav.shape[1]
                    noise_start_slice = random.randint(
                        0, noise_wav_len-(wav_len+1))
                    # print(noise_start_slice, noise_start_slice+wav_len)
                    # sum two diferents noise
                    noise_wav = noise_wav[:,
                                          noise_start_slice:noise_start_slice+wav_len]
                    # get random max amp for noise
                    max_amp = random.uniform(
                        self.c.data_aumentation['noise_min_amp'], self.c.data_aumentation['noise_max_amp'])
                    reduct_factor = max_amp/float(noise_wav.max().numpy())
                    noise_wav = noise_wav*reduct_factor
                    wav = wav + noise_wav
                #torchaudio.save('depois_controle.wav', wav, self.ap.sample_rate)

            elif self.patient_class == class_name:  # if sample is a patient sample
                for _ in range(self.c.data_aumentation['num_noise_patient']):

                    # torchaudio.save('antes_patiente.wav', wav, self.ap.sample_rate)
                    # choise random noise file
                    valid_noise = False  # sometimes the noise is smaller than the wav it causes some errors
                    while not valid_noise:
                        noise_wav = self.ap.load_wav(os.path.join(
                            self.noise_root, self.noise_list[random.randint(0, self.num_noise_files)][0]))
                        noise_wav_len = noise_wav.shape[1]
                        valid_noise = True if noise_wav_len > wav.shape[1] else False
                    wav_len = wav.shape[1]
                    #print("Wav_len",wav_len, "| noise_wav_len",noise_wav_len)
                    noise_start_slice = random.randint(
                        0, noise_wav_len-(wav_len+1))
                    # sum two diferents noise
                    noise_wav = noise_wav[:,
                                          noise_start_slice:noise_start_slice+wav_len]
                    # get random max amp for noise
                    #print("Noise",noise_wav,"| Wav_len",wav_len)
                    max_amp = random.uniform(
                        self.c.data_aumentation['noise_min_amp'], self.c.data_aumentation['noise_max_amp'])
                    reduct_factor = max_amp/float(noise_wav.max().numpy())
                    noise_wav = noise_wav*reduct_factor
                    wav = wav + noise_wav

                #torchaudio.save('depois_patient.wav', wav, self.ap.sample_rate)
        if self.c.dataset['split_wav_using_overlapping']:
            #print("Wav len:", wav.shape[1])
            #print("for",self.ap.sample_rate*self.c.dataset['window_len'], wav.shape[1], self.ap.sample_rate*self.c.dataset['step'])
            start_slice = 0
            features = []
            targets = []
            phases = []
            step = self.ap.sample_rate*self.c.dataset['step']
            for end_slice in range(self.ap.sample_rate*self.c.dataset['window_len'], wav.shape[1], step):
                #print("Slices: ", start_slice, end_slice)
                if(self.ap.feature == 'spectrogram-melspec'):
                    spec_to_get_phase = librosa.stft(wav[:, start_slice:end_slice].numpy(
                    )[0], n_fft=self.ap.n_fft, hop_length=self.ap.hop_length, win_length=self.ap.win_length)
                    mel_spec = self.ap.get_feature_from_audio(
                        wav[:, start_slice:end_slice])
                    # TODO: Aqui transformar para decibel

                else:
                    mel_spec = self.ap.get_feature_from_audio(
                        wav[:, start_slice:end_slice])
                amp_to_db = torchaudio.transforms.AmplitudeToDB()
                mel_spec = amp_to_db(mel_spec)
                spec = mel_spec.transpose(1, 2)
                magnitude, phase = librosa.magphase(spec_to_get_phase)
                # print(phase.shape)
                # phase = torch.from_numpy(phase).unsqueeze(0).transpose(1, 2)
                phase = torch.from_numpy(phase).unsqueeze(0)
                # print(phase[0, 0, 0])
                # print(phase.shape, spec.shape, mel_spec.shape)
                features.append(spec)
                targets.append(torch.FloatTensor([class_name]))
                phases.append(phase)
                start_slice += step

            if len(features) == 1:
                if(self.ap.feature == 'spectrogram-melspec'):
                    spec_to_get_phase = librosa.stft(wav[:, :self.ap.sample_rate*self.c.dataset['window_len']].numpy()[
                                                     0], n_fft=self.ap.n_fft, hop_length=self.ap.hop_length, win_length=self.ap.win_length)
                    mel_spec = self.ap.get_feature_from_audio(
                        wav[:, :self.ap.sample_rate*self.c.dataset['window_len']])
                else:
                    mel_spec = self.ap.get_feature_from_audio(
                        wav[:, :self.ap.sample_rate*self.c.dataset['window_len']])

                amp_to_db = torchaudio.transforms.AmplitudeToDB()
                mel_spec = amp_to_db(mel_spec)
                magnitude, phase = librosa.magphase(spec_to_get_phase)
                # print(mel_spec.shape)
                feature = mel_spec.transpose(1, 2)
                target = torch.FloatTensor([class_name])
                # phases.append(phase)
                phases = torch.from_numpy(phase)
            elif len(features) < 1:
                #print("ERROR: Some sample in your dataset is less than %d seconds! Change the size of the overleapping window"%self.c.dataset['window_len'])
                raise RuntimeError("ERROR: Some sample in your dataset is less than {} seconds! Change the size of the overleapping window (CONFIG.dataset['window_len'])".format(
                    self.c.dataset['window_len']))
            else:
                phases = torch.cat(phases, dim=0)
                # print(type(phases))
                feature = torch.cat(features, dim=0)
                target = torch.cat(targets, dim=0)
                # print(phases.shape, target.shape, feature.shape)
            # print(feature.shape)
        else:
            # feature shape (Batch_size, n_features, timestamp)
            if(self.ap.feature == 'spectrogram-melspec'):
                audio_class = torchaudio.transforms.Spectrogram(
                    n_fft=self.ap.n_fft, win_length=self.ap.win_length, hop_length=self.ap.hop_length)
                wav_feature = audio_class(wav)
                mel_spec = self.ap.get_feature_from_audio(wav_feature)
            else:
                mel_spec = self.ap.get_feature_from_audio(wav)
            # transpose for (Batch_size, timestamp, n_features)
            feature = mel_spec.transpose(1, 2)
            # remove batch dim = (timestamp, n_features)
            feature = feature.reshape(feature.shape[1:])
            if self.c.dataset['padding_with_max_lenght']:
                # padding for max sequence
                zeros = torch.zeros(self.max_seq_len -
                                    feature.size(0), feature.size(1))
                # append zeros before features
                feature = torch.cat([feature, zeros], 0)
                target = torch.FloatTensor([class_name])
            else:
                target = torch.zeros(feature.shape[0], 1)+class_name

        return feature, target, self.dataset_list[idx][0], phases

    def __len__(self):
        return len(self.dataset_list)


def train_dataloader(c, ap):
    return DataLoader(dataset=Dataset(c, ap, train=True),
                      batch_size=c.train_config['batch_size'],
                      shuffle=True,
                      num_workers=c.train_config['num_workers'],
                      collate_fn=own_collate_fn,
                      pin_memory=True,
                      drop_last=True,
                      sampler=None)


def eval_dataloader(c, ap, max_seq_len=None):
    return DataLoader(dataset=Dataset(c, ap, train=False, max_seq_len=max_seq_len),
                      collate_fn=own_collate_fn, batch_size=c.test_config['batch_size'],
                      shuffle=False, num_workers=c.test_config['num_workers'])


def test_dataloader(c, ap, max_seq_len=None):
    return DataLoader(dataset=Dataset(c, ap, train=False, test=True, max_seq_len=max_seq_len),
                      collate_fn=teste_collate_fn, batch_size=c.test_config['batch_size'],
                      shuffle=False, num_workers=c.test_config['num_workers'])


def own_collate_fn(batch):
    features = []
    targets = []
    for feature, target in batch:
        features.append(feature)
        # print(target.shape)
        targets.append(target)

    # if dim is 3, we have a many specs because we use a overlapping
    if len(features[0].shape) == 3:
        targets = torch.cat(targets, dim=0)
        features = torch.cat(features, dim=0)
    else:
        # padding with zeros timestamp dim
        features = pad_sequence(features, batch_first=True, padding_value=0)
        # its padding with zeros but mybe its a problem because
        targets = pad_sequence(targets, batch_first=True, padding_value=0)

    #
    targets = targets.reshape(targets.size(0), -1)
    # list to tensor
    #targets = stack(targets, dim=0)
    #features = stack(features, dim=0)
    #print(features.shape, targets.shape)
    return features, targets


def teste_collate_fn(batch):
    features = []
    targets = []
    phases = []
    slices = []
    targets_org = []
    audio_path = []
    for feature, target, path, phase in batch:
        audio_path.append(path)
        features.append(feature)
        targets.append(target)
        phases.append(phase)
        if len(feature.shape) == 3:
            slices.append(torch.tensor(feature.shape[0]))
            # its used for integrity check during unpack overlapping for calculation loss and accuracy
            targets_org.append(target[0])

    # if dim is 3, we have a many specs because we use a overlapping
    if len(features[0].shape) == 3:
        phases = torch.cat(phases, dim=0)
        targets = torch.cat(targets, dim=0)
        features = torch.cat(features, dim=0)
    else:
        # padding with zeros timestamp dim
        features = pad_sequence(features, batch_first=True, padding_value=0)
        # its padding with zeros but mybe its a problem because
        targets = pad_sequence(targets, batch_first=True, padding_value=0)

    #
    targets = targets.reshape(targets.size(0), -1)

    if slices:
        slices = stack(slices, dim=0)
        targets_org = stack(targets_org, dim=0)
    else:
        slices = None
        targets_org = None
    # list to tensor
    #targets = stack(targets, dim=0)
    #features = stack(features, dim=0)
    # print(features.shape, targets.shape,phases.shape)
    return features, targets, slices, targets_org, audio_path, phases
