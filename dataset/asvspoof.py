import os
import glob
import math
import torch
import pandas
import random
import torchaudio
from dataset.augmentations import RawBoost_Noise_Augmentation
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info

class ASVspoofDataset(Dataset):
    def __init__(self, DATA_PATH_LIST, PROTOCOL_PATH_LIST, noise_dict, noise_type="random", split="train"):
        os_path_join = os.path.join
        str_lower = str.lower
        str_endswith = str.endswith
        
        self.audioPath = []
        self.label = []
        self.attack_type = []
        self.noise = noise_type
        self.split = split
        
        self.musan_noises = None
        self.rir_noises = None
        
        musan_path = noise_dict.get("musan_path", None)
        rir_path = noise_dict.get("rir_path", None)
        
        if musan_path is not None:
            self.musan_noises = {
                'noise': [],
                'speech': [], 
                'music': []
            }
            
            for root, _, files in os.walk(musan_path):
                category = None
                if 'noise' in root.lower():
                    category = 'noise'
                elif 'speech' in root.lower():
                    category = 'speech'
                elif 'music' in root.lower():
                    category = 'music'
                
                if category:
                    for file in files:
                        if '.wav' in file:
                            self.musan_noises[category].append(os.path.join(root, file))
        
        if rir_path is not None:
            rir_folders = ['pointsource_noises', 'real_rirs_isotropic_noises', 'simulated_rirs']
            selected_folder = random.choice(rir_folders)
            folder_path = os_path_join(rir_path, selected_folder)
            
            def collect_wav_files_fast(directory):
                wav_files = []
                stack = [directory]
                
                while stack:
                    current_dir = stack.pop()
                    try:
                        with os.scandir(current_dir) as entries:
                            for entry in entries:
                                if entry.is_file():
                                    if str_endswith(str_lower(entry.name), '.wav'):
                                        wav_files.append(entry.path)
                                elif entry.is_dir():
                                    stack.append(entry.path)
                    except (OSError, PermissionError):
                        continue
                return wav_files
            
            self.rir_noises = collect_wav_files_fast(folder_path)
            
        if self.split == "train":
            self.rawBoost = RawBoost_Noise_Augmentation(self.musan_noises, self.rir_noises)
        else:
            self.rawBoost = None

        for DATA_PATH, PROTOCOL_PATH in zip(DATA_PATH_LIST, PROTOCOL_PATH_LIST):
            year = DATA_PATH.split("/")[-3]
            
            with open(PROTOCOL_PATH, 'r', buffering=16384) as file:
                data = file.read().splitlines()
            
            if year == "2024" or year == "DATA_SUBSET":
                split_data = [line.split() for line in data]
                
                audio_paths = []
                labels = []
                attack_types = []
                
                for split_line in split_data:
                    audio_paths.append(DATA_PATH + split_line[1] + '.flac')
                    labels.append(0 if split_line[-2] == 'spoof' else 1)
                    attack_types.append(split_line[-3])
                
                self.audioPath.extend(audio_paths)
                self.label.extend(labels)
                
                if attack_types:
                    attack_cat = pandas.Categorical(attack_types)
                    self.attack_type.extend(attack_cat.codes.tolist())
            else:
                data_len = len(data)
                self.audioPath.extend([DATA_PATH + path for path in data])
                self.label.extend([0] * data_len)
                self.attack_type.extend([-1] * data_len)

    def __len__(self):
        return len(self.label)
            
    def __getitem__(self, index, audio_duration:int = 64600):
        wav, seq = torchaudio.load(self.audioPath[index], num_frames=audio_duration)

        if wav.shape[-1] < audio_duration:
            needed_frames = audio_duration - wav.shape[-1]
            repeats = needed_frames // wav.shape[-1] + 1
            repeated_part = torch.cat([wav] * repeats, dim=-1)[:, :needed_frames]
            wav = torch.cat([wav, repeated_part], dim=-1)
            
        if self.split=="train" and self.rawBoost is not None and 0.80 > random.random():
            wav = torch.tensor(self.rawBoost.apply(wav.numpy()[0], seq, self.noise), dtype=torch.float32).unsqueeze(0)
        
        return (wav.squeeze(), self.label[index], self.attack_type[index])
