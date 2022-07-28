# from cProfile import label
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

class VADdataset(Dataset):

    def __init__(self, voice_path: list , voice_label: list, transform= None):
        self.voice_path = voice_path
        self.voice_label = voice_label
        self.transform = transform

    def __len__(self):
        return len(self.voice_path)

    def __getitem__(self, index):
        # 这一部分去用pd解析csv即可
        voice = pd.read_csv(self.voice_path[index], header=None)
        voice = np.array(voice)
        voice = torch.tensor(voice)
        voice = torch.unsqueeze(voice,0)
        label = self.voice_label[index]

        if self.transform is not None:
            voice = self.transform(voice)
        
        return voice, label

    @staticmethod

    
    def collate_fn(batch):
        voice, label = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(label)
        return voice, label