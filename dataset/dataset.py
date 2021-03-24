import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision
from torch.utils.data import random_split
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os, sys
import re
sys.path.append(os.path.abspath('.'))
from transforms.train_test_transform import TrainTransform, TestTransform


def MNIST_dataloader(func):
    def wrapper(*args, **kwargs):
        ret = [torch.utils.data.DataLoader(
                dataset,
                batch_size=16,
                shuffle=True if i==0 else False,
                num_workers=4
                ) 
                for i, dataset in enumerate(list(func(*args, **kwargs)))
              ]
        return ret
    return wrapper


class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, train):
        self.images = [i[0] for i in dataset]
        self.labels = [i[1] for i in dataset]
        self.trans = TrainTransform() if train else TestTransform()
            
    def __getitem__(self, idx):
        out_image = self.trans(self.images[idx])
        out_label = self.labels[idx]
        return out_image, out_label
    
    def __len__(self):
        return len(self.labels)
    

@MNIST_dataloader
def MNIST_datasets():    
    dataset_train = torchvision.datasets.MNIST(
                        root=os.path.abspath('.'), 
                        train=True, 
                        download=True
                    ) 
    len_train = int(len(dataset_train)*0.7)
    len_val = len(dataset_train) - len_train
    train, val = random_split(dataset_train, [len_train, len_val])

    test = torchvision.datasets.MNIST(
                root=os.path.abspath('.'), 
                train=False,
                download=True
           ) 
    return (MNISTDataset(train, True), 
            MNISTDataset(val, False), 
            MNISTDataset(test, False))
