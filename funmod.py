
import os
import urllib.request
from urllib.error import HTTPError

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm.notebook import tqdm
import torch.multiprocessing as mp
import Bigmodel
import torch
import torchvision
from torch import nn
from torch import optim
from torch import tensor
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10
model = Bigmodel.Autoencoder(base_channel_size=3, latent_dim=52)

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    DATASET_PATH = '/Users/JasperHilliard/Documents/Git testing Project/ganang'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])



  `
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=transform, download=True)
    pl.seed_everything(69)
    trainset, val_set = data.random_split(train_dataset, [45000, 5000])

    # Loading the test setw
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=transform, download=True)


    trainloader = data.DataLoader(trainset, batch_size=256, shuffle=True, drop_last=True, pin_memory=True,
                                  num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)


    for i in trainloader:
        res = model(i[0])
        print(res)
        break
