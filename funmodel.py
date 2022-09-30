import BigModel
#a convolutional neural net to recognize digits from mnist set - jasper hilliard
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
model = BigModel.Autoencoder(base_channel_size=3, latent_dim=52)

if __name__ == '__main__':
    DATASET_PATH = '/Users/JasperHilliard/Documents/Git testing Project/ganang'
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    # Loading the training dataset. We need to split it into a training and validation part
    train_dataset = CIFAR10(root=DATASET_PATH, train=True, transform=transform, download=True)
    pl.seed_everything(69)
    trainset, val_set = data.random_split(train_dataset, [45000, 5000])

    # Loading the test setw
    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=transform, download=True)

    # We define a set of data loaders that we can use for various purposes later.
    trainloader = data.DataLoader(trainset, batch_size=256, shuffle=True, drop_last=True, pin_memory=True,
                                  num_workers=4)
    val_loader = data.DataLoader(val_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)
    test_loader = data.DataLoader(test_set, batch_size=256, shuffle=False, drop_last=False, num_workers=4)


    for i in trainloader:
        res = model[i[0]]
        print(res)
        break