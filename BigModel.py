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
from torchvision.datasets import CIFAR10
import pytorch_lightning as pl
import matplotlib.pyplot as plt
#The input processing of the neuralnet
class Encoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.ReLU):

        super().__init__()
        channel = num_input_channels
        double_channel = channel * 2
        print('initially has the encoder init initialized.')
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, channel, kernel_size=3, padding=1, stride=2),  #32x32 => 16x16
            act_fn(),
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(channel, double_channel, kernel_size=3, padding=1, stride=2),  #16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * channel, double_channel, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * channel, double_channel, kernel_size=3, padding=1, stride=2),  # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),
            nn.Linear(2 * 16 * channel, latent_dim),
        )

   # def forward(self, x):
     #   return self.net(x)

class Decoder(nn.Module):
    def __init__(self, num_input_channels: int, base_channel_size: int, latent_dim: int, act_fn: object = nn.ReLU):

        super().__init__()
        channel = base_channel_size
        self.linear = nn.Sequential(nn.Linear(latent_dim, 2*16*channel), act_fn())
        print('initially has the decoder init initialized.')
        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                2 * channel, 2 * channel, kernel_size=3, output_padding=1, padding=1, stride=2
            ),
            act_fn(),
            nn.ConvTranspose2d(2 * channel, 2 * channel, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * channel, channel, kernel_size=3, output_padding=1, padding=1, stride=2),  # 8x8 => 16x16
            act_fn(),
            nn.ConvTranspose2d(channel, channel, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(
                channel, num_input_channels, kernel_size=3, output_padding=1, padding=1, stride=2
            ),
            nn.Tanh(),
        )

  #  def forward(self, x):
  #      x = self.linear(x)
     #   x = x.reshape(x.shape[0], -1, 4, 4)
     #   x = self.net(x)
     #   return x

class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        base_channel_size: int,
        latent_dim: int,
        encoder_class: object = Encoder,
        decoder_class: object = Decoder,
        num_input_channels: int = 3,
        width: int = 32,
        height: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size, latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size, latent_dim)


    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
