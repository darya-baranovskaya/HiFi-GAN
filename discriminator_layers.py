import torch
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import torch.nn.functional as F
from typing import List


class MPSubdisc(torch.nn.Module):
    def __init__(self, kernel_size:int=5, stride:int=3, period:int=2, leaky:float=0.1):
        super(MPSubdisc, self).__init__()
        self.period = period
        self.leaky = leaky
        padding = (kernel_size - 1) // 2
        channels = [1, 32, 128, 512, 1024, 1024]
        self.convs = nn.ModuleList([weight_norm(nn.Conv2d(channels[i], channels[i + 1], (kernel_size, 1), (stride, 1), padding=(padding, 0)))
                                    for i in range(len(channels) - 1)])
        self.out_conv = nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0))

    def forward(self, x):
        features = []

        # 1d to 2d
        b, c, t = x.shape

        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad

        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, self.leaky)
            features.append(x)

        x = self.out_conv(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)
        return x, features



class MSSubdisc(torch.nn.Module):
    def __init__(self, leaky:float=0.1):
        super(MSSubdisc, self).__init__()
        self.leaky = leaky
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(1, 128, 15, 1, padding=7)),
            weight_norm(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            weight_norm(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            weight_norm(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            weight_norm(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.out_conv = weight_norm(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        features = []

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, self.leaky)
            features.append(x)

        x = self.out_conv(x)
        features.append(x)
        x = torch.flatten(x, 1, -1)
        return x, features

