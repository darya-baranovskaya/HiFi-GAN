import torch
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from discriminator_layers import *


class Discriminator(torch.nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.mpd = MPD(config)
        self.msd = MSD(config)


class MPD(torch.nn.Module):
    def __init__(self, config):
        super(MPD, self).__init__()

        self.periods = config.periods
        self.subs = nn.ModuleList([
            MPSubdisc(config.mpd_kernel_size, config.mpd_stride, period, config.leaky) for period in self.periods])

    def forward(self, x):
        features = []
        outs = []
        for layer in self.subs:
            pred, feature = layer(x)
            features.append(feature)
            outs.append(pred)

        return outs, features




class MSD(torch.nn.Module):
    def __init__(self, config):
        super(MSD, self).__init__()

        self.subs = nn.ModuleList([
            MSSubdisc(config.leaky) for _ in range(config.n_msd_layers)
        ])

        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2)
        ])


    def forward(self, x):
        features = []
        outs = []

        out, feature_map = self.subs[0](x)
        features.append(feature_map)
        outs.append(out)
        for i in range(1, len(self.subs)):
            x = self.meanpools[i-1](x)
            out, feature_map = self.subs[i](x)
            features.append(feature_map)
            outs.append(out)

        return outs, features

