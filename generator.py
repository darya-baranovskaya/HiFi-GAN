import torch
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from generator_layers import *


class Generator(torch.nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.pre_net = weight_norm(nn.Conv1d(config.mel_h, config.n_out_initial_channels,
                                              kernel_size=7, stride=1, padding=3))
        n_upsamples = len(config.kernel_size)
        in_channels = [config.n_out_initial_channels // 2**i for i in range(n_upsamples)]
        self.net = nn.Sequential(*list(
            UpSampleBlock(in_channels[i], config.kernel_size[i], config.mrf_kernel_sizes, config.dilation, config.leaky)
            for i in range(n_upsamples)
        ))
        self.post_net = weight_norm(nn.Conv1d(in_channels=config.n_out_initial_channels // (2**len(config.kernel_size)),
                                               out_channels=1, kernel_size=7, stride=1, padding=3))
        # self.init_weights()

    def init_weights(self):
        for name, module in self.named_modules():
            if type(module) is nn.Conv1d or type(module) is nn.Conv2d:
                with torch.no_grad():
                    module.weight.normal_(0, 0.01)

    def forward(self, x):
        x = self.pre_net(x)
        x = self.net(x)
        x = F.leaky_relu(x, 0.1)
        x = self.post_net(x).squeeze(1)
        x = F.tanh(x)
        return x


    def remove_weight_norm(self):
        print('Removing weight norm...')
        remove_weight_norm(self.pre_net)
        for l in self.net:
            l.remove_weight_norm()
        remove_weight_norm(self.post_net)