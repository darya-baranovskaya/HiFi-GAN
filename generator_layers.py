import torch
from torch import nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import torch.nn.functional as F
from typing import List


class MRF(torch.nn.Module):
    def __init__(self, channels, kernel_sizes:List[int]=(3,7,11), dilations:List[List[int]]=[[1,3,5], [1,3,5], [1,3,5]], leaky: float=1e-1):
        super(MRF, self).__init__()
        self.n_blocks = len(kernel_sizes)
        self.resblocks = nn.ModuleList([ResBlock(channels, kernel_size=kernel_sizes[i], dilation=dilations[i], leaky=leaky) for i in range(self.n_blocks)])

    def forward(self, input):
        out = 0
        for i in range(self.n_blocks):
            out += self.resblocks[i](input)
        return out / self.n_blocks

    def remove_weight_norm(self):
        for i in range(self.n_blocks):
            self.resblocks[i].remove_weight_norm()




class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation:List[int], leaky: float):
        super().__init__()
        self.leaky = leaky
        paddings = [(kernel_size*d - d) // 2 for d in dilation]
        self.convs1 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=dilation[i],padding=paddings[i]))
            for i in range(len(dilation))])
        padding = (kernel_size - 1) // 2
        self.convs2 = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1,
                               padding=padding)) for _ in range(len(dilation))])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, self.leaky)
            xt = c1(xt)
            xt = F.leaky_relu(xt, self.leaky)
            xt = c2(xt)
            x = xt + x
        return x




class UpSampleBlock(nn.Module):
    def __init__(self,
                 channels: int, kernel_size: int, mrf_kernel_sizes: List[int], dilations:List[List[int]], leaky: float = 1e-1):
        super(UpSampleBlock, self).__init__()
        upsample_kernel_size = kernel_size // 2
        padding = (kernel_size - upsample_kernel_size) // 2
        out_channels = channels // 2
        self.conv = weight_norm(nn.ConvTranspose1d(channels, out_channels, kernel_size, stride=upsample_kernel_size, padding=padding))
        self.mrf = MRF(out_channels, mrf_kernel_sizes, dilations, leaky)

    def forward(self, x):
        x = F.leaky_relu(x, 0.1)
        x = self.conv(x)
        x = self.mrf(x)
        return x