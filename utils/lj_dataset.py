import torchaudio
import torch
import random
from typing import Tuple, Dict, Optional, List, Union
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from utils.melspectrogram import MelSpectrogramConfig, MelSpectrogram

featurizer = MelSpectrogram(MelSpectrogramConfig())

@dataclass
class Batch:
    waveform: torch.Tensor
    waveforn_length: torch.Tensor
    melspec: torch.Tensor = None

    def to(self, device: torch.device) -> 'Batch':
        raise NotImplementedError


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root, max_len=16384):
        super().__init__(root=root)
        self.max_len = max_len

    def __getitem__(self, index: int): #8192
        waveform, _, _, transcript = super().__getitem__(index)
        waveform_length = torch.tensor([waveform.shape[-1]]).int()

        start_idx = random.randint(0, waveform.shape[1] - self.max_len)
        waveform = waveform[:, start_idx: start_idx + self.max_len]

        return waveform, waveform_length


class LJSpeechCollator:

    def __call__(self, instances: List[Tuple]) -> Dict:
        waveform, waveform_length = list(
            zip(*instances)
        )

        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveform_length = torch.cat(waveform_length)

        melspec = featurizer(waveform)

        batch = Batch(waveform, waveform_length, melspec)

        return batch