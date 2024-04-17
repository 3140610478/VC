import os, sys
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F

base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from config import SAMPLE_RATE, N_FFT, WIN_LENGTH, HOP_LENGTH, N_MELS


class Vocoder(torch.nn.Module):
    def __init__(self):
        super(Vocoder, self).__init__()
        self.vocoder = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        self.IMel = T.InverseMelScale(
            n_stft=N_FFT//2+1, n_mels=N_MELS, sample_rate=SAMPLE_RATE,
        )
        self.ISpec = T.GriffinLim(
            n_fft=N_FFT, win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
        )

    def forward(self, input: torch.Tensor, orig_freq = 22050) -> torch.Tensor:
        input = F.resample(input, orig_freq, SAMPLE_RATE)
        return self.vocoder(input)

    def inverse(self, input: torch.Tensor) -> torch.Tensor:
        return self.ISpec(self.IMel(input))
