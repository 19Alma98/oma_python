from enum import IntEnum, unique

import numpy as np
import numpy.typing as npt
import torch

BASE_DTYPE = np.float64
COMPLEX_DTYPE = np.complex128
OMA_COMPLEX_DTYPE = np.complex64
TORCH_COMPLEX_DTYPE = torch.complex64

SignalT = npt.NDArray[BASE_DTYPE]

@unique
class NFFT(IntEnum):
    """Number of points in the FastFourier transform."""

    NFFT_64 = 64
    NFFT_128 = 128
    NFFT_256 = 256
    NFFT_512 = 512
    NFFT_1024 = 1024
    NFFT_2048 = 2048
    NFFT_4096 = 4096
    NFFT_8192 = 8192
