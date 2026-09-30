from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from scipy.signal import _signaltools
from scipy.signal.windows._windows import get_window

from dynoma.typing import DetrendT, ModeT


def prepare_arguments(
    *,
    sampling_frequency: float,
    number_of_observations: int,
    window: Literal["hann", "hamming"],
    nperseg: int,
    noverlap: int | None,
    nfft: int | None,
    scaling: str,
    mode: ModeT,
    outdtype: npt.DTypeLike,
    win_periodic_bins: bool = True,
):
    """Prepare window and scaling arguments for CSD computations.

    Args:
        sampling_frequency: The sampling frequency of the signal.
        number_of_observations: The number of observations in the signal.
        window: The window to use for the analysis.
        nperseg: The number of samples in each segment.
        noverlap: The number of samples to overlap between segments.
        nfft: The number of FFT points to use.
        scaling: The scaling mode to use.
        mode: The power spectral density mode.
        outdtype: The dtype to use for the window.
        win_periodic_bins: If True, window will be periodic; if False, symmetric.
    """
    nperseg = min(int(nperseg), number_of_observations)

    win = get_window(window, nperseg, fftbins=win_periodic_bins).astype(outdtype)
    nfft = int(nfft or nperseg)
    noverlap = int(noverlap or nperseg // 2)

    scale = 1.0 / (sampling_frequency * (win * win).sum()) if scaling == "density" else 1.0 / win.sum() ** 2

    if mode == "stft":
        scale = np.sqrt(scale)

    return win, nperseg, noverlap, nfft, scale


def handle_detrend_method(detrend: DetrendT = None) -> Callable:
    """Return the detrend function based on the detrend input method.

    Args:
        detrend: The method to detrend the signal.

    Returns:
        The detrend callable (identity when detrend is None).
    """
    return (lambda d: _signaltools.detrend(d, type=detrend, axis=-1)) if detrend else lambda d: d


def compute_pxy(
    *,
    fft_helpers: npt.NDArray,
    channel_1: int,
    channel_2: int,
    scale: float,
    mode: ModeT,
    axis: int,
) -> Any:
    """Compute cross-spectral density for a pair of channels.

    Args:
        fft_helpers: The array containing the FFT of the signal.
        channel_1: The first channel index.
        channel_2: The second channel index.
        scale: The scaling factor to apply.
        mode: The mode for the cross-spectral density.
        axis: The axis to compute the cross-spectral density along.

    Returns:
        The cross-spectral density for the pair of channels.
    """
    pxy = np.conjugate(fft_helpers[channel_1, :, :]) * fft_helpers[channel_2, :, :]
    pxy *= scale

    if mode == "psd":
        pxy[..., 1:-1] *= 2

    pxy = np.moveaxis(pxy, -1, axis)
    if len(pxy.shape) >= 2 and pxy.size > 0:
        pxy = pxy.mean(axis=-1) if pxy.shape[-1] > 1 else np.reshape(pxy, pxy.shape[:-1])
    return pxy
