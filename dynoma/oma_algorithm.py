from typing import ClassVar, Final, Literal

import numpy as np
import numpy.typing as npt
import torch
from django.utils.translation import gettext_lazy
from pydantic import Field, model_validator
from pydantic.dataclasses import dataclass
from scipy.signal import csd
from scipy.signal._spectral_py import _fft_helper  # type:ignore
from scipy.signal.windows._windows import get_window

from core.algorithms.coherence_analysis.typing import ModeT
from core.algorithms.coherence_analysis.utils import compute_pxy, handle_detrend_method, prepare_arguments
from core.algorithms.constants import BASE_DTYPE, OMA_COMPLEX_DTYPE, TORCH_COMPLEX_DTYPE
from core.algorithms.exceptions import ModalIdentificationError
from core.algorithms.typing import SignalT

CSD_COMPUTATION_AXIS: Final[int] = 0


@dataclass(slots=True, kw_only=True)
class OmaAlgorithm:
    """Class that handles the common operations between the OMA algorithms.

    It is not meant to be used directly, but rather to be inherited by other algorithms.

    It computes the cross spectral density matrix and the singular value decomposition of the cross spectral density matrix,
    that are the required by both the FDD and the COV-SSI algorithms.
    """

    frequency_max: float = Field(gt=0)
    frequency_min: float = Field(ge=0)
    number_of_fft_points: int = Field(gt=0)
    num_svd_plots: int = Field(gt=0)

    WINDOW: ClassVar[Final[Literal["hann", "hamming"]]] = "hamming"

    @model_validator(mode="after")
    def validate_frequency_range(self) -> "OmaAlgorithm":
        """Validate the frequency range is valid."""
        if self.frequency_min > self.frequency_max:
            raise ModalIdentificationError(
                gettext_lazy(
                    "Frequency minimum must be less than frequency maximum: given {freq_min}, {freq_max}"
                ).format(freq_min=self.frequency_min, freq_max=self.frequency_max)
            )
        return self

    def _initialize_csd_matrix(self, number_of_columns: int, number_of_fft_points: int):
        """Return the cross spectral density matrix initialized with zeros.

        Args:
            number_of_columns (int): The number of columns in the signal.
            number_of_fft_points (int): The number of FFT points.

        Returns:
            npt.NDArray[OMA_COMPLEX_DTYPE]: The cross spectral density matrix.
        """
        return np.zeros(
            (
                number_of_columns,
                number_of_columns,
                int(number_of_fft_points / 2 + 1),
            ),
            dtype=OMA_COMPLEX_DTYPE,
        )

    def _find_window_length(self, number_of_observations: int, number_of_fft_points: int) -> int:
        """Return the window length based on the number of observations and the number of FFT points.

        Args:
            number_of_observations (int): The number of observations in the signal.
            number_of_fft_points (int): The number of FFT points.

        Returns:
            int: The window length to be used for the FFT.
        """
        return (
            int(number_of_fft_points)
            if number_of_fft_points <= number_of_observations
            else int(2 ** np.floor(np.log2(number_of_observations)))
        )

    def _get_csd_cross_signal(self, signal: SignalT, sampling_frequency: float, nperseg: int):
        """Return the cross spectral density matrix of the input signal computed using the original scipy csd function.

        Args:
            signal (SignalT): The input signal.
            sampling_frequency (float): The sampling frequency of the signal.
            nperseg (int): The number of points in each segment.

        Returns:
            npt.NDArray[OMA_COMPLEX_DTYPE]: The cross spectral density matrix.
        """
        _, number_of_columns = signal.shape
        csd_matrix = self._initialize_csd_matrix(number_of_columns, nperseg)
        for channel_1 in range(number_of_columns):
            for channel_2 in range(channel_1, number_of_columns):
                _, cross_spectral_density = csd(
                    signal[:, channel_1],
                    signal[:, channel_2],
                    nperseg=nperseg,
                    window=get_window(  # type:ignore
                        self.WINDOW,
                        nperseg,
                        fftbins=False,
                    ),
                    detrend=handle_detrend_method(),
                    fs=sampling_frequency,
                )
                csd_matrix[channel_1, channel_2, :] = cross_spectral_density
                csd_matrix[channel_2, channel_1, :] = np.conjugate(cross_spectral_density)

        return csd_matrix

    def _get_csd_cross_signal_optimized(
        self,
        signal: SignalT,
        sampling_frequency: float,
        nperseg: int,
        noverlap: int | None = None,
        nfft: int | None = None,
        scaling: str = "density",
        mode: ModeT = "psd",
    ):
        """Return the cross spectral density matrix of the input signal using optimized computations.

        The result is the same as the one obtained using the original scipy csd function.

        Args:
            signal (SignalT): The input signal.
            sampling_frequency (float): The sampling frequency of the signal.
            nperseg (int): The number of points in each segment.
            noverlap (int | None, optional): The number of points to overlap between segments. Defaults to None.
            nfft (int | None, optional): The number of points in the FFT. Defaults to None.
            scaling (str, optional): The scaling of the cross spectral density. Defaults to "density".
            mode (ModeT, optional): The mode of the cross spectral density. Defaults to "psd".

        Returns:
            npt.NDArray[OMA_COMPLEX_DTYPE]: The cross spectral density matrix.
        """
        number_of_observations, number_of_columns = signal.shape
        csd_matrix = self._initialize_csd_matrix(number_of_columns, nperseg)
        outdtype = np.result_type(signal, OMA_COMPLEX_DTYPE)
        win, nperseg, noverlap, nfft, scale = prepare_arguments(
            sampling_frequency=sampling_frequency,
            number_of_observations=number_of_observations,
            window=self.WINDOW,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            scaling=scaling,
            mode=mode,
            outdtype=outdtype,
            win_periodic_bins=False,
        )
        fft_helpers = _fft_helper(signal.T, win, handle_detrend_method(), nperseg, noverlap, nfft, "onesided")
        for channel_1 in range(number_of_columns):
            for channel_2 in range(channel_1, number_of_columns):
                pxy = compute_pxy(
                    fft_helpers=fft_helpers,
                    channel_1=channel_1,
                    channel_2=channel_2,
                    scale=scale,
                    mode=mode,
                    axis=CSD_COMPUTATION_AXIS,
                )
                csd_matrix[channel_1, channel_2, :] = pxy
                csd_matrix[channel_2, channel_1, :] = np.conjugate(pxy)
        return csd_matrix

    def compute_signal_svd(
        self,
        *,
        signal: SignalT,
        sampling_frequency: float,
        optimize_csd_computation: bool = True,
    ) -> tuple[npt.NDArray[BASE_DTYPE], npt.NDArray[BASE_DTYPE], npt.NDArray[OMA_COMPLEX_DTYPE]]:
        """Return the singular value decomposition of the cross spectral density matrix together with the associated frequencies.

        Args:
            signal (SignalT): The input signal.
            sampling_frequency (float): The sampling frequency of the signal.
            frequency_max (float): The maximum frequency.
            frequency_min (float): The minimum frequency.
            number_of_fft_points (int): The number of FFT points.
            num_svd_plots (int): The number of SVD lines we want to plot.
            optimize_csd_computation (bool, optional): Whether to use the optimized CSD computation. Defaults to True.

        Returns:
            tuple[npt.NDArray[BASE_DTYPE], npt.NDArray[OMA_COMPLEX_DTYPE], npt.NDArray[OMA_COMPLEX_DTYPE]]: The frequencies, the eigenvalues and the eigenvectors.
        """
        if (len(signal.shape) != 2) or (signal.size == 0):
            raise ModalIdentificationError(gettext_lazy("Input data must be a bi-dimensional non-empty array"))
        if self.frequency_max > sampling_frequency / 2:
            raise ModalIdentificationError(
                gettext_lazy(
                    "Frequency maximum must be less than or equal to half of the sampling frequency: given {freq_max}, {sampling_frequency}"
                ).format(freq_max=self.frequency_max, sampling_frequency=sampling_frequency)
            )
        number_of_observations, _ = signal.shape
        window_length = self._find_window_length(number_of_observations, self.number_of_fft_points)
        frequencies = (np.arange(0, int(window_length / 2 + 1) - 1) * (sampling_frequency / window_length)).astype(
            BASE_DTYPE
        )
        try:
            if optimize_csd_computation:
                csd_matrix = self._get_csd_cross_signal_optimized(signal, sampling_frequency, window_length)
            else:
                csd_matrix = self._get_csd_cross_signal(signal, sampling_frequency, window_length)
        except Exception as error:
            raise ModalIdentificationError(
                gettext_lazy("Cross spectral decomposition failed: {error}").format(error=error)
            ) from error

        csd_tensor = torch.tensor(csd_matrix.transpose(2, 0, 1), dtype=TORCH_COMPLEX_DTYPE)
        try:
            u_tensor, s_tensor, _ = torch.linalg.svd(csd_tensor)
        except Exception as error:
            raise ModalIdentificationError(
                gettext_lazy("Singular value decomposition failed: {error}").format(error=error)
            ) from error

        valid_indices = np.nonzero((frequencies >= self.frequency_min) & (frequencies <= self.frequency_max))[0]
        if valid_indices.size == 0:
            raise ModalIdentificationError(gettext_lazy("No frequencies found for the given frequency range"))

        eigenvectors_matrix = (
            u_tensor.transpose(1, 2).numpy().T[:, :, valid_indices[0] : valid_indices[-1] + 1]
        ).astype(OMA_COMPLEX_DTYPE)
        eigenvalues_matrix = s_tensor.numpy().T[: self.num_svd_plots, valid_indices[0] : valid_indices[-1] + 1]
        eigenvalues_matrix = np.where(eigenvalues_matrix == 0, 1, eigenvalues_matrix)
        eigenvalues_matrix = (10 * np.log10(eigenvalues_matrix)).astype(BASE_DTYPE)

        return frequencies[valid_indices], eigenvalues_matrix, eigenvectors_matrix
