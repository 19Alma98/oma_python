import numpy as np
import numpy.typing as npt
from django.utils.translation import gettext_lazy
from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass
from scipy.signal import find_peaks

from core.algorithms.constants import BASE_DTYPE, OMA_COMPLEX_DTYPE
from core.algorithms.exceptions import ModalIdentificationError
from core.algorithms.modal_identification.oma_algorithm import OmaAlgorithm
from core.algorithms.modal_identification.typing import FDDResults
from core.algorithms.modal_identification.utils import complex_mode_to_real_mode
from core.algorithms.typing import SignalT


@dataclass(slots=True, kw_only=True, config=ConfigDict(arbitrary_types_allowed=True))
class FDD(OmaAlgorithm):
    """Frequency Domain Decomposition Algorithm.

    This class is used to perform the modal identification using the Frequency Domain Decomposition algorithm.
    The FFD method requires the user to select the frequency range over which the peaks are to be found.
    The peaks are then used to identify the modal parameters.

    This means that the algorithm follows two steps:
    - Compute the singular value decomposition of the cross spectral density matrix. Those are the lines over which the user will select peaks.
    - Find the peaks over the selected areas and compute the modal parameters.

    Example:
    ```python
    # Initialize the algorithm
    fdd_algorithm = FDD(
        frequency_max=100, frequency_min=0, number_of_fft_points=1024, num_svd_plots=10
    )

    # Compute the singular value decomposition of the cross spectral density matrix.
    frequencies, eigenvalues_matrix, eigenvectors_matrix = fdd_algorithm.compute_signal_svd(
        signal=signal, sampling_frequency=1000, optimize_csd_computation=True
    )

    # Eigenvalues matrix and frequencies are the data we want to plot
    # Find the peaks over the selected areas and compute the modal parameters.
    peaks_range = [[0.06, 0.064], [1.8, 1.9]]
    fdd_results = fdd_algorithm.apply(
        signal=signal, sampling_frequency=1000, peaks_range=peaks_range
    )

    # The results are:
    frequencies = fdd_results.frequencies
    complex_mode_shapes = fdd_results.complex_mode_shapes
    real_mode_shapes = fdd_results.real_mode_shapes

    # This method does not compute cluster dimension and damping ratio as for the OMA algorithm.
    ```
    """

    frequency_max: float = Field(ge=0)
    frequency_min: float = Field(ge=0)
    number_of_fft_points: int = Field(gt=0)
    num_svd_plots: int = Field(gt=0)
    frequencies: npt.NDArray[BASE_DTYPE] | None = None
    eigenvectors_matrix: npt.NDArray[OMA_COMPLEX_DTYPE] | None = None
    eigenvalues_matrix: npt.NDArray[BASE_DTYPE] | None = None

    def _validate_peak_range(self, area: list[float]) -> None:
        """Validate the peak range.

        Args:
            area (list[float]): The range of frequencies selected by the user over which we want to find the peaks.

        Raises:
            ModalIdentificationError: If the selected frequency range is invalid. They can be invalid if:
            - The selected frequency range is not a list of two elements.
            - The first entry must be less than the second entry.
        """
        if len(area) != 2:
            raise ModalIdentificationError(
                gettext_lazy("Selected frequency range must be a list of two elements: min and max x values.")
            )
        if area[0] > area[1]:
            raise ModalIdentificationError(
                gettext_lazy("Selected frequency range in FDD is invalid: first entry must be less than second entry.")
            )

    def _find_peak_over_specific_area(
        self,
        user_selected_areas: list[list[float]],
        signal_eigenvalues: npt.NDArray[BASE_DTYPE],
        frequencies: npt.NDArray[BASE_DTYPE],
    ) -> npt.NDArray[np.int32]:
        """Return the indices of the frequency peaks over the specific areas.

        Args:
            user_selected_areas (list[list[float]]): The range of frequencies selected by the user over which we want to find the peaks.
            signal_eigenvalues (npt.NDArray[BASE_DTYPE]): The eigenvalues of the signal.
            frequencies (npt.NDArray[BASE_DTYPE]): The array of frequencies of the signal.

        Returns:
            npt.NDArray[np.int32]: The indices of the frequency peaks over the specific areas.
        """
        peak_indices = []
        for area in user_selected_areas:
            self._validate_peak_range(area)
            try:
                start_idx = np.nonzero(frequencies >= area[0])[0][0]
                end_idx = np.nonzero(frequencies >= area[1])[0][0]
            except IndexError as error:
                raise ModalIdentificationError(gettext_lazy("Selected frequency range in FDD is invalid.")) from error

            sample_signal = signal_eigenvalues[start_idx:end_idx]
            try:
                peaks, _ = find_peaks(sample_signal)
            except ValueError as error:
                raise ModalIdentificationError(gettext_lazy("Peak detection in FDD failed.")) from error
            if not len(peaks):
                raise ModalIdentificationError(gettext_lazy("No peaks found in the selected area for FDD algorithm."))

            peak_values = sample_signal[peaks]
            peak_frequencies = frequencies[start_idx:end_idx][peaks][np.argmax(peak_values)]
            peak_indices.append(int(np.argmin((frequencies - peak_frequencies) ** 2)))

        return np.sort(np.array(peak_indices, dtype=np.int32))

    def apply(self, *, signal: SignalT, sampling_frequency: float, peaks_range: list[list[float]]):
        """Return the modal parameters coming from FDD analysis.

        Args:
            signal (SignalT): The input signal.
            sampling_frequency (float): The sampling frequency of the signal.
            peaks_range (list[list[float]]): The range of frequencies over which we want to find the peaks.

        Returns:
            FDDResults: The modal parameters coming from FDD analysis.
        """
        if not len(peaks_range):
            raise ModalIdentificationError(gettext_lazy("User's selected peaks not available"))
        if self.frequencies is None or self.eigenvalues_matrix is None or self.eigenvectors_matrix is None:
            frequencies, eigenvalues_matrix, eigenvectors_matrix = self.compute_signal_svd(
                signal=signal, sampling_frequency=sampling_frequency
            )
        else:
            frequencies = self.frequencies
            eigenvectors_matrix = self.eigenvectors_matrix
            eigenvalues_matrix = self.eigenvalues_matrix
        peaks_indexes = self._find_peak_over_specific_area(peaks_range, eigenvalues_matrix[0, :], frequencies)
        complex_modes = np.vstack([np.conjugate(eigenvectors_matrix[:, 0, peak]) for peak in peaks_indexes])
        peaks_frequency = np.sort(frequencies[peaks_indexes])
        selected_complex_modes = complex_modes[np.argsort(peaks_frequency), :]
        selected_real_modes = np.array(
            [complex_mode_to_real_mode(complex_mode) for complex_mode in selected_complex_modes], dtype=BASE_DTYPE
        ).T
        return FDDResults(
            frequencies=peaks_frequency,
            complex_mode_shapes=selected_complex_modes.T,
            real_mode_shapes=selected_real_modes,
        )
