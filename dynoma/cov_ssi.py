from typing import Final

import numpy as np
import numpy.typing as npt
import scipy
import torch
from pydantic import Field
from pydantic.dataclasses import dataclass
from scipy import fft as sp_fft
from scipy.signal._signaltools import _apply_conv_mode, _reverse_and_conj  # type:ignore

from dynoma.constants import BASE_DTYPE, OMA_COMPLEX_DTYPE, TORCH_COMPLEX_DTYPE
from dynoma.exceptions import ModalIdentificationError
from dynoma.fdd import OmaAlgorithm
from dynoma.typing import (
    CovSSIDashboardData,
    CovSSIResults,
    ModalDashboardData,
    ModalIdentification,
    StabilityCheck,
    StabilityModalIdentification,
)
from dynoma.utils import (
    complex_mode_to_real_mode,
    compute_mac_value,
    maximum_correlation_rotation,
    modal_phase_collinearity,
)

@dataclass(slots=True, kw_only=True)
class CovSSI(OmaAlgorithm):
    """Covariance SSI Algorithm.

    This class is used to perform the modal identification using the Covariance SSI algorithm.

    Example:
    ```python
    # Initialize the algorithm
    cov_ssi_algorithm = CovSSI(
        frequency_max=100,
        frequency_min=0,
        order_max=80,
        order_min=40,
        order_steps=2,
        time_lag=1.2,
        min_mpc=0.6,
        damping_max_value=10,
        frequency_noise_threshold=0.03,
        damping_noise_threshold=0.03,
        mac_noise_threshold=0.03,
        minimum_cluster_dimension=3,
        maximum_distance=0.03,
        number_of_fft_points=2**8,
        num_svd_plots=3,
    )

    # Compute the modal identification
    modal_identification = cov_ssi_algorithm.apply(
        signal=signal,
        sampling_frequency=sampling_frequency,
        optimized=True,
        shuffle=True,
    )

    # Signal and sampling  frequency are the main inputs of the algorithm.
    # The other parameters are used in this way:
    # - The continuous_mode parameter is used to understand if it is necessary to compute SVD lines over the signal.
    # - The optimized parameter is used to select the computation method of the impulse response function.
    # - The shuffle parameter is used to shuffle the clustering process ad give more robustness to the results.

    # The results are a dictionary containing modal parameters:
    # - frequencies: The frequencies of the modes.
    # - damping_ratios: The damping ratios of the modes.
    # - mode_shapes_complex: The complex mode shapes.
    # - mode_shapes_real: The real part of the mode shapes.
    # - cluster_dimension: The dimension of the cluster associated with the specific frequency.
    # - frequency_bounds: The bounds of the frequencies (95% confidence interval).
    # - damping_bounds: The bounds of the damping ratios (95% confidence interval).

    # Moreover we have a second output `dashboard_data`(None if continuous_mode is True) containing info related to the FE plots

    # To get info for the FE plots we can use the `get_svd_plot_data` method.
    # This method returns a dictionary containing the data to be plotted in the SVD plot.
    # The keys of the dictionary are:
    # - "y_axis_signal": The eigenvalues of the modes.
    # - "x_axis_frequencies": The frequencies of the modes.
    # - "frequencies": The frequencies of the modes.
    # - "model_orders": The model orders used to compute the modal parameters.

    svd_plot_data = cov_ssi_algorithm.get_svd_plot_data(
        signal=signal, sampling_frequency=sampling_frequency, dashboard_data=dashboard_data
    )

    # The results are a dictionary containing the data to be plotted in the SVD plot.
    # The keys of the dictionary are:
    # - "y_axis_signal": The eigenvalues of the modes (lines to be plotted).
    # - "x_axis_frequencies": The frequencies of the modes  (the x-axis of the plot).
    # - "frequencies": The pre-clustering frequencies of the modes (to be plotted as points, these are the y info).
    # - "model_orders": The pre-clustering model orders (to be plotted as points, these are the x info).
    ```
    """

    frequency_max: float = Field(ge=0)
    frequency_min: float = Field(ge=0)
    order_max: int = Field(gt=0, default=80)
    order_min: int = Field(gt=0, default=40)
    order_steps: int = Field(gt=0, default=2)
    time_lag: float = Field(ge=0, default=1.2)
    min_mpc: float = Field(ge=0, le=1, default=0.6)
    number_of_fft_points: int = Field(gt=0, default=2**8)
    num_svd_plots: int = Field(gt=0, default=3)
    damping_max_value: float = Field(ge=0, default=10)
    frequency_noise_threshold: float = Field(ge=0, default=0.03)
    damping_noise_threshold: float = Field(ge=0, default=0.03)
    mac_noise_threshold: float = Field(ge=0, default=0.03)
    minimum_cluster_dimension: int = Field(gt=0, default=3)
    maximum_distance: float = Field(ge=0, default=0.03)
    continuous_mode: bool = Field(default=True)

    DECAY_RATE: Final[float] = Field(ge=0, le=1, default=0.75)

    def _compute_impulse_response_optimized(self, signal: SignalT, time_step: float) -> npt.NDArray[BASE_DTYPE]:
        """Compute the impulse response function of the signal.

        This version of the function is optimized for speed avoiding not necessary operations in the scipy functions.

        Args:
            signal (SignalT): The signal to compute the impulse response function of.
            time_step (float): The time step of the signal.

        Returns:
            npt.NDArray[BASE_DTYPE]: The impulse response function of the signal.
        """
        try:
            number_of_observations, number_of_channels = signal.shape
        except ValueError as error:
            raise ModalIdentificationError(
                f"Input signals must be 2-dimensional, given input with {len(signal.shape)} dimensions"
            ) from error
        if number_of_observations > number_of_channels:
            signal = signal.T
            number_of_observations, number_of_channels = signal.shape

        impulse_channels = round(2 * self.time_lag / (time_step) - 1)
        if impulse_channels > number_of_channels:
            self.time_lag = int((number_of_channels + 1) * time_step) / 2
            impulse_channels = round(2 * self.time_lag / (time_step) - 1)

        impulse_response_function = np.zeros((number_of_observations, number_of_observations, impulse_channels + 1))
        start_correlation_index = int(number_of_channels - impulse_channels - 1)
        end_correlation_index = int(number_of_channels + impulse_channels)

        result_type = np.dtype(BASE_DTYPE)
        axes = [0]
        s1 = s2 = (number_of_channels,)
        shape = [max((s1[i], s2[i])) if i not in axes else s1[i] + s2[i] - 1 for i in range(1)]
        fshape = [sp_fft.next_fast_len(shape[a], True) for a in axes]

        fft, ifft = sp_fft.rfftn, sp_fft.irfftn
        ffts = fft(signal.T - np.mean(signal, axis=1), fshape, axes=axes)
        reversed_ffts = fft(_reverse_and_conj(signal.T - np.mean(signal, axis=1)), fshape, axes=axes)[:, ::-1]

        for index_first_signal in range(number_of_observations):
            for index_second_signal in range(index_first_signal, number_of_observations):
                correlation = ifft(
                    ffts[:, index_first_signal] * reversed_ffts[:, index_second_signal], fshape, axes=axes
                )
                correlation = correlation[tuple([slice(sz) for sz in shape])]
                out = _apply_conv_mode(correlation, s1, s2, "full", axes)
                out = out.astype(result_type)
                correlation = out[start_correlation_index:end_correlation_index] / (
                    number_of_channels - abs(np.arange(-impulse_channels, impulse_channels + 1))
                )
                symmetric_correlation = out[::-1][start_correlation_index:end_correlation_index] / (
                    number_of_channels - abs(np.arange(-impulse_channels, impulse_channels + 1))
                )
                correlation_index = round(len(correlation) / 2)
                impulse_response_factor_coefficient = np.exp(
                    -self.DECAY_RATE * time_step * np.arange(0, correlation_index)
                )
                impulse_response_function[index_first_signal, index_second_signal, :] = (
                    correlation[-correlation_index:] * impulse_response_factor_coefficient
                )
                impulse_response_function[index_second_signal, index_first_signal, :] = (
                    symmetric_correlation[-correlation_index:] * impulse_response_factor_coefficient
                )
        del correlation, symmetric_correlation, out, ffts, reversed_ffts
        return impulse_response_function

    def _compute_impulse_response(self, signal: SignalT, time_step: float) -> npt.NDArray[BASE_DTYPE]:
        """Compute the impulse response function of the signal.

        Args:
            signal (SignalT): The signal to compute the impulse response function of.
            time_step (float): The time step of the signal.

        Returns:
            npt.NDArray[BASE_DTYPE]: The impulse response function of the signal.
        """
        try:
            number_of_observations, number_of_channels = signal.shape
        except ValueError as error:
            raise ModalIdentificationError(
                f"Input signals must be 2-dimensional, given input with {len(signal.shape)} dimensions"
            ) from error

        if number_of_observations > number_of_channels:
            signal = signal.T
            number_of_observations, number_of_channels = signal.shape
        impulse_channels = round(2 * self.time_lag / (time_step) - 1)
        if impulse_channels > number_of_channels:
            self.time_lag = int((number_of_channels + 1) * time_step) / 2
            impulse_channels = round(2 * self.time_lag / (time_step) - 1)

        impulse_response_function = np.zeros((number_of_observations, number_of_observations, impulse_channels + 1))
        start_correlation_index = int(number_of_channels - impulse_channels - 1)
        end_correlation_index = int(number_of_channels + impulse_channels)
        normalizer = number_of_channels - abs(np.arange(-impulse_channels, impulse_channels + 1))
        for index_first_signal in range(number_of_observations):
            for index_second_signal in range(index_first_signal, number_of_observations):
                first_correlation_input = signal[index_first_signal, :] - np.mean(signal[index_first_signal, :])
                second_correlation_input = signal[index_second_signal, :] - np.mean(signal[index_second_signal, :])
                correlation_vector = scipy.signal.correlate(
                    first_correlation_input,
                    second_correlation_input,
                    mode="full",
                    method="fft",
                )
                correlation = correlation_vector[start_correlation_index:end_correlation_index] / normalizer
                symmetric_correlation = (
                    correlation_vector[::-1][start_correlation_index:end_correlation_index] / normalizer
                )
                correlation_index = round(len(correlation) / 2)
                impulse_response_coefficient = np.exp(-self.DECAY_RATE * time_step * np.arange(0, correlation_index))
                impulse_response_function[index_first_signal, index_second_signal, :] = (
                    correlation[-correlation_index:] * impulse_response_coefficient
                )
                impulse_response_function[index_second_signal, index_first_signal, :] = (
                    symmetric_correlation[-correlation_index:] * impulse_response_coefficient
                )
        del (
            correlation_vector,
            correlation,
            symmetric_correlation,
            normalizer,
            first_correlation_input,
            second_correlation_input,
        )
        return impulse_response_function

    def _build_hankel_matrix(self, impulse_response: npt.NDArray[BASE_DTYPE]) -> npt.NDArray[BASE_DTYPE]:
        """Build the Hankel matrix from the correlation matrix.

        Args:
            impulse_response (npt.NDArray[BASE_DTYPE]): The correlation matrix to build the Hankel matrix from.

        Returns:
            npt.NDArray[BASE_DTYPE]: The Hankel matrix.
        """
        _, number_of_observations, impulse_channels = impulse_response.shape
        hankel_number_of_rows = round(impulse_channels / 2) - 1
        hankel_matrix = np.zeros((dim := hankel_number_of_rows * number_of_observations, dim))

        for i in range(hankel_number_of_rows):
            for j in range(i, hankel_number_of_rows):
                r_start, r_end = i * number_of_observations, (i + 1) * number_of_observations
                c_start, c_end = j * number_of_observations, (j + 1) * number_of_observations
                hankel_matrix[r_start:r_end, c_start:c_end] = impulse_response[:, :, hankel_number_of_rows + i - j]
                if i != j:
                    hankel_matrix[c_start:c_end, r_start:r_end] = impulse_response[:, :, hankel_number_of_rows + j - i]
        return hankel_matrix

    def _perform_modal_identification(
        self,
        u: torch.Tensor,
        s: torch.Tensor,
        number_of_channels: int,
        time_step: float,
        number_of_steps: int,
    ) -> tuple[npt.NDArray[BASE_DTYPE], npt.NDArray[BASE_DTYPE], npt.NDArray[OMA_COMPLEX_DTYPE]]:
        """Perform the modal identification over each possible step.

        Args:
            u (torch.Tensor[TORCH_COMPLEX_DTYPE]): The left singular vectors of the Hankel matrix.
            s (torch.Tensor[TORCH_COMPLEX_DTYPE]): The singular values of the Hankel matrix.
            number_of_channels (int): The number of channels of the signal.
            time_step (float): The time step of the signal.
            number_of_steps (int): The number of steps to be considered.

        Returns:
            npt.NDArray[BASE_DTYPE]: The frequencies of the modes.
            npt.NDArray[BASE_DTYPE]: The damping ratios of the modes.
            npt.NDArray[OMA_COMPLEX_DTYPE]: The mode shapes of the modes.
        """
        frequencies = np.zeros((self.order_max, number_of_steps))
        damping_ratios = np.zeros((self.order_max, number_of_steps))
        mode_shapes = np.zeros((number_of_channels, self.order_max, number_of_steps), dtype=OMA_COMPLEX_DTYPE)

        steps = np.arange(self.order_min, self.order_max + self.order_steps, self.order_steps)
        for index, step in enumerate(steps):
            if step >= len(s):
                observability_matrix = torch.matmul(u, torch.sqrt(s.diag()))
            else:
                observability_matrix = torch.matmul(u[:, :step], torch.sqrt(s[:step].diag()))
            observability_matrix = observability_matrix.type(TORCH_COMPLEX_DTYPE)
            degrees_of_freedom = int(np.minimum(number_of_channels, observability_matrix.shape[0]))
            modes_matrix = observability_matrix[:degrees_of_freedom, :]
            observability_index = int(
                degrees_of_freedom * (round(observability_matrix.shape[0] / degrees_of_freedom) - 1)
            )
            try:
                modal_frequency_matrix = torch.linalg.lstsq(
                    observability_matrix[:observability_index, :],
                    observability_matrix[-observability_index:, :],
                    driver="gels",
                ).solution
                frequency_eigenvalues, frequency_eigenvectors = torch.linalg.eig(modal_frequency_matrix)
            except ValueError as error:
                raise ModalIdentificationError("Modal identification cannot be performed.") from error

            frequency_eigenvalues[frequency_eigenvalues == 0] = 1
            frequency_poles = np.array([torch.log(frequency) for frequency in frequency_eigenvalues]) / time_step
            step_frequencies = np.abs(frequency_poles) / (2 * np.pi)
            step_damping_ratios = np.zeros_like(step_frequencies)
            nonzero_mask = step_frequencies != 0
            step_damping_ratios[nonzero_mask] = (
                -100 * np.real(frequency_poles[nonzero_mask]) / np.abs(frequency_poles[nonzero_mask])
            )
            step_mode_shapes = torch.matmul(modes_matrix, frequency_eigenvectors).numpy()

            frequencies[: len(step_frequencies), index] = step_frequencies
            damping_ratios[: len(step_frequencies), index] = step_damping_ratios
            mode_shapes[:, : len(step_frequencies), index] = step_mode_shapes
            del step_frequencies, step_mode_shapes, step_damping_ratios
        del modal_frequency_matrix, frequency_poles, frequency_eigenvalues, frequency_eigenvectors, observability_matrix
        return frequencies, damping_ratios, mode_shapes

    def _remove_zero_frequency_modes(
        self,
        frequencies: npt.NDArray[BASE_DTYPE],
        damping_ratios: npt.NDArray[BASE_DTYPE],
        mode_shapes: npt.NDArray[OMA_COMPLEX_DTYPE],
    ) -> tuple[npt.NDArray[BASE_DTYPE], npt.NDArray[BASE_DTYPE], npt.NDArray[OMA_COMPLEX_DTYPE]]:
        """Remove the zero frequency modes from the modal parameters.

        Args:
            frequencies (npt.NDArray[BASE_DTYPE]): The frequencies of the modes.
            damping_ratios (npt.NDArray[BASE_DTYPE]): The damping ratios of the modes.
            mode_shapes (npt.NDArray[OMA_COMPLEX_DTYPE]): The mode shapes of the modes.

        Returns:
            tuple[npt.NDArray[BASE_DTYPE], npt.NDArray[BASE_DTYPE], npt.NDArray[OMA_COMPLEX_DTYPE]]: The filtered modal parameters.
        """
        steps = np.arange(self.order_min, self.order_max + self.order_steps, self.order_steps)
        for step in steps:
            relative_step_index = int((step - self.order_min) / self.order_steps)
            non_zero_frequencies = np.nonzero(frequencies[:, relative_step_index] > 0)[0]
            for index in non_zero_frequencies[:-1]:
                first_mode = mode_shapes[:, index, relative_step_index]
                second_mode = mode_shapes[:, index + 1, relative_step_index]
                conjugate_modes = np.matmul(np.conjugate(second_mode), second_mode) * np.matmul(
                    np.conjugate(first_mode), first_mode
                )
                if (conjugate_modes != 0) & (frequencies[index, relative_step_index] != 0):
                    mac_value = compute_mac_value(first_mode, second_mode)
                    frequency_delta = (
                        np.abs(frequencies[index, relative_step_index] - frequencies[index + 1, relative_step_index])
                        / frequencies[index, relative_step_index]
                    )
                    damping_delta = (
                        np.abs(
                            damping_ratios[index, relative_step_index] - damping_ratios[index + 1, relative_step_index]
                        )
                        / damping_ratios[index, relative_step_index]
                    )
                    if (frequency_delta < 0.001) & (damping_delta < 0.001) & (1 - mac_value < 0.001):
                        frequencies[index, relative_step_index] = np.mean(
                            [frequencies[index + 1, relative_step_index], frequencies[index, relative_step_index]]
                        )
                        frequencies[index + 1, relative_step_index] = 0
                        damping_ratios[index, relative_step_index] = np.mean(
                            [damping_ratios[index + 1, relative_step_index], damping_ratios[index, relative_step_index]]
                        )
                        damping_ratios[index + 1, relative_step_index] = 0
        for step in steps:
            true_index = int((step - self.order_min) / self.order_steps)
            valid_index_position = np.nonzero(frequencies[:, true_index] > 0)[0]
            zeros_final_index = self.order_max + 1
            frequencies[: len(valid_index_position), true_index] = frequencies[valid_index_position, true_index]
            frequencies[len(valid_index_position) : zeros_final_index, true_index] = 0
            damping_ratios[: len(valid_index_position), true_index] = damping_ratios[valid_index_position, true_index]
            damping_ratios[len(valid_index_position) : zeros_final_index, true_index] = 0
            mode_shapes[:, : len(valid_index_position), true_index] = mode_shapes[:, valid_index_position, true_index]
            mode_shapes[:, len(valid_index_position) : zeros_final_index, true_index] = 0
        del valid_index_position, conjugate_modes, first_mode, second_mode
        return frequencies, damping_ratios, mode_shapes

    def _cut_off_modal_parameters(
        self,
        frequencies: npt.NDArray[BASE_DTYPE],
        damping_ratios: npt.NDArray[BASE_DTYPE],
        mode_shapes: npt.NDArray[OMA_COMPLEX_DTYPE],
        number_of_steps: int,
    ) -> tuple[ModalIdentification, ModalDashboardData | None]:
        """Cut off the modal parameters based on the frequency, damping ratio and mac value thresholds.

        Args:
            frequencies (npt.NDArray[BASE_DTYPE]): The frequencies of the modes.
            damping_ratios (npt.NDArray[BASE_DTYPE]): The damping ratios of the modes.
            mode_shapes (npt.NDArray[OMA_COMPLEX_DTYPE]): The mode shapes of the modes.
            number_of_steps (int): The number of steps/orders to be considered.

        Returns:
            tuple[ModalIdentification, ModalDashboardData]: The cut off modal parameters.
        """
        steps = np.arange(self.order_min, self.order_max + self.order_steps, self.order_steps)
        final_frequencies = np.zeros_like(frequencies)
        final_damping_ratios = np.zeros_like(damping_ratios)
        final_modes = np.zeros_like(mode_shapes)
        frequencies_dashboard = []
        damping_ratios_dashboard = []
        mode_shapes_dashboard: list = []
        order_steps_dashboard: list = []

        for step in range(number_of_steps):
            valid_mask = (
                (frequencies[:, step] < self.frequency_max)
                & (frequencies[:, step] > self.frequency_min)
                & (damping_ratios[:, step] > 0)
                & (damping_ratios[:, step] < self.damping_max_value)
            )
            valid_indices = np.nonzero(valid_mask)[0]
            if len(valid_indices) == 0:
                raise ModalIdentificationError(
                    f"No frequencies found with thresholds: freq_min={self.frequency_min}, freq_max={self.frequency_max}, damp_max={self.damping_max_value}"
                )
            indexes_to_be_removed = modal_phase_collinearity(mode_shapes[:, valid_indices, step].T) < self.min_mpc
            valid_indices = np.delete(valid_indices, np.nonzero(indexes_to_be_removed)[0])
            if np.size(valid_indices) == 0:
                raise ModalIdentificationError(
                    f"No frequencies found with thresholds: freq_min={self.frequency_min}, freq_max={self.frequency_max}, damp_max={self.damping_max_value}, min_mpc={self.min_mpc}"
                )
            frequencies_dashboard.extend(frequencies[valid_indices, step].tolist())
            damping_ratios_dashboard.extend(damping_ratios[valid_indices, step].tolist())
            mode_shapes_dashboard.append(mode_shapes[:, valid_indices, step].tolist())
            order_steps_dashboard.extend(np.tile(steps[step], len(valid_indices)).tolist())
            final_frequencies[valid_indices, step] = frequencies[valid_indices, step]
            final_damping_ratios[valid_indices, step] = damping_ratios[valid_indices, step]
            final_modes[:, valid_indices, step] = mode_shapes[:, valid_indices, step]

        end_lines = np.nonzero(~np.all(final_frequencies == 0, axis=1))[0]
        if len(end_lines):
            end_lines = end_lines[-1] + 1
        else:
            raise ModalIdentificationError("No stable frequencies have been found.")
        final_frequencies = final_frequencies[:end_lines, :]
        final_damping_ratios = final_damping_ratios[:end_lines, :]
        final_modes = final_modes[:, :end_lines, :]

        for step in range(number_of_steps):
            sorting_index = np.argsort(final_frequencies[:, step])[::-1]
            final_frequencies[:, step] = final_frequencies[sorting_index, step]
            final_damping_ratios[:, step] = final_damping_ratios[sorting_index, step]
            final_modes[:, :, step] = final_modes[:, sorting_index, step]

        temp_mode_shapes_dashboard = [
            [mode_shapes_dashboard[index_row][index_col] for index_row in range(len(mode_shapes_dashboard))]
            for index_col in range(len(mode_shapes_dashboard[0]))
        ]

        mode_shapes_dashboard_array = np.array(
            [
                sub_item
                for sub_sublist in [item for sublist in temp_mode_shapes_dashboard for item in sublist]
                for sub_item in sub_sublist
            ]
        ).reshape(-1, len(frequencies_dashboard))
        del temp_mode_shapes_dashboard, mode_shapes_dashboard

        return ModalIdentification(
            frequencies=final_frequencies, damping_ratios=final_damping_ratios, mode_shapes=final_modes
        ), ModalDashboardData(
            frequencies=np.array(frequencies_dashboard),
            damping_ratios=np.array(damping_ratios_dashboard),
            mode_shapes=mode_shapes_dashboard_array,
            model_orders=np.array(order_steps_dashboard),
        )

    def _get_stability_pole_status(
        self, frequency_stability: int, damping_stability: int, mode_shape_stability: int
    ) -> int:
        """Get the stability status of the pole based on the frequency, damping ratio and mode shape stability.

        Args:
            frequency_stability (int): The stability status of the frequency.
            damping_stability (int): The stability status of the damping ratio.
            mode_shape_stability (int): The stability status of the mode shape.

        Returns:
            int: The stability status of the pole.
        """
        if frequency_stability == 0:
            return 0

        stability_map = {
            (1, 1, 1): 1,
            (1, 0, 1): 2,
            (1, 1, 0): 3,
            (1, 0, 0): 4,
        }

        return stability_map.get((frequency_stability, damping_stability, mode_shape_stability), 0)

    def _oma_stability_check(
        self,
        frequencies_1: npt.NDArray[BASE_DTYPE],
        damping_ratio_1: npt.NDArray[BASE_DTYPE],
        mode_shape_1: npt.NDArray[OMA_COMPLEX_DTYPE],
        frequencies_2: npt.NDArray[BASE_DTYPE],
        damping_ratio_2: npt.NDArray[BASE_DTYPE],
        mode_shape_2: npt.NDArray[OMA_COMPLEX_DTYPE],
    ) -> StabilityCheck:
        """Check the stability of the pole based on the frequency, damping ratio and mode shape stability.

        Args:
            frequencies_1 (npt.NDArray[BASE_DTYPE]): The frequencies of the first pole.
            damping_ratio_1 (npt.NDArray[BASE_DTYPE]): The damping ratios of the first pole.
            mode_shape_1 (npt.NDArray[OMA_COMPLEX_DTYPE]): The mode shapes of the first pole.
            frequencies_2 (npt.NDArray[BASE_DTYPE]): The frequencies of the second pole.
            damping_ratio_2 (npt.NDArray[BASE_DTYPE]): The damping ratios of the second pole.
            mode_shape_2 (npt.NDArray[OMA_COMPLEX_DTYPE]): The mode shapes of the second pole.

        Returns:
            StabilityCheck: The stability status of the pole.
        """
        stability_status = []
        frequencies = []
        damping_ratios = []
        mode_shapes = []
        mac_values = []

        for first_frequency, first_damping, first_mode in zip(
            frequencies_1, damping_ratio_1, mode_shape_1.T, strict=True
        ):
            for second_frequency, second_damping, second_mode in zip(
                frequencies_2, damping_ratio_2, mode_shape_2.T, strict=True
            ):
                frequency_stability = int(
                    np.abs(1 - first_frequency / second_frequency) < self.frequency_noise_threshold
                )
                damping_stability = int(np.abs(1 - first_damping / second_damping) < self.damping_noise_threshold)
                mac_value = compute_mac_value(
                    np.array(first_mode, dtype=OMA_COMPLEX_DTYPE), np.array(second_mode, dtype=OMA_COMPLEX_DTYPE)
                )
                mac_stability = int(mac_value > (1 - self.mac_noise_threshold))
                stability = self._get_stability_pole_status(frequency_stability, damping_stability, mac_stability)
                frequencies.append(second_frequency)
                damping_ratios.append(second_damping)
                mode_shapes.append(second_mode)
                mac_values.append(mac_value)
                stability_status.append(stability)

        sorting_index = np.argsort(frequencies)

        return StabilityCheck(
            frequencies=np.array(frequencies)[sorting_index],
            damping_ratios=np.array(damping_ratios)[sorting_index],
            mode_shapes=np.array(mode_shapes, dtype=np.complex64)[sorting_index, :].T,
            mac_values=np.array(mac_values)[sorting_index],
            stability_status=np.array(stability_status)[sorting_index],
        )

    def _stability_poles_analysis(
        self,
        input_frequencies: npt.NDArray[BASE_DTYPE],
        input_damping_ratios: npt.NDArray[BASE_DTYPE],
        input_modes: npt.NDArray[OMA_COMPLEX_DTYPE],
    ) -> StabilityModalIdentification:
        """Analyze the stability of the poles.

        Args:
            input_frequencies (npt.NDArray[BASE_DTYPE]): The frequencies of the poles.
            input_damping_ratios (npt.NDArray[BASE_DTYPE]): The damping ratios of the poles.
            input_modes (npt.NDArray[BASE_DTYPE]): The mode shapes of the poles.

        Returns:
            StabilityModalIdentification: The modal parameters and their stability status.
        """
        _, number_of_steps = input_frequencies.shape
        frequencies = []
        damping_ratios = []
        mode_shapes = []
        stability_status = []

        steps = np.arange(number_of_steps, 0, -1) - 1
        for step in steps[1:]:
            first_frequency = input_frequencies[:, step + 1]
            second_frequency = input_frequencies[:, step]
            first_damping = input_damping_ratios[:, step + 1]
            second_damping = input_damping_ratios[:, step]
            first_mode_shape = input_modes[:, :, step + 1]
            second_mode_shape = input_modes[:, :, step]
            position_zeros = np.nonzero(second_frequency == 0)[0]
            if position_zeros.size > 0:
                first_zero = position_zeros[0]
                second_frequency = second_frequency[:first_zero]
                second_damping = second_damping[:first_zero]
                second_mode_shape = second_mode_shape[:, :first_zero]
            try:
                oma_parameters = self._oma_stability_check(
                    first_frequency,
                    first_damping,
                    first_mode_shape,
                    second_frequency,
                    second_damping,
                    second_mode_shape,
                )
            except IndexError:
                oma_parameters = StabilityCheck(
                    frequencies=np.array([]),
                    damping_ratios=np.array([]),
                    mode_shapes=np.array([[]]),
                    stability_status=np.array([]),
                    mac_values=np.array([]),
                )
            frequencies.append(oma_parameters["frequencies"])
            damping_ratios.append(oma_parameters["damping_ratios"])
            mode_shapes.append(oma_parameters["mode_shapes"])
            stability_status.append(oma_parameters["stability_status"])
        del (
            oma_parameters,
            first_frequency,
            first_damping,
            first_mode_shape,
            second_frequency,
            second_damping,
            second_mode_shape,
        )
        return StabilityModalIdentification(
            frequencies=frequencies,
            damping_ratios=damping_ratios,
            mode_shapes=mode_shapes,
            stability_status=stability_status,
        )

    def _stability_pole_filter(
        self,
        stability_status: list[npt.NDArray[np.int32]],
        frequencies: list[npt.NDArray[BASE_DTYPE]],
        damping_ratios: list[npt.NDArray[BASE_DTYPE]],
        mode_shapes: list[npt.NDArray[OMA_COMPLEX_DTYPE]],
    ) -> tuple[npt.NDArray[BASE_DTYPE], npt.NDArray[BASE_DTYPE], npt.NDArray[OMA_COMPLEX_DTYPE]]:
        """Filter the poles based on the stability status.

        Args:
            stability_status (npt.NDArray[BASE_DTYPE]): The stability status of the poles.
            frequencies (npt.NDArray[BASE_DTYPE]): The frequencies of the poles.
            damping_ratios (npt.NDArray[BASE_DTYPE]): The damping ratios of the poles.
            mode_shapes (npt.NDArray[OMA_COMPLEX_DTYPE]): The mode shapes of the poles.

        Returns:
            tuple[npt.NDArray[BASE_DTYPE], npt.NDArray[BASE_DTYPE], npt.NDArray[OMA_COMPLEX_DTYPE]]: The filtered poles.
        """
        stable_frequencies: list[BASE_DTYPE] = []
        stable_damping_ratios: list[BASE_DTYPE] = []
        stable_modes: list[npt.NDArray[OMA_COMPLEX_DTYPE]] = []

        stable_pole_index = 1
        for pole_index in np.arange(len(stability_status) - 1, -1, -1):
            ind = np.nonzero(stability_status[pole_index] == stable_pole_index)[0]
            stable_frequencies.extend(np.array(frequencies[pole_index])[ind])
            stable_damping_ratios.extend(np.array(damping_ratios[pole_index])[ind])
            stable_modes.append(np.array(mode_shapes[pole_index])[:, ind])

        stable_modes_reorder: list[list[npt.NDArray[OMA_COMPLEX_DTYPE]]] = [
            [stable_modes[index_row][index_col] for index_row in range(len(stable_modes))]
            for index_col in range(len(stable_modes[0]))
        ]
        final_stable_modes = np.array(
            [
                sub_item
                for sub_sublist in [item for sublist in stable_modes_reorder for item in sublist]
                for sub_item in sub_sublist
            ]
        ).reshape(-1, len(stable_frequencies))
        del stable_modes_reorder
        return np.array(stable_frequencies), np.array(stable_damping_ratios), final_stable_modes

    def _initialize_cluster(
        self,
        cluster_id: int,
        freq: npt.NDArray[BASE_DTYPE],
        damp: npt.NDArray[BASE_DTYPE],
        mode_real: npt.NDArray[BASE_DTYPE],
        mode_complex: npt.NDArray[OMA_COMPLEX_DTYPE],
        clusters: dict[str, dict[int, npt.NDArray[BASE_DTYPE | OMA_COMPLEX_DTYPE]]],
    ):
        """Initialize the cluster.

        Args:
            cluster_id (int): The id of the cluster.
            freq (npt.NDArray[BASE_DTYPE]): The frequencies of the cluster.
            damp (npt.NDArray[BASE_DTYPE]): The damping ratios of the cluster.
            mode_real (npt.NDArray[BASE_DTYPE]): The real part of the mode shapes of the cluster.
            mode_complex (npt.NDArray[OMA_COMPLEX_DTYPE]): The complex part of the mode shapes of the cluster.
            clusters (dict): The clusters.
        """
        clusters["freq"][cluster_id] = freq
        clusters["damp"][cluster_id] = damp
        clusters["mode_real"][cluster_id] = mode_real.reshape(1, -1)
        clusters["mode_complex"][cluster_id] = mode_complex.reshape(1, -1)

    def _compute_cluster_averages(
        self, clusters: dict[str, dict[int, npt.NDArray]]
    ) -> tuple[
        dict[int, npt.NDArray[BASE_DTYPE]],
        dict[int, npt.NDArray[BASE_DTYPE]],
        dict[int, npt.NDArray[BASE_DTYPE]],
        dict[int, npt.NDArray[OMA_COMPLEX_DTYPE]],
    ]:
        """Compute the averages of the clusters.

        Args:
            clusters (dict): The clusters.

        Returns:
            tuple[dict[int, npt.NDArray[BASE_DTYPE]], dict[int, npt.NDArray[BASE_DTYPE]], dict[int, npt.NDArray[OMA_COMPLEX_DTYPE]], dict[int, npt.NDArray[OMA_COMPLEX_DTYPE]]]: The averages of the clusters.
        """
        freq_avg: dict[int, npt.NDArray[BASE_DTYPE]] = {}
        damp_avg: dict[int, npt.NDArray[BASE_DTYPE]] = {}
        mode_avg: dict[int, npt.NDArray[BASE_DTYPE]] = {}
        mode_avg_complex: dict[int, npt.NDArray[OMA_COMPLEX_DTYPE]] = {}
        for idx, key in enumerate(clusters["freq"]):
            freq_avg[idx] = np.mean(clusters["freq"][key])
            damp_avg[idx] = np.median(clusters["damp"][key])
            mode_avg[idx] = np.mean(abs(clusters["mode_real"][key]), axis=0) * np.sign(clusters["mode_real"][key][0])
            mode_avg_complex[idx] = np.mean(clusters["mode_complex"][key], axis=0)
        return freq_avg, damp_avg, mode_avg, mode_avg_complex

    def _compute_cluster_distances(
        self,
        start_freq: npt.NDArray[BASE_DTYPE],
        start_mode: npt.NDArray[BASE_DTYPE],
        freq_avg: dict[int, npt.NDArray[BASE_DTYPE]],
        mode_avg: dict[int, npt.NDArray[BASE_DTYPE]],
    ) -> dict[int, npt.NDArray[BASE_DTYPE]]:
        """Compute the distances between the start mode and the mode averages.

        Args:
            start_freq (npt.NDArray[BASE_DTYPE]): The frequencies of the start mode.
            start_mode (npt.NDArray[OMA_COMPLEX_DTYPE]): The mode shapes of the start mode.
            freq_avg (dict): The averages of the frequencies of the clusters.
            mode_avg (dict): The averages of the mode shapes of the clusters.

        Returns:
            dict: The distances between the start mode and the mode averages.
        """
        distances = {}
        for idx, freq in freq_avg.items():
            phi = mode_avg[idx]
            freq_diff = abs(start_freq - freq) / freq
            modal_overlap = (abs(np.matmul(phi.T, start_mode))) ** 2 / (
                np.matmul(np.conjugate(phi), phi) * np.matmul(np.conjugate(start_mode), start_mode)
            )
            distances[idx] = (freq_diff + 1 - modal_overlap)[0]
        return distances

    def _compute_confidence_bounds(
        self,
        means: npt.NDArray[BASE_DTYPE],
        std_devs: npt.NDArray[BASE_DTYPE],
        sizes: npt.NDArray[np.int32],
        z_score: float = 1.96,
    ) -> npt.NDArray[BASE_DTYPE]:
        """Compute the confidence bounds of the means.

        Args:
            means (npt.NDArray[BASE_DTYPE]): The means of the data.
            std_devs (npt.NDArray[BASE_DTYPE]): The standard deviations of the data.
            sizes (npt.NDArray[BASE_DTYPE]): The sizes of the data.
            z_score (float): The z-score for the confidence interval.

        Returns:
            npt.NDArray[BASE_DTYPE]: The confidence bounds of the means.
        """
        margin = z_score * std_devs / np.sqrt(sizes)
        return np.column_stack((means - margin, means + margin))

    def _aggregate_cluster_data(
        self, clusters: dict[str, dict[int, npt.NDArray[BASE_DTYPE]]], num_channels: int
    ) -> CovSSIResults:
        """Aggregate the data of the clusters to compute final modal parameters.

        Args:
            clusters (dict): The clusters.
            num_channels (int): The number of channels of the mode shapes.

        Returns:
            CovSSIResults: The modal parameters, not ordered, of the clusters.
        """
        num_modes = len(clusters["freq"])
        freq_avg = np.zeros(num_modes)
        freq_std = np.zeros(num_modes)
        damp_avg = np.zeros(num_modes)
        damp_std = np.zeros(num_modes)
        cluster_size = np.zeros(num_modes, dtype=np.int32)
        mode_avg = np.zeros((num_channels, num_modes))
        mode_complex_avg = np.zeros((num_channels, num_modes), dtype=OMA_COMPLEX_DTYPE)

        for i in range(num_modes):
            freq, damp = clusters["freq"][i], clusters["damp"][i]
            mode_real, mode_complex = clusters["mode_real"][i], clusters["mode_complex"][i]

            freq_avg[i] = np.mean(freq)
            freq_std[i] = np.std(freq, ddof=1) if len(freq) > 1 else 0.0
            damp_avg[i] = np.median(damp)
            damp_std[i] = np.std(damp, ddof=1) if len(damp) > 1 else 0.0
            cluster_size[i] = len(freq)

            mode_avg[:, i] = np.mean(abs(mode_real), axis=0) * np.sign(mode_real[0])
            real = np.mean(abs(np.real(mode_complex)), axis=0) * np.sign(np.real(mode_complex[0]))
            imag = np.mean(abs(np.imag(mode_complex)), axis=0) * np.sign(np.imag(mode_complex[0]))
            mode_complex_avg[:, i] = real + 1j * imag

        return CovSSIResults(
            frequencies=freq_avg,
            damping_ratios=damp_avg,
            real_mode_shapes=mode_avg,
            complex_mode_shapes=mode_complex_avg,
            cluster_dimensions=cluster_size,
            frequency_bounds=freq_std,
            damping_bounds=damp_std,
        )

    def _cluster_modal_parameters(
        self,
        frequencies: npt.NDArray[BASE_DTYPE],
        damping_ratios: npt.NDArray[BASE_DTYPE],
        mode_shapes: npt.NDArray[OMA_COMPLEX_DTYPE],
        shuffle: bool = False,
    ) -> CovSSIResults:
        """Cluster the modal parameters.

        Args:
            frequencies (npt.NDArray[BASE_DTYPE]): The frequencies of the modal parameters.
            damping_ratios (npt.NDArray[BASE_DTYPE]): The damping ratios of the modal parameters.
            mode_shapes (npt.NDArray[OMA_COMPLEX_DTYPE]): The mode shapes of the modal parameters.
            shuffle (bool): Whether to shuffle the modal parameters.

        Returns:
            dict: The clustered modal parameters.
        """
        num = len(frequencies)
        permute = np.random.Generator(np.random.PCG64(10)).permutation(num)

        clusters: dict[str, dict[int, npt.NDArray]] = {"freq": {}, "damp": {}, "mode_real": {}, "mode_complex": {}}

        for raw_idx in range(num):
            idx = permute[raw_idx] if shuffle else raw_idx
            start_freq = np.array([frequencies[idx]])
            start_damp = np.array([damping_ratios[idx]])
            start_mode = complex_mode_to_real_mode(mode_shapes[:, idx])
            start_mode_complex = mode_shapes[:, idx]

            if not clusters["freq"]:
                self._initialize_cluster(0, start_freq, start_damp, start_mode, start_mode_complex, clusters)
                continue

            freq_avg, _, mode_avg, _ = self._compute_cluster_averages(clusters)
            distances = self._compute_cluster_distances(start_freq, start_mode, freq_avg, mode_avg)
            best_cluster = int(np.argmin(list(distances.values())))

            if 0 <= distances[best_cluster] <= self.maximum_distance:
                for key, val in zip(
                    ["freq", "damp", "mode_real", "mode_complex"],
                    [start_freq, start_damp, start_mode, start_mode_complex],
                    strict=False,
                ):
                    clusters[key][best_cluster] = (
                        np.vstack((clusters[key][best_cluster], val))
                        if "mode" in key
                        else np.append(clusters[key][best_cluster], val)
                    )
            else:
                new_id = max(clusters["freq"].keys()) + 1
                self._initialize_cluster(new_id, start_freq, start_damp, start_mode, start_mode_complex, clusters)

        cluster_results = self._aggregate_cluster_data(clusters, mode_shapes.shape[0])
        del clusters, start_freq, start_damp, start_mode, start_mode_complex

        sorting_index = np.argsort(cluster_results["frequencies"])
        for key in cluster_results:
            cluster_results[key] = cluster_results[key][..., sorting_index]  # type: ignore

        selected = np.nonzero(cluster_results["cluster_dimensions"] > self.minimum_cluster_dimension)[0]
        for key in (
            "frequencies",
            "damping_ratios",
            "real_mode_shapes",
            "complex_mode_shapes",
            "cluster_dimensions",
            "frequency_bounds",
            "damping_bounds",
        ):
            cluster_results[key] = cluster_results[key][..., selected]  # type: ignore

        freq_bounds = self._compute_confidence_bounds(
            cluster_results["frequencies"], cluster_results["frequency_bounds"], cluster_results["cluster_dimensions"]
        )
        damp_bounds = self._compute_confidence_bounds(
            cluster_results["damping_ratios"], cluster_results["damping_bounds"], cluster_results["cluster_dimensions"]
        )
        return CovSSIResults(
            frequencies=cluster_results["frequencies"],
            damping_ratios=cluster_results["damping_ratios"],
            real_mode_shapes=cluster_results["real_mode_shapes"],
            complex_mode_shapes=cluster_results["complex_mode_shapes"],
            cluster_dimensions=cluster_results["cluster_dimensions"],
            frequency_bounds=freq_bounds,
            damping_bounds=damp_bounds,
        )

    def apply(
        self,
        *,
        signal: SignalT,
        sampling_frequency: float,
        optimized: bool = True,
        shuffle: bool = True,
    ) -> tuple[CovSSIResults, ModalDashboardData | None]:
        """Apply the COV-SSI algorithm to the signal.

        Args:
            signal (SignalT): The signal to apply the algorithm to.
            sampling_frequency (float): The sampling frequency of the signal.
            continuous_mode (bool): Whether to use continuous mode.
            optimized (bool): Whether to use the optimized version of the algorithm.
            shuffle (bool): Whether to shuffle the modal parameters.

        Returns:
            tuple[CovSSIResults, ModalDashboardData | None]: The clustered modal parameters and the dashboard data if continuous mode is used, otherwise None.
        """
        if (len(signal.shape) != 2) or (np.size(signal) == 0):
            raise ModalIdentificationError(
                "Input signals must be 2-dimensional, given input with different dimension or empty"
            )
        time_step = 1 / sampling_frequency
        number_of_steps = round((self.order_max - self.order_min) / self.order_steps + 1)
        _, number_of_channels = signal.shape
        try:
            if optimized:
                impulse_response = self._compute_impulse_response_optimized(signal, time_step)
            else:
                impulse_response = self._compute_impulse_response(signal, time_step)
        except Exception as error:
            raise ModalIdentificationError(
                f"Impulse response computation failed: {error}"
            ) from error

        hankel_matrix = self._build_hankel_matrix(impulse_response)
        del impulse_response
        tensor_hankel_matrix = torch.tensor(hankel_matrix)
        u, s, _ = torch.linalg.svd(tensor_hankel_matrix)
        del tensor_hankel_matrix, hankel_matrix

        frequencies, damping_ratios, mode_shapes = self._perform_modal_identification(
            u, s, number_of_channels, time_step, number_of_steps
        )
        del u, s
        frequencies, damping_ratios, mode_shapes = self._remove_zero_frequency_modes(
            frequencies, damping_ratios, mode_shapes
        )
        modal_parameters, dashboard_data = self._cut_off_modal_parameters(
            frequencies, damping_ratios, mode_shapes, number_of_steps
        )
        if self.continuous_mode:
            dashboard_data = None

        modal_parameters_with_stability = self._stability_poles_analysis(
            modal_parameters["frequencies"], modal_parameters["damping_ratios"], modal_parameters["mode_shapes"]
        )
        del modal_parameters

        frequencies, damping_ratios, mode_shapes = self._stability_pole_filter(
            modal_parameters_with_stability["stability_status"],
            modal_parameters_with_stability["frequencies"],
            modal_parameters_with_stability["damping_ratios"],
            modal_parameters_with_stability["mode_shapes"],
        )
        del modal_parameters_with_stability

        cov_results = self._cluster_modal_parameters(frequencies, damping_ratios, mode_shapes, shuffle)
        del frequencies, damping_ratios, mode_shapes

        for index in range(cov_results["complex_mode_shapes"].shape[1]):
            cov_results["complex_mode_shapes"][:, index] = maximum_correlation_rotation(
                cov_results["complex_mode_shapes"][:, index]
            )

        return cov_results, dashboard_data

    def _create_dashboard_graph_data(
        self, stability_results: StabilityModalIdentification, model_orders: npt.NDArray[np.int32]
    ) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
        """Return frequency and stability status to plot in the dashboard.

        Args:
            stability_results (StabilityModalIdentification): dictionaries with all the modal parameters together with their stability.
            model_orders (NDArray[np.int32]): array containing all the possible orders of the model.

        Returns:
            tuple[dict[str, npt.NDArray[BASE_DTYPE]], dict[str, npt.NDArray[np.int32]]]: dictionaries containing frequency and model orders with respect to stability status.
        """
        frequency_by_stability = {}
        model_order_by_stability = {}
        possible_stability_status = [
            "new_pole",
            "stable_pole",
            "stable_frequency_and_mac",
            "stable_frequency_and_damping_ratios",
            "stable_frequency",
        ]

        for pole_index, order in enumerate(possible_stability_status):
            model_order = []
            frequency: list[float] = []
            for index in np.arange(len(stability_results["stability_status"]) - 1, -1, -1):
                stable_indexes = np.nonzero(stability_results["stability_status"][index] == pole_index)[0]
                if stable_indexes.size:
                    frequency.extend(list(map(float, stability_results["frequencies"][index][stable_indexes])))
                    model_order.extend(
                        (np.ones(len(stable_indexes), dtype=np.int32) * model_orders[-((index + 1) + 1)]).tolist()
                    )
            frequency_by_stability[order] = frequency
            model_order_by_stability[order] = model_order
        return frequency_by_stability, model_order_by_stability

    def _reorder_poles(
        self,
        frequencies: np.ndarray,
        damping_ratios: np.ndarray,
        mode_shapes: np.ndarray,
        model_orders: np.ndarray,
        steps: np.ndarray,
    ) -> ModalIdentification:
        """Reorder the poles.

        Args:
            frequencies (npt.NDArray[BASE_DTYPE]): The frequencies of the poles.
            damping_ratios (npt.NDArray[BASE_DTYPE]): The damping ratios of the poles.
            mode_shapes (npt.NDArray[OMA_COMPLEX_DTYPE]): The mode shapes of the poles.
            model_orders (npt.NDArray[BASE_DTYPE]): The model orders of the poles.
            steps (npt.NDArray[BASE_DTYPE]): The steps of the model.

        Returns:
            ModalIdentification: The reordered poles.
        """
        ordered_frequencies = np.zeros((len(frequencies), len(steps)))
        ordered_damping_ratios = np.zeros((len(frequencies), len(steps)))
        ordered_mode_shapes = np.zeros((mode_shapes.shape[0], len(frequencies), len(steps)), dtype=complex)
        for i, step in enumerate(steps):
            idx = np.nonzero(model_orders == step)[0]
            n = len(idx)
            ordered_frequencies[:n, i] = frequencies[idx]
            ordered_damping_ratios[:n, i] = damping_ratios[idx]
            ordered_mode_shapes[:, :n, i] = mode_shapes[:, idx]

        nonzero_rows = np.any(ordered_frequencies != 0, axis=1)
        max_valid_index = np.argmax(~nonzero_rows) if not nonzero_rows.all() else len(frequencies)

        return ModalIdentification(
            frequencies=ordered_frequencies[:max_valid_index],
            mode_shapes=ordered_mode_shapes[:, :max_valid_index],
            damping_ratios=ordered_damping_ratios[:max_valid_index],
        )

    def get_svd_plot_data(self, *, signal: SignalT, sampling_frequency: float, dashboard_data) -> CovSSIDashboardData:
        """Get the SVD plot data.

        Args:
            signal (SignalT): The signal to apply the algorithm to.
            dashboard_data (dict): The dashboard data.
            sampling_frequency (float): The sampling frequency of the signal.

        Returns:
            CovSSIDashboardData: The SVD plot data.
        """
        frequencies, eigenvalues_matrix, _ = self.compute_signal_svd(
            signal=signal, sampling_frequency=sampling_frequency
        )
        steps = np.arange(self.order_min, self.order_max + self.order_steps, self.order_steps)
        reordered_oma = self._reorder_poles(
            dashboard_data["frequencies"],
            dashboard_data["damping_ratios"],
            dashboard_data["mode_shapes"],
            dashboard_data["model_orders"],
            steps,
        )
        reordered_poles_parameters = self._stability_poles_analysis(
            reordered_oma["frequencies"], reordered_oma["damping_ratios"], reordered_oma["mode_shapes"]
        )
        del reordered_oma
        frequency_by_stability, model_order_by_stability = self._create_dashboard_graph_data(
            reordered_poles_parameters, steps
        )
        del reordered_poles_parameters

        return CovSSIDashboardData(
            y_axis_signal=eigenvalues_matrix,
            x_axis_frequencies=frequencies,
            frequencies=frequency_by_stability,
            model_orders=model_order_by_stability,
        )
