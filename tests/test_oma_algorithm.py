import numpy as np
import pytest
from pydantic_core import ValidationError
from pytest_mock import MockerFixture

from dynoma.exceptions import ModalIdentificationError
from dynoma.oma_algorithm import OmaAlgorithm

TOLERANCE = 1e-6

base_oma_algorithm = OmaAlgorithm(
    frequency_max=5,
    frequency_min=0,
    number_of_fft_points=2**3,
    num_svd_plots=3,
)


@pytest.mark.parametrize(
    "sampling_frequency, frequency_min, frequency_max, number_of_fft_points, error_message",
    [
        (20, -5, 10, 2**4, "Input should be greater than or equal to 0"),
        (20, 10, 5, 2**4, "Frequency minimum must be less than frequency maximum: given 10.0, 5.0"),
        (
            15,
            5,
            10,
            2**4,
            "Frequency maximum must be less than or equal to half of the sampling frequency: given 10.0, 15",
        ),
        (20, 5, 10, 0, "Input should be greater than 0"),
    ],
)
def test_compute_signal_svd_should_raise_error_if_parameters_not_well_defined(
    sampling_frequency,
    frequency_min,
    frequency_max,
    number_of_fft_points,
    error_message,
):
    signals = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    with pytest.raises((ValidationError, ModalIdentificationError), match=error_message):
        oma_algorithm = OmaAlgorithm(
            frequency_max=frequency_max,
            frequency_min=frequency_min,
            number_of_fft_points=number_of_fft_points,
            num_svd_plots=2,
        )
        oma_algorithm.compute_signal_svd(signal=signals, sampling_frequency=sampling_frequency)


def test_compute_signal_svd_should_raise_error_if_input_signal_is_not_bi_dimensional():
    signals = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    oma_algorithm = OmaAlgorithm(frequency_max=10, frequency_min=5, number_of_fft_points=2**4, num_svd_plots=2)
    with pytest.raises(ModalIdentificationError, match="Input data must be a bi-dimensional non-empty array"):
        oma_algorithm.compute_signal_svd(signal=signals, sampling_frequency=20)


def test_compute_signal_svd_should_raise_error_if_csd_raises_error(
    mocker: MockerFixture,
):
    mock_csd = mocker.patch("dynoma.oma_algorithm.csd")
    mock_csd.side_effect = ValueError

    input_array = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])

    with pytest.raises(ModalIdentificationError, match="Cross spectral decomposition failed: "):
        base_oma_algorithm.compute_signal_svd(signal=input_array, sampling_frequency=10, optimize_csd_computation=False)


def test_compute_signal_svd_should_raise_error_if_fft_helper_raises_error(
    mocker: MockerFixture,
):
    mock_fft_helper = mocker.patch("dynoma.oma_algorithm._fft_helper")
    mock_fft_helper.side_effect = ValueError

    input_array = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]])

    with pytest.raises(ModalIdentificationError, match="Cross spectral decomposition failed: "):
        base_oma_algorithm.compute_signal_svd(signal=input_array, sampling_frequency=10, optimize_csd_computation=True)


def test_compute_signal_svd_should_raise_error_if_svd_raises_error(
    mocker: MockerFixture,
):
    mock_svd = mocker.patch("dynoma.oma_algorithm.torch.linalg.svd")
    mock_svd.side_effect = ValueError

    with pytest.raises(ModalIdentificationError, match="Singular value decomposition failed: "):
        base_oma_algorithm.compute_signal_svd(
            signal=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]), sampling_frequency=10, optimize_csd_computation=True
        )


@pytest.mark.parametrize(
    "optimize_csd_computation",
    [True, False],
)
def test_compute_signal_svd_should_raise_error_if_no_valid_frequencies_are_found(optimize_csd_computation):
    input_array = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    oma_algorithm = OmaAlgorithm(frequency_max=5, frequency_min=4.5, number_of_fft_points=3, num_svd_plots=2)
    with pytest.raises(ModalIdentificationError, match="No frequencies found for the given frequency range"):
        oma_algorithm.compute_signal_svd(
            signal=input_array, sampling_frequency=10, optimize_csd_computation=optimize_csd_computation
        )


@pytest.mark.parametrize(
    "optimize_csd_computation",
    [True, False],
)
def test_compute_signal_svd_should_return_expected_results(optimize_csd_computation):
    input_array = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])

    expected_frequencies = np.array([0, 2.5, 5.0])
    actual_results = base_oma_algorithm.compute_signal_svd(
        signal=input_array, sampling_frequency=10, optimize_csd_computation=optimize_csd_computation
    )
    np.testing.assert_allclose(expected_frequencies, actual_results[0].round(3), atol=TOLERANCE)
