import numpy as np
import pytest
from pytest_mock import MockerFixture

from dynoma.constants import BASE_DTYPE
from dynoma.exceptions import ModalIdentificationError
from dynoma.fdd import FDD

TOLERANCE = 1e-6


@pytest.fixture(scope="module")
def fdd() -> FDD:
    return FDD(
        frequency_max=5,
        frequency_min=0,
        number_of_fft_points=2**4,
        num_svd_plots=2,
    )


def test_fdd_find_peak_over_specific_area_should_raise_error_if_no_frequencies_found(fdd: FDD):
    """Dynoma raises invalid-range when area exceeds frequency grid (Prisma expected no-peaks)."""
    user_selected_areas = [[5, 10]]
    signal = np.array([44.53, 43.99, 45.99, 40.16, 37.36, 34.22, 31.29, 29.69, 32.22, 44.37, 35.22])
    frequencies = np.array([3.80, 3.83, 3.86, 3.89, 3.92, 3.95, 3.98, 4.01, 4.04, 4.07, 4.10])
    with pytest.raises(ModalIdentificationError, match="Selected frequency range in FDD is invalid."):
        fdd._find_peak_over_specific_area(
            user_selected_areas=user_selected_areas, signal_eigenvalues=signal, frequencies=frequencies
        )


def test_fdd_find_peak_over_specific_area_should_raise_error_if_find_peaks_raises_error(
    mocker: MockerFixture, fdd: FDD
):
    mock_peaks = mocker.patch("dynoma.fdd.find_peaks")
    mock_peaks.side_effect = ValueError
    user_selected_areas = [[3.83, 3.98], [3.92, 4.04]]
    signal = np.array([44.53, 43.99, 45.99, 40.16, 37.36, 34.22, 31.29, 29.69, 32.22, 44.37, 35.22])
    frequencies = np.array([3.80, 3.83, 3.86, 3.89, 3.92, 3.95, 3.98, 4.01, 4.04, 4.07, 4.10])
    with pytest.raises(ModalIdentificationError, match="Peak detection in FDD failed."):
        fdd._find_peak_over_specific_area(
            user_selected_areas=user_selected_areas, signal_eigenvalues=signal, frequencies=frequencies
        )


def test_find_peak_over_specific_area_should_raise_error_if_no_peaks_found(fdd: FDD):
    signal = np.array([-4, -4, -4, -4, -4, -4, -4, -4, -4, -4])
    frequencies = np.array([3.80, 3.83, 3.86, 3.89, 3.92, 3.95, 3.98, 4.01, 4.04, 4.07])
    user_selected_areas = [[3.83, 3.98], [3.92, 4.04]]
    with pytest.raises(ModalIdentificationError, match="No peaks found in the selected area for FDD algorithm."):
        fdd._find_peak_over_specific_area(
            user_selected_areas=user_selected_areas, signal_eigenvalues=signal, frequencies=frequencies
        )


def test_find_peak_over_specific_area_should_return_expected_array_of_int(fdd: FDD):
    user_selected_areas = [[3.80, 3.95], [4.07, 4.21]]
    signal_subset = np.array(
        [44.53, 43.99, 41.99, 44.16, 37.36, 34.22, 31.29, 29.69, 35.22, 34.35, 36.15, 42.95, 40.41, 42.54, 44.37]
    )
    frequency_subset = np.array(
        [3.80, 3.83, 3.86, 3.89, 3.92, 3.95, 3.98, 4.01, 4.04, 4.07, 4.10, 4.13, 4.16, 4.18, 4.21]
    )
    actual_frequency_position = fdd._find_peak_over_specific_area(
        user_selected_areas=user_selected_areas, signal_eigenvalues=signal_subset, frequencies=frequency_subset
    )
    expected_freq_position = np.array([3, 11])
    np.testing.assert_allclose(expected_freq_position, actual_frequency_position)


def test_fdd_apply_should_raise_error_if_user_selected_peaks_not_available(fdd: FDD):
    input_array = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]])
    with pytest.raises(ModalIdentificationError, match="User's selected peaks not available"):
        fdd.apply(signal=input_array, sampling_frequency=10, peaks_range=[])


def test_fdd_apply_should_raise_error_if_peaks_range_is_invalid(fdd: FDD):
    signal = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=float)
    with pytest.raises(
        ModalIdentificationError, match="Selected frequency range must be a list of two elements: min and max x values."
    ):
        fdd.apply(signal=signal, sampling_frequency=10, peaks_range=[[0.06], []])


def test_fdd_apply_should_raise_error_if_peak_is_not_valid_having_first_entry_greater_than_second_entry(
    fdd: FDD,
):
    signal = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=float)
    with pytest.raises(
        ModalIdentificationError,
        match="Selected frequency range in FDD is invalid: first entry must be less than second entry.",
    ):
        fdd.apply(signal=signal, sampling_frequency=10, peaks_range=[[0.06, 0.05]])


def test_fdd_apply_should_return_expected_results(fdd: FDD, mocker: MockerFixture):
    frequencies = np.array([0.0, 0.625, 1.25, 1.875, 2.5], dtype=BASE_DTYPE)
    eigenvalues = np.ones((2, 5), dtype=BASE_DTYPE)
    eigenvectors = np.zeros((2, 2, 5), dtype=np.complex64)
    eigenvectors[:, 0, :] = np.array([[1.0 + 0.5j], [2.0 + 1.0j]], dtype=np.complex64)
    mocker.patch(
        "dynoma.fdd.FDD.compute_signal_svd",
        return_value=(frequencies, eigenvalues, eigenvectors),
    )
    mocker.patch(
        "dynoma.fdd.FDD._find_peak_over_specific_area",
        return_value=np.array([1, 3]),
    )
    signal = np.zeros((16, 2), dtype=float)
    results = fdd.apply(
        signal=signal,
        sampling_frequency=10.0,
        peaks_range=[[0.5, 0.7], [1.7, 2.0]],
    )
    np.testing.assert_allclose(results["frequencies"], np.array([0.625, 1.875]), atol=TOLERANCE)
    assert results["complex_mode_shapes"].shape[1] == 2
    assert results["real_mode_shapes"].shape[1] == 2
