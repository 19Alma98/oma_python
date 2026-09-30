import re

import numpy as np
import pytest
from pytest_mock import MockerFixture

from dynoma.exceptions import ModalIdentificationError
from dynoma.utils import (
    complex_mode_to_real_mode,
    compute_mac_value,
    find_angle_sign,
    maximum_correlation_rotation,
    modal_phase_collinearity,
)

TOLERANCE = 1e-6


@pytest.mark.parametrize(
    "input_vector",
    [np.array([]), np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])],
)
def test_maximum_correlation_rotation_should_raise_error_if_input_not_mono_dimensional(input_vector):
    with pytest.raises(ModalIdentificationError, match="Input must be a non-empty 1D array"):
        maximum_correlation_rotation(input_vector)


def test_maximum_correlation_rotation_should_raise_error_if_polynomial_fit_fails(mocker: MockerFixture):
    mocker.patch("numpy.polyfit", side_effect=np.linalg.LinAlgError)
    with pytest.raises(ModalIdentificationError, match="Least squares polynomial fit failed"):
        maximum_correlation_rotation(np.array([1.0, 2.0, 3.0, 4.0]))


def test_maximum_correlation_rotation_should_return_expected_array():
    input_array = np.array([0 + 1j, 0 + 0j, 0.1 + 1j, 1 + 0j, 1 + 0j, 2 + 0.3j, 2 + 1.5j])
    actual_output = maximum_correlation_rotation(input_array)
    expected_output = np.array(
        [
            0.082493280843906 + 0.996591620783362j,
            0.000000000000000 + 0.000000000000000j,
            0.182152442922242 + 0.988342292698971j,
            0.996591620783362 - 0.082493280843906j,
            0.996591620783362 - 0.082493280843906j,
            2.017931225819896 + 0.133990924547197j,
            2.116923162832583 + 1.329900869487231j,
        ]
    )

    np.testing.assert_allclose(actual_output, expected_output, atol=TOLERANCE)


def test_sign_finder_should_raise_error_if_input_value_not_in_correct_range_of_values():
    input_angle = 10
    with pytest.raises(ModalIdentificationError, match=re.escape("Complex angles must be in the range (-π, π]")):
        find_angle_sign(input_angle)


@pytest.mark.parametrize(
    "input_angle,expected_value",
    [(0.5, +1), (2.5, -1), (-2.5, -1), (-0.5, +1)],
)
def test_sign_finder_should_return_expected_value(input_angle, expected_value):
    np.testing.assert_array_equal(
        find_angle_sign(np.asarray(input_angle)),
        np.asarray(expected_value),
    )


@pytest.mark.parametrize(
    "input_vector",
    [np.array([]), np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])],
)
def test_complex_mode_to_real_mode_should_raise_error_if_input_not_mono_dimensional(input_vector):
    with pytest.raises(ModalIdentificationError, match="Input must be a non-empty 1D array"):
        complex_mode_to_real_mode(input_vector)


@pytest.mark.parametrize(
    "input_value,expected_value",
    [
        (
            np.array([0 + 1j, 0 + 0j, 0.1 + 1j, 1 + 0j, 2 + 0j]),
            np.array([-0.500000000000000, 0, -0.502493781056045, 0.500000000000000, 1]),
        ),
        (
            np.array([1 + 1j, 0 + 0j, 0 + 1j, 0 + 0j, 1 + 0j]),
            np.array([1, 0, 0.707106781186548, 0, 0.707106781186548]),
        ),
    ],
)
def test_complex_mode_to_real_mode_should_return_expected_array(input_value, expected_value):
    actual_value = complex_mode_to_real_mode(input_value)
    np.testing.assert_allclose(actual_value, expected_value, atol=TOLERANCE)


@pytest.mark.parametrize(
    "input_array, expected_coefficient",
    [
        (
            np.array(
                [
                    -2.50155790272220e-06 - 3.25601943797008e-06j,
                    -0.000515547396818131 - 0.000516218487420379j,
                    -0.000960236589309833 - 0.000460088370616296j,
                    2.81774450565676e-05 - 0.000100310505827090j,
                    -0.00862364101195196 - 0.00420980172323539j,
                    -0.00308157353238295 - 0.00179158707534183j,
                    1.14026829002432e-05 + 8.36838247510715e-05j,
                    4.51689147044627e-05 - 1.73247105973111e-05j,
                ]
            ),
            np.array([0.9948843119087739]),
        ),
        (
            np.array(
                [
                    -1.41453904842451e-05 + 4.83279558031198e-05j,
                    -0.000686410571096166 - 0.000197261175636621j,
                    -0.000376049593808473 - 3.88724256596064e-05j,
                    8.92931978021882e-05 + 0.000231208559510647j,
                    -0.00109819598242087 - 0.000732575133652312j,
                    -0.000398890919296312 - 0.000247518188416777j,
                    0.000194541192368417 - 0.000136353326300543j,
                    2.41452926440341e-05 + 4.01146554287520e-06j,
                ]
            ),
            np.array([0.7540798351197143]),
        ),
        (
            np.array(
                [
                    0 - 3.25601943797008e-06j,
                    0 - 0.000516218487420379j,
                    0 - 0.000460088370616296j,
                    0 - 0.000100310505827090j,
                    0 - 0.00420980172323539j,
                    0 - 0.00179158707534183j,
                    0 + 8.36838247510715e-05j,
                    0 - 1.73247105973111e-05j,
                ]
            ),
            np.array([1]),
        ),
    ],
)
def test_modal_phase_collinearity_should_return_expected_float(input_array, expected_coefficient):
    actual_mpc = modal_phase_collinearity(input_array)
    np.testing.assert_array_equal(actual_mpc, expected_coefficient)


def test_compute_mac_value_should_return_expected_float():
    mode_1 = np.array(
        [
            -2.50155790272220e-06 - 3.25601943797008e-06j,
            -0.000515547396818131 - 0.000516218487420379j,
            -0.000960236589309833 - 0.000460088370616296j,
            2.81774450565676e-05 - 0.000100310505827090j,
            -0.00862364101195196 - 0.00420980172323539j,
            -0.00308157353238295 - 0.00179158707534183j,
            1.14026829002432e-05 + 8.36838247510715e-05j,
            4.51689147044627e-05 - 1.73247105973111e-05j,
        ]
    )
    mode_2 = np.array(
        [
            -1.41453904842451e-05 + 4.83279558031198e-05j,
            -0.000686410571096166 - 0.000197261175636621j,
            -0.000376049593808473 - 3.88724256596064e-05j,
            8.92931978021882e-05 + 0.000231208559510647j,
            -0.00109819598242087 - 0.000732575133652312j,
            -0.000398890919296312 - 0.000247518188416777j,
            0.000194541192368417 - 0.000136353326300543j,
            2.41452926440341e-05 + 4.01146554287520e-06j,
        ]
    )
    actual_mac = compute_mac_value(mode_1, mode_2)
    np.testing.assert_array_equal(actual_mac, 0.7816592173468412)


def test_compute_mac_value_should_return_zero_if_input_is_zero():
    mode_1 = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    mode_2 = np.array(
        [
            -1.41453904842451e-05 + 4.83279558031198e-05j,
            -0.000686410571096166 - 0.000197261175636621j,
            -0.000376049593808473 - 3.88724256596064e-05j,
            8.92931978021882e-05 + 0.000231208559510647j,
            -0.00109819598242087 - 0.000732575133652312j,
            -0.000398890919296312 - 0.000247518188416777j,
            0.000194541192368417 - 0.000136353326300543j,
            2.41452926440341e-05 + 4.01146554287520e-06j,
        ]
    )
    actual_mac = compute_mac_value(mode_1, mode_2)
    np.testing.assert_array_equal(actual_mac, 0.0)
