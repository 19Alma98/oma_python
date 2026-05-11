from typing import Any

import numpy as np
import numpy.typing as npt

from dynoma.constants import BASE_DTYPE, COMPLEX_DTYPE, OMA_COMPLEX_DTYPE
from dynoma.exceptions import ModalIdentificationError


def maximum_correlation_rotation(input_vector: npt.NDArray[OMA_COMPLEX_DTYPE]) -> npt.NDArray[OMA_COMPLEX_DTYPE]:
    """Return input data rotated with respect to the maximum correlation line.

    Args:
        input_vector (npt.NDArray[OMA_COMPLEX_DTYPE]): input array to be rotated.

    Returns:
        max_corr_rotation (npt.NDArray[OMA_COMPLEX_DTYPE]): rotated array.
    """
    if input_vector.ndim != 1 or input_vector.size == 0:
        raise ModalIdentificationError("Input must be a non-empty 1D array")

    try:
        intercept = np.polyfit(input_vector.real, input_vector.imag, deg=1)[0]
    except np.linalg.LinAlgError as error:
        raise ModalIdentificationError("Least squares polynomial fit failed") from error

    theta_angle = -np.arctan(intercept)
    rotation_matrix = np.array(
        [[np.cos(theta_angle), -np.sin(theta_angle)], [np.sin(theta_angle), np.cos(theta_angle)]]
    )

    rotated = np.matmul(rotation_matrix, np.vstack((input_vector.real, input_vector.imag)))
    return np.array(rotated[0] + 1j * rotated[1], dtype=COMPLEX_DTYPE)


def find_angle_sign(angle: npt.NDArray[np.floating[Any]]) -> npt.NDArray[np.int32]:
    """Return the sign of an angle based on radiants.

    Args:
        angle (npt.NDArray[np.floating[Any]]): the input angle on the complex plane, in the range (-π, π].

    Returns:
        sign (npt.NDArray[np.int32]): -1 if the angle is in II or III quarter, +1 if the angle in in I or IV quarter.
    """
    if np.any((angle <= -np.pi) | (angle > np.pi)):
        raise ModalIdentificationError("Complex angles must be in the range (-π, π]")

    return np.where(
        ((angle >= 0) & (angle < np.pi / 2)) | ((angle >= -np.pi / 2) & (angle < 0)),
        +1,
        -1,
    )


def complex_mode_to_real_mode(complex_mode: npt.NDArray[OMA_COMPLEX_DTYPE]) -> npt.NDArray[BASE_DTYPE]:
    """Return the real valued mode shape starting from the complex one.

    Args:
        complex_mode (npt.NDArray[OMA_COMPLEX_DTYPE]): complex mode shape.

    Returns:
        real_mode (npt.NDArray[BASE_DTYPE]): real mode shape.
    """
    if complex_mode.ndim != 1 or complex_mode.size == 0:
        raise ModalIdentificationError("Input must be a non-empty 1D array")

    rotated = maximum_correlation_rotation(complex_mode)
    magnitudes = np.abs(rotated)
    magnitudes /= np.max(magnitudes)

    signs = find_angle_sign(np.angle(rotated))
    signs[magnitudes == 0] = 0

    return np.real(magnitudes * signs)


def modal_phase_collinearity(mode_shapes: npt.NDArray[OMA_COMPLEX_DTYPE]) -> npt.NDArray[BASE_DTYPE]:
    """Return the modal phase collinearity of a mode shape.

    Args:
        mode_shapes (npt.NDArray[OMA_COMPLEX_DTYPE]): mode shape.

    Returns:
        mpc (npt.NDArray[BASE_DTYPE]): modal phase collinearity.
    """
    if mode_shapes.ndim == 1:
        mode_shapes = mode_shapes.reshape(1, -1)
    centered = (mode_shapes.T - np.mean(mode_shapes, axis=1)).T
    real_part = np.real(centered)
    imag_part = np.imag(centered)

    real_component = np.sum(real_part**2, axis=1)
    imaginary_component = np.sum(imag_part**2, axis=1)
    real_imaginary_component = np.sum(real_part * imag_part, axis=1)

    coef = np.divide(
        imaginary_component - real_component,
        2 * real_imaginary_component,
        out=np.zeros_like(real_imaginary_component),
        where=real_imaginary_component != 0,
    )
    beta = coef + np.sign(real_imaginary_component) * np.sqrt(coef**2 + 1)
    angle = np.arctan(beta)

    mpc_first = real_component + np.divide(
        real_imaginary_component * (2 * (coef**2 + 1) * (np.sin(angle) ** 2) - 1),
        coef,
        out=np.zeros_like(coef),
        where=coef != 0,
    )
    mpc_second = imaginary_component - np.divide(
        real_imaginary_component * (2 * (coef**2 + 1) * (np.sin(angle) ** 2) - 1),
        coef,
        out=np.zeros_like(coef),
        where=coef != 0,
    )
    return (
        np.array(
            2
            * (
                np.divide(
                    mpc_first,
                    (mpc_first + mpc_second),
                    out=np.zeros_like(mpc_first),
                    where=(mpc_first + mpc_second) != 0,
                )
                - 0.5
            )
        )
        ** 2
    )


def compute_mac_value(
    mode_shape_1: npt.NDArray[OMA_COMPLEX_DTYPE], mode_shape_2: npt.NDArray[OMA_COMPLEX_DTYPE]
) -> float:
    """Return the modal assurance criterion value between two mode shapes.

    Args:
        mode_shape_1 (npt.NDArray[OMA_COMPLEX_DTYPE]): first mode shape.
        mode_shape_2 (npt.NDArray[OMA_COMPLEX_DTYPE]): second mode shape.

    Returns:
        mac_value (float): modal assurance criterion value.
    """
    numerator = np.abs(np.matmul(np.conjugate(mode_shape_1), mode_shape_2)) ** 2
    denominator_first_factor = np.matmul(np.conjugate(mode_shape_1), mode_shape_1)
    denominator_second_factor = np.matmul(np.conjugate(mode_shape_2), mode_shape_2)
    if denominator_first_factor * denominator_second_factor == 0:
        mac_value = 0.0
    else:
        mac_value = numerator / np.real(denominator_first_factor * denominator_second_factor)
    return float(mac_value)
