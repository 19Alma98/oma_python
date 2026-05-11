from typing import TypedDict

import numpy as np
import numpy.typing as npt

from dynoma.constants import BASE_DTYPE, OMA_COMPLEX_DTYPE


class FDDResults(TypedDict):
    """FDD algorithm results dictionary."""

    frequencies: npt.NDArray[BASE_DTYPE]
    complex_mode_shapes: npt.NDArray[OMA_COMPLEX_DTYPE]
    real_mode_shapes: npt.NDArray[BASE_DTYPE]


class ModalIdentification(TypedDict):
    """Modal identification results dictionary."""

    frequencies: npt.NDArray[BASE_DTYPE]
    damping_ratios: npt.NDArray[BASE_DTYPE]
    mode_shapes: npt.NDArray[OMA_COMPLEX_DTYPE]


class ModalDashboardData(ModalIdentification):
    """Modal identification dashboard data dictionary."""

    model_orders: npt.NDArray[np.int32]


class StabilityModalIdentification(TypedDict):
    """Stability modal identification results dictionary."""

    frequencies: list[npt.NDArray[BASE_DTYPE]]
    damping_ratios: list[npt.NDArray[BASE_DTYPE]]
    mode_shapes: list[npt.NDArray[OMA_COMPLEX_DTYPE]]
    stability_status: list[npt.NDArray[np.int32]]


class StabilityCheck(ModalIdentification):
    """Stability check results dictionary."""

    stability_status: npt.NDArray[np.int32]
    mac_values: npt.NDArray[BASE_DTYPE]


class CovSSIResults(FDDResults):
    """Covariance SSI algorithm results dictionary."""

    damping_ratios: npt.NDArray[BASE_DTYPE]
    cluster_dimensions: npt.NDArray[np.int32]
    frequency_bounds: npt.NDArray[BASE_DTYPE]
    damping_bounds: npt.NDArray[BASE_DTYPE]


class CovSSIDashboardData(TypedDict):
    """Covariance SSI dashboard data dictionary."""

    y_axis_signal: npt.NDArray[BASE_DTYPE]
    x_axis_frequencies: npt.NDArray[BASE_DTYPE]
    frequencies: dict[str, list[float]]
    model_orders: dict[str, list[int]]
