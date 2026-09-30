from importlib.metadata import PackageNotFoundError, version

from dynoma.cov_ssi import CovSSI
from dynoma.exceptions import ModalIdentificationError
from dynoma.fdd import FDD
from dynoma.oma_algorithm import OmaAlgorithm

try:
    __version__ = version("oma-python")
except PackageNotFoundError:  # source tree without install
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "FDD",
    "CovSSI",
    "ModalIdentificationError",
    "OmaAlgorithm",
]
