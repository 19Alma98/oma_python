# oma-python

Open-source **Operational Modal Analysis (OMA)** algorithms for Python.
Estimate natural frequencies, damping ratios, and mode shapes from ambient vibration measurements.

Install the distribution as `oma-python`; import the library as `dynoma`.

## Install

```bash
pip install oma-python
```

For local development with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
```

## Quick start (FDD)

Frequency Domain Decomposition on a synthetic multi-channel signal:

```python
import numpy as np
from dynoma import FDD

rng = np.random.default_rng(0)
fs = 100.0  # Hz
t = np.arange(0, 40, 1 / fs)

# Ambient-like response on 3 channels (modes near 2.5 Hz and 6.0 Hz)
signal = np.column_stack(
    [
        np.sin(2 * np.pi * 2.5 * t) + 0.4 * np.sin(2 * np.pi * 6.0 * t),
        0.8 * np.sin(2 * np.pi * 2.5 * t + 0.2) + np.sin(2 * np.pi * 6.0 * t),
        0.5 * np.sin(2 * np.pi * 2.5 * t) + 0.7 * np.sin(2 * np.pi * 6.0 * t + 0.1),
    ]
)
signal += 0.05 * rng.standard_normal(signal.shape)

fdd = FDD(
    frequency_min=0.0,
    frequency_max=20.0,
    number_of_fft_points=512,
    num_svd_plots=3,
)

# Optional: inspect SVD lines before picking peaks
frequencies, eigenvalues, eigenvectors = fdd.compute_signal_svd(
    signal=signal,
    sampling_frequency=fs,
)

# Results are TypedDict mappings (use bracket access)
results = fdd.apply(
    signal=signal,
    sampling_frequency=fs,
    peaks_range=[[2.0, 3.0], [5.5, 6.5]],
)

print(results["frequencies"])       # ~2.54 Hz, ~6.05 Hz
print(results["complex_mode_shapes"].shape)  # (n_channels, n_modes)
print(results["real_mode_shapes"].shape)
```

## Covariance SSI

```python
from dynoma import CovSSI

cov_ssi = CovSSI(
    frequency_min=0.0,
    frequency_max=20.0,
    order_min=20,
    order_max=40,
    order_steps=2,
    time_lag=1.2,
    number_of_fft_points=256,
    num_svd_plots=3,
    continuous_mode=True,
)

results, dashboard = cov_ssi.apply(
    signal=signal,
    sampling_frequency=fs,
    optimized=True,
    shuffle=False,
)

print(results["frequencies"])
print(results["damping_ratios"])  # percent scale
# dashboard is None when continuous_mode=True (default, headless).
# For SVD / stability dashboard plots, set continuous_mode=False on CovSSI,
# then: svd_plot_data = cov_ssi.get_svd_plot_data(
#     signal=signal, sampling_frequency=fs, dashboard_data=dashboard
# )
```

## Public API

```python
from dynoma import FDD, CovSSI, OmaAlgorithm, ModalIdentificationError, __version__
```

| Symbol | Role |
|--------|------|
| `FDD` | Frequency Domain Decomposition |
| `CovSSI` | Covariance-driven Stochastic Subspace Identification |
| `OmaAlgorithm` | Shared CSD / SVD base (usually subclassed) |
| `ModalIdentificationError` | Domain errors from validation / identification |

Signal layout is `(n_samples, n_channels)`.

## Requirements

- Python ≥ 3.10
- NumPy, SciPy, Pydantic, PyTorch

CPU-only PyTorch wheels are preferred for development (`uv` is configured for the official CPU index).

## Changelog

See [CHANGELOG.md](https://github.com/19Alma98/oma_python/blob/main/CHANGELOG.md).

## License

MIT — see [LICENSE](https://github.com/19Alma98/oma_python/blob/main/LICENSE).
