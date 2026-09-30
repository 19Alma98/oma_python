# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- CovSSI cluster filtering now keeps clusters with size **greater than or equal to**
  `minimum_cluster_dimension` (inclusive minimum). Callers who previously relied on
  `minimum_cluster_dimension=1` to require at least two modes now get size-1 clusters as well.
- Documented that exact zero singular values become **0 dB** in `compute_signal_svd`
  (replaced with `1` before `10*log10`).

### Fixed

- `get_svd_plot_data` now raises `ModalIdentificationError` when `dashboard_data` is
  `None` (typical when `continuous_mode=True`), instead of failing with a TypeError.

## [0.1.0] - 2026-09-30

### Added

- Initial public release of `oma-python` (`dynoma`): FDD and Covariance SSI algorithms.
- CI on Python 3.10–3.13 and PyPI publish via Trusted Publishing on `v*` tags.

### Release process

1. Bump `version` in `pyproject.toml`.
2. Add a new section under this changelog.
3. Push a matching git tag `vX.Y.Z` to trigger the publish workflow.

[Unreleased]: https://github.com/19Alma98/oma_python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/19Alma98/oma_python/releases/tag/v0.1.0
