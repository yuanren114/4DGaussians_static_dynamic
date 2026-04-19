# Changelog

## 2026-04-19

- Created sibling experimental copy `4DGaussians_static_dynamic` from `4DGaussians`.
- Added dashed CLI aliases for existing underscore options in `arguments/__init__.py`.
- Added motion-separation flags:
  - `--motion-separation`
  - `--motion-mask-lambda`
  - `--motion-gate-rot-scale`
- Added a lightweight motion mask head in `scene/deformation.py`.
- Gated position deformation as `x' = x + m * dx` when motion separation is enabled.
- Kept baseline deformation unchanged when motion separation is disabled.
- Added optional scale/rotation gating behind `--motion-gate-rot-scale`.
- Added motion-mask loss, stats, TensorBoard scalars/histograms, JSONL logging, and red/blue PLY diagnostics.
- Added `scripts/compare_motion_separation.py` for baseline vs motion-separation runs.
- Added setup/build/architecture/report documentation.
