# Experiment Log

## 2026-04-19

Created the new sibling copy and implemented the minimal motion-separation variant.

Smoke checks run:

- `python -m py_compile arguments\__init__.py scene\deformation.py scene\gaussian_model.py train.py scripts\compare_motion_separation.py render.py metrics.py`
  - Result: passed.
- Base Python `python train.py --help`
  - Result: failed before parsing because the active Python environment has no Torch.
  - Exact error: `ModuleNotFoundError: No module named 'torch'`
- Base Python `python scripts\compare_motion_separation.py --help`
  - Result: failed because the active Python environment has no Matplotlib.
  - Exact error: `ModuleNotFoundError: No module named 'matplotlib'`
- `conda run -n 4dgs_bw python train.py --help`
  - Result: failed because that environment has no Torch.
  - Exact error: `ModuleNotFoundError: No module named 'torch'`
- `conda run -n sac_clean python train.py --help`
  - Result: failed because that environment is modern Torch 2.12/cu128 and lacks original repo dependencies.
  - Exact error: `ModuleNotFoundError: No module named 'lpips'`
- `conda run -n sac_clean python scripts\compare_motion_separation.py --help`
  - Result: passed.
- Parser-only check for `ModelHiddenParams`:
  - Command parsed `--motion-separation --motion-mask-lambda 0.001 --motion-gate-rot-scale`.
  - Result: passed, values were `True 0.001 True`.

Full baseline-vs-motion training was not run during implementation because it requires a prepared dataset and built CUDA extensions. Use `scripts/compare_motion_separation.py` for the reproducible comparison once the legacy 4DGS environment is active.
