# Final Report

## What Changed

This is a copied experimental variant of vanilla 4DGS with a minimal static/dynamic Gaussian separation mechanism.

The new behavior is disabled by default. Baseline commands run through the original deformation path. Passing `--motion-separation` adds one motion mask head to the deformation MLP and gates position deformation:

```python
m = sigmoid(phi_m(fd))
x_prime = x + m * dx
```

Optional regularization:

```python
loss += motion_mask_lambda * mean(m)
```

## Files Changed

- `arguments/__init__.py`
- `scene/deformation.py`
- `scene/gaussian_model.py`
- `train.py`
- `scripts/compare_motion_separation.py`
- `SETUP_LOG.md`
- `environment_same_as_original.yml`
- `BUILD_NOTES.md`
- `ARCHITECTURE_NOTES.md`
- `CHANGELOG_CODEX.md`
- `COMMANDS_CHEATSHEET.md`
- `EXPERIMENT_LOG.md`

## Implementation Details

- The motion mask head is created only when `motion_separation=True`.
- Position deformation is gated first, as requested.
- Scale and rotation gating are off by default and enabled only by `--motion-gate-rot-scale`.
- Opacity and SH deformation behavior is unchanged.
- Dataset loading, renderer API, camera formats, and output directory conventions remain unchanged.

## Original Dependency Environment

The variant keeps the original stack:

- Python 3.7
- PyTorch 1.13.1, preferably CUDA 11.6 wheels
- `mmcv==1.6.0`
- original editable rasterizer and simple-knn submodules

See `BUILD_NOTES.md` and `environment_same_as_original.yml`.

## How To Run

Baseline:

```bash
python train.py -s <dataset_path> --port 6017 --expname "<name>_baseline" --configs <config.py>
```

Motion separation:

```bash
python train.py -s <dataset_path> --port 6017 --expname "<name>_motion" --configs <config.py> --motion-separation --motion-mask-lambda 0.001
```

Comparison:

```bash
python scripts/compare_motion_separation.py -s <dataset_path> --configs <config.py> --expname <dataset_name> --motion-mask-lambda 0.001
```

## Metrics

Real reconstruction metrics from the original evaluation path:

- PSNR
- SSIM
- LPIPS-vgg
- LPIPS-alex

Proxy diagnostics added by the comparison script:

- background stability proxy: temporal variance in border pixels
- dynamic sharpness proxy: center-region Laplacian variance
- motion-mask summary: mean, standard deviation, static fraction, dynamic fraction

The proxy diagnostics are not formal benchmarks and should be reported as practical diagnostics only.

## Current Results and Testing

No full dataset comparison was completed during implementation. The pipeline writes results to:

```text
outputs/comparison/<dataset_name>/
```

after it is run in a built 4DGS environment with a prepared dataset.

Smoke testing completed:

- Python syntax compilation passed for modified scripts.
- New motion-separation CLI flags parsed successfully.
- Comparison script help parsed successfully in the available `sac_clean` environment.

Blocked tests:

- Training startup could not be tested in the active Python environment: `ModuleNotFoundError: No module named 'torch'`.
- The existing `4dgs_bw` conda environment also lacks Torch: `ModuleNotFoundError: No module named 'torch'`.
- The available `sac_clean` conda environment has modern Torch 2.12/cu128, not the target 4DGS stack, and lacks `lpips`: `ModuleNotFoundError: No module named 'lpips'`.

Because of those environment gaps, baseline/motion training startup, finite loss checks, checkpoint creation, and a real comparison run still need to be executed after creating the original-compatible environment from `BUILD_NOTES.md`.

## Limitations

- The motion mask is learned without external supervision, segmentation, or optical flow.
- The sparsity term can collapse motion if `--motion-mask-lambda` is too large.
- The default mode gates only positions; scale/rotation gating is intentionally optional.
- Proxy metrics are coarse diagnostics and should not be treated as formal motion-disentanglement benchmarks.

## Next Steps

- Run the comparison script on one small D-NeRF scene.
- Sweep `--motion-mask-lambda` over small values such as `0`, `1e-4`, and `1e-3`.
- Inspect `motion_mask_colors.ply` and `motion_mask_stats.jsonl` for collapse or all-dynamic behavior.
