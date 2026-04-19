# Architecture Notes

## Entry Points

- Training: `train.py`
- Rendering: `render.py`
- Evaluation metrics: `metrics.py`
- Automated comparison: `scripts/compare_motion_separation.py`

## Argument Structure

Arguments are declared in `arguments/__init__.py`.

- `ModelParams`: dataset paths, output path, image settings
- `OptimizationParams`: iteration counts, learning rates, densification settings
- `PipelineParams`: renderer-side switches
- `ModelHiddenParams`: deformation-network options and regularization weights

This variant adds to `ModelHiddenParams`:

- `motion_separation`, exposed as `--motion-separation`
- `motion_mask_lambda`, exposed as `--motion-mask-lambda`
- `motion_gate_rot_scale`, exposed as `--motion-gate-rot-scale`

Underscore spellings still work, for example `--motion_separation`.

## Dataset Loading

`scene/__init__.py` detects the dataset type from the unchanged source directory structure:

- COLMAP: `sparse`
- D-NeRF/Blender: `transforms_train.json`
- DyNeRF: `poses_bounds.npy`
- Nerfies/HyperNeRF: `dataset.json`
- PanopticSports: `train_meta.json`
- MultipleView: `points3D_multipleview.ply`

The variant does not change dataset layout or camera loading.

## Gaussian State

Gaussian parameters live in `scene/gaussian_model.py`:

- Positions: `_xyz`
- SH features: `_features_dc`, `_features_rest`
- Scale: `_scaling`
- Rotation: `_rotation`
- Opacity: `_opacity`
- Deformation network: `_deformation`

Checkpoints and point clouds are saved under `output/<expname>/point_cloud/...`.

## Baseline Deformation Path

`gaussian_renderer/__init__.py` calls `pc._deformation(...)` during the fine stage. The deformation network in `scene/deformation.py` uses the hexplane grid feature and `feature_out` MLP, then predicts:

- `dx` through `pos_deform`
- `ds` through `scales_deform`
- `dr` through `rotations_deform`
- optional opacity and SH deltas

Baseline behavior is preserved when `--motion-separation` is not passed.

## Motion Gate Insertion

The new motion gate is inserted in `scene/deformation.py` immediately after the shared deformation feature:

```python
hidden = self.query_time(...)
m = sigmoid(phi_m(hidden))
dx = phi_x(hidden)
x_prime = x + m * dx
```

By default, only position deformation is gated. Scale and rotation can also be gated with `--motion-gate-rot-scale`.

## Losses and Logging

`train.py` adds the optional sparsity term only in motion-separation mode:

```python
loss += motion_mask_lambda * mean(m)
```

Logged diagnostics:

- motion mask mean
- motion mask std
- fraction with `m > 0.5`
- fraction with `m <= 0.5`
- JSONL log at `motion_mask_stats.jsonl`
- TensorBoard histograms when TensorBoard is available
- optional red/blue `motion_mask_colors.ply` diagnostic on save
