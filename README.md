# 4DGaussians Static-Dynamic Motion Separation Fork

This repository is based on the original **4D Gaussian Splatting** codebase:

- upstream project page: <https://guanjunwu.github.io/4dgs/index.html>
- upstream paper: <https://arxiv.org/abs/2310.08528>

This fork keeps the original 4DGS training and rendering pipeline, and adds a lightweight **motion-separation** mechanism for static-dynamic analysis.

## What This Fork Adds

The main modification is a **soft motion mask** predicted inside the deformation network.

For each Gaussian at time \(t\), the code predicts a scalar motion coefficient \(m_i(t)\in[0,1]\) and uses it to gate deformation:

\[
\mu_i(t) = \mu_i^0 + m_i(t)\Delta \mu_i(t)
\]

Optionally, the same gate can also be applied to scale and rotation updates.

The current motion-separation implementation supports:

- `--motion-separation`
- `--motion-gate-rot-scale`
- `--static-deform-lambda`
- `--motion-bin-lambda`

The current cleaned codebase **does not** keep the earlier sparsity-only prototype that used `--motion-mask-lambda`.

## Current Repository Scope

This repository is currently organized around three workflows:

1. **Baseline 4DGS training and rendering**
2. **Motion-separation training and rendering**
3. **Project analysis / reporting for the modified method**

The most relevant top-level files are:

- [train.py](./train.py): training entry point
- [render.py](./render.py): rendering entry point
- [metrics.py](./metrics.py): quantitative evaluation
- [MOTION_SEPARATION_COMMANDS.md](./MOTION_SEPARATION_COMMANDS.md): current one-line training/render/eval commands
- [FINAL_PROJECT_REPORT.md](./FINAL_PROJECT_REPORT.md): project report
- [ARCHITECTURE_FLOW_INPUT_TO_OUTPUT.md](./ARCHITECTURE_FLOW_INPUT_TO_OUTPUT.md): architecture explanation

## Environment Setup

The environment assumptions are still close to upstream 4DGS.

Typical setup:

```bash
git clone <this-repo>
cd 4DGaussians_static_dynamic
git submodule update --init --recursive

conda create -n Gaussians4D python=3.7
conda activate Gaussians4D

pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

The experiments in this repo were run with PyTorch `1.13.1+cu116`.

## Supported Data Layouts

### 1. D-NeRF-style synthetic scenes

Examples:

- `data/dnerf/bouncingballs`
- `data/dnerf/jumpingjacks`
- `data/dnerf/lego`

These scenes use `transforms_train.json` / `transforms_test.json` style camera metadata.

### 2. HyperNeRF scenes

Examples:

- `data/hypernerf/interp/chickchicken`
- `data/hypernerf/virg/...`

These require the scene data plus a COLMAP-generated or pregenerated point cloud such as `points3D_downsample2.ply`.

### 3. Custom D-NeRF-style scenes

The repo includes [scripts/make_dnerf_transforms.py](./scripts/make_dnerf_transforms.py), which can convert a `transforms.json` file into `transforms_train.json` and `transforms_test.json` while adding per-frame `time` values.

Example:

```bash
python scripts/make_dnerf_transforms.py data/custom/M200 --time-step 0.5 --zero-base --hold 8
```

This is useful when you have a single moving-camera video or a manually processed custom sequence and want to turn it into the Blender/D-NeRF-style format expected by this codebase.

## Training

### Baseline 4DGS

Example:

```bash
python train.py -s data/dnerf/bouncingballs --model_path output/dnerf/bouncingballs_baseline --port 6017 --expname "bouncingballs_baseline" --configs arguments/dnerf/bouncingballs.py
```

### Motion Separation

Example:

```bash
python train.py -s data/dnerf/bouncingballs --model_path output/dnerf/bouncingballs_motion_fixed_static1e-3_bin1e-3 --port 6021 --expname "bouncingballs_motion_fixed_static1e-3_bin1e-3" --configs arguments/dnerf/bouncingballs.py --motion-separation --motion-gate-rot-scale --static-deform-lambda 0.001 --motion-bin-lambda 0.001
```

Meaning of the motion-separation flags:

- `--motion-separation`  
  Enable the motion-mask branch.

- `--motion-gate-rot-scale`  
  Also gate scale and rotation deformation with the same motion mask.

- `--static-deform-lambda`  
  Penalize deformation in low-mask regions:
  \[
  \mathcal{L}_{\text{static}} = \frac{1}{N}\sum_i (1-m_i)\|\Delta \mu_i\|_2
  \]

- `--motion-bin-lambda`  
  Encourage masks toward \(0\) or \(1\):
  \[
  \mathcal{L}_{\text{bin}} = \frac{1}{N}\sum_i m_i(1-m_i)
  \]

For more tested commands, see [MOTION_SEPARATION_COMMANDS.md](./MOTION_SEPARATION_COMMANDS.md).

## Rendering

Example:

```bash
python render.py --model_path output/dnerf/bouncingballs_baseline --skip_train --configs arguments/dnerf/bouncingballs.py
```

For a motion-separation run:

```bash
python render.py --model_path output/dnerf/bouncingballs_motion_fixed_static1e-3_bin1e-3 --skip_train --configs arguments/dnerf/bouncingballs.py --motion-separation --motion-gate-rot-scale
```

Notes:

- `--skip_train` means "do not render the training split".
- `render.py` can render `train`, `test`, and `video` splits depending on flags.
- rendered outputs are written under the model directory, for example:
  - `output/.../test/ours_20000/renders`
  - `output/.../test/ours_20000/gt`
  - `output/.../video/ours_20000/video_rgb.mp4`

## Evaluation

Example:

```bash
python metrics.py -m output/dnerf/bouncingballs_baseline output/dnerf/bouncingballs_motion_fixed_static1e-3_bin1e-3
```

The code reports:

- PSNR
- SSIM
- LPIPS-VGG
- LPIPS-Alex
- MS-SSIM
- D-SSIM

## Motion-Mask Diagnostics

When motion separation is enabled, training also logs:

- `motion_mask_stats.jsonl`
- `motion_mask_colors.ply`
- `motion_mask_last.pt`

These are useful for inspecting whether the learned motion mask is meaningful.

Common diagnostics include:

- mask mean / standard deviation
- dynamic fraction (`m > 0.5`)
- softer thresholds such as fraction `m > 0.4`
- static-weighted deformation magnitude
- binarization diagnostic

## HyperNeRF Notes

For HyperNeRF scenes in this repo:

1. place the scene under `data/hypernerf/...`
2. ensure a point cloud such as `points3D_downsample2.ply` is available
3. use `arguments/hypernerf/default.py` or a scene-specific config

Example baseline:

```bash
python train.py -s data/hypernerf/interp/chickchicken --model_path output/hypernerf/interp/chickchicken_baseline_bs1 --port 6031 --expname "hypernerf/interp/chickchicken_baseline_bs1" --configs arguments/hypernerf/default.py --batch-size 1 --densify-until-iter 6000
```

Example motion-separation run:

```bash
python train.py -s data/hypernerf/interp/chickchicken --model_path output/hypernerf/interp/chickchicken_motion_fixed_static2e-3_bin1e-3_bs1 --port 6032 --expname "hypernerf/interp/chickchicken_motion_fixed_static2e-3_bin1e-3_bs1" --configs arguments/hypernerf/default.py --motion-separation --motion-gate-rot-scale --static-deform-lambda 0.002 --motion-bin-lambda 0.001 --batch-size 1 --densify-until-iter 6000
```

## Custom Scene Notes

This repo can support custom scenes, but the required preprocessing depends on the format:

- **Blender/D-NeRF style**: use `transforms_train.json` / `transforms_test.json`
- **COLMAP / HyperNeRF-style**: use reconstructed camera poses and point clouds

For custom scenes with a moving camera and frame timestamps, the easiest path in this fork is usually:

1. prepare a `transforms.json`
2. generate `transforms_train.json` / `transforms_test.json` with `scripts/make_dnerf_transforms.py`
3. train with a D-NeRF-style config

## Viewer and Auxiliary Tools

- Viewer usage: [docs/viewer_usage.md](./docs/viewer_usage.md)
- `export_perframe_3DGS.py`: export Gaussian point clouds at each timestamp
- `merge_many_4dgs.py`: merge trained 4DGS outputs
- `colmap.sh`: generate point clouds from input data

## Current Project Documentation

If you are trying to understand the modified code rather than just run it, start with:

1. [FINAL_PROJECT_REPORT.md](./FINAL_PROJECT_REPORT.md)
2. [ARCHITECTURE_FLOW_INPUT_TO_OUTPUT.md](./ARCHITECTURE_FLOW_INPUT_TO_OUTPUT.md)
3. [MOTION_SEPARATION_COMMANDS.md](./MOTION_SEPARATION_COMMANDS.md)

## Acknowledgment

This repository is built on top of the original 4D Gaussian Splatting project and its dependencies. The upstream authors deserve full credit for the baseline 4DGS architecture and release. This fork focuses on a motion-separation extension and related project documentation.
