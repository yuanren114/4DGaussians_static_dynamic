# Motion Separation Training Commands

All commands below are one-line commands for the Linux cluster shell. Use `/` path separators, not Windows `\`.

This file keeps only the currently supported command paths. Early sparsity-only experiments using `--motion-mask-lambda` were removed from the codebase during cleanup and are intentionally not listed here.

## Baseline Experiments

### Lego baseline train

```bash
python train.py -s data/dnerf/lego --model_path output/dnerf/lego_baseline --port 6017 --expname "lego_baseline" --configs arguments/dnerf/lego.py
```

### Lego baseline render

```bash
python render.py --model_path output/dnerf/lego_baseline --skip_train --configs arguments/dnerf/lego.py
```

### Bouncingballs baseline train

```bash
python train.py -s data/dnerf/bouncingballs --model_path output/dnerf/bouncingballs_baseline --port 6017 --expname "bouncingballs_baseline" --configs arguments/dnerf/bouncingballs.py
```

### Bouncingballs baseline render

```bash
python render.py --model_path output/dnerf/bouncingballs_baseline --skip_train --configs arguments/dnerf/bouncingballs.py
```

## Motion Gate Experiments

These commands use the current motion-separation losses:

- `--static-deform-lambda`: penalizes `(1 - mask) * ||dx||`, so low-mask Gaussians are discouraged from moving.
- `--motion-bin-lambda`: penalizes `mask * (1 - mask)`, encouraging masks to move toward 0 or 1.
- `--motion-gate-rot-scale`: also gates scale and rotation deformation, reducing bypass paths.

### Bouncingballs fixed motion gate conservative train

```bash
python train.py -s data/dnerf/bouncingballs --model_path output/dnerf/bouncingballs_motion_fixed_static1e-3_bin1e-3 --port 6021 --expname "bouncingballs_motion_fixed_static1e-3_bin1e-3" --configs arguments/dnerf/bouncingballs.py --motion-separation --motion-gate-rot-scale --static-deform-lambda 0.001 --motion-bin-lambda 0.001
```

### Bouncingballs fixed motion gate conservative render

```bash
python render.py --model_path output/dnerf/bouncingballs_motion_fixed_static1e-3_bin1e-3 --skip_train --configs arguments/dnerf/bouncingballs.py --motion-separation --motion-gate-rot-scale
```

### Bouncingballs fixed motion gate stronger static penalty train

```bash
python train.py -s data/dnerf/bouncingballs --model_path output/dnerf/bouncingballs_motion_fixed_static1e-2_bin1e-3 --port 6022 --expname "bouncingballs_motion_fixed_static1e-2_bin1e-3" --configs arguments/dnerf/bouncingballs.py --motion-separation --motion-gate-rot-scale --static-deform-lambda 0.01 --motion-bin-lambda 0.001
```

### Bouncingballs fixed motion gate stronger static penalty render

```bash
python render.py --model_path output/dnerf/bouncingballs_motion_fixed_static1e-2_bin1e-3 --skip_train --configs arguments/dnerf/bouncingballs.py --motion-separation --motion-gate-rot-scale
```

### Lego fixed motion gate conservative train

```bash
python train.py -s data/dnerf/lego --model_path output/dnerf/lego_motion_fixed_static1e-3_bin1e-3 --port 6023 --expname "lego_motion_fixed_static1e-3_bin1e-3" --configs arguments/dnerf/lego.py --motion-separation --motion-gate-rot-scale --static-deform-lambda 0.001 --motion-bin-lambda 0.001
```

### Lego fixed motion gate conservative render

```bash
python render.py --model_path output/dnerf/lego_motion_fixed_static1e-3_bin1e-3 --skip_train --configs arguments/dnerf/lego.py --motion-separation --motion-gate-rot-scale
```

## Metrics Commands

### Compare Bouncingballs baseline and current motion setting

```bash
python metrics.py -m output/dnerf/bouncingballs_baseline output/dnerf/bouncingballs_motion_fixed_static1e-3_bin1e-3
```

### Compare Bouncingballs fixed ablations

```bash
python metrics.py -m output/dnerf/bouncingballs_baseline output/dnerf/bouncingballs_motion_fixed_static1e-3_bin1e-3 output/dnerf/bouncingballs_motion_fixed_static1e-2_bin1e-3
```

## Mask Diagnostics

### Show final Lego fixed mask stats

```bash
tail -n 10 output/dnerf/lego_motion_fixed_static1e-3_bin1e-3/motion_mask_stats.jsonl
```

### Show final Bouncingballs fixed mask stats

```bash
tail -n 10 output/dnerf/bouncingballs_motion_fixed_static1e-3_bin1e-3/motion_mask_stats.jsonl
```

### Inspect Bouncingballs mask distribution from saved tensor

```bash
python -c "import torch; m=torch.load('output/dnerf/bouncingballs_motion_fixed_static1e-3_bin1e-3/point_cloud/iteration_20000/motion_mask_last.pt'); print('mean',m.mean().item(),'std',m.std().item(),'min',m.min().item(),'max',m.max().item()); print('>0.1',(m>0.1).float().mean().item()); print('>0.2',(m>0.2).float().mean().item()); print('>0.3',(m>0.3).float().mean().item()); print('>0.4',(m>0.4).float().mean().item()); print('>0.5',(m>0.5).float().mean().item())"
```

### Open Bouncingballs fixed motion-mask point cloud with Open3D

```bash
python -c "import open3d as o3d; p=o3d.io.read_point_cloud('output/dnerf/bouncingballs_motion_fixed_static1e-3_bin1e-3/point_cloud/iteration_20000/motion_mask_colors.ply'); o3d.visualization.draw_geometries([p])"
```

## HyperNeRF Chickchicken

### HyperNeRF chickchicken baseline train

```bash
python train.py -s data/hypernerf/interp/chickchicken --model_path output/hypernerf/interp/chickchicken_baseline_bs1 --port 6031 --expname "hypernerf/interp/chickchicken_baseline_bs1" --configs arguments/hypernerf/default.py --batch-size 1 --densify-until-iter 6000
```

### HyperNeRF chickchicken baseline render

```bash
python render.py --model_path output/hypernerf/interp/chickchicken_baseline_bs1 --skip_train --configs arguments/hypernerf/default.py
```

### HyperNeRF chickchicken motion train (`static=2e-3`, `bin=1e-3`)

```bash
python train.py -s data/hypernerf/interp/chickchicken --model_path output/hypernerf/interp/chickchicken_motion_fixed_static2e-3_bin1e-3_bs1 --port 6032 --expname "hypernerf/interp/chickchicken_motion_fixed_static2e-3_bin1e-3_bs1" --configs arguments/hypernerf/default.py --motion-separation --motion-gate-rot-scale --static-deform-lambda 0.002 --motion-bin-lambda 0.001 --batch-size 1 --densify-until-iter 6000
```

### HyperNeRF chickchicken motion render (`static=2e-3`, `bin=1e-3`)

```bash
python render.py --model_path output/hypernerf/interp/chickchicken_motion_fixed_static2e-3_bin1e-3_bs1 --skip_train --configs arguments/hypernerf/default.py --motion-separation --motion-gate-rot-scale
```

### HyperNeRF chickchicken metrics compare

```bash
python metrics.py -m output/hypernerf/interp/chickchicken_baseline_bs1 output/hypernerf/interp/chickchicken_motion_fixed_static2e-3_bin1e-3_bs1
```

### HyperNeRF chickchicken motion mask stats

```bash
tail -n 10 output/hypernerf/interp/chickchicken_motion_fixed_static2e-3_bin1e-3_bs1/motion_mask_stats.jsonl
```

## Custom M200

### Custom M200 generate D-NeRF-style transforms from transforms.json

```bash
python scripts/make_dnerf_transforms.py data/custom/M200 --time-step 0.5 --zero-base --hold 8
```

### Custom M200 baseline train

```bash
python train.py -s data/custom/M200 --model_path output/custom/M200_baseline --port 6017 --expname "custom/M200_baseline" --configs arguments/dnerf/M200.py --extension ""
```

### Custom M200 baseline render

```bash
python render.py --model_path output/custom/M200_baseline --skip_train --configs arguments/dnerf/M200.py --extension ""
```

### Custom M200 baseline render without video

```bash
python render.py --model_path output/custom/M200_baseline --skip_train --skip_video --configs arguments/dnerf/M200.py --extension ""
```

### Custom M200 baseline metrics

```bash
python metrics.py -m output/custom/M200_baseline
```

## Suggested Order

1. Run `bouncingballs_baseline`.
2. Run `bouncingballs_motion_fixed_static1e-3_bin1e-3`.
3. If the mask is still too soft, run `bouncingballs_motion_fixed_static1e-2_bin1e-3`.
4. Use Lego as a secondary discussion scene because its motion is smaller and slower.
