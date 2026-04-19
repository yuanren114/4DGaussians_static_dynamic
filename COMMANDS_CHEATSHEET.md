# Commands Cheatsheet

Run commands from this folder:

```bash
cd E:/Study/ROB_430/Final_Project/code/4DGS_motion_separation/4DGaussians_static_dynamic
```

## Environment Creation

```bash
conda create -n Gaussians4D python=3.7
conda activate Gaussians4D
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install mmcv==1.6.0 matplotlib argparse lpips plyfile pytorch_msssim open3d imageio[ffmpeg]
```

Alternative:

```bash
conda env create -f environment_same_as_original.yml
conda activate Gaussians4D
```

## Extension Build

```bash
git submodule update --init --recursive
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

## Baseline Training

```bash
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs_baseline" --configs arguments/dnerf/bouncingballs.py
```

## Motion-Separation Training

```bash
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs_motion" --configs arguments/dnerf/bouncingballs.py --motion-separation --motion-mask-lambda 0.001
```

Optional scale/rotation gating:

```bash
python train.py -s data/dnerf/bouncingballs --port 6017 --expname "dnerf/bouncingballs_motion_rot_scale" --configs arguments/dnerf/bouncingballs.py --motion-separation --motion-gate-rot-scale --motion-mask-lambda 0.001
```

## Rendering

```bash
python render.py --model_path output/dnerf/bouncingballs_motion --skip_train --configs arguments/dnerf/bouncingballs.py
```

## Evaluation

`metrics.py` accepts `--model_paths`:

```bash
python metrics.py --model_paths output/dnerf/bouncingballs_motion
```

## Comparison Pipeline

```bash
python scripts/compare_motion_separation.py -s data/dnerf/bouncingballs --configs arguments/dnerf/bouncingballs.py --expname dnerf_bouncingballs --iterations 30000 --coarse-iterations 3000 --motion-mask-lambda 0.001
```

Short smoke-test version:

```bash
python scripts/compare_motion_separation.py -s data/dnerf/bouncingballs --configs arguments/dnerf/bouncingballs.py --expname smoke_bouncingballs --iterations 10 --coarse-iterations 5 --motion-mask-lambda 0.001
```

Outputs:

```text
outputs/comparison/<dataset_name>/metrics.csv
outputs/comparison/<dataset_name>/metrics.json
outputs/comparison/<dataset_name>/plots/*.png
outputs/comparison/<dataset_name>/summary.md
outputs/comparison/<dataset_name>/logs/*.log
```

## Troubleshooting

Check imports:

```bash
python -c "import torch; import diff_gaussian_rasterization; import simple_knn; import mmcv; print(torch.__version__)"
```

Check CLI parsing:

```bash
python train.py --help
```

Check CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Inspect motion-mask logs:

```bash
type output\dnerf\bouncingballs_motion\motion_mask_stats.jsonl
```
