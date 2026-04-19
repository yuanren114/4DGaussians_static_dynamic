# Build Notes

This variant intentionally reuses the original 4DGS dependency stack and build flow.

## Environment

The original README specifies:

- Python 3.7
- PyTorch 1.13.1, with the authors using `pytorch=1.13.1+cu116`
- `requirements.txt`
- editable extension installs for:
  - `submodules/depth-diff-gaussian-rasterization`
  - `submodules/simple-knn`

Recommended setup:

```bash
conda create -n Gaussians4D python=3.7
conda activate Gaussians4D
pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

For CUDA 11.6 wheels, use the same PyTorch family as the original environment:

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install mmcv==1.6.0 matplotlib argparse lpips plyfile pytorch_msssim open3d imageio[ffmpeg]
```

Then build the original extensions:

```bash
git submodule update --init --recursive
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

## Notes

- No CUDA, PyTorch, renderer, or dataset-format modernization was attempted.
- The new motion-separation code adds one MLP head and optional loss/logging only.
- If extension build fails, keep the first fix focused on the exact compiler/CUDA error rather than upgrading the whole project.
