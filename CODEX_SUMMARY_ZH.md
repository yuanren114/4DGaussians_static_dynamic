# Codex 工作总结

## 重要说明

- 我没有修改原始 `4DGaussians` 文件夹。
- 我新建并修改的是它的 sibling copy：

```text
E:\Study\ROB_430\Final_Project\code\4DGS_motion_separation\4DGaussians_static_dynamic
```

- 原始仓库路径是：

```text
E:\Study\ROB_430\Final_Project\code\4DGS_motion_separation\4DGaussians
```

- `LOW_MEMORY_4DGS_REPORT.md` 是复制原仓库时一起带进来的文件。我没有参考它的内容，也没有基于它做实现设计。

## 我完成的事情

### 1. 创建安全副本

我把原始 `4DGaussians` 文件夹完整复制成了：

```text
4DGaussians_static_dynamic
```

所有代码改动都发生在这个新文件夹里，原始仓库保持不动。

同时我写了：

```text
SETUP_LOG.md
```

里面记录了：

- 原始仓库路径
- 新仓库路径
- 复制时间
- 复制命令
- 当时的 git status

### 2. 保留原始依赖栈

我没有升级 CUDA、PyTorch、renderer 或数据格式。

新增了：

```text
environment_same_as_original.yml
BUILD_NOTES.md
```

这些文件记录了原始 4DGS 风格的环境设置方式：

- Python 3.7
- PyTorch 1.13.1 / CUDA 11.6
- `mmcv==1.6.0`
- 原始 `requirements.txt`
- 原始 editable extension build：
  - `submodules/depth-diff-gaussian-rasterization`
  - `submodules/simple-knn`

因为你后面说明本地环境不用继续试，会去学校 Great Lakes 上跑，所以我没有继续在本地强行创建或调试环境。

### 3. 分析原始代码结构

我分析了原始 4DGS 的主要路径，并写入：

```text
ARCHITECTURE_NOTES.md
```

主要结论：

- 训练入口：`train.py`
- 渲染入口：`render.py`
- 指标计算入口：`metrics.py`
- 参数定义：`arguments/__init__.py`
- 数据集识别和加载：`scene/__init__.py`
- Gaussian 参数存储：`scene/gaussian_model.py`
- deformation 网络：`scene/deformation.py`
- renderer 调用 deformation 的位置：`gaussian_renderer/__init__.py`
- loss 主要在 `train.py` 中构造
- 输出、checkpoint、point cloud 保持原始 `output/<expname>` 风格

### 4. 实现 static/dynamic motion separation

我做的是最小改动版本，没有引入 segmentation、optical flow、外部监督，也没有改 renderer API 或数据格式。

新增 CLI 参数：

```bash
--motion-separation
--motion-mask-lambda
--motion-gate-rot-scale
```

其中：

- 不加 `--motion-separation` 时，默认走原始 baseline 行为。
- 加 `--motion-separation` 时，启用 motion gate。
- `--motion-mask-lambda` 控制 motion mask sparsity regularization。
- `--motion-gate-rot-scale` 是可选项，默认不开启，用来同时 gate scale 和 rotation deformation。

核心实现位置：

```text
scene/deformation.py
```

实现逻辑：

```python
hidden = self.query_time(...)
m = sigmoid(phi_m(hidden))
dx = phi_x(hidden)
x_prime = x + m * dx
```

默认只 gate position deformation：

```python
pts = rays_pts_emb[:, :3] + motion_mask * dx
```

scale 和 rotation gating 默认关闭，只有传入：

```bash
--motion-gate-rot-scale
```

才会开启。

### 5. 添加 motion mask loss 和日志

在：

```text
train.py
scene/gaussian_model.py
```

中添加了 motion mask 相关逻辑。

如果启用：

```bash
--motion-separation --motion-mask-lambda 0.001
```

训练 loss 会额外加：

```python
loss += motion_mask_lambda * mean(m)
```

记录的 motion mask 统计包括：

- `mean(m)`
- `std(m)`
- `fraction(m > 0.5)`
- `fraction(m <= 0.5)`

输出文件：

```text
motion_mask_stats.jsonl
```

如果保存 checkpoint / point cloud，也会尝试保存 motion mask 诊断文件：

```text
motion_mask_last.pt
motion_mask_colors.ply
```

其中 `motion_mask_colors.ply` 使用：

- 红色：更 dynamic
- 蓝色：更 static

### 6. 添加自动比较脚本

新增：

```text
scripts/compare_motion_separation.py
```

它用于在同一个 dataset 上跑：

1. baseline
2. motion separation
3. render
4. metrics
5. 保存 CSV / JSON
6. 画 matplotlib 图
7. 写 summary

预期输出：

```text
outputs/comparison/<dataset_name>/metrics.csv
outputs/comparison/<dataset_name>/metrics.json
outputs/comparison/<dataset_name>/plots/*.png
outputs/comparison/<dataset_name>/summary.md
outputs/comparison/<dataset_name>/logs/*.log
```

真实指标来自原始 `metrics.py`：

- PSNR
- SSIM
- LPIPS-vgg
- LPIPS-alex

另外脚本里加了 proxy diagnostics，但文档里明确标注它们不是 formal benchmark：

- background stability proxy：border region temporal variance
- dynamic sharpness proxy：center region Laplacian variance
- motion mask summary：mean/std/static fraction/dynamic fraction

### 7. 写了运行文档

新增了这些文档：

```text
CHANGELOG_CODEX.md
COMMANDS_CHEATSHEET.md
EXPERIMENT_LOG.md
FINAL_REPORT.md
CODEX_SUMMARY_ZH.md
```

其中：

- `COMMANDS_CHEATSHEET.md`：放常用命令
- `EXPERIMENT_LOG.md`：记录本地测试情况和失败原因
- `FINAL_REPORT.md`：写完整英文报告
- `CODEX_SUMMARY_ZH.md`：本中文总结

## 修改过的主要代码文件

```text
arguments/__init__.py
scene/deformation.py
scene/gaussian_model.py
train.py
scripts/compare_motion_separation.py
```

## 本地测试情况

我本地能做的测试：

```bash
python -m py_compile arguments\__init__.py scene\deformation.py scene\gaussian_model.py train.py scripts\compare_motion_separation.py render.py metrics.py
```

结果：通过。

新参数 parser 测试通过：

```text
--motion-separation --motion-mask-lambda 0.001 --motion-gate-rot-scale
```

解析结果：

```text
True 0.001 True
```

由于本地没有可用的原始 4DGS legacy 环境，训练没有继续跑。记录到的本地环境错误包括：

```text
ModuleNotFoundError: No module named 'torch'
ModuleNotFoundError: No module named 'matplotlib'
ModuleNotFoundError: No module named 'lpips'
```

这些错误已经写入 `EXPERIMENT_LOG.md`，没有被隐藏。

## Great Lakes 上建议怎么跑

进入新副本：

```bash
cd 4DGaussians_static_dynamic
```

按原始 4DGS 方式建环境和编译 extension：

```bash
conda create -n Gaussians4D python=3.7
conda activate Gaussians4D
pip install -r requirements.txt
pip install -e submodules/depth-diff-gaussian-rasterization
pip install -e submodules/simple-knn
```

如果需要 CUDA 11.6 PyTorch wheel：

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

baseline：

```bash
python train.py -s <dataset_path> --port 6017 --expname "<name>_baseline" --configs <config.py>
```

motion separation：

```bash
python train.py -s <dataset_path> --port 6017 --expname "<name>_motion" --configs <config.py> --motion-separation --motion-mask-lambda 0.001
```

自动比较：

```bash
python scripts/compare_motion_separation.py -s <dataset_path> --configs <config.py> --expname <dataset_name> --motion-mask-lambda 0.001
```

短 smoke test 可以先用很少 iteration：

```bash
python scripts/compare_motion_separation.py -s <dataset_path> --configs <config.py> --expname smoke_test --iterations 10 --coarse-iterations 5 --motion-mask-lambda 0.001
```

## 当前限制

- 这个版本没有外部 motion supervision。
- motion mask 可能会受 `--motion-mask-lambda` 影响而过度变静态，需要 sweep 小值。
- proxy metrics 只是诊断，不是正式 benchmark。
- 本地未完成真实训练对比，需要在 Great Lakes 上用可用环境运行。

## 推荐下一步

1. 在 Great Lakes 上先跑一个 10 iteration smoke test。
2. 确认 baseline 和 motion mode 都能进入训练并保存输出。
3. 检查 `motion_mask_stats.jsonl` 是否有正常数值。
4. 用一个小场景跑完整 baseline vs motion comparison。
5. 对 `--motion-mask-lambda` 做小范围 sweep，例如：

```text
0
1e-4
1e-3
```

