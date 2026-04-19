# Low-Memory 4DGS Implementation Analysis

本文档面向一个 vanilla 4D Gaussian Splatting / 4DGS 代码仓库，目标是分析哪些模块最可能导致显存占用过高，并提出尽量保持方法主体不变的轻量化改法。重点目标是：

1. 尽可能在 12GB VRAM 显卡上跑通训练；
2. 给出 8GB VRAM 的妥协版方案；
3. 保持为 vanilla-compatible / lightweight modification，而不是替换成另一个方法。

## 1. Repo 结构与 4DGS 主流程梳理

### 关键文件

- `train.py`
  - 训练入口。
  - `training()` 创建 `GaussianModel` 和 `Scene`。
  - `scene_reconstruction()` 是主训练循环，先 coarse 后 fine。
  - 每个 iteration 采样 camera，调用 `gaussian_renderer.render()`，计算 loss，backward，densify/prune，optimizer step。

- `gaussian_renderer/__init__.py`
  - 核心 render 函数。
  - coarse 阶段直接 rasterize canonical Gaussian。
  - fine 阶段先对所有 Gaussian 调用 deformation network，再调用 `diff_gaussian_rasterization.GaussianRasterizer`。
  - 返回 rendered image、viewspace points、visibility filter、radii、depth。

- `scene/gaussian_model.py`
  - Gaussian 参数初始化、保存、加载、optimizer setup。
  - 显式 Gaussian 参数包括：
    - `_xyz`
    - `_features_dc`
    - `_features_rest`
    - `_scaling`
    - `_rotation`
    - `_opacity`
  - 还包含 densification / pruning / grow / opacity reset / regulation。

- `scene/deformation.py`
  - 4D 时间建模核心。
  - `deform_network` 包装 time / position encoding 和 `Deformation`。
  - `Deformation` 内部使用 HexPlane grid，并有多个 MLP heads：
    - `pos_deform`
    - `scales_deform`
    - `rotations_deform`
    - `opacity_deform`
    - `shs_deform`

- `scene/hexplane.py`
  - HexPlane grid 参数与多分辨率插值。
  - 使用 4D coordinate 的 2D plane combinations。
  - `grid_dimensions=2` 且 `input_coordinate_dim=4` 时有 6 个 plane。
  - `multires` 会复制多组 resolution 不同的 planes。

- `scene/__init__.py`
  - 根据数据集结构识别 Colmap / Blender / DyNeRF / HyperNeRF / PanopticSports / MultipleView。
  - 构造 train/test/video dataset。
  - 初始化 Gaussian 点云和 deformation AABB。

- `scene/dataset_readers.py`
  - 读取不同数据集格式。
  - Blender / D-NeRF 路径中存在 `PILtoTorch(image,(800,800))` 硬编码。
  - DyNeRF 使用 `points3D_downsample2.ply`，并由 `Neural3D_NDC_Dataset` 控制视频帧读取和 downsample。

- `scene/dataset.py`
  - `FourDGSdataset`，每次按 index 返回一个 `Camera`。

- `scene/cameras.py`
  - `Camera` 类，保存 image tensor、pose、projection matrix、time。
  - 当前 `original_image` 默认保留 CPU tensor，训练时再 `.cuda()`。

- `arguments/__init__.py`
  - 默认参数集中定义。
  - 关键默认值：
    - `sh_degree = 3`
    - `batch_size = 1`
    - `densify_until_iter = 15000`
    - `kplanes_config.output_coordinate_dim = 32`
    - `kplanes_config.resolution = [64, 64, 64, 25]`
    - `multires = [1, 2, 4, 8]`

- `arguments/dynerf/default.py`
  - DyNeRF 配置。
  - 关键点：
    - `batch_size = 4`
    - `net_width = 128`
    - `output_coordinate_dim = 16`
    - `resolution = [64, 64, 64, 150]`
    - `multires = [1,2]`
    - `no_dshs = False`

- `render.py`
  - 离线 render / evaluation。
  - 当前会把 `render_list`、`gt_list`、`render_images` 累积起来。
  - `render_list` 和 `gt_list` 是 GPU tensor，长视频或高分辨率评估可能额外爆显存。

### 训练主链路

1. `train.py` 解析 CLI 和 config。
2. 创建 `GaussianModel(dataset.sh_degree, hyper)`。
3. `Scene` 读取数据集、camera、点云。
4. `GaussianModel.create_from_pcd()` 初始化 canonical Gaussian。
5. coarse stage：
   - 不使用 deformation。
   - 直接 rasterize static Gaussian。
6. fine stage：
   - 对每个 sampled camera 的 time，调用 deformation network。
   - deformation 产生 time-conditioned position / scale / rotation / opacity / SH。
   - 调用 CUDA rasterizer。
7. 计算 L1、可选 SSIM、可选 temporal regulation。
8. backward。
9. 根据 viewspace gradient 做 densification stats。
10. densify / prune / opacity reset。
11. optimizer step。

## 2. 显存占用来源分析

### 2.1 batch 内多 view render graph 同时保留

代码确认位置：`train.py` 的 `scene_reconstruction()`。

当前逻辑是：

1. 对 `viewpoint_cams` 中每个 camera 调用一次 `render()`；
2. 把 `image`、`gt_image`、`radii`、`visibility_filter`、`viewspace_point_tensor` 放入 list；
3. `torch.cat(images, 0)` 和 `torch.cat(gt_images, 0)`；
4. 统一计算 loss；
5. 最后一次性 `loss.backward()`。

这意味着当 `batch_size > 1` 时，多个完整 render graph 会同时保留到 backward。

影响：

- DyNeRF 默认 `batch_size=4`，fine 阶段会保留 4 份 deformation graph 和 rasterizer graph。
- 显存峰值近似随 batch size 线性增加。
- 这是 12GB / 8GB 最应该优先处理的问题。

结论：这是代码已确认的训练时主要瓶颈。

### 2.2 fine 阶段对所有 Gaussian 做 deformation

代码确认位置：`gaussian_renderer/__init__.py`。

fine 阶段实际执行：

```python
means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
    means3D, scales, rotations, opacity, shs, time
)
```

虽然代码中存在：

```python
deformation_point = pc._deformation_table
```

并且有只对 `deformation_point` 做 deformation 的注释代码，但当前真正运行的是所有 Gaussian 都进入 deformation network。

影响：

- 每个 render 都会为全部 `N_gaussian` 产生 deformation activation。
- 中间量包括 grid features、MLP hidden、dx、ds、dr、do、dshs。
- 训练时这些 activation 要保留给 backward。
- `batch_size=4` 时这部分 graph 又会重复 4 份。

结论：这是代码已确认的 fine stage 主要显存来源。

### 2.3 Gaussian 数量和 Adam optimizer state

代码确认位置：

- `train.py` 中 densify 条件硬编码 `gaussians.get_xyz.shape[0] < 360000`。
- prune 只在 `gaussians.get_xyz.shape[0] > 200000` 时触发。
- `scene/gaussian_model.py` 使用 Adam optimizer。

默认 `sh_degree=3` 时，每个 Gaussian 显式参数大约：

- xyz: 3 floats
- SH DC: 3 floats
- SH rest: 45 floats
- scaling: 3 floats
- rotation: 4 floats
- opacity: 1 float

总计约 59 floats / Gaussian。

Adam 训练时还需要：

- parameter
- gradient
- exp_avg
- exp_avg_sq

所以仅显式 Gaussian 参数和 optimizer state 就是数百 MB 级别。更大的问题是 rasterizer 和 deformation 的 per-point 中间量也随 Gaussian 数量线性增长。

结论：Gaussian count 是训练和渲染的共同瓶颈。

### 2.4 SH degree 和 dynamic SH deformation

代码确认位置：

- `arguments/__init__.py` 默认 `sh_degree=3`。
- `scene/deformation.py` 中 `shs_deform` 固定输出 `16 * 3`。
- DyNeRF config 中 `no_dshs=False`。

问题：

- `sh_degree=3` 对应 16 个 SH coeff。
- `_features_rest` 有 45 floats / Gaussian，是 Gaussian 参数中最大的一块。
- 如果启用 dynamic SH deformation，deformation network 还会输出 `[N, 16, 3]` 的 `dshs`。
- 当前 `shs_deform` 固定假设 SH degree 3。如果直接把 `sh_degree` 改成 2，同时 `no_dshs=False`，会出现 shape mismatch。

结论：

- 低显存配置中可以降低 `sh_degree`，但应该同时设置 `no_dshs=True`。
- 如果想保留 dynamic SH，则需要修改 `shs_deform` 输出维度，使其跟 `sh_degree` 动态匹配。

### 2.5 HexPlane grid 参数和 grid_sample activation

代码确认位置：

- `scene/hexplane.py`
- `arguments/__init__.py`
- dataset-specific config files

默认配置：

```python
kplanes_config = {
    "grid_dimensions": 2,
    "input_coordinate_dim": 4,
    "output_coordinate_dim": 32,
    "resolution": [64, 64, 64, 25]
}
multires = [1, 2, 4, 8]
```

4D coordinate 取 2D plane combinations 时有 6 个 plane：

- xy
- xz
- xt
- yz
- yt
- zt

`multires` 会为每个 scale 建一套 planes。

估算：

- 默认 `output_coordinate_dim=32`、`multires=[1,2,4,8]` 时，HexPlane 参数本身约 35.7M floats，约 143MB。
- Adam + grad 后可到 500MB 级别。
- DyNeRF config 使用 `output_coordinate_dim=16`、`multires=[1,2]`，grid 参数小很多，但 batch size 和 dynamic SH 更重。

结论：HexPlane 参数不是唯一瓶颈，但在默认 global config 下并不轻。降低 `output_coordinate_dim` 和 `multires` 是有效的 vanilla-compatible 改法。

### 2.6 输入分辨率和 rasterization buffer

代码确认位置：

- `scene/dataset_readers.py` 中 Blender / D-NeRF 路径硬编码 `PILtoTorch(image,(800,800))`。
- `train.py` 中 render image 和 gt image 被 cat 成 `[B, 3, H, W]`。

影响：

- 分辨率影响 rendered image、gt image、loss buffer。
- 更重要的是影响 rasterizer 内部 per-pixel / per-tile buffer。
- 从 800x800 降到 400x400，像素相关显存约降到 1/4。

结论：降分辨率有效，但不是唯一或最核心的改法。它应该和 batch graph streaming、Gaussian budget 配合。

### 2.7 loss 组成

代码确认位置：`train.py` 和 `utils/loss_utils.py`。

当前默认：

- L1 loss 总是启用。
- `lambda_dssim=0`，所以 SSIM 默认关闭。
- LPIPS 代码被注释掉。
- fine 阶段可选 temporal regulation：
  - plane TV
  - time smoothness
  - L1 time planes

影响：

- L1 本身不重。
- SSIM 会创建 conv buffer，但默认关闭。
- LPIPS 如果启用会显著增加显存，但当前默认没有启用。
- temporal regulation 会访问 HexPlane grids，但不是主要峰值瓶颈。

结论：loss 不是当前默认配置的首要显存来源。

### 2.8 evaluation / validation / render 额外显存

代码确认位置：

- `train.py` 的 `training_report()`。
- `render.py` 的 `render_set()`。

`training_report()` 在 outer `torch.no_grad()` 中被调用，因此不会保留 autograd graph。但它会在测试 iteration 渲染多个 train/test views，会产生临时峰值。

`render.py` 更危险：

```python
render_images = []
gt_list = []
render_list = []
...
render_list.append(rendering)
gt_list.append(gt)
```

`rendering` 和 `gt` 是 GPU tensor，因此长视频 render 时显存会随帧数线性增长。

结论：训练中 validation 是临时峰值；离线 render 当前有明确的 GPU list 累积问题，应修复。

### 2.9 数据加载和 GPU cache

代码确认：

- Colmap / Blender / HyperNeRF 路径会把 image tensor 保存在 camera info 中，但多数情况下是 CPU tensor。
- `Camera.original_image` 中 `.to(self.data_device)` 被注释了。
- 训练时才通过 `viewpoint_cam.original_image.cuda()` 放到 GPU。

结论：

- 当前没有明显的一次性把所有 frame image 放入 GPU 的写法。
- 但 CPU RAM 可能占用较大。
- DataLoader `num_workers=16/32` 对 CPU 内存和进程开销不小，但不是 GPU 显存主因。

## 3. 最值得尝试的轻量化修改

### 3.1 优先级 1：batch 内改成 sequential micro-batch backward

修改内容：

- 不要把多个 view 的 render graph 全部放进 list。
- 对每个 camera：
  1. render；
  2. 计算 `loss_i / batch_size`；
  3. 立即 `backward()`；
  4. 保存 detached 的 viewspace grad / radii / visibility；
  5. 删除当前 render package。

涉及文件：

- `train.py`

推荐改法：

- 保留逻辑 batch size 的语义，但用 gradient accumulation 实现。
- optimizer step 仍然每 iteration 一次。
- densification stats 用每个 view 的 `viewspace_points.grad.detach()` 累积。

预期收益：

- 对 `batch_size > 1` 最明显。
- DyNeRF 默认 `batch_size=4` 时，峰值显存可能接近按 view 数下降。

风险/副作用：

- logging 需要改成累积 scalar。
- 需要确保 densification stats 与原逻辑一致。
- 数值上基本等价。

实现难度：中。

是否 vanilla-compatible：是。

### 3.2 优先级 2：增加 Gaussian 总数上限

修改内容：

- 新增 config 参数：
  - `max_gaussians`
  - `prune_min_gaussians`
- 替代 `train.py` 中硬编码的 `360000` 和 `200000`。
- densify 后如果超过上限，则基于 opacity / radius / gradient 做 budget prune。

涉及文件：

- `arguments/__init__.py`
- `train.py`
- `scene/gaussian_model.py`

推荐改法：

- 12GB：`max_gaussians=220000` 到 `260000`。
- 8GB：`max_gaussians=120000` 到 `160000`。
- pruning 不要等到 `>200000` 才开始。

预期收益：

- 降低 Gaussian 参数、Adam state、deformation activation、rasterizer per-point buffer。

风险/副作用：

- 点数过低会损失细节。
- 过早 cap 可能欠拟合。

实现难度：低到中。

是否 vanilla-compatible：是。

### 3.3 优先级 3：低显存 config preset

修改内容：

- 新增 12GB / 8GB preset。
- 不改变主代码时，先通过 config 控制：
  - `batch_size`
  - `sh_degree`
  - `no_dshs`
  - `net_width`
  - `kplanes_config`
  - `multires`
  - densification schedule

涉及文件：

- `arguments/__init__.py`
- `arguments/dynerf/*.py`
- `arguments/dnerf/*.py`
- `arguments/hypernerf/*.py`

推荐改法：

- 低显存下设置 `no_dshs=True`。
- 降低 `net_width` 到 64。
- 降低 `output_coordinate_dim` 到 8 或 16。
- 降低 `multires` 到 `[1,2]`。

预期收益：

- 稳定降低参数量和 activation。

风险/副作用：

- dynamic color 能力下降。
- deformation 表达变弱。

实现难度：低。

是否 vanilla-compatible：基本是。

### 3.4 优先级 4：让 `_deformation_table` 真正生效

修改内容：

- 当前 `gaussian_renderer/__init__.py` 中 masked deformation 代码被注释。
- 可以恢复为：
  - static 点直接使用原始 Gaussian 参数；
  - dynamic 点进入 deformation network；
  - deformation 结果 scatter 回 full tensor。

涉及文件：

- `gaussian_renderer/__init__.py`
- `scene/gaussian_model.py`
- `train.py`

推荐改法：

- 每隔若干 iteration 根据 `_deformation_accum` 调用 `update_deformation_table()`。
- 先用保守阈值，避免把动态点误判为静态点。

预期收益：

- 如果场景大部分区域静态，可显著减少 fine-stage deformation activation。

风险/副作用：

- mask 策略不好会损失动态细节。
- 当前 `_deformation_accum` 更新路径需要进一步检查和补齐。

实现难度：中。

是否 vanilla-compatible：是。代码里本身已经有这条路线的痕迹。

### 3.5 优先级 5：修复 render.py 的 GPU list 累积

修改内容：

- 每帧 render 后立即保存。
- 不要保留 GPU tensor list。
- 视频输出用 CPU uint8 list 或 streaming writer。

涉及文件：

- `render.py`

推荐改法：

- `torchvision.utils.save_image(rendering.cpu(), path)` 后释放 `rendering`。
- `gt` 同理。
- `render_images` 如果用于 mp4，保存为 CPU numpy uint8。

预期收益：

- 离线 render / evaluation 显存变成常数级。

风险/副作用：

- I/O 可能稍慢。

实现难度：低。

是否 vanilla-compatible：是，纯工程修复。

### 3.6 优先级 6：配置化输入分辨率

修改内容：

- 去掉 `scene/dataset_readers.py` 中的 `800x800` 硬编码。
- 增加统一的 `resolution` / `downsample` 参数。

涉及文件：

- `arguments/__init__.py`
- `scene/dataset_readers.py`
- `scene/neural_3D_dataset_NDC.py`

推荐改法：

- D-NeRF / Blender：支持 `--resolution 400`、`--resolution 600`、`--resolution 800`。
- DyNeRF：统一暴露 downsample。

预期收益：

- 降低 rasterizer buffer 和 image/loss buffer。

风险/副作用：

- 细节下降。
- 需要保证 FOV/focal 计算一致。

实现难度：中。

是否 vanilla-compatible：是。

### 3.7 优先级 7：AMP / mixed precision

修改内容：

- 当前没有 `autocast` / `GradScaler`。
- 可先只对 deformation MLP / HexPlane 使用 autocast。
- rasterizer 输入保持 fp32，降低自定义 CUDA op 风险。

涉及文件：

- `train.py`
- `gaussian_renderer/__init__.py`

预期收益：

- 主要节省 deformation activation。
- 对 Adam state 节省不明显。

风险/副作用：

- 自定义 rasterizer 不一定支持 half。
- scale / opacity / rotation 数值稳定性要验证。

实现难度：中。

是否 vanilla-compatible：是，但建议作为第二阶段。

### 3.8 优先级 8：patch-based / crop-based rendering

修改内容：

- 只渲染图像 crop 或 patch。
- 需要正确修改 projection matrix / principal point。

涉及文件：

- `gaussian_renderer/__init__.py`
- `scene/cameras.py`
- dataset / camera utils

预期收益：

- 像素相关 buffer 大幅降低。

风险/副作用：

- 实现复杂。
- 如果只裁剪 gt 而不改 camera，会训练错误的投影。
- densification 的 screen-space gradient 也会受影响。

实现难度：高。

是否 vanilla-compatible：偏工程扩展，不建议作为最小 final project contribution。

## 4. 面向 12GB 显存的可行配置建议

推荐 12GB preset：

```python
ModelParams = dict(
    sh_degree = 2,
    render_process = False,
)

ModelHiddenParams = dict(
    net_width = 64,
    defor_depth = 0,
    kplanes_config = {
        "grid_dimensions": 2,
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 16,
        "resolution": [64, 64, 64, 100],
    },
    multires = [1, 2],
    no_do = True,
    no_dshs = True,
)

OptimizationParams = dict(
    batch_size = 1,
    coarse_iterations = 3000,
    iterations = 14000,
    densify_from_iter = 800,
    densify_until_iter = 9000,
    densification_interval = 150,
    pruning_from_iter = 800,
    pruning_interval = 150,
    opacity_reset_interval = 60000,
    max_gaussians = 240000,
    prune_min_gaussians = 100000,
)
```

建议：

- 分辨率：
  - Blender / D-NeRF：`512x512` 或 `600x600`。
  - DyNeRF：约 `676x507` 或更低。
- 逻辑 batch：
  - 如果还没实现 sequential backward，固定 `batch_size=1`。
  - 如果实现了 sequential backward，可以允许逻辑 batch 2，但峰值仍按 1 view 控制。
- SH：
  - `sh_degree=2`。
  - `no_dshs=True`。
- Gaussian 上限：
  - 220k 到 260k。
- HexPlane：
  - `output_coordinate_dim=16`。
  - `multires=[1,2]`，效果不足再试 `[1,2,4]`。
- AMP：
  - 不是第一阶段必要条件。
  - baseline 跑通后再加 deformation-only AMP。
- evaluation：
  - 减少 `test_iterations`。
  - 关闭 `render_process`。

## 5. 面向 8GB 显存的妥协版建议

推荐 8GB preset：

```python
ModelParams = dict(
    sh_degree = 1,
    render_process = False,
)

ModelHiddenParams = dict(
    net_width = 64,
    defor_depth = 0,
    kplanes_config = {
        "grid_dimensions": 2,
        "input_coordinate_dim": 4,
        "output_coordinate_dim": 8,
        "resolution": [48, 48, 48, 75],
    },
    multires = [1, 2],
    no_do = True,
    no_dshs = True,
)

OptimizationParams = dict(
    batch_size = 1,
    coarse_iterations = 3000,
    iterations = 12000,
    densify_from_iter = 1200,
    densify_until_iter = 7000,
    densification_interval = 250,
    pruning_from_iter = 800,
    pruning_interval = 150,
    opacity_threshold_coarse = 0.008,
    opacity_threshold_fine_init = 0.008,
    opacity_threshold_fine_after = 0.008,
    opacity_reset_interval = 60000,
    max_gaussians = 140000,
    prune_min_gaussians = 60000,
)
```

建议：

- 分辨率：
  - Blender / D-NeRF：`400x400`。
  - DyNeRF：约 `512x384` 或更低。
- `batch_size=1`。
- 强烈建议实现 sequential backward，避免以后误设 batch 造成 OOM。
- `sh_degree=1` 起跑；如果显存允许再试 2。
- `no_dshs=True`，关闭 dynamic SH。
- `no_do=True`，关闭 dynamic opacity。
- 如仍 OOM，可尝试 `no_ds=True`，只保留 position + rotation deformation。
- Gaussian 上限：
  - 120k 到 160k。
- densification：
  - 更晚开始，更早停止，更低频。
- evaluation：
  - 训练中不跑中间 render。
  - 最终 render 必须流式保存，不累积 GPU tensor。

## 6. 最小改动实现路线

建议把 final project contribution 命名为：

**Low-Memory 4DGS** 或 **Lite-4DGS**

核心思路：

- 不换 rasterizer；
- 不换 4DGS 主体；
- 不引入完全不同的表示；
- 只做 memory-aware training 和 lightweight configuration。

最推荐的前三个改动组合：

1. Sequential View Backward
   - 把 batch 内多 view graph 累积改成逐 view backward。
   - 最大化降低峰值显存。

2. Configurable Gaussian Budget
   - 新增 `max_gaussians`。
   - 替代硬编码 360k。
   - 配合更主动 pruning。

3. Low-VRAM Presets
   - 提供 12GB / 8GB config。
   - 包含低分辨率、低 SH、轻 HexPlane、关闭 dynamic SH。

这三个改动最容易解释，也最容易在课程报告中形成清晰贡献：

> We introduce a memory-aware training variant of vanilla 4DGS by streaming per-view backward passes, enforcing a configurable Gaussian budget, and providing low-VRAM temporal model presets. The method preserves the canonical Gaussian + temporal deformation + splatting pipeline while reducing peak VRAM for commodity GPUs.

## 7. 可直接修改的代码清单

### `train.py`

建议修改：

- `scene_reconstruction()`：
  - 将 batch render 从 list-then-backward 改为 per-view backward。
  - 保留 loss scalar 用于 logging。
  - 保留 detached viewspace grad / radii / visibility 用于 densification。

- densification 部分：
  - 将 `360000` 改为 `opt.max_gaussians`。
  - 将 `200000` 改为 `opt.prune_min_gaussians`。

- validation 部分：
  - low-vram 模式下降低 `training_report()` 频率。
  - low-vram 模式下禁用 `render_process`。

### `arguments/__init__.py`

建议新增参数：

```python
self.max_gaussians = 360000
self.prune_min_gaussians = 200000
self.low_vram = False
self.use_amp = False
self.target_resolution = -1
```

也可增加：

```python
self.densify_budget_prune = True
```

### `scene/gaussian_model.py`

建议修改：

- densification 后支持 hard cap。
- 新增 budget prune helper：
  - 优先 prune low opacity；
  - 再 prune over-large screen/world scale；
  - 必要时按低 gradient prune。
- 如果要支持 `sh_degree < 3` 且 dynamic SH 开启，需要配合修改 deformation SH 输出。

### `gaussian_renderer/__init__.py`

建议修改：

- fine 阶段可恢复 masked deformation：
  - full tensors 先等于原始 Gaussian 参数；
  - 只对 `_deformation_table` 为 True 的点调用 deformation；
  - 将结果写回对应位置。
- 可选：deformation-only autocast，rasterizer 前转回 fp32。

### `scene/deformation.py`

建议修改：

- 如果保留 `no_dshs=True`，可以不改。
- 如果要支持低 SH 且 dynamic SH，必须把：

```python
self.shs_deform = nn.Sequential(..., nn.Linear(self.W, 16*3))
```

改成依赖 `sh_degree` 的输出维度。

### `scene/dataset_readers.py`

建议修改：

- 去掉 `PILtoTorch(image,(800,800))` 硬编码。
- 用 config 控制目标分辨率。
- `add_points()` 当前一次加 100000 点，low-vram 模式应禁用或改成可配置数量。

### `scene/neural_3D_dataset_NDC.py`

建议修改：

- 将 DyNeRF downsample 参数暴露到 config。
- 低显存 preset 中控制视频帧 resolution。

### `render.py`

建议修改：

- 不再累积 GPU tensor list。
- 每帧 render 后立即保存到磁盘。
- `render_images` 只保留 CPU uint8，或使用 streaming writer。

## 最终建议

如果只做一个最小但有效的 low-memory 版本，推荐先实现：

1. `train.py` sequential per-view backward；
2. `max_gaussians` 配置化；
3. 12GB / 8GB config preset；
4. `render.py` streaming save。

其中前两项是训练显存核心，第三项让实验可复现，第四项避免 evaluation/render 阶段额外 OOM。这个组合不会改变 4DGS 的主体方法，适合作为 final project 中的 lightweight improvement。
