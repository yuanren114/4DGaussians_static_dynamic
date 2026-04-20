# Motion Separation Analysis / 动静分离分析

## 1. Current Implementation Overview / 当前实现概述

**English**

The current motion-separation implementation is a lightweight mask head added to the existing 4D Gaussian deformation network, not a fully separate static/dynamic representation.

The relevant code paths are:

- `arguments/__init__.py`
  - `ModelHiddenParams.motion_separation = False`
  - `ModelHiddenParams.motion_mask_lambda = 0.0`
  - `ModelHiddenParams.motion_gate_rot_scale = False`
  - The parser also exposes dashed CLI aliases such as `--motion-separation`, `--motion-mask-lambda`, and `--motion-gate-rot-scale`.

- `scene/deformation.py`
  - `Deformation.create_net()` creates the mask head only when `args.motion_separation` is true:

    ```python
    self.motion_mask_deform = nn.Sequential(
        nn.ReLU(),
        nn.Linear(self.W, self.W),
        nn.ReLU(),
        nn.Linear(self.W, 1)
    )
    ```

  - `Deformation.forward_dynamic()` computes the mask as:

    ```python
    motion_mask = torch.sigmoid(self.motion_mask_deform(hidden))
    self.last_motion_mask = motion_mask
    ```

  - The same `hidden` feature is also used by the normal deformation heads:

    ```python
    dx = self.pos_deform(hidden)
    ds = self.scales_deform(hidden)
    dr = self.rotations_deform(hidden)
    ```

  - Position deformation is gated when `motion_separation` is enabled:

    ```python
    pts = rays_pts_emb[:, :3] + motion_mask * dx
    ```

  - Scale and rotation are only gated if `motion_gate_rot_scale` is also enabled:

    ```python
    if self.args.motion_separation and self.args.motion_gate_rot_scale:
        scales = scales_emb[:, :3] + motion_mask * ds
    else:
        scales = scales_emb[:, :3] * mask + ds
    ```

    ```python
    if self.args.motion_separation and self.args.motion_gate_rot_scale:
        rotations = rotations_emb[:, :4] + motion_mask * dr
    else:
        rotations = rotations_emb[:, :4] + dr
    ```

  - Therefore, the default behavior of `--motion-separation` gates only position. Rotation and scale can still deform without the motion mask unless `--motion-gate-rot-scale` is passed.

- `gaussian_renderer/__init__.py`
  - During the fine stage, rendering calls:

    ```python
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
        means3D, scales, rotations, opacity, shs, time
    )
    ```

  - The rasterizer receives the deformed means, scales, rotations, opacity, and SHs. The motion mask itself is not directly rendered; it only affects the rendered image through the deformation outputs.

- `train.py`
  - The normal training loss is primarily reconstruction-driven:

    ```python
    Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
    loss = Ll1
    ```

  - Fine-stage grid regularization is still applied:

    ```python
    tv_loss = gaussians.compute_regulation(...)
    loss += tv_loss
    ```

  - The only direct mask loss is a sparsity penalty:

    ```python
    if hyper.motion_separation and hyper.motion_mask_lambda != 0:
        motion_mask_loss = gaussians.motion_mask_loss()
        loss += hyper.motion_mask_lambda * motion_mask_loss
    ```

  - There is no explicit binary loss, entropy loss, anti-collapse term, lower-bound dynamic prior, or deformation-mask consistency loss.

- `scene/gaussian_model.py`
  - `motion_mask_loss()` returns:

    ```python
    return motion_mask.mean()
    ```

    This only pushes the mask downward.

  - `motion_mask_stats()` reports:

    ```python
    "mean": mask.mean().item(),
    "std": mask.std().item(),
    "dynamic_fraction": (mask > 0.5).float().mean().item(),
    "static_fraction": (mask <= 0.5).float().mean().item(),
    ```

  - `save_motion_mask_ply()` saves a point cloud where red is `mask`, blue is `1 - mask`, and the raw value is stored as a `motion_mask` vertex property.

- `render.py`
  - Rendering reconstructs `GaussianModel(dataset.sh_degree, hyperparam)` from the saved config plus CLI/config arguments. If the model was trained with `--motion-separation`, render must also instantiate the same architecture. Otherwise loading `deformation.pth` fails because the checkpoint contains `motion_mask_deform.*` weights.

One important implementation detail is that `dnerf_default.py` sets `defor_depth = 0`. In `Deformation.create_net()`, this means `feature_out` is just a single linear layer from HexPlane features to `W`, without additional hidden layers in the shared trunk. The mask head itself has two linear layers after ReLU activations, but it is still driven by the same shared feature used for all deformation heads.

**中文**

当前的 motion separation 实现是在原有 4D Gaussian deformation network 上加了一个轻量级 mask head，而不是显式维护两套独立的 static/dynamic Gaussian 表示。

相关代码路径如下：

- `arguments/__init__.py`
  - `ModelHiddenParams.motion_separation = False`
  - `ModelHiddenParams.motion_mask_lambda = 0.0`
  - `ModelHiddenParams.motion_gate_rot_scale = False`
  - 参数解析器会自动提供 `--motion-separation`、`--motion-mask-lambda`、`--motion-gate-rot-scale` 这些命令行开关。

- `scene/deformation.py`
  - `Deformation.create_net()` 只在 `args.motion_separation` 为 true 时创建 mask head：

    ```python
    self.motion_mask_deform = nn.Sequential(
        nn.ReLU(),
        nn.Linear(self.W, self.W),
        nn.ReLU(),
        nn.Linear(self.W, 1)
    )
    ```

  - `Deformation.forward_dynamic()` 中 mask 的计算方式是：

    ```python
    motion_mask = torch.sigmoid(self.motion_mask_deform(hidden))
    self.last_motion_mask = motion_mask
    ```

  - 同一个 `hidden` feature 同时被 normal deformation heads 使用：

    ```python
    dx = self.pos_deform(hidden)
    ds = self.scales_deform(hidden)
    dr = self.rotations_deform(hidden)
    ```

  - 当开启 `motion_separation` 时，position deformation 会被 mask 门控：

    ```python
    pts = rays_pts_emb[:, :3] + motion_mask * dx
    ```

  - scale 和 rotation 只有在同时开启 `motion_gate_rot_scale` 时才会被 mask 门控：

    ```python
    if self.args.motion_separation and self.args.motion_gate_rot_scale:
        scales = scales_emb[:, :3] + motion_mask * ds
    else:
        scales = scales_emb[:, :3] * mask + ds
    ```

    ```python
    if self.args.motion_separation and self.args.motion_gate_rot_scale:
        rotations = rotations_emb[:, :4] + motion_mask * dr
    else:
        rotations = rotations_emb[:, :4] + dr
    ```

  - 因此，默认 `--motion-separation` 只门控位置 `dx`。如果没有加 `--motion-gate-rot-scale`，rotation 和 scale 仍然可以绕过 motion mask 进行变形。

- `gaussian_renderer/__init__.py`
  - fine stage 渲染时会调用：

    ```python
    means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
        means3D, scales, rotations, opacity, shs, time
    )
    ```

  - rasterizer 接收的是 deformation 后的 means、scales、rotations、opacity、SH。motion mask 本身不会被直接渲染，它只通过 deformation output 间接影响图像。

- `train.py`
  - 当前训练主要由 reconstruction loss 驱动：

    ```python
    Ll1 = l1_loss(image_tensor, gt_image_tensor[:, :3, :, :])
    loss = Ll1
    ```

  - fine stage 仍然有 grid regularization：

    ```python
    tv_loss = gaussians.compute_regulation(...)
    loss += tv_loss
    ```

  - 唯一直接作用于 mask 的 loss 是稀疏项：

    ```python
    if hyper.motion_separation and hyper.motion_mask_lambda != 0:
        motion_mask_loss = gaussians.motion_mask_loss()
        loss += hyper.motion_mask_lambda * motion_mask_loss
    ```

  - 代码里没有显式二值化 loss、entropy loss、反塌缩约束、dynamic 下限先验，或者 deformation-mask consistency loss。

- `scene/gaussian_model.py`
  - `motion_mask_loss()` 返回：

    ```python
    return motion_mask.mean()
    ```

    这个项只会把 mask 往小压。

  - `motion_mask_stats()` 统计：

    ```python
    "mean": mask.mean().item(),
    "std": mask.std().item(),
    "dynamic_fraction": (mask > 0.5).float().mean().item(),
    "static_fraction": (mask <= 0.5).float().mean().item(),
    ```

  - `save_motion_mask_ply()` 会保存彩色点云，其中红色是 `mask`，蓝色是 `1 - mask`，并且原始数值会作为 `motion_mask` vertex property 存入 PLY。

- `render.py`
  - render 会根据保存的 config 和命令行/config 参数重新构造 `GaussianModel(dataset.sh_degree, hyperparam)`。如果训练时用了 `--motion-separation`，render 时也必须构造同样的 architecture，否则加载 `deformation.pth` 时会因为 checkpoint 中存在 `motion_mask_deform.*` 权重而报错。

一个重要实现细节是 `dnerf_default.py` 设置了 `defor_depth = 0`。在 `Deformation.create_net()` 中，这意味着 `feature_out` 基本只是从 HexPlane feature 到 `W` 的单层线性映射，没有额外 shared trunk 深层结构。mask head 本身有两层 Linear，但它仍然依赖和 deformation heads 相同的 shared feature。

## 2. Failure Symptoms / 失败现象

**English**

The observed behavior is consistent with the implementation above:

- On `lego_motion` with `motion_mask_lambda = 0.001`, the final mask was near zero:

  ```text
  mean ~= 0.026
  std ~= 0.043
  dynamic_fraction = 0.0
  static_fraction = 1.0
  ```

- On `lego_motion_noreg` with `motion_mask_lambda = 0`, the mask opened slightly but still stayed below the hard dynamic threshold:

  ```text
  mean ~= 0.069
  std ~= 0.066
  dynamic_fraction ~= 0.0
  static_fraction ~= 1.0
  ```

- On a more clearly dynamic scene such as `bouncingballs_motion_noreg`, the mask became larger but remained soft:

  ```text
  mean ~= 0.24
  std ~= 0.06
  dynamic_fraction ~= 0.0001
  static_fraction ~= 0.9999
  ```

- Reconstruction quality can still look reasonable, and PSNR/SSIM can remain close to the baseline. This means the deformation network can still model the dynamic scene without producing an interpretable binary static/dynamic mask.

The main symptom is therefore not simply poor rendering. The main symptom is that the learned mask behaves like a soft attenuation coefficient rather than a semantic binary motion assignment.

**中文**

观察到的现象和上述实现是吻合的：

- 在 `lego_motion` 且 `motion_mask_lambda = 0.001` 时，最终 mask 接近 0：

  ```text
  mean ~= 0.026
  std ~= 0.043
  dynamic_fraction = 0.0
  static_fraction = 1.0
  ```

- 在 `lego_motion_noreg` 且 `motion_mask_lambda = 0` 时，mask 稍微打开了一点，但仍然低于 hard dynamic threshold：

  ```text
  mean ~= 0.069
  std ~= 0.066
  dynamic_fraction ~= 0.0
  static_fraction ~= 1.0
  ```

- 在动态更明显的 `bouncingballs_motion_noreg` 上，mask 数值变大，但仍然是 soft 的：

  ```text
  mean ~= 0.24
  std ~= 0.06
  dynamic_fraction ~= 0.0001
  static_fraction ~= 0.9999
  ```

- 重建质量仍然可以看起来合理，PSNR/SSIM 也可以接近 baseline。这说明 deformation network 能够拟合动态场景，但没有学出可解释的二值 static/dynamic mask。

所以主要失败现象不是渲染质量一定很差，而是 learned mask 更像一个 soft attenuation coefficient，而不是语义上的 binary motion assignment。

## 3. Root Cause Analysis / 根因分析

**English**

The failure is caused by a combination of objective design, architecture design, training dynamics, and metric definition.

First, reconstruction loss alone does not identify semantic motion. The image loss only cares whether the final rasterized image matches the ground truth. It does not care whether a moving object was represented by a high mask, a low mask with larger displacement, scale/rotation changes, opacity/SH effects, or changes in the underlying HexPlane features. Therefore, semantic static/dynamic separation is underdetermined.

Second, the current mask has no binary pressure. In `scene/deformation.py`, the mask is produced by:

```python
motion_mask = torch.sigmoid(self.motion_mask_deform(hidden))
```

With only reconstruction gradients, sigmoid outputs are not forced to saturate near 0 or 1. A value around `0.24` can be a stable local solution if the downstream displacement head compensates for it. When `motion_mask_lambda > 0`, `motion_mask.mean()` explicitly pushes the mask toward 0, making all-static collapse even more attractive.

Third, there is a scale ambiguity between the mask and the displacement. The code implements:

```python
pts = rays_pts_emb[:, :3] + motion_mask * dx
```

This product is not identifiable. Many combinations of `motion_mask` and `dx` produce similar final positions. A low mask does not necessarily prevent motion if the network can increase `dx`.

Fourth, by default the mask gates only position. Scale and rotation are still allowed to deform unless `--motion-gate-rot-scale` is enabled. In the current code:

```python
rotations = rotations_emb[:, :4] + dr
```

and:

```python
scales = scales_emb[:, :3] * mask + ds
```

when `motion_gate_rot_scale` is false. Here `mask` is not the motion mask; it is the legacy `static_mlp`/`empty_voxel`/ones mask. In the common case it is all ones. Thus scale and rotation can bypass the motion mask.

Fifth, the current statistic `dynamic_fraction = (mask > 0.5)` is useful as one diagnostic, but it is not sufficient. If the learned mask distribution is centered around 0.24 with low variance, `dynamic_fraction` will be almost zero even though the mask is not numerically zero. This means the metric is correctly reporting that the mask is not binary-high, but it does not capture weaker relative activation.

Sixth, the saved `last_motion_mask` is the mask from the most recent forward pass, not necessarily an average over all cameras and all timesteps. It is still useful, but it should be interpreted as a snapshot produced by the last sampled batch/view/time during training or the last render call before saving. The code does not aggregate masks across time to estimate a stable per-Gaussian dynamic probability.

Finally, the implementation admits several degenerate solutions:

- All-static collapse: `motion_mask -> 0`, especially when `motion_mask_lambda > 0`.
- Low-mask-plus-large-displacement: `motion_mask` stays below 0.5 while `dx` grows to preserve the rendered motion.
- Bypass through ungated scale/rotation when `--motion-gate-rot-scale` is not used.
- Soft global attenuation: all Gaussians get similar moderate mask values, such as 0.2-0.3, without becoming semantically separated.

**中文**

这个失败是 objective design、architecture design、training dynamics 和 metric definition 共同导致的。

第一，reconstruction loss 本身不能识别语义 motion。图像 loss 只关心最终 rasterized image 是否接近 ground truth。它不关心运动物体是通过高 mask、低 mask 加大位移、scale/rotation 变化、opacity/SH 效果，还是 HexPlane feature 的变化来实现的。因此，语义 static/dynamic separation 本质上是欠约束的。

第二，当前 mask 没有二值化压力。在 `scene/deformation.py` 中，mask 由下面代码产生：

```python
motion_mask = torch.sigmoid(self.motion_mask_deform(hidden))
```

如果只有 reconstruction gradient，sigmoid 输出不会被强制推到 0 或 1。只要下游 displacement head 能补偿，`0.24` 这样的中间值也可能是稳定局部解。当 `motion_mask_lambda > 0` 时，`motion_mask.mean()` 会显式把 mask 往 0 压，使 all-static collapse 更有吸引力。

第三，mask 和 displacement 之间存在尺度不唯一性。当前代码实现的是：

```python
pts = rays_pts_emb[:, :3] + motion_mask * dx
```

这个乘积不可辨识。很多组 `motion_mask` 和 `dx` 的组合都能产生相似的最终位置。低 mask 不一定阻止运动，因为网络可以增大 `dx`。

第四，默认情况下 mask 只门控 position。除非开启 `--motion-gate-rot-scale`，scale 和 rotation 仍然可以变形。当前代码中：

```python
rotations = rotations_emb[:, :4] + dr
```

以及：

```python
scales = scales_emb[:, :3] * mask + ds
```

当 `motion_gate_rot_scale` 为 false 时，这里的 `mask` 不是 motion mask，而是旧的 `static_mlp`/`empty_voxel`/ones mask。常见情况下它就是全 1。因此 scale 和 rotation 可以绕过 motion mask。

第五，当前统计 `dynamic_fraction = (mask > 0.5)` 可以作为一个诊断指标，但远远不够。如果 learned mask 分布集中在 0.24 附近且方差很小，`dynamic_fraction` 会接近 0，即使 mask 数值上并不是 0。这说明该指标正确反映了 mask 没有变成 high binary mask，但它不能描述较弱的相对激活。

第六，保存的 `last_motion_mask` 是最近一次 forward pass 的 mask，不一定是所有 camera 和所有 timestep 的平均结果。它仍然有用，但应理解为训练中最后一个 sampled batch/view/time 或保存前最后一次 render call 的快照。代码没有跨时间聚合 mask 来估计稳定的 per-Gaussian dynamic probability。

最后，当前实现允许几种退化解：

- 全静态塌缩：`motion_mask -> 0`，尤其是 `motion_mask_lambda > 0` 时。
- 低 mask 加大位移：`motion_mask` 保持低于 0.5，同时 `dx` 变大来保持渲染运动。
- 未开启 `--motion-gate-rot-scale` 时通过 ungated scale/rotation 绕过 mask。
- 全局 soft attenuation：所有 Gaussians 得到类似的中等 mask 值，例如 0.2-0.3，但没有形成语义分离。

## 4. Theoretical Explanation / 理论解释

**English**

The current position model can be simplified as:

```text
x_t = x_0 + m(x_0, t) * Delta x(x_0, t)
```

where:

- `x_0` is the canonical Gaussian position.
- `m` is the motion mask from `sigmoid(motion_mask_deform(hidden))`.
- `Delta x` is the output of `pos_deform(hidden)`.
- The rendered image is `R(x_t, scale_t, rotation_t, opacity_t, sh_t)`.

The training objective is approximately:

```text
min  L_render(R(x_t, ...), I_t) + lambda * mean(m)
```

with additional grid regularization in the fine stage.

The key identifiability problem is:

```text
m * Delta x = (alpha m) * (Delta x / alpha)
```

for many positive `alpha`, as long as the displacement head has enough capacity and is not strongly penalized. Therefore:

```text
small m + large Delta x
```

and:

```text
large m + small Delta x
```

can produce nearly the same Gaussian positions and nearly the same rendered image. The rendering loss sees the product, not the semantic meaning of the factors.

If `lambda > 0`, the objective prefers smaller `m` whenever the model can compensate with larger `Delta x`. This makes low-mask solutions mathematically attractive. If `lambda = 0`, the objective is indifferent to the absolute scale of `m` except through optimization dynamics, initialization, sigmoid saturation, and indirect coupling to other heads. There is still no reason for `m` to become binary.

Sigmoid alone does not create binary decisions. A sigmoid output becomes close to 0 or 1 only if the logit becomes large in magnitude. Without a loss term that rewards confident assignments, the optimizer can leave logits in a moderate range. This is exactly consistent with observed masks such as:

```text
mean ~= 0.24
std ~= 0.06
```

Those values are soft gates, not discrete dynamic labels.

Semantic motion separation is also not guaranteed by reconstruction because the rendering problem has multiple equivalent explanations. A moving pixel can be explained by:

- moving canonical Gaussians through `dx`,
- changing rotation or scale,
- changing opacity or color if those paths are enabled,
- using nearby Gaussians with different visibility,
- distributing deformation over many Gaussians instead of assigning a compact dynamic object.

The current implementation does not include constraints that choose the semantic explanation among these possibilities.

**中文**

当前 position model 可以简化成：

```text
x_t = x_0 + m(x_0, t) * Delta x(x_0, t)
```

其中：

- `x_0` 是 canonical Gaussian position。
- `m` 是 `sigmoid(motion_mask_deform(hidden))` 得到的 motion mask。
- `Delta x` 是 `pos_deform(hidden)` 输出的 displacement。
- 渲染图像是 `R(x_t, scale_t, rotation_t, opacity_t, sh_t)`。

训练目标近似为：

```text
min  L_render(R(x_t, ...), I_t) + lambda * mean(m)
```

fine stage 还会有 grid regularization。

关键不可辨识性在于：

```text
m * Delta x = (alpha m) * (Delta x / alpha)
```

对于很多正的 `alpha` 都成立，只要 displacement head 有足够容量且没有被强约束。因此：

```text
small m + large Delta x
```

和：

```text
large m + small Delta x
```

可以产生几乎相同的 Gaussian 位置，也就可以产生几乎相同的渲染图像。rendering loss 看到的是乘积，而不是两个因子的语义含义。

如果 `lambda > 0`，只要模型能通过更大的 `Delta x` 补偿，目标函数就偏好更小的 `m`。这使低 mask 解在数学上更有吸引力。如果 `lambda = 0`，目标函数对 `m` 的绝对尺度基本不敏感，只有 optimization dynamics、初始化、sigmoid 饱和程度以及和其他 head 的间接耦合会影响它。仍然没有理由让 `m` 自动变成二值。

sigmoid 本身不会产生二值决策。sigmoid 输出只有在 logit 绝对值很大时才会接近 0 或 1。如果没有奖励 confident assignment 的 loss，优化器可以把 logit 留在中间区域。这和观察到的结果完全一致：

```text
mean ~= 0.24
std ~= 0.06
```

这些数值代表 soft gate，而不是离散 dynamic label。

reconstruction 也不能保证语义 motion separation，因为渲染问题有多个等价解释。一个运动像素可以由以下方式解释：

- 通过 `dx` 移动 canonical Gaussians；
- 改变 rotation 或 scale；
- 如果相关路径开启，改变 opacity 或 color；
- 使用附近不同 visibility 的 Gaussians；
- 把 deformation 分散到许多 Gaussians，而不是赋给紧凑的 dynamic object。

当前实现没有加入约束来在这些可能解释中选择语义正确的那一个。

## 5. Proposed Solutions / 解决方案

### 5.1 Loss-level changes / 损失层面的改进

**English**

1. Add a binarization or entropy-style regularizer.

   To make `m` approach 0 or 1, add a term that penalizes uncertain middle values. A simple option is:

   ```text
   L_bin = mean(m * (1 - m))
   ```

   This is largest at `m = 0.5` and smallest at 0 or 1. Another option is entropy:

   ```text
   L_entropy = mean(-m log m - (1 - m) log(1 - m))
   ```

   Minimizing entropy encourages confident assignments. This should be used carefully because it can also accelerate all-static or all-dynamic collapse if no anti-collapse term is present.

2. Replace plain sparsity with anti-collapse sparsity.

   The current `mean(m)` loss only encourages smaller masks. A better design is to combine sparsity with a target range or lower-bound prior:

   ```text
   L_balance = (mean(m) - rho)^2
   ```

   where `rho` is an expected dynamic fraction. For a scene like bouncingballs, `rho` may be larger than for Lego. This prevents immediate all-zero collapse.

3. Couple mask with deformation magnitude.

   If the goal is "static means low deformation", penalize displacement in static regions:

   ```text
   L_static_def = mean((1 - m) * ||Delta x||_2)
   ```

   This makes it expensive to hide motion behind a low mask. It directly addresses the low-mask-plus-large-displacement ambiguity.

4. Normalize or penalize displacement magnitude.

   Add:

   ```text
   L_dx = mean(||Delta x||_2^2)
   ```

   or a robust version. If `dx` cannot grow arbitrarily, the model has more incentive to use larger masks for truly moving regions.

5. Add temporal consistency.

   A semantic Gaussian should not randomly switch between static and dynamic across nearby timesteps. Penalize temporal mask variation:

   ```text
   L_temp = mean(|m(x, t + dt) - m(x, t)|)
   ```

   This requires evaluating masks for adjacent times or sampled time pairs.

6. Add spatial smoothness or object-level coherence.

   Neighboring Gaussians on the same object should have similar masks. A graph/Laplacian smoothness term over nearby points can reduce noisy masks:

   ```text
   L_spatial = sum_{i,j in N(i)} w_ij |m_i - m_j|
   ```

7. Use mask-displacement alignment diagnostics as training losses.

   Encourage high mask where displacement magnitude is high:

   ```text
   L_align = mean(|m - stopgrad(normalize(||Delta x||))|)
   ```

   This is weaker than supervised labels but can bootstrap interpretable masks.

**中文**

1. 加入二值化或 entropy regularizer。

   要让 `m` 接近 0 或 1，可以加入惩罚中间值的项。一个简单选择是：

   ```text
   L_bin = mean(m * (1 - m))
   ```

   它在 `m = 0.5` 最大，在 0 或 1 最小。另一个选择是 entropy：

   ```text
   L_entropy = mean(-m log m - (1 - m) log(1 - m))
   ```

   最小化 entropy 会鼓励 confident assignment。但它必须谨慎使用，因为如果没有反塌缩项，它也可能加速全静态或全动态塌缩。

2. 用 anti-collapse sparsity 替代单纯 sparsity。

   当前 `mean(m)` loss 只鼓励 mask 变小。更好的设计是把稀疏性和目标范围或下限先验结合：

   ```text
   L_balance = (mean(m) - rho)^2
   ```

   其中 `rho` 是预期 dynamic fraction。对于 bouncingballs，`rho` 应该比 Lego 大。这个项可以防止立刻全 0 塌缩。

3. 把 mask 和 deformation magnitude 绑定。

   如果目标是 “static 区域应该低 deformation”，可以惩罚 static 区域的位移：

   ```text
   L_static_def = mean((1 - m) * ||Delta x||_2)
   ```

   这样低 mask 下隐藏大位移会变贵，直接解决 low-mask-plus-large-displacement ambiguity。

4. 对 displacement magnitude 做归一化或惩罚。

   加入：

   ```text
   L_dx = mean(||Delta x||_2^2)
   ```

   或者鲁棒版本。如果 `dx` 不能任意变大，模型就更需要在真正运动区域使用更高 mask。

5. 加入 temporal consistency。

   一个语义 Gaussian 不应该在相邻时间随机切换 static/dynamic。可以惩罚 temporal mask variation：

   ```text
   L_temp = mean(|m(x, t + dt) - m(x, t)|)
   ```

   这需要对相邻时间或时间对额外计算 mask。

6. 加入 spatial smoothness 或 object-level coherence。

   同一物体上的邻近 Gaussians 应该有相似 mask。可以在邻近点图上加 Laplacian smoothness：

   ```text
   L_spatial = sum_{i,j in N(i)} w_ij |m_i - m_j|
   ```

   这样可以减少 noisy masks。

7. 把 mask-displacement alignment 作为训练 loss。

   鼓励大位移区域有高 mask：

   ```text
   L_align = mean(|m - stopgrad(normalize(||Delta x||))|)
   ```

   这不是强监督 label，但可以 bootstrap 更可解释的 mask。

### 5.2 Architecture-level changes / 架构层面的改进

**English**

1. Gate position, scale, and rotation together by default.

   The current default gates only `dx`. If the goal is dynamic/static separation, the mask should control all geometric deformation channels:

   ```text
   x_t = x_0 + m * Delta x_t
   s_t = s_0 + m * Delta s_t
   r_t = r_0 + m * Delta r_t
   ```

   In this codebase, that means using `--motion-gate-rot-scale` or making it the default for motion-separation experiments.

2. Expose and regularize raw deformation outputs.

   The current `forward_dynamic()` computes `dx`, `ds`, and `dr` locally but does not return them. Losses such as `L_static_def` require access to `dx` or at least its norm. A practical implementation would store `self.last_dx`, `self.last_ds`, and `self.last_dr` similarly to `self.last_motion_mask`.

3. Separate static and dynamic branches.

   Instead of one deformation field with a multiplicative mask, use explicit branches:

   ```text
   x_t = (1 - m) * x_static + m * x_dynamic_t
   ```

   or maintain separate static and dynamic Gaussian groups. This makes the representation less ambiguous than simply multiplying `dx`.

4. Use hard or straight-through gating after warm-up.

   A soft sigmoid gate is easy to optimize but hard to interpret. After warm-up, use a straight-through estimator:

   ```text
   m_hard = 1[m > tau]
   m_train = m_hard.detach() - m.detach() + m
   ```

   This produces binary forward behavior while preserving gradients.

5. Couple mask capacity with deformation capacity.

   The current mask head and deformation heads share `hidden`, but the deformation head can compensate for mask scale. One option is to parameterize displacement as:

   ```text
   Delta x_t = m * v_t
   ```

   with a bounded or normalized `v_t`, or to predict a displacement direction and magnitude separately.

**中文**

1. 默认同时门控 position、scale 和 rotation。

   当前默认只门控 `dx`。如果目标是 dynamic/static separation，mask 应该控制所有几何 deformation channel：

   ```text
   x_t = x_0 + m * Delta x_t
   s_t = s_0 + m * Delta s_t
   r_t = r_0 + m * Delta r_t
   ```

   在当前代码库里，这意味着使用 `--motion-gate-rot-scale`，或者在 motion-separation 实验中直接把它设为默认。

2. 暴露并 regularize 原始 deformation outputs。

   当前 `forward_dynamic()` 在局部计算 `dx`、`ds`、`dr`，但没有返回它们。像 `L_static_def` 这样的 loss 需要访问 `dx` 或至少它的 norm。实际实现上可以像 `self.last_motion_mask` 一样保存 `self.last_dx`、`self.last_ds`、`self.last_dr`。

3. 分离 static 和 dynamic branches。

   与其用一个 deformation field 加乘法 mask，不如使用显式分支：

   ```text
   x_t = (1 - m) * x_static + m * x_dynamic_t
   ```

   或者维护独立 static/dynamic Gaussian groups。这样比简单地乘 `dx` 更少歧义。

4. warm-up 后使用 hard 或 straight-through gating。

   soft sigmoid gate 容易优化，但难解释。warm-up 后可以使用 straight-through estimator：

   ```text
   m_hard = 1[m > tau]
   m_train = m_hard.detach() - m.detach() + m
   ```

   这样 forward 行为是二值的，同时仍然保留梯度。

5. 把 mask capacity 和 deformation capacity 绑定。

   当前 mask head 和 deformation heads 共用 `hidden`，但 deformation head 可以补偿 mask scale。一种做法是把 displacement 参数化为：

   ```text
   Delta x_t = m * v_t
   ```

   并限制或归一化 `v_t`，或者分别预测 displacement direction 和 magnitude。

### 5.3 Training strategy / 训练策略

**English**

1. Warm up deformation before enforcing binary masks.

   If binary/entropy regularization is applied too early, the model may collapse before it discovers useful motion. A practical schedule is:

   - Train normal deformation for a short warm-up.
   - Enable mask head with weak regularization.
   - Gradually increase binarization and static-deformation penalties.

2. Avoid positive `motion_mask_lambda` alone.

   Plain `lambda * mean(m)` is biased toward all-static. It should not be used by itself. If sparsity is desired, pair it with anti-collapse or reconstruction-sensitive constraints.

3. Initialize mask logits near a useful prior.

   If the expected dynamic fraction is not tiny, initialize the mask head bias so that:

   ```text
   sigmoid(bias) ~= rho
   ```

   For example, `rho = 0.2` gives `bias = log(rho / (1 - rho)) ~= -1.386`. This is better than leaving the initialization entirely to Xavier/random effects.

4. Use scene-dependent curriculum.

   Lego has small and slow motion, so it is a weak test case for separation. Bouncingballs should provide stronger signal. Start debugging on scenes with obvious motion, then check whether the same method remains stable on Lego.

5. Bootstrap masks from deformation magnitude.

   First train a baseline dynamic model. Then compute per-Gaussian deformation magnitude over time and use high-motion Gaussians as pseudo-dynamic labels. Then train the mask model using a weak supervised or distillation loss.

**中文**

1. 在强制二值 mask 前先 warm up deformation。

   如果太早施加 binary/entropy regularization，模型可能在发现有用 motion 前就塌缩。一个实用 schedule 是：

   - 先训练普通 deformation 一小段；
   - 再开启 mask head 和弱 regularization；
   - 逐步增加 binarization 和 static-deformation penalty。

2. 不要单独使用正的 `motion_mask_lambda`。

   单纯的 `lambda * mean(m)` 天然偏向全静态。它不应该单独使用。如果需要稀疏性，应配合 anti-collapse 或 reconstruction-sensitive constraints。

3. 根据先验初始化 mask logits。

   如果预期 dynamic fraction 不是极小，可以初始化 mask head bias，使：

   ```text
   sigmoid(bias) ~= rho
   ```

   例如 `rho = 0.2` 时，`bias = log(rho / (1 - rho)) ~= -1.386`。这比完全依赖 Xavier/random initialization 更可控。

4. 使用 scene-dependent curriculum。

   Lego 的运动区域小且慢，所以不是验证 separation 的强测试场景。Bouncingballs 应该提供更强信号。建议先在明显动态的场景上调通方法，再检查它在 Lego 上是否稳定。

5. 从 deformation magnitude bootstrap masks。

   先训练一个 baseline dynamic model。然后计算每个 Gaussian 跨时间的 deformation magnitude，把高运动 Gaussians 当作 pseudo-dynamic labels。再用弱监督或 distillation loss 训练 mask model。

### 5.4 Evaluation improvements / 评估改进

**English**

`dynamic_fraction = (mask > 0.5)` is not enough. It only answers whether the mask became high under a hard threshold. It does not describe soft activation, ranking quality, temporal stability, or spatial coherence.

Recommended diagnostics:

1. Report mask quantiles:

   ```text
   min, p01, p05, p10, p25, median, p75, p90, p95, p99, max
   ```

2. Report fractions at multiple thresholds:

   ```text
   mean(mask > 0.05), mean(mask > 0.10), mean(mask > 0.20),
   mean(mask > 0.30), mean(mask > 0.40), mean(mask > 0.50)
   ```

3. Report deformation-mask correlation:

   ```text
   corr(mask, ||Delta x||)
   ```

   A meaningful motion mask should correlate with actual deformation magnitude.

4. Aggregate masks over time.

   Instead of saving only `last_motion_mask`, evaluate masks across all training/test timesteps and compute:

   ```text
   mean_t m_i(t)
   max_t m_i(t)
   std_t m_i(t)
   mean_t ||Delta x_i(t)||
   ```

5. Visualize the mask PLY with raw scalar coloring.

   The current red/blue PLY is useful, but for soft masks it is better to use a colormap and inspect raw scalar ranges.

6. Compare rendered quality and separation quality separately.

   PSNR/SSIM/LPIPS evaluate reconstruction, not semantic separation. A method can match baseline rendering and still fail to produce a meaningful mask.

**中文**

`dynamic_fraction = (mask > 0.5)` 不够。它只回答 mask 是否在 hard threshold 下变高，不能描述 soft activation、排序质量、时间稳定性或空间一致性。

推荐诊断：

1. 报告 mask quantiles：

   ```text
   min, p01, p05, p10, p25, median, p75, p90, p95, p99, max
   ```

2. 报告多个阈值下的比例：

   ```text
   mean(mask > 0.05), mean(mask > 0.10), mean(mask > 0.20),
   mean(mask > 0.30), mean(mask > 0.40), mean(mask > 0.50)
   ```

3. 报告 deformation-mask correlation：

   ```text
   corr(mask, ||Delta x||)
   ```

   有意义的 motion mask 应该和实际 deformation magnitude 相关。

4. 跨时间聚合 masks。

   不要只保存 `last_motion_mask`，而是对所有 train/test timesteps 计算 mask，并统计：

   ```text
   mean_t m_i(t)
   max_t m_i(t)
   std_t m_i(t)
   mean_t ||Delta x_i(t)||
   ```

5. 用 raw scalar colormap 可视化 mask PLY。

   当前红/蓝 PLY 有用，但对于 soft masks，最好使用 colormap 并检查原始数值范围。

6. 分开比较渲染质量和分离质量。

   PSNR/SSIM/LPIPS 评估的是 reconstruction，不是语义分离。一个方法可以达到接近 baseline 的渲染质量，但仍然没有产生有意义的 mask。

## 6. Recommended Next Steps / 推荐的下一步

**English**

Priority 1: fix diagnostics before changing the model.

- Add quantiles and multi-threshold fractions to `motion_mask_stats()`.
- Save or compute time-aggregated masks instead of relying only on `last_motion_mask`.
- Report mask statistics for Lego and Bouncingballs side by side.

This is the easiest and gives immediate clarity.

Priority 2: run the existing architecture with stronger gating.

- Train Bouncingballs with:

  ```bash
  --motion-separation --motion-gate-rot-scale --motion-mask-lambda 0
  ```

- This tests whether ungated scale/rotation bypass is a major issue without requiring code changes.

Priority 3: add displacement-aware regularization.

- Modify `scene/deformation.py` to store `last_dx`.
- Modify `scene/gaussian_model.py` to expose a loss such as:

  ```text
  mean((1 - m) * ||Delta x||)
  ```

- This directly attacks the low-mask-plus-large-dx failure mode.

Priority 4: add binary and anti-collapse losses.

- Add `mean(m * (1 - m))` to encourage binary masks.
- Do not use it alone; combine it with a balance prior or target dynamic fraction:

  ```text
  (mean(m) - rho)^2
  ```

Priority 5: for the final project, present the current result honestly and then show an improved diagnostic or ablation.

The most convincing course/final-project result does not require proving that the first implementation works perfectly. A strong result can be:

- baseline rendering metrics,
- motion-mask failure analysis,
- demonstration that naive reconstruction-only gating is underdetermined,
- one principled fix or ablation showing improved mask distribution or better correlation with deformation.

**中文**

优先级 1：先修诊断，再改模型。

- 在 `motion_mask_stats()` 中加入 quantiles 和多阈值比例。
- 保存或计算跨时间聚合 mask，不要只依赖 `last_motion_mask`。
- 并排报告 Lego 和 Bouncingballs 的 mask statistics。

这是最容易实现、也最能立即澄清问题的步骤。

优先级 2：用现有架构测试更强 gating。

- 用下面参数训练 Bouncingballs：

  ```bash
  --motion-separation --motion-gate-rot-scale --motion-mask-lambda 0
  ```

- 这可以在不改代码的情况下测试 ungated scale/rotation bypass 是否是主要问题。

优先级 3：加入 displacement-aware regularization。

- 修改 `scene/deformation.py` 保存 `last_dx`。
- 修改 `scene/gaussian_model.py` 暴露类似下面的 loss：

  ```text
  mean((1 - m) * ||Delta x||)
  ```

- 这会直接打击 low-mask-plus-large-dx 失败模式。

优先级 4：加入 binary 和 anti-collapse losses。

- 加入 `mean(m * (1 - m))` 鼓励二值 mask。
- 不要单独使用它；应结合 balance prior 或目标 dynamic fraction：

  ```text
  (mean(m) - rho)^2
  ```

优先级 5：final project 中诚实呈现当前结果，并展示一个改进诊断或 ablation。

一个有说服力的课程/期末项目结果不一定要求第一个实现完美成功。一个强结果可以是：

- baseline rendering metrics；
- motion-mask failure analysis；
- 证明 naive reconstruction-only gating 是欠约束的；
- 展示一个原则性修复或 ablation，使 mask distribution 或 mask-deformation correlation 更合理。

## 7. Conclusion / 结论

**English**

The current code implements a soft multiplicative motion gate, not a fully identifiable static/dynamic separation model. The mask is predicted from the same hidden deformation feature and is trained mostly through reconstruction loss. The only direct mask regularizer is `mean(mask)`, which encourages lower masks and can cause all-static collapse. Because the rendered result depends on the product `mask * dx`, the model can preserve reconstruction quality with low masks and larger displacement. By default, scale and rotation also bypass the motion mask unless `--motion-gate-rot-scale` is enabled. Finally, `dynamic_fraction = (mask > 0.5)` is too narrow to diagnose soft masks.

Therefore, the underperformance is not just a dataset issue. Lego's small slow motion can make the problem harder, but Bouncingballs showing `mean ~= 0.24` and `dynamic_fraction ~= 0` indicates a deeper objective/architecture ambiguity. Meaningful motion separation requires additional constraints: binary pressure, anti-collapse priors, deformation-mask coupling, broader gating, time-aggregated diagnostics, and possibly explicit static/dynamic branches.

**中文**

当前代码实现的是 soft multiplicative motion gate，而不是完全可辨识的 static/dynamic separation model。mask 从同一个 hidden deformation feature 预测，主要通过 reconstruction loss 训练。唯一直接的 mask regularizer 是 `mean(mask)`，它会鼓励更低 mask，并可能导致全静态塌缩。由于渲染结果依赖 `mask * dx` 的乘积，模型可以用低 mask 加大 displacement 来保持重建质量。默认情况下，如果没有开启 `--motion-gate-rot-scale`，scale 和 rotation 也会绕过 motion mask。最后，`dynamic_fraction = (mask > 0.5)` 对 soft mask 的诊断过于狭窄。

因此，这个问题不只是数据集问题。Lego 的小幅慢速运动确实会让 separation 更难，但 Bouncingballs 中出现 `mean ~= 0.24` 且 `dynamic_fraction ~= 0` 表明存在更深的 objective/architecture ambiguity。要实现有意义的 motion separation，需要额外约束：二值化压力、反塌缩先验、deformation-mask coupling、更完整的 gating、跨时间聚合诊断，以及可能的显式 static/dynamic branches。
