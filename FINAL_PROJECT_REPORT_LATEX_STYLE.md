# Motion-Aware Regularization for 4D Gaussian Splatting

## Abstract

This project studies motion-aware dynamic scene reconstruction in a modified 4D Gaussian Splatting (4DGS) codebase. The original 4DGS framework models time-varying scenes through a deformation network applied to canonical Gaussians. However, the baseline method does not explicitly indicate which Gaussians should remain static and which should participate in motion. To address this, we implemented a lightweight motion-mask branch inside the deformation network. The mask acts as a soft coefficient that gates deformation and is learned jointly with reconstruction.

An early prototype used only a sparsity-style mask regularizer, but later experiments showed that this term mainly encouraged collapse and was not useful in the final method. The final method reported in this project instead uses a soft motion mask together with a binarization loss and a static-deformation loss. The method is unsupervised with respect to motion labels: it uses only image reconstruction and regularization, without ground-truth static/dynamic masks. Verified experiments on D-NeRF scenes show that the final regularized motion-mask variant produces more meaningful soft motion localization and yields modest reconstruction gains on strongly dynamic scenes such as Bouncingballs and Jumpingjacks. A HyperNeRF chickchicken experiment shows that the method does not necessarily improve reconstruction metrics on real-world data, but stronger static-deformation regularization can produce cleaner mask diagnostics.

## 1. Introduction

Reconstructing a 3D scene from images is a central problem in computer vision and graphics. Neural Radiance Fields (NeRF) represent scenes as continuous neural functions, while 3D Gaussian Splatting (3DGS) instead uses an explicit set of anisotropic Gaussian primitives for more efficient rendering. For dynamic scenes, 4D Gaussian Splatting extends this idea by combining canonical Gaussians with a time-dependent deformation model.

Dynamic scene reconstruction is challenging because geometry, appearance, and motion are entangled. A deformation network can explain temporal variation, but the resulting representation does not necessarily reveal which structures are inherently static and which are dynamic. This matters both for interpretability and for robust dynamic modeling.

At the beginning of this project, an alternative direction considered using external segmentation models such as GroundingDINO and SAM2 together with an Instant-NGP-style 3D pipeline. That approach has several limitations: it depends on prompt-based segmentation, masks can be temporally inconsistent, and segmentation is not inherently motion-aware. Most importantly, such masks are external to the differentiable reconstruction process.

The final project therefore focuses on learning a motion-aware soft coefficient directly inside the 4DGS deformation pipeline.

## 2. Related Work

### 2.1 4D Gaussian Splatting

Original 4DGS models a dynamic scene by storing Gaussians in a canonical space and predicting time-dependent deformation. In simplified notation:

$$
\mu_t = \mu_0 + \Delta \mu_t.
$$

The same idea is implemented in this codebase through a HexPlane-based deformation network, which predicts position, scale, and rotation changes before rasterization.

### 2.2 SDD-4DGS

SDD-4DGS proposes a static-dynamic decoupling framework for 4D Gaussian Splatting. Based on the public paper description and the reviewed formulation, it introduces a dynamic perception coefficient associated with each Gaussian and uses it to decouple dynamic and static components during optimization.

The current project is related to that idea, but it is not a full reproduction of SDD-4DGS. In particular, our implementation does not store a persistent per-Gaussian Bernoulli coefficient. Instead, it predicts a soft mask from the deformation-network feature. This difference is discussed in detail in Section 3.2.

### 2.3 Segmentation-Driven Mask-Based Methods

Segmentation-driven methods use external masks from object detectors or segmentation models to guide scene decomposition. These methods can separate semantic objects, but they rely on extra supervision and do not necessarily correspond to motion. They are also outside the reconstruction training loop.

This project does not use external segmentation masks during 4DGS training.

## 3. Method

### 3.1 Overview

The goal is to augment the 4DGS deformation model with a motion-aware soft gate. The core idea is:

1. predict a scalar motion mask from the deformation feature,
2. use it to gate deformation,
3. add regularization so that the mask becomes more interpretable.

The final method reported in this project is the regularized variant with:

- motion-mask gating,
- optional scale/rotation gating,
- binarization loss,
- static-deformation loss.

An earlier sparsity-only prototype was also implemented, but it is treated as a preliminary failed attempt rather than part of the final method.

### 3.2 Difference from SDD-4DGS

#### Representation

SDD-4DGS is described as using a dynamic perception coefficient associated with each Gaussian. In contrast, our implementation predicts a motion mask from the shared deformation feature:

```python
motion_mask = torch.sigmoid(self.motion_mask_deform(hidden))
```

Thus, our coefficient is:

$$
m_i(t) = \sigma(g_\phi(h_i(t))),
$$

which is a network-predicted soft gate rather than a persistent per-Gaussian parameter.

#### Training

SDD-4DGS uses a decoupling framework with a probabilistic dynamic coefficient. Our implementation uses the original 4DGS reconstruction pipeline and augments it with regularization terms. The final method uses:

$$
\mathcal{L}_{\text{img}} + \mathcal{L}_{\text{grid}} + \lambda_{\text{bin}}\mathcal{L}_{\text{bin}} + \lambda_{\text{static}}\mathcal{L}_{\text{static-def}}.
$$

The earlier sparsity term was tested but not retained in the final method.

#### Supervision

Like SDD-4DGS, this project does not use ground-truth motion masks. The mask is learned from reconstruction and regularization.

#### Advantages and Limitations

The advantage of our implementation is simplicity and ease of integration. Since the mask is predicted by a network head, there is no need to modify Gaussian cloning, pruning, saving, or optimizer-state logic to support a new per-Gaussian parameter.

The limitation is weaker identifiability. A predicted soft mask is easier to optimize but less directly interpretable than an explicit static/dynamic coefficient per Gaussian.

### 3.3 Motion Mask Formulation

The motion mask is a scalar soft coefficient in $[0,1]$. It is computed from the shared deformation feature:

$$
m_i(t) = \sigma(g_\phi(h_i(t))).
$$

For position deformation, the implemented update is:

$$
x_i(t) = x_i^0 + m_i(t)\Delta x_i(t).
$$

In code:

```python
dx = self.pos_deform(hidden)
pts = rays_pts_emb[:, :3] + motion_mask * dx
```

If `--motion-gate-rot-scale` is enabled, scale and rotation are also gated:

$$
s_i(t) = s_i^0 + m_i(t)\Delta s_i(t), \qquad r_i(t) = r_i^0 + m_i(t)\Delta r_i(t).
$$

The mask is not directly applied to opacity. It is also not a hard binary variable.

### 3.4 Integration into the 4DGS Pipeline

The implementation follows the original 4DGS pipeline with one added branch.

1. **Gaussian initialization.** Canonical Gaussians are initialized from the input scene point cloud.
2. **Deformation feature computation.** A HexPlane field is queried at the canonical Gaussian position and current time.
3. **Shared feature projection.** The queried feature is mapped to a shared deformation feature `hidden`.
4. **Mask prediction.** The new motion-mask head predicts a scalar mask from `hidden`.
5. **Deformation prediction.** Position, scale, and rotation heads predict deformation deltas.
6. **Mask-gated update.** The mask gates the deformation before rasterization.
7. **Rendering.** The deformed Gaussians are rasterized to produce the image.
8. **Loss computation.** Image reconstruction loss and final regularization terms are computed.

### 3.5 Differences from the Segmentation-Based Approach

The final method differs from the earlier segmentation-based idea in four important ways:

1. It does not depend on external prompt-based masks.
2. It is integrated into differentiable reconstruction training.
3. It is motion-aware rather than purely semantic.
4. It can be evaluated through both reconstruction and mask diagnostics.

## 4. Experiments

### 4.1 Setup

The main reported experiments use D-NeRF scenes with stronger motion:

- Bouncingballs
- Jumpingjacks

The report also includes one real-world HyperNeRF scene:

- HyperNeRF chickchicken

Lego was only used in early pilot experiments before the final regularized method was established.

The default D-NeRF configuration in this codebase includes:

- `iterations = 20000`
- `coarse_iterations = 3000`
- `defor_depth = 0`
- `net_width = 64`
- `multires = [1, 2]`

### 4.2 Metrics

The available evaluation script computes:

- PSNR
- SSIM
- LPIPS-vgg
- LPIPS-alex
- MS-SSIM
- D-SSIM

The codebase does not provide RMSE in the standard evaluation script. Training time was not logged in a way that supports a verified quantitative comparison, so it is not reported here.

Additional motion-mask diagnostics come from `motion_mask_stats.jsonl`.

### 4.3 Early Pilot Results

An early prototype used either:

- a naive soft mask gate, or
- a sparsity-only regularizer on the mask.

These early experiments did not produce convincing static/dynamic separation and are not treated as the final method. Lego belongs to this exploratory stage.

### 4.4 Main Results: Bouncingballs

Verified Bouncingballs reconstruction metrics:

| Method | SSIM | PSNR | LPIPS-vgg | LPIPS-alex | MS-SSIM | D-SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.9942868 | 40.6763 | 0.0153625 | 0.0060306 | 0.9953953 | 0.0023024 |
| Regularized variant ($\lambda_{\text{static}}=10^{-3}, \lambda_{\text{bin}}=10^{-3}$) | 0.9945415 | 40.8666 | 0.0143814 | 0.0056324 | 0.9955825 | 0.0022088 |
| Regularized variant ($\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=10^{-3}$) | 0.9944983 | **40.9611** | 0.0144220 | 0.0052267 | 0.9956228 | 0.0021886 |
| Regularized variant ($\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=2\times10^{-3}$) | **0.9947041** | 40.8955 | **0.0136226** | **0.0050899** | **0.9956791** | **0.0021604** |
| Over-regularized variant ($\lambda_{\text{static}}=10^{-2}, \lambda_{\text{bin}}=10^{-3}$) | 0.9942789 | 40.7233 | 0.0155924 | 0.0059685 | 0.9953101 | 0.0023450 |

These results show a scene-level tradeoff. The setting $\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=2\times10^{-3}$ gives the best reconstruction quality on Bouncingballs, while $\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=10^{-3}$ gives the clearest soft-mask separation.

Mask diagnostics:

| Method | Final mean | Final std | Final dynamic fraction | Final fraction $m>0.4$ | Static deformation | Binarization |
|---|---:|---:|---:|---:|---:|---:|
| Early no-sparsity mask | 0.2475 | 0.0641 | 0.0001 | not logged | not logged | not logged |
| Regularized variant ($\lambda_{\text{static}}=10^{-3}, \lambda_{\text{bin}}=10^{-3}$) | 0.1851 | 0.1904 | 0.0117 | 0.2162 | 0.0934 | 0.1146 |
| Regularized variant ($\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=10^{-3}$) | 0.2179 | 0.2487 | 0.2604 | 0.3879 | 0.0523 | 0.1086 |
| Regularized variant ($\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=2\times10^{-3}$) | 0.1400 | 0.2137 | 0.0487 | 0.2615 | 0.0499 | 0.0747 |
| Over-regularized variant ($\lambda_{\text{static}}=10^{-2}, \lambda_{\text{bin}}=10^{-3}$) | 0.9986 | 0.0022 | 1.0000 | 1.0000 | 0.0002 | 0.0014 |

Thus, best reconstruction and best mask separability are not the same operating point for this scene.

### 4.5 Main Results: Jumpingjacks

Verified Jumpingjacks reconstruction metrics:

| Method | SSIM | PSNR | LPIPS-vgg | LPIPS-alex | MS-SSIM | D-SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.9855952 | 35.4000 | 0.0199500 | 0.0126626 | 0.9936216 | 0.0031892 |
| Regularized variant ($\lambda_{\text{static}}=10^{-3}, \lambda_{\text{bin}}=10^{-3}$) | **0.9863845** | 35.5784 | 0.0189655 | **0.0123328** | **0.9940146** | **0.0029927** |
| Regularized variant ($\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=10^{-3}$) | 0.9863592 | **35.6238** | **0.0189545** | 0.0126150 | 0.9939969 | 0.0030016 |
| Regularized variant ($\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=2\times10^{-3}$) | 0.9858798 | 35.3964 | 0.0198835 | 0.0131377 | 0.9936382 | 0.0031809 |

Both $\lambda_{\text{static}}=10^{-3}, \lambda_{\text{bin}}=10^{-3}$ and $\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=10^{-3}$ improve over baseline. However, unlike Bouncingballs, increasing the binarization weight to $2\times10^{-3}$ degrades Jumpingjacks reconstruction.

Mask diagnostics:

| Method | Final mean | Final std | Final dynamic fraction | Final fraction $m>0.4$ | Static deformation | Binarization |
|---|---:|---:|---:|---:|---:|---:|
| Regularized variant ($\lambda_{\text{static}}=10^{-3}, \lambda_{\text{bin}}=10^{-3}$) | 0.3684 | 0.1295 | 0.0179 | 0.5809 | 0.2352 | 0.2159 |
| Regularized variant ($\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=10^{-3}$) | 0.5114 | 0.0520 | 0.6763 | 0.9613 | 0.1491 | 0.2472 |
| Regularized variant ($\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=2\times10^{-3}$) | 0.4679 | 0.1911 | 0.4817 | 0.8345 | 0.1540 | 0.2125 |

Increasing $\lambda_{\text{static}}$ from $10^{-3}$ to $2\times10^{-3}$ reduces static-weighted deformation from about $0.2352$ to about $0.1491$. The best reconstruction setting, $\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=10^{-3}$, has high dynamic fraction but a weakly binary mask: its binarization diagnostic is close to the maximum possible value of $0.25$. The stronger binarization setting, $\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=2\times10^{-3}$, produces a more spread-out mask distribution, but it hurts reconstruction. Therefore, Jumpingjacks also shows that mask separability and image quality are related but distinct objectives.

### 4.6 Main Results: HyperNeRF Chickchicken

HyperNeRF chickchicken is a more difficult real-world scene than the clean synthetic D-NeRF scenes. The runs below use the HyperNeRF default configuration, 14,000 fine iterations, and batch size 1.

Verified HyperNeRF chickchicken reconstruction metrics:

| Method | SSIM | PSNR | LPIPS-vgg | LPIPS-alex | MS-SSIM | D-SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (`bs1`) | **0.7968500** | **26.9153** | **0.3368862** | **0.1853653** | 0.9106787 | 0.0446607 |
| Regularized variant ($\lambda_{\text{static}}=10^{-3}, \lambda_{\text{bin}}=10^{-3}$, `bs1`) | 0.7967568 | 26.8727 | 0.3421589 | 0.1861793 | **0.9110526** | **0.0444737** |
| Regularized variant ($\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=10^{-3}$, `bs1`) | 0.7967100 | 26.9141 | 0.3440263 | 0.1903088 | 0.9106759 | 0.0446621 |

On this scene, motion-mask regularization does not improve the main reconstruction metrics. The baseline is slightly better on SSIM, PSNR, LPIPS-vgg, and LPIPS-alex. The $\lambda_{\text{static}}=10^{-3}$ variant slightly improves MS-SSIM and D-SSIM, but the difference is small.

However, the mask diagnostics show a clearer separation effect:

| Method | Final mean | Final std | Final dynamic fraction | Final fraction $m>0.4$ | Static deformation | Binarization |
|---|---:|---:|---:|---:|---:|---:|
| Regularized variant ($\lambda_{\text{static}}=10^{-3}, \lambda_{\text{bin}}=10^{-3}$, `bs1`) | 0.2056 | 0.2649 | 0.2190 | 0.3844 | 0.1120 | 0.0932 |
| Regularized variant ($\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=10^{-3}$, `bs1`) | 0.4152 | 0.4878 | 0.4188 | 0.4192 | 0.0050 | 0.0049 |

Increasing $\lambda_{\text{static}}$ from $10^{-3}$ to $2\times10^{-3}$ greatly reduces deformation in low-mask regions. The static-weighted deformation diagnostic drops from about $0.1120$ to about $0.0050$, and the binarization diagnostic drops from about $0.0932$ to about $0.0049$. Thus, on HyperNeRF chickchicken, $\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=10^{-3}$ is better for motion-mask interpretability, even though it is not better for image metrics.

### 4.7 Qualitative Summary

The final regularized method is most convincing on scenes with stronger motion. On Bouncingballs, the learned mask provides nontrivial soft motion localization on the moving balls, and a tradeoff appears between reconstruction quality and mask separability. On Jumpingjacks, the same regularization family improves reconstruction over baseline, but the stronger binarization setting does not generalize for image quality, even though it produces a more spread-out mask. On HyperNeRF chickchicken, reconstruction metrics remain close to baseline rather than improving, but the mask diagnostics show that the stronger static-deformation penalty produces a much cleaner decomposition. This suggests that the added motion-aware regularization is useful mainly as an interpretability and decomposition mechanism, while reconstruction gains are scene-dependent.

## 5. Discussion

### 5.1 When the Motion Mask Helps

The motion mask helps in two ways:

1. It adds interpretability by showing which Gaussians participate more strongly in motion.
2. With appropriate regularization, it can slightly improve reconstruction on strongly dynamic scenes.

### 5.2 Failure Cases

The main failure modes are:

1. **All-static collapse**, especially in early sparsity-based experiments.
2. **Soft ambiguous masks**, where the mask stays in the middle range.
3. **All-dynamic collapse**, when static-deformation regularization is too strong.

### 5.3 Cross-Scene Hyperparameter Finding

The new experiments support one practical conclusion. If a single setting must be selected across scenes, the most robust choice is

$$
\lambda_{\text{static}} = 2\times10^{-3}, \qquad
\lambda_{\text{bin}} = 10^{-3}.
$$

This choice remains strong on both Bouncingballs and Jumpingjacks, and it produces the cleanest tested HyperNeRF chickchicken mask diagnostics. In contrast, the stronger binarization setting

$$
\lambda_{\text{static}} = 2\times10^{-3}, \qquad
\lambda_{\text{bin}} = 2\times10^{-3}
$$

is attractive for Bouncingballs reconstruction alone, but does not generalize to Jumpingjacks.

### 5.4 Limitations vs. SDD-4DGS

Compared with SDD-4DGS, this project uses a simpler soft-gating implementation rather than a full per-Gaussian decoupling framework. This makes the method easier to integrate, but also less principled and less directly interpretable as a binary static/dynamic assignment.

## 6. Conclusion

This project introduces a motion-aware soft mask into an existing 4D Gaussian Splatting codebase. The final method uses the mask to gate deformation and regularizes it through binarization and static-deformation penalties. The earlier sparsity-only prototype was not successful and was not retained as part of the final method.

The verified experiments support the following conclusion: the final regularized motion-mask formulation can produce meaningful soft motion localization and modest reconstruction improvements on strongly dynamic D-NeRF scenes such as Bouncingballs and Jumpingjacks. On HyperNeRF chickchicken, the method does not improve reconstruction metrics, but the mask diagnostics show that $\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=10^{-3}$ produces a much cleaner decomposition than $\lambda_{\text{static}}=10^{-3}, \lambda_{\text{bin}}=10^{-3}$.

The experiments also show that best reconstruction and best mask separability are not always achieved by the same setting. Across the tested scenes, $\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=10^{-3}$ is the most robust overall choice for interpretable motion separation, while $\lambda_{\text{static}}=2\times10^{-3}, \lambda_{\text{bin}}=2\times10^{-3}$ is better interpreted as a scene-specific Bouncingballs reconstruction optimum. The current method should be presented as a soft, unsupervised motion-decomposition mechanism rather than a guaranteed reconstruction-quality improvement or a fully binary static/dynamic separator.
