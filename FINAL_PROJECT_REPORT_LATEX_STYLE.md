# Motion-Aware Static-Dynamic Separation in 4D Gaussian Splatting

## Abstract

This project studies motion-aware reconstruction for dynamic scenes using a modified 4D Gaussian Splatting (4DGS) codebase. The baseline 4DGS formulation models scene dynamics through a deformation network that maps canonical Gaussians to time-dependent positions, scales, and rotations. While this can reconstruct dynamic scenes, the learned deformation field does not explicitly indicate which Gaussians correspond to static structure and which participate in motion.

We implemented a lightweight motion-mask extension that predicts a scalar soft motion coefficient from the deformation-network feature and uses it to gate time-dependent Gaussian deformation. The implementation is unsupervised: it uses image reconstruction loss and optional regularization terms, but no ground-truth motion masks. The project also includes experiments on D-NeRF scenes, including Lego and Bouncingballs. Verified results show that the proposed mask does not substantially improve image reconstruction metrics on Lego, but the regularized variant produces more meaningful soft motion localization on Bouncingballs than the naive mask gate. The final method should therefore be interpreted as a motion-aware diagnostic and soft separation mechanism, not a complete supervised object-level dynamic segmentation method.

## 1. Introduction

Reconstructing a static 3D scene from posed images is a central problem in computer vision and graphics. Neural Radiance Fields (NeRF) introduced a continuous neural representation that synthesizes novel views through volumetric rendering. 3D Gaussian Splatting (3DGS) later replaced slow volumetric sampling with an explicit set of anisotropic 3D Gaussians, enabling high-quality real-time rendering. For dynamic scenes, 4D Gaussian Splatting extends this idea by introducing time-dependent deformation, allowing Gaussians in a canonical space to move and change over time.

Dynamic reconstruction is more difficult than static reconstruction because scene appearance, geometry, and motion are entangled. A deformation model can explain frame-to-frame changes, but it does not necessarily reveal which components are actually static and which are dynamic. This matters because many real scenes contain both stable background structure and moving foreground objects. Treating all Gaussians as equally deformable can reduce interpretability and may allow dynamic motion to contaminate otherwise static regions.

At the beginning of the project, an alternative direction was considered: using external segmentation models such as GroundingDINO and SAM2 to produce object masks, then using those masks with an Instant-NGP-style 3D reconstruction pipeline. That approach has several limitations. It depends strongly on text prompts or detector quality, mask predictions may be inconsistent across time, and segmentation masks are not necessarily motion-aware. A segmented object can be static, and a moving region may be missed if it is not captured by the prompt. In addition, external mask generation is not naturally differentiable inside the reconstruction optimization.

The final direction of this project therefore focuses on integrating a motion-aware mask directly into the 4DGS deformation pipeline. The goal is not to use external mask supervision, but to learn a latent soft motion coefficient during reconstruction.

## 2. Related Work

### 2.1 4D Gaussian Splatting

4D Gaussian Splatting models a dynamic scene by maintaining Gaussians in a canonical space and applying a deformation network conditioned on time. In a simplified form, a canonical Gaussian position $\mu_0$ is transformed into a time-dependent position

$$
\mu_t = \mu_0 + \Delta \mu_t.
$$

The codebase used in this project follows this deformation-based structure. In `gaussian_renderer/__init__.py`, the renderer calls `pc._deformation(...)` during the fine stage to obtain time-dependent Gaussian means, scales, rotations, opacities, and SH features before rasterization. The deformation network is implemented in `scene/deformation.py` and predicts position, scale, rotation, opacity, and SH deltas through separate heads. In the default D-NeRF configuration used here, opacity and SH deformation are disabled by `no_do=True` and `no_dshs=True`.

This baseline formulation is effective for dynamic reconstruction, but the motion is implicit. The deformation network can move Gaussians without assigning a persistent semantic meaning such as "static" or "dynamic" to each Gaussian.

### 2.2 SDD-4DGS: Static-Dynamic Decoupling

SDD-4DGS, "Static-Dynamic Aware Decoupling in Gaussian Splatting for 4D Scene Reconstruction," proposes a static-dynamic decoupled reconstruction framework for 4D Gaussian Splatting. The public paper page describes the method as using a probabilistic dynamic perception coefficient integrated into the Gaussian reconstruction pipeline to separate static and dynamic components. In the formulation discussed in the project notes, this coefficient is associated with each Gaussian and gates time-dependent deformation, conceptually resembling

$$
\mu'_t = \mu_0 + w \Delta \mu_t,
$$

and similarly for covariance or geometric deformation terms. The paper further frames the coefficient probabilistically and encourages a single static or dynamic state through binary-style regularization and threshold-based separation.

The implementation in this project is related to that idea but is not a reproduction of SDD-4DGS. The main differences are detailed in Section 3.2.

### 2.3 Segmentation-Driven Mask-Based Methods

Another family of methods uses external masks from segmentation or detection models to guide reconstruction. Examples include object-aware NeRF pipelines or workflows that use foundation models to segment foreground objects before reconstruction. These methods can provide object-level supervision, but their quality depends on the segmentation model, prompts, and temporal consistency of masks. They also do not necessarily detect motion; they detect semantic or prompted objects.

This project does not use segmentation masks during 4DGS training. The motion mask is a latent variable predicted by the deformation network and optimized through reconstruction and regularization losses.

## 3. Method

### 3.1 Overview

The goal is to introduce motion-aware separation into 4DGS while preserving the original reconstruction pipeline. The key idea is to add a scalar motion mask to the deformation network. This mask modulates how strongly a Gaussian participates in time-dependent deformation.

The implementation is intentionally lightweight:

- The baseline 4DGS deformation pipeline remains unchanged when `--motion-separation` is not enabled.
- When `--motion-separation` is enabled, a mask head is added to the deformation network.
- The mask is used as a soft coefficient in the deformation equation.
- Optional regularizers are added to reduce trivial soft-mask behavior.

The method is unsupervised with respect to motion masks. The only ground truth used by the reconstruction pipeline is the image supervision already present in 4DGS.

### 3.2 Difference from SDD-4DGS

This section is important because the current implementation is inspired by static-dynamic decoupling ideas but differs substantially from SDD-4DGS.

#### Representation

SDD-4DGS describes a dynamic perception coefficient associated with each Gaussian ellipsoid. It can be interpreted as a per-Gaussian coefficient that indicates whether the Gaussian belongs to the dynamic or static component.

In this project, the mask is not stored as a persistent learnable parameter on each Gaussian. Instead, it is predicted by a neural mask head in `scene/deformation.py`:

```python
motion_mask = torch.sigmoid(self.motion_mask_deform(hidden))
```

where `hidden` is the shared spatiotemporal deformation feature. Thus the implemented mask is better described as

$$
m_i(t) = \sigma(g_\phi(h_i(t))),
$$

not as a fixed per-Gaussian parameter $w_i$. This makes the mask a network-predicted soft motion response rather than a persistent Gaussian identity.

#### Training

SDD-4DGS is described as a decoupling framework with a probabilistic coefficient and binary-style regularization. The public description also mentions optimization strategies for static-dynamic separation.

The implemented method uses the original 4DGS reconstruction loop in `train.py`. The baseline loss is image reconstruction:

$$
\mathcal{L}_{\text{render}} = \| \hat{I} - I \|_1,
$$

with fine-stage grid regularization from the original 4DGS implementation. The motion-mask extension adds optional terms:

$$
\mathcal{L}_{\text{sparse}} = \operatorname{mean}(m),
$$

$$
\mathcal{L}_{\text{bin}} = \operatorname{mean}(m(1-m)),
$$

$$
\mathcal{L}_{\text{static-def}} = \operatorname{mean}((1-m)\|\Delta x\|_2).
$$

The full implemented objective is therefore

$$
\mathcal{L}
= \mathcal{L}_{\text{render}}
+ \mathcal{L}_{\text{grid}}
+ \lambda_{\text{mask}}\mathcal{L}_{\text{sparse}}
+ \lambda_{\text{bin}}\mathcal{L}_{\text{bin}}
+ \lambda_{\text{static}}\mathcal{L}_{\text{static-def}},
$$

where the last three terms are enabled only by their corresponding command-line weights.

#### Supervision

Both SDD-4DGS and this project avoid external ground-truth static/dynamic masks in the core reconstruction setup. However, this implementation does not include the full self-supervised decoupling strategy described by SDD-4DGS. It only uses reconstruction loss and the explicitly implemented regularization terms listed above.

#### Advantages and Limitations

The advantage of this implementation is that it is simple and minimally invasive. Since the motion mask is predicted by a network head, the code does not need to add a new persistent parameter to every Gaussian or modify densification and pruning logic to copy and delete that parameter. It also allows the mask to depend on time and local deformation features.

The limitation is weaker identifiability. A network-predicted soft mask is not guaranteed to become a binary static/dynamic label. Without careful regularization, the mask can collapse to all static, all dynamic, or remain soft and ambiguous.

### 3.3 Motion Mask Formulation

The implemented motion mask is a scalar soft coefficient in $[0,1]$. It is not a binary label during training. It is computed in `Deformation.forward_dynamic()` from the shared deformation feature:

$$
m_i(t) = \sigma(g_\phi(h_i(t))).
$$

For position deformation, the implementation uses

$$
x_i(t) = x_i^0 + m_i(t)\Delta x_i(t).
$$

In code:

```python
dx = self.pos_deform(hidden)
pts = rays_pts_emb[:, :3] + motion_mask * dx
```

Scale and rotation are gated only if `--motion-gate-rot-scale` is enabled:

$$
s_i(t) = s_i^0 + m_i(t)\Delta s_i(t),
$$

$$
r_i(t) = r_i^0 + m_i(t)\Delta r_i(t).
$$

Otherwise, scale and rotation deformation remain ungated. The mask does not directly gate opacity. The D-NeRF default configuration already disables opacity deformation through `no_do=True`, and SH deformation through `no_dshs=True`.

The mask evolves during training through gradients from the rendered image loss and any enabled regularization losses. It is not updated by a ground-truth mask. The code stores the most recent mask in `self.last_motion_mask`, which is then used for logging and diagnostics. This is a snapshot of the latest forward pass, not a time-averaged per-Gaussian label.

### 3.4 Integration into the 4DGS Pipeline

The implemented pipeline follows the original 4DGS structure with an additional motion-mask branch.

#### Step 1: Gaussian Initialization

`scene/__init__.py` loads the dataset and initializes Gaussians from the scene point cloud when no trained model is loaded. Gaussian parameters are stored in `scene/gaussian_model.py`, including positions `_xyz`, SH features, scale `_scaling`, rotation `_rotation`, opacity `_opacity`, and the deformation network `_deformation`.

#### Step 2: Deformation Modeling

During fine-stage rendering, `gaussian_renderer/__init__.py` passes canonical Gaussian parameters and time to the deformation network:

```python
means3D_final, scales_final, rotations_final, opacity_final, shs_final = pc._deformation(
    means3D, scales, rotations, opacity, shs, time
)
```

The deformation network computes a shared spatiotemporal feature and predicts deformation deltas.

#### Step 3: Mask Computation and Update

When `--motion-separation` is enabled, `Deformation.create_net()` constructs the mask head:

```python
self.motion_mask_deform = nn.Sequential(
    nn.ReLU(),
    nn.Linear(self.W, self.W),
    nn.ReLU(),
    nn.Linear(self.W, 1)
)
```

The mask is computed by applying a sigmoid to the output of this head. It is saved as `last_motion_mask` for diagnostic losses and logging.

#### Step 4: Rendering

The renderer receives the deformed Gaussian parameters and rasterizes them using the existing Gaussian rasterization path. The motion mask is not rendered as an image. It only affects rendering indirectly through deformation.

#### Step 5: Loss Computation

The main reconstruction loss is the image-space $L_1$ loss. During the fine stage, the original grid regularization is also applied. If enabled, the motion-mask losses are added:

- `--motion-mask-lambda`: encourages small masks via $\operatorname{mean}(m)$.
- `--motion-bin-lambda`: encourages binary masks via $\operatorname{mean}(m(1-m))$.
- `--static-deform-lambda`: penalizes deformation in low-mask regions via $\operatorname{mean}((1-m)\|\Delta x\|_2)$.

The outputs `motion_mask_stats.jsonl`, `motion_mask_last.pt`, and `motion_mask_colors.ply` provide diagnostics. In the PLY visualization, red corresponds to high mask values and blue corresponds to low mask values.

### 3.5 Differences from the Segmentation-Based Approach

The earlier GroundingDINO + SAM2 + Instant-NGP direction differs from the final method in several ways:

1. It relies on external segmentation or prompt-based detection.
2. Mask consistency across time is not guaranteed.
3. The masks are semantic or object-driven, not necessarily motion-driven.
4. The masks are not produced by the differentiable reconstruction model.
5. It operates outside the current 4DGS deformation optimization loop.

The implemented 4DGS motion mask avoids those dependencies by learning a latent motion response inside the deformation model. However, because it lacks external supervision, it requires regularization and still does not guarantee clean object-level separation.

## 4. Experiments

### 4.1 Setup

Experiments were run on D-NeRF scenes available in the local project structure, including Lego and Bouncingballs. The configuration files are located in `arguments/dnerf/`. The D-NeRF default configuration sets:

- `iterations = 20000`
- `coarse_iterations = 3000`
- `defor_depth = 0`
- `net_width = 64`
- `multires = [1, 2]`
- fine-stage grid regularization terms inherited from the original 4DGS configuration

The following types of runs were used:

1. Baseline 4DGS without motion separation.
2. Naive soft motion gate with `--motion-separation`.
3. Motion gate without sparsity regularization.
4. Regularized motion gate with `--motion-gate-rot-scale`, `--static-deform-lambda`, and `--motion-bin-lambda`.

The regularized Bouncingballs commands included:

```bash
python train.py -s data/dnerf/bouncingballs --model_path output/dnerf/bouncingballs_motion_fixed_static1e-3_bin1e-3 --port 6021 --expname "bouncingballs_motion_fixed_static1e-3_bin1e-3" --configs arguments/dnerf/bouncingballs.py --motion-separation --motion-gate-rot-scale --motion-mask-lambda 0 --static-deform-lambda 0.001 --motion-bin-lambda 0.001
```

and a stronger static-deformation variant:

```bash
python train.py -s data/dnerf/bouncingballs --model_path output/dnerf/bouncingballs_motion_fixed_static1e-2_bin1e-3 --port 6022 --expname "bouncingballs_motion_fixed_static1e-2_bin1e-3" --configs arguments/dnerf/bouncingballs.py --motion-separation --motion-gate-rot-scale --motion-mask-lambda 0 --static-deform-lambda 0.01 --motion-bin-lambda 0.001
```

### 4.2 Metrics

The available evaluation script `metrics.py` computes:

- PSNR
- SSIM
- LPIPS-vgg
- LPIPS-alex
- MS-SSIM
- D-SSIM

The current code does not compute RMSE in the standard evaluation script. Formal training time was not logged in a way that supports a verified comparison, so training-time numbers are not reported.

Motion-mask diagnostics are taken from `motion_mask_stats.jsonl`:

- mean mask value
- standard deviation
- `dynamic_fraction = mean(mask > 0.5)`
- `static_fraction = mean(mask <= 0.5)`
- additional fractions above thresholds 0.1, 0.2, 0.3, and 0.4 for newer runs
- deformation-norm diagnostics for the regularized variant

### 4.3 Verified Quantitative Results

The following Lego reconstruction metrics were read from local `results.json` files. Higher is better for PSNR, SSIM, and MS-SSIM. Lower is better for LPIPS and D-SSIM.

| Method | SSIM | PSNR | LPIPS-vgg | LPIPS-alex | MS-SSIM | D-SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Lego baseline | 0.9376 | 25.0273 | 0.0563 | 0.0381 | 0.9533 | 0.0234 |
| Lego motion, $\lambda_{\text{mask}}=0.001$ | 0.9367 | 25.0465 | 0.0590 | 0.0406 | 0.9531 | 0.0235 |
| Lego motion, no sparsity | 0.9367 | 25.0325 | 0.0584 | 0.0398 | 0.9532 | 0.0234 |

These results show no meaningful reconstruction improvement from the motion mask on Lego. The sparse-mask version gives a small PSNR increase, but it is accompanied by worse SSIM and LPIPS. The no-sparsity version is also close to baseline but does not clearly outperform it.

For Bouncingballs, reconstruction metrics for the fixed variants were not found in local `results.json` files at the time of writing. Therefore, only verified mask diagnostics are reported:

| Method | Final mean | Final std | Final dynamic fraction | Final fraction $m>0.4$ | Qualitative mask observation |
|---|---:|---:|---:|---:|---|
| Bouncingballs motion no sparsity | 0.2475 | 0.0641 | 0.0001 | not logged | soft but nearly uniform mask |
| Bouncingballs regularized, $\lambda_{\text{static}}=10^{-3}$, $\lambda_{\text{bin}}=10^{-3}$ | 0.1851 | 0.1904 | 0.0117 | 0.2162 | soft localization; bouncing balls show purple regions |
| Bouncingballs regularized, $\lambda_{\text{static}}=10^{-2}$, $\lambda_{\text{bin}}=10^{-3}$ | 0.9986 | 0.0022 | 1.0000 | 1.0000 | all-dynamic collapse; point cloud is almost entirely red |

The Bouncingballs results are useful diagnostically. The naive no-sparsity mask remains soft and nearly uniform. The regularized $10^{-3}$ static-deformation setting produces a wider mask distribution and visible soft localization on the moving balls. The stronger $10^{-2}$ setting collapses to all dynamic, showing that the static-deformation penalty must be balanced.

### 4.4 Qualitative Observations

The rendered results for the $10^{-3}$ and $10^{-2}$ Bouncingballs regularized variants appear similar by visual reconstruction quality. However, their learned masks are very different:

- $\lambda_{\text{static}}=10^{-2}$ produces an almost entirely red PLY, corresponding to mask values near 1 everywhere.
- $\lambda_{\text{static}}=10^{-3}$ produces many purple points on the bouncing balls, corresponding to intermediate mask values.

This demonstrates that reconstruction quality alone is insufficient to evaluate motion separation. Two models can render similarly while learning very different internal decompositions.

## 5. Discussion

### 5.1 When the Motion Mask Helps

The motion mask is most useful as an interpretability and diagnostic tool. In Bouncingballs, the regularized variant produces a non-uniform mask distribution and qualitatively highlights moving ball regions with intermediate mask values. This suggests that the model learns a motion-aware response even without explicit mask supervision.

The mask can also support future downstream tasks, such as extracting high-motion Gaussians, visualizing deformation participation, or designing stronger static/dynamic constraints.

### 5.2 Failure Cases

Several failure modes were observed:

1. **All-static collapse.** With a positive `motion_mask_lambda`, the loss $\operatorname{mean}(m)$ can push the mask toward zero.
2. **Soft-mask ambiguity.** Without sufficient regularization, the mask may stay in a soft range and never become binary.
3. **All-dynamic collapse.** A strong static-deformation penalty can make the easiest solution $m \rightarrow 1$ everywhere.
4. **Scale ambiguity.** Since position uses $x_t=x_0+m\Delta x_t$, low $m$ and large $\Delta x_t$ can produce a similar deformation to high $m$ and small $\Delta x_t$.
5. **Snapshot logging.** `last_motion_mask` records the latest forward pass, not a fully time-aggregated dynamic identity.

### 5.3 Limitations Compared with SDD-4DGS

Compared with SDD-4DGS, the current method is simpler but less principled. It does not implement a persistent per-Gaussian Bernoulli coefficient, does not perform formal static/dynamic Gaussian set optimization, and does not include the full self-supervised decoupling strategy described by SDD-4DGS. The mask is a network-predicted soft gate rather than a stable Gaussian-level identity.

The advantage is ease of integration. The code does not need to change Gaussian storage, densification, pruning, or optimizer-state handling for a new per-Gaussian parameter. The limitation is that identifiability is weaker, and the learned mask requires careful interpretation.

### 5.4 What the Current Results Support

The verified results support the following conservative conclusions:

- The motion mask does not improve Lego reconstruction quality in a meaningful way.
- The naive motion mask can collapse or remain soft.
- The regularized motion mask can produce more structured soft localization on Bouncingballs.
- Stronger regularization is not automatically better; it can cause all-dynamic collapse.
- Motion-mask quality must be evaluated separately from image reconstruction quality.

The results do not support a claim of fully successful binary static/dynamic separation. They support a claim of partial, soft motion-aware localization.

## 6. Conclusion

This project modified a 4D Gaussian Splatting codebase to include a learned motion mask for static-dynamic analysis. The implemented mask is a scalar soft gate predicted by the deformation network. It modulates time-dependent deformation and can be regularized through sparsity, binarization, and static-deformation penalties.

The method is not a direct reproduction of SDD-4DGS. It is a lightweight soft-gating variant that avoids external segmentation supervision and avoids adding persistent per-Gaussian coefficients. This makes the method easy to integrate into the existing 4DGS pipeline, but it also makes the mask less identifiable.

Experiments show that reconstruction quality can remain similar across different mask behaviors. On Lego, the motion mask does not clearly improve reconstruction metrics. On Bouncingballs, the regularized $10^{-3}$ static-deformation setting produces a more meaningful soft mask distribution than the naive version, while the stronger $10^{-2}$ setting collapses to all dynamic. These observations suggest that motion-aware regularization can reveal useful motion saliency, but additional work is required to achieve robust binary static/dynamic decomposition.

Future work should include time-aggregated mask estimation, better balance priors, stronger deformation-mask coupling, and a more principled comparison to per-Gaussian coefficient methods such as SDD-4DGS.

## References

- B. Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis."
- B. Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering."
- G. Wu et al., "4D Gaussian Splatting for Real-Time Dynamic Scene Rendering."
- D. Sun, H. Guan, K. Zhang, X. Xie, and S. K. Zhou, "SDD-4DGS: Static-Dynamic Aware Decoupling in Gaussian Splatting for 4D Scene Reconstruction," arXiv:2503.09332, 2025.
