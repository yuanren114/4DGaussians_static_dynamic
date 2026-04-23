# Motion-Aware Regularization for 4D Gaussian Splatting

## Abstract

This project studies static-dynamic separation in a modified 4D Gaussian Splatting (4DGS) codebase. The baseline 4DGS model represents a dynamic scene by storing Gaussian primitives in a canonical space and predicting time-dependent deformation before rasterization. However, the baseline does not explicitly indicate which Gaussians should remain static and which should participate in motion. To address this limitation, this project adds a lightweight motion-mask branch inside the deformation network. The mask is a learned scalar gate that modulates deformation strength and is optimized jointly with image reconstruction.

The final method is fully unsupervised with respect to motion labels: it does not use ground-truth static/dynamic masks or external segmentation during training. Instead, it combines the original reconstruction objective with two additional regularizers: a binarization loss and a static-deformation loss. Experiments on D-NeRF and HyperNeRF scenes show that the proposed method can produce a more interpretable motion decomposition and can modestly improve reconstruction on strongly dynamic synthetic scenes. However, the results also show a consistent tradeoff between reconstruction quality and mask separability, and they do not indicate universal gains on more challenging real-world data.

## 1. Introduction

Dynamic scene reconstruction extends static 3D reconstruction by adding the temporal dimension. Instead of modeling only geometry and appearance, a dynamic representation must also explain how the scene changes over time. NeRF-based methods address this problem with continuous volumetric functions, while 3D Gaussian Splatting (3DGS) replaces dense ray samples with explicit Gaussian primitives for faster rendering. 4D Gaussian Splatting (4DGS) further extends this representation to dynamic scenes by learning a time-dependent deformation field over canonical Gaussians.

Although baseline 4DGS can reconstruct dynamic scenes effectively, it does not explicitly separate static and dynamic components. In practice, the deformation network is free to move any Gaussian if doing so reduces image reconstruction loss. This entangles static structure and dynamic motion, which weakens interpretability and can produce ambiguous motion explanations.

An early direction for this project considered external segmentation-based masking using models such as GroundingDINO and SAM2 together with a 3D reconstruction pipeline. That direction was not adopted as the final method for three reasons. First, segmentation masks are external supervision and are not jointly optimized with the reconstruction model. Second, semantic masks are not necessarily motion-aware. Third, prompt-based segmentation can be temporally inconsistent across frames.

The final project therefore focuses on learning a motion-aware coefficient directly inside the 4DGS deformation pipeline.

## 2. Related Work

### 2.1 Baseline 4D Gaussian Splatting

In baseline 4DGS, each Gaussian is stored in a canonical space and then deformed at time \(t\). In simplified notation, the canonical Gaussian center is \(\mu_i^0 \in \mathbb{R}^3\), and the deformed center is

\[
\mu_i(t) = \mu_i^0 + \Delta \mu_i(t).
\]

The same idea is used for scale and rotation through predicted deformations \(\Delta s_i(t)\) and \(\Delta r_i(t)\). In this repository, these deformations are produced by a HexPlane-based deformation network and then passed to the rasterizer.

### 2.2 SDD-4DGS

SDD-4DGS proposes a static-dynamic decoupling framework in which each Gaussian is associated with a dynamic perception coefficient. The paper presents this coefficient as a mechanism for separating dynamic and static components during optimization. Relative to baseline 4DGS, the paper reports improved PSNR and SSIM on several benchmarks.

Two representative quantitative comparisons reported by the paper are summarized below.

| Benchmark | Baseline 4DGS | SDD-4DGS | Reported Change |
|---|---:|---:|---:|
| D-NeRF PSNR \(\uparrow\) | 34.14 | **34.82** | +0.68 |
| D-NeRF SSIM \(\uparrow\) | 0.94 | **0.96** | +0.02 |
| HyperNeRF mean PSNR \(\uparrow\) | 22.33 | **22.62** | +0.29 |
| HyperNeRF mean SSIM \(\uparrow\) | 0.765 | **0.772** | +0.007 |

These numbers indicate that SDD-4DGS is presented as an improvement over baseline 4DGS. However, the current project is not a reproduction of SDD-4DGS. The implementation here is simpler and uses a predicted soft motion gate rather than a persistent per-Gaussian dynamic parameter.

### 2.3 Segmentation-Driven Masking

Segmentation-driven approaches use external object masks to isolate parts of a scene. Such methods can help with semantic object extraction, but the masks may not align with physical motion and are usually outside the differentiable training loop. The final method in this project does not use segmentation masks during 4DGS training.

## 3. Baseline Pipeline and Notation

### 3.1 Core Variables

The following notation is used throughout the report.

| Symbol | Meaning | Code-level realization |
|---|---|---|
| \(i\) | Gaussian index | point index |
| \(t\) | frame timestamp | `time`, `time_emb`, `times_sel` |
| \(\mu_i^0\) | canonical Gaussian center | `pc.get_xyz`, `rays_pts_emb[:, :3]` |
| \(s_i^0\) | canonical Gaussian scale | `pc._scaling`, `scales_emb[:, :3]` |
| \(r_i^0\) | canonical Gaussian rotation | `pc._rotation`, `rotations_emb[:, :4]` |
| \(\Delta \mu_i(t)\) | predicted positional deformation | `dx` |
| \(\Delta s_i(t)\) | predicted scale deformation | `ds` |
| \(\Delta r_i(t)\) | predicted rotation deformation | `dr` |
| \(F_i(t)\) | HexPlane spatiotemporal feature | `grid_feature` |
| \(h_i(t)\) | shared hidden feature after projection | `hidden` |
| \(m_i(t)\) | learned motion mask in \([0,1]\) | `motion_mask` |

### 3.2 What HexPlane Is

HexPlane is not a multilayer perceptron (MLP). It is a learned factorized 4D feature field. The input is a 4D coordinate \([x,y,z,t]\), where \((x,y,z)\) is the Gaussian position and \(t\) is time. Instead of storing one dense 4D feature volume, HexPlane decomposes the 4D domain into six learned 2D planes:

\[
(x,y),\ (x,z),\ (x,t),\ (y,z),\ (y,t),\ (z,t).
\]

For each input coordinate, the implementation bilinearly samples all six planes, multiplies the resulting feature vectors within each level, and concatenates features across multiple resolution levels. The output is a spatiotemporal feature vector \(F_i(t)\), not another position-time coordinate. That feature is then processed by small MLP heads that predict deformation.

In the default D-NeRF configuration, the HexPlane field outputs \(64\) channels after multiresolution concatenation. In the default HyperNeRF configuration, it outputs \(48\) channels.

### 3.3 Baseline 4DGS Forward Path

The forward path implemented in this repository can be summarized as follows.

1. Start from canonical Gaussian attributes: position, scale, rotation, opacity, and spherical-harmonic color features.
2. Query the HexPlane field using canonical position and timestamp.
3. Project the queried HexPlane feature through a shared MLP block to obtain a hidden feature \(h_i(t)\).
4. Predict deformation heads for position, scale, and rotation.
5. Apply those deformations to obtain deformed Gaussian attributes at time \(t\).
6. Rasterize the deformed Gaussians into the target view.
7. Compute image reconstruction loss against the ground-truth image.

The relevant implementation is primarily in:

- `scene/deformation.py`
- `scene/hexplane.py`
- `gaussian_renderer/__init__.py`
- `scene/gaussian_model.py`
- `train.py`

## 4. Proposed Method

### 4.1 Motion Mask Formulation

The proposed method adds a new scalar branch to the deformation network:

\[
m_i(t) = \sigma(g_\phi(h_i(t))),
\]

where \(g_\phi(\cdot)\) is a small MLP head and \(\sigma(\cdot)\) is the sigmoid function. Therefore, \(m_i(t)\in[0,1]\) is a soft motion coefficient predicted from the same hidden deformation feature used by the original deformation heads.

The default positional update becomes

\[
\mu_i(t) = \mu_i^0 + m_i(t)\,\Delta \mu_i(t).
\]

In code, this is implemented as

```python
dx = self.pos_deform(hidden)
pts = rays_pts_emb[:, :3] + motion_mask * dx
```

This means that a Gaussian with low mask value receives little positional deformation, while a Gaussian with high mask value is allowed to move more.

### 4.2 Optional Gating of Scale and Rotation

The code also provides an optional flag `--motion-gate-rot-scale`. When enabled, the same motion mask gates scale and rotation deformation:

\[
s_i(t) = s_i^0 + m_i(t)\,\Delta s_i(t),
\]

\[
r_i(t) = r_i^0 + m_i(t)\,\Delta r_i(t).
\]

When this flag is disabled, scale and rotation are still deformed, but only positional deformation is explicitly motion-gated.

### 4.3 What the Method Does Not Change

The method does not introduce ground-truth motion supervision. It also does not add a new persistent Gaussian parameter stored directly as part of the Gaussian state. Instead, the motion coefficient is predicted on the fly from the deformation-network hidden feature.

This design choice keeps the implementation lightweight, but it also means that motion separation is learned indirectly through reconstruction and regularization rather than through direct static/dynamic labels.

### 4.4 Final Training Objective

The baseline training objective is image reconstruction, with optional grid regularization and DSSIM as already present in the codebase. The final method adds two motion-related losses:

\[
\mathcal{L}_{\text{bin}} = \frac{1}{N}\sum_i m_i(t)\bigl(1-m_i(t)\bigr),
\]

\[
\mathcal{L}_{\text{static}} = \frac{1}{N}\sum_i \bigl(1-m_i(t)\bigr)\,\|\Delta \mu_i(t)\|_2.
\]

The final objective is

\[
\mathcal{L}
=
\mathcal{L}_{\text{img}}
\;+\;
\mathcal{L}_{\text{baseline-reg}}
\;+\;
\lambda_{\text{bin}}\mathcal{L}_{\text{bin}}
\;+\;
\lambda_{\text{static}}\mathcal{L}_{\text{static}}.
\]

Here:

- \(\mathcal{L}_{\text{img}}\) is the baseline image loss,
- \(\mathcal{L}_{\text{baseline-reg}}\) denotes the original regularization terms already used by the codebase,
- \(\lambda_{\text{bin}}\) is the binarization weight,
- \(\lambda_{\text{static}}\) is the static-deformation weight.

An earlier prototype also tested a simple mask-mean penalty

\[
\mathcal{L}_{\text{mask}}=\frac{1}{N}\sum_i m_i(t),
\]

weighted by a separate coefficient in the early prototype. This term simply pushes the average mask value downward, so it acts as a crude sparsity prior on mask activation. However, it did not produce useful static-dynamic separation and is not part of the final method reported in this project.

### 4.5 Why These Losses Are Needed

Image reconstruction alone does not identify static and dynamic components. The main ambiguity is

\[
\mu_i(t)=\mu_i^0+m_i(t)\Delta\mu_i(t).
\]

If only the product matters for rendering, then different combinations can produce similar images:

- small \(m_i(t)\) and large \(\Delta \mu_i(t)\),
- large \(m_i(t)\) and small \(\Delta \mu_i(t)\).

Therefore, the model needs additional pressure if the mask is expected to become interpretable. The binarization loss discourages persistent mid-range mask values, and the static-deformation loss penalizes motion in regions whose mask is small.

## 5. Experimental Setup

### 5.1 Datasets

Experiments were run on:

- D-NeRF `bouncingballs`
- D-NeRF `jumpingjacks`
- HyperNeRF `chickchicken`

The D-NeRF experiments used the standard dynamic synthetic data layout already supported by the repository. The HyperNeRF experiment used the generated COLMAP-based data path and the HyperNeRF default config already present in the codebase.

### 5.2 Compared Variants

The most relevant variants are:

- **Baseline 4DGS**: no motion-mask regularization
- **Motion 1e-3 / 1e-3**: `static_deform_lambda = 1e-3`, `motion_bin_lambda = 1e-3`
- **Motion 2e-3 / 1e-3**: `static_deform_lambda = 2e-3`, `motion_bin_lambda = 1e-3`
- **Motion 2e-3 / 2e-3**: `static_deform_lambda = 2e-3`, `motion_bin_lambda = 2e-3`

An earlier unregularized or sparsity-only prototype is discussed only as a pilot failure case.

### 5.3 Reconstruction Metrics

The image quality metrics reported by the codebase are:

- **PSNR**: peak signal-to-noise ratio; higher is better
- **SSIM**: structural similarity index; higher is better
- **MS-SSIM**: multi-scale SSIM; higher is better
- **LPIPS-VGG** and **LPIPS-Alex**: perceptual distance metrics; lower is better
- **D-SSIM**: dissimilarity derived from SSIM; lower is better

### 5.4 Motion-Mask Diagnostic Metrics

Because there is no ground-truth dynamic mask, the project also uses internal diagnostics logged in `motion_mask_stats.jsonl`.

Let \(m_i(t)\) be the mask and \(\Delta \mu_i(t)\) the predicted positional deformation. Then:

- **Mean**: \(\frac{1}{N}\sum_i m_i(t)\)
- **Std**: standard deviation of the mask values
- **Dynamic fraction**: fraction of Gaussians satisfying \(m_i(t) > 0.5\)
- **Fraction \(m>0.4\)**: a softer indicator of moderately active motion regions
- **Static deformation**:
  \[
  \frac{1}{N}\sum_i (1-m_i(t))\|\Delta \mu_i(t)\|_2
  \]
  Lower values indicate that low-mask regions are moving less.
- **Binarization**:
  \[
  \frac{1}{N}\sum_i m_i(t)(1-m_i(t))
  \]
  This quantity lies in \([0,0.25]\). Lower values indicate that mask values are closer to \(0\) or \(1\), while higher values indicate softer, more ambiguous masks.

These diagnostics do not measure ground-truth motion accuracy. They only quantify the internal behavior of the learned motion mask.

## 6. Results

### 6.1 Early Pilot Observation

The earliest motion-mask prototype used only the mask-mean penalty

\[
\mathcal{L}_{\text{mask}}=\frac{1}{N}\sum_i m_i(t),
\]

without the later binarization and static-deformation terms. This design did not produce meaningful static-dynamic separation. On both slow-motion and strong-motion scenes, some runs kept the mask soft with near-zero dynamic fraction, while other runs under stronger regularization collapsed toward a trivial all-dynamic solution. This pilot result motivated the final method based on binarization and static-deformation regularization.

### 6.2 D-NeRF Bouncingballs

#### Reconstruction Metrics

| Method | SSIM \(\uparrow\) | PSNR \(\uparrow\) | LPIPS-VGG \(\downarrow\) | LPIPS-Alex \(\downarrow\) | MS-SSIM \(\uparrow\) | D-SSIM \(\downarrow\) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 4DGS | 0.9942868 | 40.6763 | 0.0153625 | 0.0060306 | 0.9953953 | 0.0023024 |
| Motion 1e-3 / 1e-3 | 0.9945415 | 40.8666 | 0.0143814 | 0.0056324 | 0.9955825 | 0.0022088 |
| Motion 2e-3 / 1e-3 | 0.9944983 | **40.9611** | 0.0144220 | 0.0052267 | 0.9956228 | 0.0021886 |
| Motion 2e-3 / 2e-3 | **0.9947041** | 40.8955 | **0.0136226** | **0.0050899** | **0.9956791** | **0.0021604** |
| Motion 1e-2 / 1e-3 | 0.9942789 | 40.7233 | 0.0155924 | 0.0059685 | 0.9953101 | 0.0023450 |

#### Motion-Mask Diagnostics

| Method | Mean | Std | Dynamic Fraction | Fraction \(m>0.4\) | Static Deformation | Binarization |
|---|---:|---:|---:|---:|---:|---:|
| Early unregularized pilot | 0.2475 | 0.0641 | 0.0001 | not logged | not logged | not logged |
| Motion 1e-3 / 1e-3 | 0.1851 | 0.1904 | 0.0117 | 0.2162 | 0.0934 | 0.1146 |
| Motion 2e-3 / 1e-3 | 0.2179 | 0.2487 | 0.2604 | 0.3879 | 0.0523 | 0.1086 |
| Motion 2e-3 / 2e-3 | 0.1400 | 0.2137 | 0.0487 | 0.2615 | 0.0499 | 0.0747 |
| Motion 1e-2 / 1e-3 | 0.9986 | 0.0022 | 1.0000 | 1.0000 | 0.0002 | 0.0014 |

#### Interpretation

Three conclusions are important.

First, the final regularized method improves reconstruction over baseline on this scene. The PSNR gain is approximately \(+0.19\) to \(+0.28\) dB depending on the setting, and perceptual metrics also improve.

Second, the best reconstruction and the best motion separation are not achieved by the same setting. The configuration `2e-3 / 2e-3` yields the best image metrics overall, but `2e-3 / 1e-3` yields the most interpretable mask statistics among the non-collapsed runs, including the highest dynamic fraction and the largest fraction above \(m>0.4\).

Third, excessive static regularization can collapse the model into an almost all-dynamic solution. The `1e-2 / 1e-3` run shows this failure mode clearly.

### 6.3 D-NeRF Jumpingjacks

#### Reconstruction Metrics

| Method | SSIM \(\uparrow\) | PSNR \(\uparrow\) | LPIPS-VGG \(\downarrow\) | LPIPS-Alex \(\downarrow\) | MS-SSIM \(\uparrow\) | D-SSIM \(\downarrow\) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 4DGS | 0.9855952 | 35.4000 | 0.0199500 | 0.0126626 | 0.9936216 | 0.0031892 |
| Motion 1e-3 / 1e-3 | **0.9863845** | 35.5784 | 0.0189655 | **0.0123328** | **0.9940146** | **0.0029927** |
| Motion 2e-3 / 1e-3 | 0.9863592 | **35.6238** | **0.0189545** | 0.0126150 | 0.9939969 | 0.0030016 |
| Motion 2e-3 / 2e-3 | 0.9858798 | 35.3964 | 0.0198835 | 0.0131377 | 0.9936382 | 0.0031809 |

#### Motion-Mask Diagnostics

| Method | Mean | Std | Dynamic Fraction | Fraction \(m>0.4\) | Static Deformation | Binarization |
|---|---:|---:|---:|---:|---:|---:|
| Motion 1e-3 / 1e-3 | 0.3684 | 0.1295 | 0.0179 | 0.5809 | 0.2352 | 0.2159 |
| Motion 2e-3 / 1e-3 | 0.5114 | 0.0520 | 0.6763 | 0.9613 | 0.1491 | 0.2472 |
| Motion 2e-3 / 2e-3 | 0.4679 | 0.1911 | 0.4817 | 0.8345 | 0.1540 | 0.2125 |

#### Interpretation

On Jumpingjacks, the regularized motion-mask variants again improve reconstruction over baseline. The configuration `2e-3 / 1e-3` gives the best PSNR and a strong SSIM improvement, while `2e-3 / 2e-3` slightly hurts image metrics.

The mask diagnostics show a different pattern from Bouncingballs. Increasing \(\lambda_{\text{static}}\) from \(1e{-3}\) to \(2e{-3}\) substantially reduces low-mask motion, as static deformation decreases from \(0.2352\) to \(0.1491\). However, the best reconstruction setting also has a binarization value of \(0.2472\), which is close to the theoretical maximum of \(0.25\). In other words, the best-performing Jumpingjacks run still uses a relatively soft mask. This indicates that good reconstruction does not require a near-binary decomposition.

### 6.4 HyperNeRF Chickchicken

#### Reconstruction Metrics

| Method | SSIM \(\uparrow\) | PSNR \(\uparrow\) | LPIPS-VGG \(\downarrow\) | LPIPS-Alex \(\downarrow\) | MS-SSIM \(\uparrow\) | D-SSIM \(\downarrow\) |
|---|---:|---:|---:|---:|---:|---:|
| Baseline 4DGS | **0.7968500** | **26.9152870** | **0.3368862** | **0.1853653** | 0.9106787 | 0.0446607 |
| Motion 1e-3 / 1e-3 | 0.7967568 | 26.8726616 | 0.3421589 | 0.1861793 | **0.9110526** | **0.0444737** |
| Motion 2e-3 / 1e-3 | 0.7967100 | 26.9141216 | 0.3440263 | 0.1903088 | 0.9106759 | 0.0446621 |

#### Motion-Mask Diagnostics

| Method | Mean | Std | Dynamic Fraction | Fraction \(m>0.4\) | Static Deformation | Binarization |
|---|---:|---:|---:|---:|---:|---:|
| Motion 1e-3 / 1e-3 | 0.2056 | 0.2649 | 0.2190 | 0.3844 | 0.1120 | 0.0932 |
| Motion 2e-3 / 1e-3 | 0.4152 | 0.4878 | 0.4188 | 0.4192 | 0.0050 | 0.0049 |

#### Interpretation

Unlike the D-NeRF scenes, HyperNeRF Chickchicken does not show reconstruction improvement from the motion-mask method. Baseline 4DGS remains slightly better on the main image metrics, and the gap is small.

Nevertheless, the motion-mask diagnostics reveal a meaningful internal difference. Increasing the static-deformation weight from \(1e{-3}\) to \(2e{-3}\) reduces the static-deformation metric from \(0.1120\) to \(0.0050\), while the binarization diagnostic also drops from \(0.0932\) to \(0.0049\). This indicates that the stronger static-deformation penalty successfully suppresses motion in low-mask regions and produces a much cleaner decomposition, even though the image metrics remain almost unchanged.

### 6.5 Overall Quantitative Summary

Relative to baseline 4DGS, the current method provides modest but real reconstruction gains on strongly dynamic synthetic scenes:

- about \(+0.22\) to \(+0.28\) dB PSNR on Bouncingballs,
- about \(+0.22\) dB PSNR on Jumpingjacks,
- approximately no PSNR gain on HyperNeRF Chickchicken.

Therefore, the method should not be described as a universal reconstruction improvement. A more accurate summary is that it improves or preserves reconstruction on some scenes while also providing a motion-aware decomposition signal whose quality depends strongly on regularization.

## 7. Discussion

### 7.1 What the Method Improves

The main technical contribution of this project is not simply a higher benchmark score. The contribution is a lightweight mechanism for injecting motion awareness into the 4DGS deformation pipeline without redesigning the entire Gaussian state.

The method has three practical strengths.

First, it is easy to integrate into an existing 4DGS codebase because the mask is implemented as an additional deformation head rather than as a new persistent Gaussian parameter.

Second, it can improve image reconstruction on strongly dynamic synthetic scenes.

Third, even when image quality does not improve, the method can still produce cleaner motion diagnostics, as seen on HyperNeRF Chickchicken.

### 7.2 Why Reconstruction and Mask Quality Diverge

The experiments repeatedly show that the best reconstruction setting is not always the setting with the cleanest motion mask. This is expected from the underlying ambiguity in

\[
\mu_i(t)=\mu_i^0+m_i(t)\Delta\mu_i(t).
\]

The rasterizer mainly observes the final deformed Gaussian. It does not directly care whether the deformation was explained by a large mask and small displacement or a smaller mask and larger displacement. Therefore, the regularizers shape interpretability, but they also alter the optimization landscape. As a result, stronger regularization can improve decomposition while leaving reconstruction unchanged, or improve reconstruction while leaving the mask relatively soft.

### 7.3 Comparison with SDD-4DGS

From the reported paper numbers, SDD-4DGS shows larger reconstruction gains over baseline 4DGS than the current lightweight method. For example, the paper reports a \(+0.68\) dB PSNR gain on D-NeRF relative to baseline 4DGS, whereas the current implementation produces gains closer to \(+0.22\) to \(+0.28\) dB on the tested D-NeRF scenes. The paper also reports consistent HyperNeRF improvements, while the current HyperNeRF result mainly improves mask diagnostics rather than reconstruction.

This comparison does not invalidate the present method. The two approaches have different design goals. SDD-4DGS is a fuller decoupling framework, while the current project intentionally pursues a smaller code modification that can be inserted into an existing 4DGS implementation with limited structural change.

### 7.4 Limitations

Several limitations remain.

1. The motion mask is still unsupervised and therefore not guaranteed to align with semantic motion.
2. The mask is predicted from hidden features and is not a persistent Gaussian-level state variable.
3. The position-gated formulation still leaves some ambiguity between mask magnitude and deformation magnitude.
4. The method does not uniformly improve real-world reconstruction quality.
5. The mask diagnostics are internal indicators rather than ground-truth motion accuracy metrics.

## 8. Conclusion

This project introduced a motion-aware soft-gating mechanism into an existing 4D Gaussian Splatting codebase. The implemented method predicts a motion mask from the deformation-network hidden feature, uses that mask to gate deformation, and regularizes the resulting behavior with binarization and static-deformation losses. The method requires no ground-truth motion labels and no external segmentation.

The experimental results support three main conclusions.

First, the final regularized motion-mask method improves reconstruction on strongly dynamic D-NeRF scenes, but the gains are modest rather than dramatic. Second, the method produces useful motion-separation behavior, especially when the static-deformation penalty is tuned appropriately. Third, reconstruction quality and motion-mask interpretability are different objectives, and the best setting for one is not always the best setting for the other.

Overall, the project demonstrates that a lightweight motion-aware modification can be integrated into 4DGS with limited code changes and can yield a more interpretable dynamic representation. At the same time, the results also show that meaningful static-dynamic separation remains an underconstrained problem, especially on complex real-world scenes.
