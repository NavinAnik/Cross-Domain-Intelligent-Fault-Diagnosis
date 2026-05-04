# Technical architecture specification

This document provides **exact, code-verified** details of every backbone, attention module, hybrid architecture, and loss function implemented in this repository. Every parameter default, tensor shape, and mathematical expression below is traced directly to the source files listed in each section header. **Nothing is paraphrased or approximated** — all formulas match the PyTorch operations line for line.

---

## 1. CNN backbones via timm

**Source:** [`src/modules/spectrogram_cnn_models.py`](../src/modules/spectrogram_cnn_models.py), class `CNNBaseline` (lines 25–57).

All CNN backbones are instantiated through the [timm](https://github.com/huggingface/pytorch-image-models) library with two critical flags:

```python
timm.create_model(backbone, pretrained=True, num_classes=0, global_pool="")
```

| Flag | Effect |
|------|--------|
| `num_classes=0` | Removes timm's built-in classifier head entirely. |
| `global_pool=""` | Disables timm's internal global average pooling, so `forward()` returns a **4-D feature map** `(B, C, h, w)` instead of a pooled vector. |

The code then applies its own **global average pooling** (`feat.mean(dim=(-2, -1))`) and a single `nn.Linear(num_features, num_classes)` head.

### 1.1 ResNet-50 (default backbone)

| Property | Value |
|----------|-------|
| timm model name | `resnet50` |
| Weights | ImageNet-1K pretrained (IMAGENET1K_V1 via timm) |
| Architecture | 4 residual stages: [3, 4, 6, 3] bottleneck blocks |
| Input | `(B, 3, 224, 224)` (RGB, ImageNet-normalized) |
| Output feature map | `(B, 2048, 7, 7)` |
| `backbone.num_features` | **2048** |
| After GAP | `(B, 2048)` |
| After `nn.Linear(2048, K)` | `(B, K)` logits |
| Total parameters (backbone only) | ~23.5 M (frozen or fine-tuned depending on script) |

### 1.2 EfficientNet-B0

| Property | Value |
|----------|-------|
| timm model name | `efficientnet_b0` |
| Weights | ImageNet-1K pretrained |
| Architecture | MBConv stages with squeeze-and-excitation, width multiplier 1.0, depth multiplier 1.0 |
| Input | `(B, 3, 224, 224)` |
| Output feature map | `(B, 1280, 7, 7)` |
| `backbone.num_features` | **1280** |
| After GAP + Linear | `(B, K)` |
| Total parameters (backbone only) | ~5.3 M |

### 1.3 Other supported backbones

The `build_model` factory also accepts `resnet34` (`num_features` = 512) and `efficientnet_b1` (`num_features` = 1280). Any timm model name is accepted through the `cnn_baseline` path.

---

## 2. Spatial attention module (CBAM-style)

**Source:** [`src/modules/spectrogram_cnn_models.py`](../src/modules/spectrogram_cnn_models.py), classes `SpatialAttention2D` (lines 60–73) and `CNNSpatialAttention` (lines 76–109).

### 2.1 `SpatialAttention2D` — exact layer structure

This implements the **spatial attention** branch of CBAM (Woo et al., ECCV 2018), **without** the channel attention branch. The module has exactly **one** learnable layer.

**Layers:**

| Layer | Type | Parameters |
|-------|------|------------|
| `self.conv` | `nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, padding=3, bias=False)` | 7 × 7 × 2 × 1 = **98 parameters**, no bias |

**Forward pass (exact operations):**

Given input feature map `x` of shape `(B, C, h, w)`:

```
1.  avg = x.mean(dim=1, keepdim=True)           # (B, 1, h, w)  — channel-wise average
2.  mx  = x.max(dim=1, keepdim=True).values     # (B, 1, h, w)  — channel-wise max
3.  cat = torch.cat([avg, mx], dim=1)            # (B, 2, h, w)
4.  M_s = sigmoid(Conv2d_7x7(cat))              # (B, 1, h, w)  — spatial attention map
5.  out = x * M_s                                # (B, C, h, w)  — element-wise gating
```

**Mathematical formulation:**

Given a feature map **F** of shape (C, h, w) at one spatial location (i, j):

$$
M_s(i,j) = \sigma\!\Bigl(\mathbf{W}_{7 \times 7} * \bigl[\mathrm{AvgPool}_c(\mathbf{F});\, \mathrm{MaxPool}_c(\mathbf{F})\bigr](i,j)\Bigr)
$$

where:
- $\sigma$ is the sigmoid function
- $\mathbf{W}_{7 \times 7}$ is a single convolution kernel of spatial size 7 × 7 operating on 2 input channels and producing 1 output channel
- $\mathrm{AvgPool}_c$ and $\mathrm{MaxPool}_c$ are average and max pooling **across the channel dimension** (not spatial)
- $[~;~]$ denotes concatenation along the channel axis
- $*$ denotes 2-D convolution with zero-padding = 3 (same-size output)

The gated output is: $\mathbf{F}' = \mathbf{F} \odot M_s$

### 2.2 `CNNSpatialAttention` — full model

| Component | Detail |
|-----------|--------|
| `self.backbone` | `timm.create_model(backbone, pretrained, num_classes=0, global_pool="")` — same as CNNBaseline |
| `self.spatial_attn` | `SpatialAttention2D(kernel_size=7)` — the 98-parameter module above |
| `self.head` | `nn.Linear(num_features, num_classes)` |

**Forward pass:**

```
feat = backbone(x)                 # (B, C, h, w)    e.g. (B, 2048, 7, 7)
feat, attn = spatial_attn(feat)    # feat: (B, C, h, w), attn: (B, h, w)
pooled = feat.mean(dim=(-2, -1))   # (B, C)           global average pool
logits = head(pooled)              # (B, K)
```

**Total added parameters** over the baseline: **98** (the 7 × 7 conv kernel). The backbone and head are identical to `CNNBaseline`.

**Default hyperparameters** (from `build_model`):

| Parameter | Default |
|-----------|---------|
| `backbone` | `"resnet50"` |
| `attn_kernel` | `7` |
| `pretrained` | `True` |

---

## 3. CNN–Transformer hybrid

**Source:** [`src/modules/spectrogram_cnn_models.py`](../src/modules/spectrogram_cnn_models.py), class `CNNTransformerHybrid` (lines 138–240) and helper `_build_2d_sincos_pe` (lines 112–135).

### 3.1 Architecture overview

The model treats the CNN feature map as a grid of spatial tokens, projects them to a transformer width, adds 2-D sinusoidal positional encodings, optionally prepends a CLS token, passes through a small transformer encoder, and classifies from the CLS token or mean-pooled output.

### 3.2 Layer-by-layer specification

| Layer / Component | Type | Shape transformation | Parameters |
|-------------------|------|---------------------|------------|
| `self.backbone` | timm CNN (same flags as above) | `(B, 3, 224, 224)` → `(B, C_bb, h, w)` | Backbone params (e.g. ~23.5 M for ResNet-50) |
| Flatten + transpose | (no learnable params) | `(B, C_bb, h, w)` → `(B, h*w, C_bb)` | 0 |
| `self.proj` | `nn.Linear(C_bb, d_model)` | `(B, h*w, C_bb)` → `(B, h*w, d_model)` | C_bb × d_model + d_model |
| 2-D sinusoidal PE | (no learnable params) | adds `(1, h*w, d_model)` | 0 |
| `self.cls_token` | `nn.Parameter(1, 1, d_model)` init N(0, 0.02) | prepended → `(B, 1+h*w, d_model)` | d_model |
| `self.encoder` | `nn.TransformerEncoder(layer, num_layers)` | `(B, L, d_model)` → `(B, L, d_model)` | see below |
| `self.norm` | `nn.LayerNorm(d_model)` | `(B, d_model)` → `(B, d_model)` | 2 × d_model |
| `self.head` | `nn.Linear(d_model, num_classes)` | `(B, d_model)` → `(B, K)` | d_model × K + K |

### 3.3 Transformer encoder layer (each of `num_layers` copies)

Each layer is `nn.TransformerEncoderLayer` with these exact constructor arguments:

```python
nn.TransformerEncoderLayer(
    d_model=256,           # token dimension
    nhead=8,               # attention heads (head_dim = 256/8 = 32)
    dim_feedforward=1024,  # inner dimension of the 2-layer FFN
    dropout=0.1,
    activation="gelu",     # GELU non-linearity in FFN
    batch_first=True,
    norm_first=True,       # Pre-LN variant (LayerNorm before attention/FFN)
)
```

**Internal sub-layers of one TransformerEncoderLayer (Pre-LN order):**

```
Input x
  │
  ├─ LayerNorm(d_model)          ← self.norm1
  ├─ MultiheadAttention(d_model, nhead)   ← self.self_attn
  │   ├─ Q, K, V projections: 3 × Linear(d_model, d_model)
  │   └─ Output projection:   1 × Linear(d_model, d_model)
  ├─ Dropout(0.1)
  ├─ Residual add
  │
  ├─ LayerNorm(d_model)          ← self.norm2
  ├─ FFN:
  │   ├─ Linear(d_model, dim_feedforward)    256 → 1024
  │   ├─ GELU
  │   ├─ Dropout(0.1)
  │   ├─ Linear(dim_feedforward, d_model)    1024 → 256
  │   └─ Dropout(0.1)
  └─ Residual add
```

**Parameter count per encoder layer:**

| Sub-component | Count |
|---------------|-------|
| `norm1` (LayerNorm) | 2 × 256 = 512 |
| `self_attn` Q, K, V projections | 3 × (256 × 256 + 256) = 197,376 |
| `self_attn` output projection | 256 × 256 + 256 = 65,792 |
| `norm2` (LayerNorm) | 2 × 256 = 512 |
| FFN linear 1 | 256 × 1024 + 1024 = 263,168 |
| FFN linear 2 | 1024 × 256 + 256 = 262,400 |
| **Total per layer** | **789,760** |
| **Total for 2 layers** | **1,579,520** |

### 3.4 2-D sinusoidal positional encoding

**Source:** `_build_2d_sincos_pe(h, w, d_model, device, dtype)` (lines 112–135).

The PE tensor has shape `(1, h × w, d_model)`. The first `d_model // 2` dimensions encode **row** position; the remaining dimensions encode **column** position. Row and column coordinates are normalized to [0, 1] via `torch.linspace(0.0, 1.0, steps=h)` and `torch.linspace(0.0, 1.0, steps=w)`.

For row encoding (dimensions 0 … d_model//2 − 1), at position (r, c) flattened to index `r * w + c`:

$$
\text{PE}[\text{idx}, 2k]   = \sin\!\bigl(r_{\text{norm}} \cdot \omega_k\bigr), \quad
\text{PE}[\text{idx}, 2k+1] = \cos\!\bigl(r_{\text{norm}} \cdot \omega_k\bigr)
$$

where $\omega_k = \exp\!\bigl(-\frac{2k}{d/2} \cdot \ln 10000\bigr)$, $r_{\text{norm}} \in [0,1]$.

Column encoding occupies dimensions `d_model//2` … `d_model − 1` with the same sinusoidal formula applied to $c_{\text{norm}}$.

This is **not learnable** — it is recomputed from the feature-map spatial size at each forward pass, so it adapts to any input resolution.

### 3.5 Default hyperparameters (from `build_model`)

| Parameter | Default | Note |
|-----------|---------|------|
| `backbone` | `"resnet50"` | C_bb = 2048 for ResNet-50 |
| `d_model` | `256` | Transformer token width |
| `nhead` | `8` | Attention heads (head_dim = 32) |
| `num_layers` | `2` | Transformer encoder depth |
| `dim_feedforward` | `1024` | FFN inner dimension |
| `dropout` | `0.1` | Applied in attention and FFN |
| `use_cls_token` | `True` | Prepend learnable [CLS] token |
| `pretrained` | `True` | ImageNet backbone weights |

### 3.6 Complete parameter budget (ResNet-50 + hybrid head)

| Component | Parameters |
|-----------|-----------|
| ResNet-50 backbone | ~23,508,032 |
| Linear projection (2048 → 256) | 524,544 |
| CLS token | 256 |
| Transformer encoder (2 layers) | 1,579,520 |
| Final LayerNorm | 512 |
| Classification head (256 → 2) | 514 |
| **Total** | **~25,613,378** |

---

## 4. Asymmetric Focal Contrastive Loss (AFCL)

**Source:** [`src/modules/afcl_loss.py`](../src/modules/afcl_loss.py) — full file, 484 lines.

AFCL is a **three-term composite loss** with epoch-dependent warmup. It has two variants: base `AsymmetricFocalContrastiveLoss` and `AFCLWithHardMining` (extends the base with hard-negative mining).

### 4.1 Overall loss (line 268–272)

$$
\mathcal{L}_{\text{AFCL}} = \mathcal{L}_{\text{focal}} + w(t) \cdot \lambda_{\text{contrast}} \cdot \mathcal{L}_{\text{contrastive}} + w(t) \cdot \lambda_{\text{align}} \cdot \mathcal{L}_{\text{alignment}}
$$

where the **warmup multiplier** is (line 250–253):

$$
w(t) = \min\!\Bigl(1,\;\frac{t}{T_{\text{warmup}}}\Bigr)
$$

with $t$ = current epoch and $T_{\text{warmup}}$ = `warmup_epochs` (default **10**). For $T_{\text{warmup}} \leq 0$, $w(t) = 1$ always.

**Default weights:**

| Symbol | Parameter | Default |
|--------|-----------|---------|
| $\lambda_{\text{contrast}}$ | `lambda_contrast` | 0.1 |
| $\lambda_{\text{align}}$ | `lambda_align` | 0.05 |
| $T_{\text{warmup}}$ | `warmup_epochs` | 10 |

### 4.2 Term 1: Focal Loss (`FocalLossComponent`, lines 31–67)

**Input:** logits `(B, K)`, integer labels `(B,)`.

$$
\mathcal{L}_{\text{focal}} = \frac{1}{B}\sum_{i=1}^{B} \alpha_{y_i} \cdot (1 - p_{t,i})^{\gamma} \cdot \text{CE}(i)
$$

where:
- $\text{CE}(i) = -\log p_{t,i}$ is the standard cross-entropy for sample $i$
- $p_{t,i} = \exp(-\text{CE}(i))$ = softmax probability of the ground-truth class
- $\gamma$ = `focal_gamma` (default **2.0**)
- $\alpha_{y_i}$ = per-class weight looked up from vector $\alpha$

**Exact code sequence:**

```python
ce = F.cross_entropy(logits, targets, reduction="none")   # (B,)
pt = torch.exp(-ce)                                        # (B,)
focal_weight = (1 - pt) ** self.gamma                      # (B,)
if self.alpha is not None:
    alpha_t = self.alpha[targets]                           # (B,)
    focal_weight = alpha_t * focal_weight
loss = (focal_weight * ce).mean()
```

**Factory `create_afcl_loss` alpha computation** (lines 452–456):

Given imbalance ratio $r$ (default 5.0):

$$
\alpha_0 = \frac{1}{1 + r}, \qquad \alpha_1 = \frac{r}{1 + r}
$$

Gamma is dynamically set:

$$
\gamma = \min\!\bigl(3.0,\; 1.5 + 0.3 \cdot \ln(r + 1)\bigr)
$$

For $r = 5.0$: $\alpha = [0.1\overline{6}, 0.8\overline{3}]$, $\gamma \approx 2.037$.

### 4.3 Term 2: Asymmetric Contrastive Loss (`AsymmetricContrastiveLoss`, lines 70–138)

**Input:** L2-normalized features `(B, D)`, integer labels `(B,)`.

**Step 1 — Cosine similarity matrix:**

$$
S_{ij} = \frac{\mathbf{z}_i \cdot \mathbf{z}_j}{\tau}
$$

where $\tau$ = `temperature` (default **0.5** in AFCL; 0.07 standalone default).

**Step 2 — Masks:**

- $P_{ij} = \mathbb{1}[y_i = y_j] \;\wedge\; (i \neq j)$ — positive pairs (same class, not self)
- $N_{ij} = \mathbb{1}[y_i \neq y_j]$ — negative pairs
- $a_i = \mathbb{1}[y_i = \text{abnormal\_class}]$ — abnormal indicator

**Step 3 — Asymmetric pair weights (lines 122–125):**

Positive pair weight:

$$
w^+_{ij} = 1 + (w_{\text{abn}} - 1) \cdot a_i \cdot a_j
$$

where $w_{\text{abn}}$ = `abnormal_weight` (default **2.0**). So abnormal–abnormal pairs get weight 2.0; all others get 1.0.

Negative pair weight:

$$
w^-_{ij} = 1 + (w_{\text{abn}} - 1) \cdot \min(a_i + a_j,\; 1)
$$

Any pair involving at least one abnormal sample gets weight 2.0.

**Step 4 — Weighted InfoNCE (lines 127–136):**

```python
exp_sim = exp(S)
neg_sum_i = sum_j( exp_sim[i,j] * N[i,j] * w_neg[i,j] )         # for each i
log_prob[i,j] = S[i,j] - log( exp_sim[i,j] + neg_sum_i + 1e-8 )
L_contrastive = -mean_i( sum_j(log_prob[i,j] * P[i,j] * w_pos[i,j]) / |P_i| )
```

In mathematical notation:

$$
\mathcal{L}_{\text{contrastive}} = -\frac{1}{B}\sum_{i=1}^{B} \frac{1}{|P_i|} \sum_{j \in P_i} w^+_{ij} \left[ \frac{S_{ij}}{\tau} - \log\!\left(e^{S_{ij}/\tau} + \sum_{k \in N_i} w^-_{ik}\, e^{S_{ik}/\tau}\right) \right]
$$

### 4.4 Term 3: Prototype Alignment Loss (`PrototypeAlignmentLoss`, lines 141–187)

**Learnable parameters:** class prototype matrix $\mathbf{P} \in \mathbb{R}^{K \times D}$, initialized with Xavier uniform.

**Forward pass:**

Both features and prototypes are L2-normalized before computing cosine similarity.

$$
\text{sim}_{ik} = \bar{\mathbf{z}}_i \cdot \bar{\mathbf{p}}_k
$$

For each sample $i$ with label $y_i$:

$$
s^+ = \text{sim}_{i, y_i}, \qquad s^- = \max_{k \neq y_i} \text{sim}_{ik}
$$

$$
\mathcal{L}_{\text{align}} = \frac{1}{B}\sum_{i=1}^{B} \max\!\bigl(0,\; s^- - s^+ + m\bigr)
$$

where $m$ = `margin` (default **0.5**). This is a **triplet-style margin loss** that pushes each feature closer to its own class prototype and further from the nearest wrong prototype.

**Default hyperparameters:**

| Parameter | Default |
|-----------|---------|
| `feature_dim` | 768 |
| `num_classes` | 2 |
| `margin` | 0.5 |

### 4.5 Hard-negative mining variant (`AFCLWithHardMining`, lines 296–431)

Extends the base AFCL. Before computing the focal term, it identifies **hard negatives**: normal-class samples that are **closest to the abnormal centroid** in cosine distance.

**Mining procedure** (lines 341–379):

```
1. Compute abnormal centroid:  c_abn = mean(z_i for i where y_i = abnormal)  (L2-normalized features)
2. For each normal sample j:   dist_j = 1 - z_j · c_abn
3. Select top-k closest:       k = floor(0.3 * n_normal)
4. Set sample_weight[hard_j] = hard_weight (default 2.0)
5. Set sample_weight[abnormal] = hard_weight (default 2.0)
6. All other samples: weight = 1.0
```

The focal loss is then computed **per-sample** and multiplied by `sample_weights` before averaging:

$$
\mathcal{L}_{\text{focal}}^{\text{hard}} = \frac{1}{B}\sum_{i=1}^{B} w_i \cdot \alpha_{y_i} \cdot (1 - p_{t,i})^{\gamma} \cdot \text{CE}(i)
$$

where $w_i$ is the sample weight (1.0, or 2.0 for hard negatives and all abnormal samples).

Contrastive and alignment terms remain unchanged from the base variant.

**Additional default hyperparameters:**

| Parameter | Default |
|-----------|---------|
| `hard_negative_ratio` | 0.3 (30% of normal samples) |
| `hard_weight` | 2.0 |

### 4.6 Factory function summary (`create_afcl_loss`, lines 434–483)

| Argument | Default | Effect |
|----------|---------|--------|
| `feature_dim` | 768 | Prototype matrix width |
| `num_classes` | 2 | Number of prototypes |
| `imbalance_ratio` | 5.0 | Computes focal alpha and gamma |
| `use_hard_mining` | True | Returns `AFCLWithHardMining` if True, `AsymmetricFocalContrastiveLoss` otherwise |
| `warmup_epochs` | 10 | Ramp contrastive + alignment from 0 → full weight |
| `temperature` | 0.5 | Contrastive similarity scaling |
| `lambda_contrast` | 0.1 | Weight on contrastive term |
| `lambda_align` | 0.05 | Weight on alignment term |

---

## 5. Standalone focal loss (spectrogram fault pipeline)

**Source:** [`src/modules/focal_loss.py`](../src/modules/focal_loss.py), class `FocalLoss` (lines 13–83).

Used independently in the spectrogram supervised pipeline (`train_spectrogram_fault.py`) when `--loss focal` is selected. Same formula as AFCL's focal component:

$$
\text{FL}(p_t) = -\alpha_t \cdot (1 - p_t)^{\gamma} \cdot \log(p_t)
$$

In `train_spectrogram_fault.py` `_make_criterion` (line 99–103), alpha is computed as inverse frequency per class on the **training subset only**:

```python
alpha[c] = n_train / (num_classes * count[c])
```

Default gamma = 2.0. This is **not** AFCL — it is a standalone focal loss with no contrastive or alignment terms.

---

## 6. Summary of all model variants and their exact default parameters

| CLI name | Class | Backbone features | Added head params | Key hyperparameters |
|----------|-------|------------------:|------------------:|---------------------|
| `resnet50` | `CNNBaseline` | 2048 | ~4 K | — |
| `efficientnet_b0` | `CNNBaseline` | 1280 | ~2.6 K | — |
| `cnn_se` | `CNNSpatialAttention` | 2048 | ~4.1 K | attn_kernel=7 |
| `cnn_transformer` | `CNNTransformerHybrid` | 2048 | ~2.1 M | d_model=256, nhead=8, num_layers=2, dim_ff=1024, dropout=0.1, use_cls=True |

All models accept `num_classes` (inferred from data) and `pretrained` (default True = ImageNet weights).
