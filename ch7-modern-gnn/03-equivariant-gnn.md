# 03. Equivariant GNN — E(3), SO(3) Equivariance

## 🎯 핵심 질문

- Translation equivariance (CNN) 이 분자·물리에서 왜 부족한가? E(3) 또는 SO(3) equivariance 가 필요한 이유?
- EGNN (Satorras 2021) 의 coordinate-feature 분리와 $x_i^{(l+1)} = x_i^{(l)} + \sum_j (x_i - x_j) \phi_x(\ldots)$ 의 equivariance 증명?
- SE(3)-Transformer (Fuchs 2020) 의 steerable filter (spherical harmonics basis) 의 의미?
- Tensor Field Network (TFN, Thomas 2018) 의 higher-order tensor feature?
- QM9 / MD17 molecular dynamics 에서 equivariant GNN 의 실전 성능?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

CNN 의 translation equivariance (Ch1-CNN) 가 image 에 충분. 그러나 **3D scientific data**:
- Molecule: 원자 3D coordinate, 회전·반사·translation 대칭
- Physics simulation: Particle position, velocity
- Protein: 3D structure

이들에는 **E(3)** (translation + rotation + reflection) 또는 **SE(3)** (without reflection) equivariance 필요. 일반 GNN 은 3D 정보 무시 — graph structure 만 사용.

**Equivariant GNN**:
- EGNN (Satorras 2021): Simple, practical
- SE(3)-Transformer (Fuchs 2020): Higher-order tensor features
- Tensor Field Network (TFN, Thomas 2018): Mathematical foundation

이들은 AlphaFold (protein folding), Open Catalyst (catalyst discovery) 등 modern 과학 응용의 기반.

---

## 📐 수학적 선행 조건

- [CNN Deep Dive Ch1-02](https://github.com/iq-ai-lab/cnn-deep-dive): Translation equivariance
- [Group Theory](group-theoretic 정의 reminder): Group, representation
- [Transformer Deep Dive](https://github.com/iq-ai-lab/transformer-deep-dive): Attention

---

## 📖 직관적 이해

### Translation vs Rotation Equivariance

**Translation equivariance** (CNN):
$$
\phi(x + t) = \phi(x) + t \quad \text{(shift input → shift output)}
$$

**Rotation equivariance** (EGNN 등):
$$
\phi(Rx) = R \phi(x) \quad \text{(rotate input → rotate output)}
$$

**Combined E(3) equivariance**:
$$
\phi(Rx + t) = R \phi(x) + t
$$

### 왜 Equivariance?

분자 의 특성 (energy, force) 이 **orientation-independent**. 분자 를 공간에서 임의 회전해도 energy 같음.

그러나 position coordinate 자체는 equivariant — 회전 후 force vector 도 회전.

따라서:
- **Invariant**: Energy (scalar)
- **Equivariant**: Force, velocity (vector)

Equivariant GNN 이 두 모두 handle.

### EGNN 의 간단함

EGNN 의 핵심 insight: **coordinate 와 feature 를 분리**, coordinate update 에 $(x_i - x_j)$ 형태만 사용 → automatic equivariance.

$$
x_i^{(l+1)} = x_i^{(l)} + \sum_j (x_i - x_j) \phi_x(\|x_i - x_j\|^2, h_i, h_j)
$$

- $x_i - x_j$ 는 rotation 아래 equivariant
- $\|x_i - x_j\|^2$ 는 invariant (distance)
- $\phi_x$ 가 invariant features 만 사용 → coordinate update 가 자동 equivariant

Feature update 는 invariant:
$$
h_i^{(l+1)} = \phi_h(h_i, \sum_j m_{ij})
$$

### Steerable Filter (SE(3)-Transformer)

더 sophisticated: spherical harmonics basis 를 사용, feature 가 tensor (higher-order representation).

- Type-0 feature: scalar (invariant)
- Type-1 feature: vector (equivariant)
- Type-2 feature: higher-order tensor

각 type 이 SO(3) representation — Clebsch-Gordan tensor product 로 결합.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Group E(3) / SE(3) / SO(3)

- **SO(3)**: 3D rotation group (orthogonal matrices with $\det = 1$)
- **SE(3)**: rotation + translation
- **E(3)**: rotation + translation + reflection

### 정의 3.2 — Equivariance

함수 $\phi: V^n \to V^n$ ($V$ = vector space) 가 group $G$-equivariant:
$$
\phi(g \cdot x) = g \cdot \phi(x) \quad \forall g \in G, x \in V^n
$$

**Invariant**: Special case — $g \cdot \phi(x) = \phi(x)$ (scalar output).

### 정의 3.3 — EGNN Layer (Satorras 2021)

**Input**: Node feature $h_i \in \mathbb R^d$ (invariant), coordinate $x_i \in \mathbb R^3$ (equivariant).

**Message**:
$$
m_{ij} = \phi_e(h_i, h_j, \|x_i - x_j\|^2, a_{ij})
$$

($a_{ij}$: edge attribute, optional)

**Coordinate update**:
$$
x_i^{(l+1)} = x_i^{(l)} + \sum_{j \neq i} (x_i^{(l)} - x_j^{(l)}) \phi_x(m_{ij})
$$

$\phi_x: \mathbb R^d \to \mathbb R$ (scalar scaling).

**Feature update**:
$$
h_i^{(l+1)} = \phi_h\left(h_i^{(l)}, \sum_{j \neq i} m_{ij}\right)
$$

### 정의 3.4 — SE(3)-Transformer (Fuchs 2020)

**Steerable feature**: Feature 가 irreducible representation (irrep) 으로 분해:
$$
h_i = (h_i^{(0)}, h_i^{(1)}, h_i^{(2)}, \ldots)
$$

- $h_i^{(0)}$: scalar (SO(3) invariant)
- $h_i^{(1)}$: vector (equivariant)
- $h_i^{(l)}$: $(2l+1)$-dim irrep

**Attention**:
$$
\alpha_{ij} = \text{softmax}_j(Q_i \cdot K_j^{\text{steerable}})
$$

Key 가 spherical harmonics $Y_l(\hat r_{ij})$ 로 encoded, $\hat r_{ij}$ = 단위 방향 벡터.

### 정의 3.5 — Tensor Field Network (Thomas 2018)

**Tensor product** of representations with Clebsch-Gordan coefficients:
$$
(h^{(l_1)} \otimes h^{(l_2)})^{(l_{\text{out}})} = \sum_{m_1, m_2} C^{l_{\text{out}}, m_{\text{out}}}_{l_1, m_1; l_2, m_2} h^{(l_1)}_{m_1} h^{(l_2)}_{m_2}
$$

(Clebsch-Gordan decomposition)

---

## 🔬 정리와 결과

### 정리 3.1 — EGNN Equivariance

**Theorem**: EGNN layer 는 E(3)-equivariant:
$$
\text{EGNN}(Rx + t, h) = (R \cdot \text{EGNN}_x(x, h) + t, \text{EGNN}_h(x, h))
$$

**증명** (key step):

**Distance 의 invariance**: $\|Rx_i + t - (Rx_j + t)\| = \|R(x_i - x_j)\| = \|x_i - x_j\|$ (orthogonal $R$). 따라서 $m_{ij}$ 가 invariant.

**Coordinate update equivariance**:
$$
(Rx_i + t)^{(l+1)} = (Rx_i + t) + \sum_j (Rx_i + t - Rx_j - t) \phi_x(m_{ij})
$$
$$
= Rx_i + t + R \sum_j (x_i - x_j) \phi_x(m_{ij})
$$
$$
= R(x_i + \sum_j (x_i - x_j) \phi_x(m_{ij})) + t = R x_i^{(l+1)} + t
$$

✓ Equivariant.

**Feature invariance**: $m_{ij}$ invariant ⟹ $h_i^{(l+1)}$ invariant. $\square$

### 정리 3.2 — SE(3)-Transformer 의 expressive power

**Theorem (informal)**: Higher-order tensor features (type $l \geq 1$) 가 vector-only 보다 strict 표현력 강. Clebsch-Gordan tensor product 로 arbitrary angular dependence 학습.

**Limitation**: Type 을 어디까지 사용할지 choice — 보통 $l \leq 2$ 실전적.

### 정리 3.3 — 실전 성능 (QM9)

**QM9** (13 molecular properties 예측, ~130k molecules):

| Model | Avg MAE |
|-------|---------|
| SchNet (distance-based) | 0.061 |
| DimeNet++ (angular info) | 0.038 |
| **EGNN** | 0.029 |
| **SE(3)-Transformer** | 0.026 |
| PaiNN | 0.027 |
| NequIP (E(3)-equivariant) | 0.020 |

Equivariant GNN 이 distance-only baseline 보다 2-3x 낮은 MAE.

### 정리 3.4 — Complexity

| Model | 표현력 | Cost |
|-------|--------|------|
| SchNet (invariant only) | Limited | $O(m d)$ |
| EGNN (scalar + vector) | Basic equivariance | $O(m d)$ |
| SE(3)-Transformer | Higher-order tensor | $O(m d l^3)$ |
| TFN | Full tensor | $O(m d l^3)$ |

EGNN 이 simplicity-expressiveness trade-off 의 sweet spot.

### 정리 3.5 — Information Loss without Equivariance

**Theorem**: Non-equivariant GNN 이 3D structure 에 대해:
- Augmentation 으로 학습 가능 (random rotate input) — 단 sample 효율 낮음
- Canonical form 으로 학습 가능 — 단 canonical 선택 brittle

Equivariant GNN 이 **design 에서 자연적** → sample efficiency + generalization.

---

## 💻 구현

### 실험 1 — EGNN Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

class EGNNLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        # Message: m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2)
        self.phi_e = nn.Sequential(
            nn.Linear(2 * d + 1, d), nn.SiLU(),
            nn.Linear(d, d), nn.SiLU()
        )
        # Coord update: phi_x(m_ij) 스칼라
        self.phi_x = nn.Sequential(
            nn.Linear(d, d), nn.SiLU(),
            nn.Linear(d, 1)
        )
        # Feature update
        self.phi_h = nn.Sequential(
            nn.Linear(2 * d, d), nn.SiLU(),
            nn.Linear(d, d)
        )
    
    def forward(self, h, x, edge_index):
        """
        h: [n, d] invariant feature
        x: [n, 3] coordinate (equivariant)
        edge_index: [2, m]
        """
        src, dst = edge_index
        
        # Relative coordinate
        dx = x[src] - x[dst]   # [m, 3]
        dist2 = (dx ** 2).sum(-1, keepdim=True)   # [m, 1]
        
        # Message
        m = self.phi_e(torch.cat([h[src], h[dst], dist2], dim=-1))
        
        # Coordinate update
        coord_weight = self.phi_x(m)   # [m, 1]
        dx_update = dx * coord_weight   # [m, 3]
        x_new = x + scatter_add(dx_update, dst, dim=0, dim_size=x.size(0))
        
        # Feature update
        agg = scatter_add(m, dst, dim=0, dim_size=h.size(0))
        h_new = h + self.phi_h(torch.cat([h, agg], dim=-1))
        
        return h_new, x_new

# Equivariance 확인
n = 5
h = torch.randn(n, 16)
x = torch.randn(n, 3)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)

layer = EGNNLayer(d=16)
h_new, x_new = layer(h, x, edge_index)

# Random rotation
R = torch.linalg.qr(torch.randn(3, 3))[0]
x_rot = x @ R.T
h_rot_new, x_rot_new = layer(h, x_rot, edge_index)

# Verify: x_new @ R.T == x_rot_new
x_new_rot = x_new @ R.T
diff_coord = (x_new_rot - x_rot_new).abs().max().item()
diff_feat = (h_new - h_rot_new).abs().max().item()
print(f'Coordinate equivariance error: {diff_coord:.6e}')
print(f'Feature invariance error: {diff_feat:.6e}')
# 둘 다 거의 0
```

### 실험 2 — Translation Equivariance 확인

```python
# Translation
t = torch.randn(3)
x_trans = x + t

h_trans_new, x_trans_new = layer(h, x_trans, edge_index)
x_new_trans = x_new + t
diff_coord = (x_new_trans - x_trans_new).abs().max().item()
print(f'Translation equivariance: {diff_coord:.6e}')
```

### 실험 3 — SE(3)-Transformer (simplified)

```python
import numpy as np

class SE3SimpleLayer(nn.Module):
    """Simplified SE(3): scalar + vector features."""
    def __init__(self, d_scalar, d_vector):
        super().__init__()
        self.d_s = d_scalar
        self.d_v = d_vector
        # Scalar-to-scalar, scalar-to-vector, vector-to-scalar, vector-to-vector
        self.W_ss = nn.Linear(d_scalar, d_scalar)
        self.W_sv = nn.Linear(d_scalar, d_vector)
        self.W_vs = nn.Linear(d_vector, d_scalar)
        self.W_vv = nn.Linear(d_vector, d_vector)
    
    def forward(self, s, v, edge_index):
        """
        s: [n, d_s] scalar (invariant)
        v: [n, d_v, 3] vector (equivariant)
        """
        src, dst = edge_index
        # Message: combine scalar & vector info
        s_msg = self.W_ss(s[src])
        v_msg = self.W_vv(v[src].transpose(-1, -2)).transpose(-1, -2)
        s_to_v = self.W_sv(s[src]).unsqueeze(-1) * ...
        # ... (전체 mixing logic 복잡)
        return s, v

# SE(3)-Transformer 는 제대로 구현하려면 spherical harmonics + Clebsch-Gordan
# 여기선 개념 보이는 수준
```

### 실험 4 — QM9 Style Molecular Energy

```python
class EGNNModel(nn.Module):
    def __init__(self, d_in, d_hid, num_layers=3):
        super().__init__()
        self.embed = nn.Linear(d_in, d_hid)
        self.layers = nn.ModuleList([EGNNLayer(d_hid) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(d_hid, d_hid), nn.SiLU(),
            nn.Linear(d_hid, 1)
        )
    
    def forward(self, h, x, edge_index):
        h = self.embed(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        # Molecular energy: sum over atoms
        energy_per_atom = self.readout(h)
        return energy_per_atom.sum()

# Toy molecule: 5 atoms
h = torch.randn(5, 8)   # atomic features (element type, charge)
x = torch.randn(5, 3)   # 3D positions
ei = torch.tensor([[0,1,1,2,2,3,3,4,4,0], [1,0,2,1,3,2,4,3,0,4]], dtype=torch.long)

model = EGNNModel(d_in=8, d_hid=32, num_layers=3)
energy = model(h, x, ei)
print(f'Predicted molecular energy: {energy.item():.4f}')

# Check invariance: energy should be SO(3) invariant
R = torch.linalg.qr(torch.randn(3, 3))[0]
x_rot = x @ R.T
energy_rot = model(h, x_rot, ei)
print(f'Energy after rotation: {energy_rot.item():.4f}')
print(f'Diff: {abs(energy.item() - energy_rot.item()):.6e}  (should be ~0)')
```

### 실험 5 — 비교: GIN vs EGNN on 3D Data

```python
# GIN: 3D 정보 무시 (graph structure 만)
# EGNN: 3D 정보 활용

# 3D structure 가 중요한 task 에서 EGNN 이 GIN 우월 예상

# 단순 test: 분자의 "conformer energy" (3D 만 결정) 예측
# GIN: 같은 graph 의 다른 conformer 에 대해 같은 prediction
# EGNN: 3D 에 따라 다른 prediction

# (실제 benchmark 는 QM9, MD17 등)
```

---

## 🔗 실전 활용

### 1. AlphaFold 2 / Protein Structure

- Protein folding: E(3)-equivariant attention (evoformer)
- Distillation from SE(3)-Transformer 원리

### 2. Open Catalyst Project (OC20)

- Catalyst discovery with DFT simulation
- EGNN, SchNet, PaiNN, GemNet 등 benchmark
- Equivariant models 가 SOTA

### 3. AlphaEarth / Material Discovery

- Crystal structure prediction
- E(3)-equivariant + periodicity

### 4. Molecular Dynamics (MD17)

- Force prediction (vector output!) — equivariant 필수
- SchNet, PaiNN, NequIP
- MACE (2022): tensor field + message passing

### 5. PyG Implementation

```python
from e3nn import o3  # Equivariant group library
# ... SE(3)-Transformer, TFN 구현
```

PyG 공식 EGNN: `torch_geometric.nn.models.EGNN`.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Euclidean 3D space | Non-euclidean (hyperbolic) 별도 |
| Point cloud representation | Voxel, mesh 는 다른 처리 |
| Higher-order tensors ($l \geq 2$) 비용 | SE(3)-Transformer 의 $O(l^3)$ |
| Global equivariance only | Local equivariance 가 더 복잡 |
| 분자 의 rigid 가정 | Flexibility 는 dynamics 필요 |
| Pre-defined representation types | Adaptive type 선택 어려움 |

---

## 📌 핵심 정리

$$\boxed{\text{EGNN: } x_i^{(l+1)} = x_i^{(l)} + \sum_j (x_i - x_j) \phi_x(\ldots)}$$

$$\boxed{\phi(Rx + t) = R \phi(x) + t \quad \text{(E(3) equivariance)}}$$

| Model | Coordinate 처리 | 표현력 | Cost |
|-------|----------------|--------|------|
| **GIN (no 3D)** | ✗ | Graph only | $O(m d)$ |
| **SchNet** | Distance-based | Invariant only | $O(m d)$ |
| **EGNN** | Scalar + vector | Basic equivariant | $O(m d)$ |
| **PaiNN** | Scalar + vector refined | Improved equivariant | $O(m d)$ |
| **SE(3)-Transformer** | Higher-order tensor | Tensor equivariant | $O(m d l^3)$ |
| **NequIP / MACE** | Full tensor network | Rich equivariant | $O(m d l^3)$ |

핵심: **E(3) equivariance 가 분자·물리 GNN 의 필수 inductive bias**. Translation-only (CNN) 부족.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Water molecule (H2O) 을 각도 $\theta$ 로 회전했을 때 energy 와 dipole moment 의 변화는?

<details>
<summary>해설</summary>

**Energy (scalar, SO(3) invariant)**:
$$
E(R x) = E(x) \quad \forall R
$$

회전해도 energy 같음 — physical principle. 분자 가 어떻게 놓여있든 binding energy 동일.

**Dipole moment (vector, SO(3) equivariant)**:
$$
\mu(R x) = R \mu(x)
$$

Dipole 은 방향성 있음 — 회전 시 같이 회전. H2O 의 dipole 이 H 에서 O 방향.

**Equivariant GNN 의 역할**:

- **Energy 예측**: Invariant scalar output → 마지막 layer 의 scalar feature 만.
- **Dipole 예측**: Equivariant vector output → 마지막 layer 의 vector feature.

**EGNN** 이 둘 다 handle:
- $h^{(L)}$ → energy (scalar, invariant)
- $x^{(L)} - x^{(0)}$ (coordinate difference) 또는 vector feature → dipole (equivariant)

**General pattern**:

- Molecular property (energy, gap, heat capacity): scalar invariant
- Force, dipole, velocity: vector equivariant
- Polarizability tensor: rank-2 tensor equivariant (SE(3)-Transformer / TFN 필요)

**실전**: Energy regression 은 EGNN 충분, force/dipole 은 equivariant 필수, polarizability 는 higher-order tensor 필요.

</details>

**문제 2** (심화): EGNN 의 coordinate update $x_i^{(l+1)} = x_i^{(l)} + \sum_j (x_i - x_j) \phi_x(\ldots)$ 의 equivariance 가 "$(x_i - x_j)$" 형태에서 오는 이유를 explicitly 증명하라.

<details>
<summary>해설</summary>

**Claim**: $(x_i - x_j)$ 형태가 translation + rotation equivariance 보장.

**Translation invariance of $(x_i - x_j)$**:

$$
(x_i + t) - (x_j + t) = x_i - x_j
$$

Translation 이 차이에서 소멸. ✓ (invariant)

**Rotation equivariance of $(x_i - x_j)$**:

$$
R x_i - R x_j = R (x_i - x_j)
$$

Linear transform 이 subtraction 와 commute. ✓ (equivariant to rotation)

**EGNN coordinate update 의 equivariance**:

$$
x_i^{(l+1)} = x_i^{(l)} + \sum_j (x_i^{(l)} - x_j^{(l)}) \phi_x(\|x_i - x_j\|^2, h_i, h_j)
$$

**Translation** $x \to x + t$:
- $x_i^{(l)} + t$ → 첫 term: $x_i + t$
- $(x_i + t) - (x_j + t) = x_i - x_j$ (translation invariant)
- $\|x_i - x_j\|^2$ invariant → $\phi_x$ 같음
- 합: $(x_i + t) + \sum_j (x_i - x_j) \phi_x = x_i^{(l+1)} + t$ ✓

**Rotation** $x \to Rx$:
- $R x_i^{(l)}$ → 첫 term: $R x_i$
- $R(x_i) - R(x_j) = R(x_i - x_j)$
- $\|R(x_i) - R(x_j)\|^2 = \|R(x_i - x_j)\|^2 = \|x_i - x_j\|^2$ (orthogonal $R$) → $\phi_x$ 같음
- 합: $R x_i + \sum_j R(x_i - x_j) \phi_x = R(x_i + \sum_j (x_i - x_j) \phi_x) = R x_i^{(l+1)}$ ✓

**If not $(x_i - x_j)$**:

Alternative: $x_i \cdot \phi_x$ (coordinate 자체):
- Translation: $x_i + t$ → $(x_i + t) \phi_x$ ≠ $x_i \phi_x + t$ (unless $\phi_x = 1$)
- Non-equivariant.

Alternative: $x_i \times x_j$ (cross product):
- Rotation: $Rx_i \times Rx_j \neq R(x_i \times x_j)$ (actually true, since cross product is equivariant, but complex)

**Conclusion**: $(x_i - x_j)$ 가 가장 natural, simplest form 으로 E(3) equivariance 보장. EGNN 의 우아한 design. $\square$

</details>

**문제 3** (논문 비평): SE(3)-Transformer 가 이론적 더 powerful 하지만 EGNN 이 실전에서 competitive. 이유와 trade-off?

<details>
<summary>해설</summary>

**SE(3)-Transformer 의 이론적 우위**:

- Higher-order tensor (type-2, type-3) 로 arbitrary angular dependence 학습
- Clebsch-Gordan tensor product 로 rich geometric interaction
- 이론상 universal equivariant function approximator

**EGNN 의 simplicity**:

- Scalar + vector (type-0, type-1) 만
- 단순한 $(x_i - x_j)$ 기반 coordinate update
- Implementation 간단, optimization 쉬움

**실전 비교 (QM9)**:

| Model | Avg MAE | Cost |
|-------|---------|------|
| SchNet | 0.061 | Low |
| EGNN | 0.029 | Low |
| SE(3)-Transformer | 0.026 | High |
| NequIP | 0.020 | High |

**Trade-off 분석**:

**SE(3)-Transformer의 우위 (marginal)**:
- Higher-order geometric pattern (angular, dihedral) 학습 가능
- 일부 task (다체 물리 simulation) 에서 더 나음

**SE(3)-Transformer의 단점**:
- Computational cost: $l^3$ growth with tensor order
- Implementation complexity: Clebsch-Gordan, spherical harmonics
- Optimization: 더 많은 parameter, initialization sensitive
- Sample efficiency: 많은 training data 필요

**EGNN의 우위**:
- Simplicity → fewer bugs, faster iteration
- Sample efficiency → 작은 dataset (QM9 sub) 에서 competitive
- Computational efficiency → large-scale (AlphaFold) 에서 스케일 가능

**Modern 추세**:

- **Scientific computation (AlphaFold 2, OC20)**: Higher-order equivariant (SE(3)-Transformer, NequIP, MACE) — precision 중요
- **Drug discovery (fast screening)**: EGNN, SchNet — speed 중요
- **Small dataset / prototyping**: EGNN 충분

**MACE (Batatia 2022)**:

Hybrid: Message passing + Equivariant + higher-order. "Equivariant 의 ResNet" — SOTA on OC20, HEA-25.

**AlphaFold Principle**:

실증적 효과 = (이론 표현력) + (inductive bias quality) + (data availability) + (compute budget). 각 system 마다 optimal choice 다름.

**General lesson**:

- Prototype: EGNN
- Production chemistry: MACE / NequIP
- Extreme precision (DFT-level): Full tensor network

따라서 equivariant GNN 의 hierarchy 가 existential — trade-off 인지 중요.

</details>

---

<div align="center">

[◀ 이전](./02-gnn-attention-unification.md) | [📚 README](../README.md) | [다음 ▶](./04-scaling-and-future.md)

</div>
