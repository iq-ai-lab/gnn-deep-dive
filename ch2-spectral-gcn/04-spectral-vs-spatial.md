# 04. Spectral vs Spatial 관점의 통합

## 🎯 핵심 질문

- GCN 은 spectral 출발이지만 결과가 spatial 1-hop aggregation — 두 관점이 어떻게 동치인가?
- 일반적으로 polynomial spectral filter ↔ localized spatial aggregation 의 동치 관계는 무엇인가?
- "Spectral 의 장점은 이론, spatial 의 장점은 구현" 이라는 Defferrard 의 주장의 의미는?
- Message Passing (Ch3) 으로 가는 자연스러운 다리는 어떻게 만들어지는가?
- 현대 GNN (GAT, GIN, Graphormer) 에서도 spectral 관점이 유용한가?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

GNN 의 두 분기 — spectral GCN (Bruna → ChebNet → GCN) 과 spatial MPNN (GraphSAGE → GAT → GIN) — 은 처음에는 다른 출발점을 가졌지만, **수학적으로 동치 또는 위계 관계** 를 가집니다:

1. **GCN 의 양면성** — Spectral 유도, spatial 결과
2. **모든 polynomial spectral filter = localized spatial filter** (정리 1.3 Ch2-01)
3. **Inductive bias 설계의 통합 framework** — 어떤 spatial aggregator 가 어떤 spectral 의미를 가지는지

이 문서에서는 두 관점의 **수학적 동치성**을 정리하고, Message Passing (Ch3) 으로의 자연스러운 다리를 만듭니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [03-gcn-derivation.md](./03-gcn-derivation.md), [02-chebnet.md](./02-chebnet.md), [01-spectral-convolution.md](./01-spectral-convolution.md)
- [Ch1-05](../ch1-graph-laplacian/05-graph-fourier.md): Graph Fourier transform
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Polynomial of matrix, function calculus

---

## 📖 직관적 이해

### 두 관점의 표

| 관점 | Spectral | Spatial |
|------|----------|---------|
| **출발점** | $L$ 의 고유분해 | 노드 이웃 aggregation |
| **Filter** | $\hat g(\lambda)$ — frequency response | $W$ — 이웃별 weight |
| **Locality** | Polynomial $\Rightarrow$ K-hop | 직접 K-hop neighbor |
| **Inductive** | 일반 spectral X, polynomial $\bigcirc$ | $\bigcirc$ |
| **Eigendecomposition** | 필요 (general), 불필요 (polynomial) | 불필요 |
| **이론 분석** | 쉬움 (closed-form spectrum) | 어려움 (multi-step composition) |
| **표현력** | Polynomial $K$ — explicit | $K$-layer $\Rightarrow$ $K$-hop, but composition 비자명 |

### Polynomial 의 양면

$g_\theta(L) = \sum_{k=0}^K \theta_k L^k$:

**Spectral 관점**: $\hat g(\lambda) = \sum_k \theta_k \lambda^k$ — frequency $\lambda$ 별 response $\hat g(\lambda)$ 를 polynomial 로 modeling.

**Spatial 관점**: $g(L) x = \sum_k \theta_k L^k x$ — $k$-hop neighbor information 의 weighted sum. 각 $L^k x$ 는 $k$-step diffusion.

**같은 식, 다른 해석**. 어느 관점에서 봐도 표현력 동일.

### GCN 의 spatial 의미 vs spectral 의미

**Spatial**: $\hat A H$ = 노드 자신 + 이웃의 normalized weighted sum.

**Spectral**: $\hat A = I - \tilde L_{\text{sym}}$ — 고유값이 $1 - \tilde\lambda$ 인 low-pass filter (작은 $\tilde\lambda$ 의 frequency 가 강조됨, 즉 smooth signal 통과).

따라서 GCN 한 layer = "low-pass smoothing + linear + ReLU" — spectral 관점이 over-smoothing 을 자연스럽게 설명.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Spatial GNN Layer (General Form)

$$
h_i^{(l+1)} = \phi_l\left( h_i^{(l)}, \bigoplus_{j \in N(i)} \psi_l(h_i^{(l)}, h_j^{(l)}) \right)
$$

- $\psi_l$: edge function (message)
- $\bigoplus$: aggregation (sum, mean, max)
- $\phi_l$: update function

이는 Ch3-01 의 MPNN framework 일반 형태.

### 정의 4.2 — Spectral GNN Layer

$$
H^{(l+1)} = \sigma(g_{\theta^{(l)}}(L) H^{(l)} W^{(l)})
$$

$g$ 가 임의 spectral filter — polynomial 일 때 spatial 형태로 환원.

### 정의 4.3 — Equivalence Classes

두 GNN layer 가 **함수적으로 동치**: 같은 input 에 같은 output. **표현력 동치**: 학습 가능한 함수의 집합이 동일.

---

## 🔬 정리와 증명

### 정리 4.1 — Polynomial Spectral ↔ Localized Spatial 동치

**Theorem**: $g_\theta(L) = \sum_{k=0}^K \theta_k L^k$ 인 spectral filter 는 다음 spatial form 과 정확히 동일:
$$
y_i = \sum_{k=0}^K \theta_k \sum_{j: \text{hop}(i,j) \leq k} (L^k)_{ij} x_j
$$

**증명**: $L^k$ 의 $(i,j)$ 성분은 hop$(i,j) \leq k$ 일 때만 nonzero. Matrix 곱 펼치면 자동으로 spatial weighted sum. $\square$

### 정리 4.2 — Spatial $K$-Hop Aggregation은 Polynomial Spectral

**Theorem (역방향)**: $K$-hop locality 와 graph-equivariance 를 만족하는 spatial aggregator 는 (선형 부분만) polynomial of $L$ 으로 표현 가능.

**Proof sketch**: Permutation equivariance + locality 두 조건이 spatial filter 의 표현 공간을 polynomial of $L$ + sparse pattern 으로 제한. Bronstein et al. (Geometric Deep Learning proto-book) 의 일반 정리.

**의미**: Spectral polynomial 과 spatial localized aggregation 은 **동치 클래스** — 어떤 관점에서 시작해도 같은 함수 family.

### 정리 4.3 — GCN 의 Spectral 해석

GCN propagation $\hat A = \tilde D^{-1/2} \tilde A \tilde D^{-1/2}$ 의 고유분해:
$$
\hat A = U_{\tilde A} \Lambda_{\hat A} U_{\tilde A}^T, \quad \lambda(\hat A) = 1 - \tilde\lambda(L_{\text{sym}}^{(\text{aug})})
$$

따라서 GCN 의 spectral filter 는:
$$
\hat g_{\text{GCN}}(\tilde\lambda) = 1 - \tilde\lambda
$$

**Low-pass**: 작은 $\tilde\lambda$ (smooth eigenvector) 일수록 1 에 가까움. 큰 $\tilde\lambda$ (oscillatory) 는 0 근처. 따라서 GCN = low-pass filter.

**의미**: 깊은 layer = 강한 low-pass = oscillatory components 제거 → 모든 노드 feature 평탄화 (over-smoothing).

### 정리 4.4 — GAT 의 Spectral 해석 (제한적)

GAT (Ch3-03) 의 attention-weighted aggregation 은 input-dependent (data-dependent). 따라서 fixed spectral filter 가 아닌 **input-dependent filter**.

**Theorem (제한)**: GAT 한 layer 는 fixed input 에 대해 polynomial of order 1 in (data-dependent) propagation matrix. 단, 학습 중에는 propagation matrix 자체가 변해 spectral 분석 어려움.

이것이 spectral 분석이 spatial GAT 에서 한계가 있는 이유 — 현대 GNN 의 spectral-spatial 통합은 여전히 활발 연구.

### 정리 4.5 — Defferrard 의 Trade-off 정리 (informal)

**"Spectral 의 장점은 이론, spatial 의 장점은 구현"** — Defferrard 2016:

| 차원 | Spectral | Spatial |
|------|----------|---------|
| Closed-form 표현력 분석 | ✓ | ✗ |
| Transferability (graph-invariant param) | ✓ (polynomial 만) | ✓ |
| 계산 효율 | $U$ 시 X, polynomial 시 ✓ | ✓ |
| Inductive learning | $\bigcirc$ (polynomial) | ✓ |
| Implementation 단순성 | 복잡 | 단순 |
| Modern GNN frameworks (PyG/DGL) | $\triangle$ | ✓ |

이 trade-off 가 **GCN 이 polynomial spectral 을 단순한 spatial form 으로 환원**하는 동기.

### 정리 4.6 — Message Passing 으로의 다리

$\hat A H = \tilde D^{-1/2} \tilde A \tilde D^{-1/2} H$ 의 row $i$:
$$
(\hat A H)_i = \sum_{j} \frac{\tilde A_{ij}}{\sqrt{\tilde d_i \tilde d_j}} h_j = \sum_{j \in N(i) \cup \{i\}} \frac{1}{\sqrt{\tilde d_i \tilde d_j}} h_j
$$

이는 정확히 message passing form (Gilmer 2017, Ch3-01):
$$
m_{ij} = \frac{1}{\sqrt{\tilde d_i \tilde d_j}} h_j, \quad h_i^{\text{new}} = \sum_j m_{ij}
$$

따라서 **GCN 은 specific aggregator (degree-normalized weighted sum) 의 message passing**. 이 관점이 Ch3 의 출발점.

---

## 💻 구현 검증

### 실험 1 — Spectral vs Spatial 같은 결과

```python
import numpy as np
import networkx as nx

G = nx.karate_club_graph()
n = G.number_of_nodes()
A = nx.adjacency_matrix(G).toarray().astype(float)
deg = A.sum(1)
D_inv_sqrt = np.diag(1/np.sqrt(deg))
L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

# Polynomial filter: g(λ) = 1 - λ + 0.5 λ²
def poly_filter_spectral(L_sym, x, coeffs):
    """Spectral form: U g(Λ) U^T x"""
    eigvals, U = np.linalg.eigh(L_sym)
    g_diag = sum(c * eigvals**k for k, c in enumerate(coeffs))
    return U @ np.diag(g_diag) @ U.T @ x

def poly_filter_spatial(L_sym, x, coeffs):
    """Spatial form: sum_k c_k L^k x"""
    Lk_x = x.copy()
    out = coeffs[0] * Lk_x
    for k in range(1, len(coeffs)):
        Lk_x = L_sym @ Lk_x
        out = out + coeffs[k] * Lk_x
    return out

x = np.random.randn(n)
coeffs = [1.0, -1.0, 0.5]   # 1 - λ + 0.5 λ²

y_spec = poly_filter_spectral(L_sym, x, coeffs)
y_spat = poly_filter_spatial(L_sym, x, coeffs)
print(f'Max diff (spectral - spatial): {np.abs(y_spec - y_spat).max():.2e}')
# 매우 작음 ≈ 0
```

### 실험 2 — GCN 의 Spectral Filter 시각화

```python
# GCN propagation: A_hat = D̃^(-1/2) Ã D̃^(-1/2)
A_tilde = A + np.eye(n)
d_tilde = A_tilde.sum(1)
A_hat = np.diag(1/np.sqrt(d_tilde)) @ A_tilde @ np.diag(1/np.sqrt(d_tilde))

# A_hat 의 spectrum
eig_hat = np.sort(np.linalg.eigvalsh(A_hat))
# L_sym^(aug) = I - A_hat 의 spectrum
eig_aug_L = 1 - eig_hat
print(f'Aug. L_sym eigenvalues range: [{eig_aug_L.min():.3f}, {eig_aug_L.max():.3f}]')
print(f'A_hat eigenvalues range:        [{eig_hat.min():.3f}, {eig_hat.max():.3f}]')

# GCN 한 layer 의 spectral response = A_hat 의 eigenvalue = 1 - tilde_lambda
import matplotlib.pyplot as plt
plt.plot(eig_aug_L, eig_hat, 'o-')
plt.xlabel('$\\tilde\\lambda$ (augmented L_sym)')
plt.ylabel('GCN filter response = $1 - \\tilde\\lambda$')
plt.title('GCN as low-pass filter')
plt.axhline(0, color='k', linestyle='--', alpha=0.3)
plt.grid(); plt.show()
```

### 실험 3 — Multi-Layer GCN 의 Spectral Response

```python
# L 번 적용 → A_hat^L
def gcn_filter_response(eig_aug_L, num_layers):
    return (1 - eig_aug_L) ** num_layers

fig, ax = plt.subplots(figsize=(8, 5))
for L in [1, 2, 4, 8, 16]:
    response = gcn_filter_response(eig_aug_L, L)
    ax.plot(eig_aug_L, response, 'o-', label=f'L={L}')

ax.set_xlabel('$\\tilde\\lambda$')
ax.set_ylabel('Filter response')
ax.set_title('Deep GCN = strong low-pass filter (over-smoothing 의 spectral 설명)')
ax.legend(); ax.grid(); plt.show()
```

**관찰**: $L$ 커질수록 small $\tilde\lambda$ (smooth) 만 통과, 다른 모든 frequency 가 0 근처로 → over-smoothing 의 spectral 해석.

### 실험 4 — Spatial Form 의 1-hop Equivalence

```python
# GCN spatial form: h_i^(new) = sum_{j ∈ N(i) ∪ {i}} 1/sqrt(d̃_i d̃_j) h_j
def gcn_spatial(A_hat, H):
    return A_hat @ H

# Vs explicit loop
def gcn_explicit(A, H):
    n = len(A)
    A_t = A + np.eye(n)
    d_t = A_t.sum(1)
    out = np.zeros_like(H)
    for i in range(n):
        for j in range(n):
            if A_t[i, j]:
                out[i] += H[j] / np.sqrt(d_t[i] * d_t[j])
    return out

H_in = np.random.randn(n, 8)
H_mat = gcn_spatial(A_hat, H_in)
H_loop = gcn_explicit(A, H_in)
print(f'Matrix vs loop: {np.abs(H_mat - H_loop).max():.2e}')   # ≈ 0
```

### 실험 5 — GIN vs GCN 의 Spectral 차이

```python
# GIN aggregator (sum, no normalization): h^(new) = (1+ε) h_i + sum_{j ∈ N(i)} h_j
# 등가: h^(new) = ((1+ε) I + A) H = M H

epsilon = 0.1
M_GIN = (1 + epsilon) * np.eye(n) + A
eig_GIN = np.sort(np.linalg.eigvalsh(M_GIN))

# GCN A_hat eigenvalues
eig_GCN = np.sort(np.linalg.eigvalsh(A_hat))

print(f'GCN A_hat eig:  range [{eig_GCN.min():.3f}, {eig_GCN.max():.3f}]')
print(f'GIN M eig:      range [{eig_GIN.min():.3f}, {eig_GIN.max():.3f}]')
# GIN spectrum 은 매우 다름 (정규화 X) → spectral 해석 다름

# 하지만 GIN 의 표현력 (1-WL) 가 GCN 보다 높음 (Ch4)
```

---

## 🔗 실전 활용

### 1. 어느 관점을 언제 쓰는가

- **Spectral**: Theoretical analysis (over-smoothing, expressive power), filter design (heterophily, band-pass)
- **Spatial / MPNN**: Implementation (PyG, DGL), inductive learning, large-scale

대부분의 modern GNN 라이브러리는 spatial 구현. Spectral 은 polynomial 로 환원 후 spatial 로 implement.

### 2. Heterophilic Graphs 와 Spectral

GCN/GAT 가 가정하는 "이웃과 비슷" 가정 (homophily) 이 깨지는 graph (heterophilic) 에서는 high-pass filter 필요. **GPRGNN** (Chien 2021), **FAGCN** (Bo 2021), **JacobiConv** (Wang 2022) 등 spectral filter 학습 — polynomial coefficient 자체를 학습.

### 3. Specformer (Bo 2023)

Spectral 관점의 부활: Transformer 를 사용해 eigenvalue token 위에서 attention. 각 frequency 별 학습 가능 filter. Polynomial 보다 더 풍부한 spectral 표현.

### 4. Diffusion Models on Graphs

Graph diffusion model = heat equation on graph = spectral filter. Generative graph model (GDSS, DiGress) 의 이론적 기반.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Linear filter (polynomial of $L$) | Non-linear filter (GAT attention) 는 spectral 분석 제한적 |
| Permutation equivariance | Position-aware (LapPE) 는 추가 처리 (Ch4-05) |
| Static spectral filter | Input-dependent filter (attention) 는 spectral 정의 모호 |
| Single propagation matrix | Multi-relational (R-GCN) 는 relation별 spectrum |
| Symmetric Laplacian | Directed graph 는 별도 일반화 |

---

## 📌 핵심 정리

$$\boxed{\text{Polynomial Spectral} \equiv \text{Localized Spatial}}$$

$$\boxed{\hat g_{\text{GCN}}(\tilde\lambda) = 1 - \tilde\lambda \quad \text{(low-pass filter)}}$$

| 관점 | Spectral | Spatial |
|------|----------|---------|
| **GCN** | Low-pass filter $1 - \tilde\lambda$ | $\tilde D^{-1/2} \tilde A \tilde D^{-1/2}$ |
| **ChebNet** | Polynomial $\sum \theta_k T_k$ | $\sum \theta_k T_k(\tilde L) X$ |
| **GAT** | Input-dependent (제한적 분석) | Attention-weighted sum |
| **GIN** | $(1+\epsilon)I + A$ — high spectrum | Sum + MLP |
| **APPNP** | Rational $(I - \alpha \hat A)^{-1}$ | PPR diffusion |
| **표현력** | Polynomial degree $K$ | $K$-layer MPNN ($K$-hop) |
| **이론 분석** | 쉬움 | 어려움 |
| **구현** | Polynomial → spatial form | 직접 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): GCN 의 spectral filter 가 $1 - \tilde\lambda$ 임을 직접 derive 하라.

<details>
<summary>해설</summary>

GCN propagation matrix: $\hat A = \tilde D^{-1/2} \tilde A \tilde D^{-1/2} = I - \tilde L_{\text{sym}}^{(\text{aug})}$

(여기서 $\tilde L_{\text{sym}}^{(\text{aug})} = I - \tilde D^{-1/2} \tilde A \tilde D^{-1/2}$, 즉 $\tilde A$ 의 normalized Laplacian)

$\tilde L^{(\text{aug})}_{\text{sym}}$ 의 spectrum $\tilde\lambda \in [0, 2)$.

$\hat A$ 의 spectrum $= 1 - \tilde\lambda \in (-1, 1]$.

따라서 GCN 한 layer 적용 = "각 frequency 의 spectral component $\hat x_k$ 에 $1 - \tilde\lambda_k$ 곱하기":

$$
\widehat{(\hat A x)}_k = (1 - \tilde\lambda_k) \hat x_k
$$

이는 $\hat g(\tilde\lambda) = 1 - \tilde\lambda$ 형태의 spectral filter — 작은 $\tilde\lambda$ 통과 (low-pass), 큰 $\tilde\lambda$ 약화. $\square$

</details>

**문제 2** (심화): $L$-layer GCN 의 effective spectral filter 가 $(1 - \tilde\lambda)^L$ 임을 보이고, 이것이 over-smoothing 을 어떻게 정량화하는지 설명하라.

<details>
<summary>해설</summary>

**Effective filter**:

$L$-layer GCN: $H^{(L)} = \hat A^L X (\prod W^{(l)})$ (linear part). $\hat A^L = U \text{diag}((1-\tilde\lambda_k)^L) U^T$.

따라서 effective spectral filter:
$$
\hat g_L(\tilde\lambda) = (1 - \tilde\lambda)^L
$$

**Over-smoothing 정량화**:

- $\tilde\lambda = 0$ (smooth, $\hat A$ 의 largest eigenvalue 1): $(1)^L = 1$ — 보존
- $\tilde\lambda > 0$ (other): $|1 - \tilde\lambda| < 1$ → $(1-\tilde\lambda)^L \to 0$ exponentially
- 따라서 $L \to \infty$ 에서 $\hat A^L X \to$ (largest eigenvector $\sqrt{\tilde d}$ 방향의 projection)

**수렴 속도**:

Second largest eigenvalue $\mu_2 = 1 - \tilde\lambda_2$ ($\tilde\lambda_2$ = augmented $L_{\text{sym}}$ 의 second smallest eigenvalue, Fiedler-like).

$\| \hat A^L X - $ projection $\| \sim (\mu_2)^L = (1 - \tilde\lambda_2)^L$.

**Spectral gap $\tilde\lambda_2$ 가 클수록 빠른 over-smoothing** — well-connected graph (expander) 에서 소수 layer 만에 collapse.

이는 Ch5-02 에서 정리.

</details>

**문제 3** (논문 비평): GAT 의 attention-weighted aggregation 이 input-dependent 라 spectral 분석이 제한적이다. 그럼에도 spectral 관점에서 GAT 를 보는 것이 어떤 통찰을 줄 수 있는가?

<details>
<summary>해설</summary>

**Input-dependent filter 의 spectral 해석 시도**:

각 input $X$ 에 대해 GAT 의 effective propagation matrix $P_{\text{GAT}}(X)$ 가 결정 — 이를 "data-dependent filter" 로 본다면:
- Linear regime (small attention variation): GCN-like spectrum + small perturbation
- Strong attention (one-hot $\alpha$): Selectively dropping edges → graph 자체가 sparser → spectrum 변화

**다음 통찰**:

1. **Adaptive locality**: 일부 edge 의 attention $\alpha_{ij} \to 0$ 이면 "soft edge removal" → spectrum 변화. 이는 DropEdge (Rong 2020, Ch5-03) 와 유사한 효과 — over-smoothing 완화 가능.

2. **Heterophily 대응**: Negative attention 학습 가능하면 high-pass filter 효과. 실제로 FAGCN (Bo 2021) 이 signed attention 으로 high-pass.

3. **Frequency selection**: Multi-head GAT 가 각 head 별로 다른 effective filter — multi-band filter bank 와 유사. 이론적으로 multi-aggregator (PNA, Corso 2020, Ch7-02) 와 같은 모티베이션.

4. **Sparse attention as graph perturbation**: GAT 의 attention 이 "graph 자체를 학습 중에 변형" 하는 것 — graph rewiring 과 같은 효과.

**결론**: GAT 의 spectral 분석은 정확하지 않지만, "input-dependent low-pass + adaptive edge weighting" 으로 해석 가능. 이는 spectral GNN literature 에서 점점 활성화되는 영역 — Ch7-02 의 GNN-Transformer 통합으로 이어짐.

</details>

---

<div align="center">

[◀ 이전](./03-gcn-derivation.md) | [📚 README](../README.md) | [다음 ▶](../ch3-message-passing/01-mpnn-framework.md)

</div>
