# 02. Over-smoothing의 Laplacian 분석

## 🎯 핵심 질문

- GCN propagation matrix $P = \tilde D^{-1/2} \tilde A \tilde D^{-1/2}$ 의 spectrum 분석에서 어떤 사실이 over-smoothing 을 결정하는가?
- $P^L x \to$ dominant eigenvector projection 임을 정확히 증명할 수 있는가?
- 수렴 속도 $O((\lambda_2)^L)$ 의 정확한 statement 와 spectral gap 의 의미?
- Connected graph 에서 모든 노드 feature 가 상수로 collapse 하는 정확한 한계 vector?
- Disconnected graph 의 경우 $\ker(L_{\text{sym}}^{(\text{aug})})$ 차원이 component 수와 같음을 어떻게 over-smoothing 에 적용?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

Ch5-01 의 over-smoothing 현상을 **수학적으로 정확히 증명** 합니다. Li et al. 2018 의 결정적 분석:

1. **GCN propagation = power iteration of $P$** — 같은 행렬을 반복 곱
2. **$P^L$ 이 dominant eigenvector 으로 collapse** — Perron-Frobenius
3. **수렴 속도 = spectral gap** — 정량적 bound
4. **$\ker(L_{\text{sym}}^{(\text{aug})})$ 가 over-smoothing 의 attractor**

이 정리는 over-smoothing 이 implementation 문제가 아니라 **수학적 inevitability** 임을 보임. 해결책 (DropEdge, PairNorm, APPNP) 의 동기를 정확히 제공.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-phenomenon.md](./01-phenomenon.md)
- [Ch1-04](../ch1-graph-laplacian/04-spectral-theory.md) — Spectral theorem, eigenvalue/eigenvector
- [Ch2-03](../ch2-spectral-gcn/03-gcn-derivation.md) — GCN propagation
- 선형대수: Spectral decomposition, Perron-Frobenius

---

## 📖 직관적 이해

### Power Iteration 으로서의 Multi-layer GCN

$L$-layer GCN (linear part 만):
$$
H^{(L)} = P^L H^{(0)} (\prod_l W^{(l)})
$$

Linear classifier 측면에서 $H^{(L)}$ 의 information 은 $P^L H^{(0)}$ 에 포함. 따라서 over-smoothing 분석 = $P^L$ 의 limit 분석.

**$P^L$ 의 spectral 행동**:
- $P = U \Lambda U^T$, $\Lambda = \text{diag}(1, \mu_2, \mu_3, \ldots)$
- Largest eigenvalue $\mu_1 = 1$, 다른 $|\mu_k| < 1$
- $P^L = U \Lambda^L U^T$
- $L \to \infty$: $\mu_1^L = 1$ (보존), $\mu_k^L \to 0$ for $k \geq 2$
- 따라서 $P^L \to v_1 v_1^T$ (rank-1)

**결과**: 모든 노드 feature 가 $v_1$ 방향으로 collapse → over-smoothing.

### Dominant Eigenvector 의 정체

$P = \tilde D^{-1/2} \tilde A \tilde D^{-1/2}$ 의 largest eigenvector $v_1 = \sqrt{\tilde d} / \|\sqrt{\tilde d}\|$ (Ch2-03 문제 2 참조).

따라서:
$$
\lim_{L \to \infty} (P^L H^{(0)})_i = \frac{\sqrt{\tilde d_i}}{\|\sqrt{\tilde d}\|} \cdot \langle \sqrt{\tilde d}, H^{(0)} \rangle / \|\sqrt{\tilde d}\|
$$

각 노드의 final feature 가 $\sqrt{\tilde d_i}$ 에 비례 — 모든 노드가 같은 방향 (after 정규화 cosine sim = 1).

### Spectral Gap의 결정적 역할

$|\mu_2|$ 가 "second-largest absolute eigenvalue" — 수렴 속도 결정:
$$
\|P^L H - v_1 v_1^T H\| \leq |\mu_2|^L \cdot \|H_\perp\|
$$

작은 $|\mu_2|$ (큰 spectral gap) → 빠른 over-smoothing.

**Bipartite-like graph**: $|\mu_n| \approx 1$ — slow mixing, but 다른 oscillation 위험.
**Expander graph**: $|\mu_2|$ 매우 작음 (e.g., $1/\sqrt{d}$) — 빠른 collapse.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — GCN Propagation Matrix

$$
P := \tilde D^{-1/2} \tilde A \tilde D^{-1/2}, \quad \tilde A = A + I, \tilde D = D + I
$$

Symmetric, $n \times n$. Spectral radius $\rho(P) = 1$.

### 정의 2.2 — Augmented Normalized Laplacian

$$
\tilde L_{\text{sym}} := I - P = I - \tilde D^{-1/2} \tilde A \tilde D^{-1/2}
$$

이는 self-loop 추가된 graph $\tilde G$ 의 normalized Laplacian. Eigenvalue $\tilde \lambda \in [0, 2)$ (self-loop 덕에 $< 2$ strict).

### 정의 2.3 — Dominant Eigenvector

$P$ 의 largest eigenvalue $1$ 의 eigenvector:
$$
v_1 = \frac{\sqrt{\tilde d}}{\|\sqrt{\tilde d}\|}, \quad (\sqrt{\tilde d})_i = \sqrt{d_i + 1}
$$

(Ch2-03 문제 2 에서 증명)

### 정의 2.4 — Spectral Gap

$$
\gamma = 1 - |\mu_2(P)| = \tilde \lambda_2(\tilde L_{\text{sym}})
$$

(Augmented Laplacian 의 second smallest eigenvalue. $\gamma > 0$ ⟺ connected graph $\tilde G$, always satisfied since $\tilde G$ has self-loops.)

### 정의 2.5 — Over-smoothing 의 Formal Limit

$$
\lim_{L \to \infty} P^L H = v_1 v_1^T H
$$

(rank-1 limit)

---

## 🔬 정리와 증명

### 정리 2.1 — $P$ 의 Spectral 성질

**Theorem**: $P = \tilde D^{-1/2} \tilde A \tilde D^{-1/2}$ 는:
1. Symmetric (real eigenvalues, orthonormal eigenvectors)
2. $\rho(P) = 1$ (largest eigenvalue exactly 1)
3. 다른 eigenvalue $\mu_k < 1$ for $k \geq 2$ (assuming connected $\tilde G$)
4. $|\mu_n| < 1$ — bipartite 효과 self-loop 으로 사라짐

**증명**:

(1) $P^T = \tilde D^{-1/2} \tilde A^T \tilde D^{-1/2} = P$ (symmetric $\tilde A$). Spectral theorem 적용.

(2) $\rho(P) = 1$:
- $P \sqrt{\tilde d} = \tilde D^{-1/2} \tilde A \tilde D^{-1/2} \sqrt{\tilde d} = \tilde D^{-1/2} \tilde A \mathbb 1 = \tilde D^{-1/2} \tilde d = \sqrt{\tilde d}$
- 따라서 $\sqrt{\tilde d}$ 가 eigenvalue 1 의 eigenvector.
- $\tilde L_{\text{sym}} = I - P \succeq 0$ (PSD by Ch1-02 정리 2.2 일반화) → $P \preceq I$ → $\rho(P) \leq 1$.
- 두 결합: $\rho(P) = 1$.

(3) Connected $\tilde G$ ⟹ $\dim \ker(\tilde L_{\text{sym}}) = 1$ ⟹ multiplicity of eigenvalue 0 is 1 in $\tilde L_{\text{sym}}$ ⟹ eigenvalue 1 of $P$ has multiplicity 1.

(4) Bipartite 시 standard normalized Laplacian 의 $\lambda_n = 2$ → $P$ 의 $\mu_n = -1$. Self-loop 추가 시 $\tilde G$ 가 bipartite 아님 (모든 노드 self-loop) → $|\mu_n| < 1$.

$\square$

### 정리 2.2 — $P^L H$ 의 Limit

**Theorem**: $H \in \mathbb R^{n \times d}$ 와 $P$ 위와 같음. 그러면:
$$
\lim_{L \to \infty} P^L H = v_1 v_1^T H = \frac{\sqrt{\tilde d} (\sqrt{\tilde d}^T H)}{\|\sqrt{\tilde d}\|^2}
$$

(Rank-1 outer product. 각 column $j$ 에 대해 $(P^L H)_{:, j} \to v_1 \cdot \alpha_j$ where $\alpha_j = v_1^T H_{:, j}$.)

**증명**: Spectral 분해 $P = \sum_k \mu_k v_k v_k^T$, $|\mu_1| = 1 > |\mu_2| \geq \ldots$.

$$
P^L = \sum_k \mu_k^L v_k v_k^T = v_1 v_1^T + \sum_{k \geq 2} \mu_k^L v_k v_k^T
$$

$|\mu_k| < 1$ for $k \geq 2$ ⟹ $\mu_k^L \to 0$. 따라서:
$$
\lim_L P^L = v_1 v_1^T
$$

$P^L H \to v_1 v_1^T H$. $\square$

### 정리 2.3 — 수렴 속도

**Theorem**: $\|P^L H - v_1 v_1^T H\|_F \leq |\mu_2|^L \cdot \|H\|_F$.

**증명**: $P^L H - v_1 v_1^T H = \sum_{k \geq 2} \mu_k^L v_k (v_k^T H)$.

Frobenius norm:
$$
\|\ldots\|_F^2 = \sum_{k \geq 2} \mu_k^{2L} \|v_k^T H\|^2 \leq |\mu_2|^{2L} \sum_{k \geq 2} \|v_k^T H\|^2 \leq |\mu_2|^{2L} \|H\|_F^2
$$

(Bessel-like inequality + $\mu_k^2 \leq \mu_2^2$.)

$\sqrt{} \Rightarrow \|P^L H - v_1 v_1^T H\|_F \leq |\mu_2|^L \|H\|_F$. $\square$

### 정리 2.4 — Connected Graph 에서 Constant Vector Collapse

**Corollary**: Connected $\tilde G$ 에서 over-smoothing 의 limit 이 constant vector:
$$
\lim_L (P^L H)_i = c \cdot \sqrt{\tilde d_i}
$$

(각 column 의 $i$-th component 가 $\sqrt{\tilde d_i}$ 에 비례)

**Cosine similarity**: $\cos(h_i^{(L)}, h_j^{(L)}) \to 1$ for all pairs $(i, j)$ as $L \to \infty$.

### 정리 2.5 — Disconnected Graph 의 경우

**Theorem**: Graph $G$ 가 $k$ connected component 면, $\dim \ker(\tilde L_{\text{sym}}) = k$ — eigenvalue 1 of $P$ has multiplicity $k$.

$P^L H \to $ projection onto $\ker(\tilde L_{\text{sym}})$ (dim $k$ subspace). 각 component 내에서 over-smoothing, but components 사이에는 정보 흐름 없음.

(Ch1-02 정리 2.3 의 일반화)

### 정리 2.6 — Nonlinearity 와 Bias 의 효과

**Caveat**: 위 분석은 linear propagation. Real GCN 은 ReLU 와 weight matrix $W^{(l)}$ 적용:
$$
H^{(l+1)} = \sigma(P H^{(l)} W^{(l)})
$$

ReLU 가 약간의 nonlinearity 를 추가 — over-smoothing 약간 완화. 단 essential dynamics 는 같음 (Oono & Suzuki 2020 정리: $\sigma$ 도 over-smoothing 를 못 막음).

또한 $W^{(l)}$ 가 학습으로 over-smoothing 을 더 가속할 수도 (특히 weight norm 이 작으면). 따라서 정리 2.4 는 **upper bound on possible expressiveness**.

---

## 💻 NumPy 구현 검증

### 실험 1 — $P^L$ 의 Rank-1 Convergence

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
n = G.number_of_nodes()
A = nx.adjacency_matrix(G).toarray().astype(float)

# GCN P matrix
A_tilde = A + np.eye(n)
d_tilde = A_tilde.sum(1)
P = np.diag(1/np.sqrt(d_tilde)) @ A_tilde @ np.diag(1/np.sqrt(d_tilde))

# Eigenvalues
eigvals, eigvecs = np.linalg.eigh(P)
print(f'P eigenvalues (sorted descending):')
print(f'  μ_1 = {eigvals[-1]:.6f}')   # ≈ 1
print(f'  μ_2 = {eigvals[-2]:.6f}')
print(f'  |μ_2| = {abs(eigvals[-2]):.6f}')
print(f'  μ_n = {eigvals[0]:.6f}')

# Dominant eigenvector
v1 = eigvecs[:, -1]
v1_theory = np.sqrt(d_tilde) / np.linalg.norm(np.sqrt(d_tilde))
print(f'v_1 == sqrt(d̃)/||sqrt(d̃)||? {np.allclose(np.abs(v1), np.abs(v1_theory))}')
```

### 실험 2 — Convergence Rate Verification

```python
np.random.seed(0)
H0 = np.random.randn(n, 4)
v1_outer_H0 = v1.reshape(-1, 1) @ (v1.reshape(1, -1) @ H0)

errors = []
H_l = H0.copy()
for L in range(50):
    err = np.linalg.norm(H_l - v1_outer_H0, 'fro')
    errors.append(err)
    H_l = P @ H_l

theoretical_rate = abs(eigvals[-2])
plt.semilogy(errors, 'o-', label='actual')
plt.semilogy([errors[0] * theoretical_rate**L for L in range(50)], '--',
             label=f'$|\\mu_2|^L$ = {theoretical_rate:.3f}^L')
plt.xlabel('L'); plt.ylabel('||P^L H - v_1 v_1^T H||_F (log)')
plt.title(f'Over-smoothing rate (Karate Club)')
plt.legend(); plt.grid(); plt.show()
```

### 실험 3 — 수렴 후의 Cosine Similarity

```python
for L in [0, 5, 10, 20, 50]:
    H_L = np.linalg.matrix_power(P, L) @ H0
    H_norm = H_L / (np.linalg.norm(H_L, axis=1, keepdims=True) + 1e-8)
    cos_sim = H_norm @ H_norm.T
    avg = (cos_sim.sum() - cos_sim.diagonal().sum()) / (n * (n - 1))
    print(f'L={L:3d}: avg pairwise cos sim = {avg:.4f}')
```

**예상**: $L = 50$ 시 $\approx 1$.

### 실험 4 — Disconnected Graph 의 Component-wise Over-smoothing

```python
# 2 components: {0..4} 와 {5..9}
G_disc = nx.Graph()
G_disc.add_nodes_from(range(10))
G_disc.add_edges_from([(0,1), (1,2), (2,3), (3,4), (5,6), (6,7), (7,8), (8,9)])

A_d = nx.adjacency_matrix(G_disc, nodelist=range(10)).toarray().astype(float)
A_td = A_d + np.eye(10)
d_td = A_td.sum(1)
P_d = np.diag(1/np.sqrt(d_td)) @ A_td @ np.diag(1/np.sqrt(d_td))

eig_d, U_d = np.linalg.eigh(P_d)
print(f'Disconnected: # eigenvalues = 1: {np.sum(np.abs(eig_d - 1) < 1e-6)}')
# Should be 2 (= # components)

# 매우 깊은 layer 후
H = np.random.randn(10, 4)
H_inf = np.linalg.matrix_power(P_d, 100) @ H
print('\nAfter L=100, feature norms by component:')
print(f'  Comp 1 (nodes 0-4): {np.linalg.norm(H_inf[:5], axis=1)}')
print(f'  Comp 2 (nodes 5-9): {np.linalg.norm(H_inf[5:], axis=1)}')
# 각 component 내에서 collapse, 하지만 components 사이에는 다른 값
```

### 실험 5 — Spectral Gap 비교 (Karate vs Cycle)

```python
def get_spectral_gap(G):
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    A_t = A + np.eye(n)
    d_t = A_t.sum(1)
    P = np.diag(1/np.sqrt(d_t)) @ A_t @ np.diag(1/np.sqrt(d_t))
    eigs = np.sort(np.linalg.eigvalsh(P))[::-1]
    return 1 - abs(eigs[1])   # spectral gap

for name, G_test in [
    ('Karate Club', nx.karate_club_graph()),
    ('Path P_10', nx.path_graph(10)),
    ('Cycle C_10', nx.cycle_graph(10)),
    ('Complete K_10', nx.complete_graph(10)),
    ('Erdos-Renyi (n=20, p=0.3)', nx.erdos_renyi_graph(20, 0.3, seed=0)),
]:
    gap = get_spectral_gap(G_test)
    print(f'{name}: spectral gap = {gap:.4f}, '
          f'over-smoothing rate per layer = {1 - gap:.4f}')
```

**관찰**: Path graph 는 spectral gap 작음 (느린 over-smoothing). Complete graph 는 큼 (즉시 collapse).

---

## 🔗 실전 활용

### 1. Spectral Gap 측정으로 Over-smoothing 예측

학습 전 spectral gap 측정으로 GNN 깊이 예산 결정:
- 큰 spectral gap → 적은 layer (collapse 빠름)
- 작은 spectral gap → 더 많은 layer 가능 (Path-like graph)

### 2. APPNP 의 동기 (Ch5-05)

APPNP propagation: $Z = (1 - \alpha)(I - \alpha P)^{-1} H_0$. 

$(I - \alpha P)^{-1}$ 의 spectral 분해:
$$
(I - \alpha P)^{-1} = \sum_k \frac{1}{1 - \alpha \mu_k} v_k v_k^T
$$

각 eigenvector $v_k$ 의 weight $\frac{1}{1 - \alpha \mu_k}$ — **모든 spectral 방향 보존** (specifically dominant 만 collapse 안 됨). 이것이 over-smoothing 회피.

### 3. JKN 의 동기 (Ch5-05)

JKN: $H_{\text{final}} = \text{concat}(H^{(1)}, \ldots, H^{(L)})$. 각 layer 의 representation 보존 → final 이 over-smoothed 라도 earlier layer 가 정보 보유.

### 4. GCNII 의 동기 (Chen 2020)

Initial residual $H^{(l+1)} = \sigma((1-\alpha)P H^{(l)} + \alpha H^{(0)}) W$. $\alpha H^{(0)}$ teleport 으로 $\ker$ collapse 회피 — 64-layer 까지 가능.

### 5. 표현력 vs Over-smoothing

GIN (sum aggregator) 는 over-smoothing 더 심함 (sum 의 magnitude 폭발). PairNorm 또는 batch norm 으로 완화.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Connected graph $\tilde G$ | Disconnected → component-wise (정리 2.5) |
| Linear propagation 분석 | Real GCN 의 ReLU 가 약간 다른 dynamics |
| Trained $W^{(l)}$ 무시 | Weight matrix 가 over-smoothing 가속 가능 |
| Static graph | Dynamic graph 의 evolving spectrum |
| Same $P$ 매 layer | DropEdge 등은 stochastic $P$ |
| Mean cosine sim metric | 다른 metric 의 dynamics 약간 다름 |

---

## 📌 핵심 정리

$$\boxed{P^L H \to v_1 v_1^T H \quad \text{as } L \to \infty}$$

$$\boxed{\|P^L H - v_1 v_1^T H\|_F \leq |\mu_2|^L \|H\|_F \quad \text{(rate)}}$$

$$\boxed{v_1 = \sqrt{\tilde d}/\|\sqrt{\tilde d}\|, \quad \mu_2 = 1 - \tilde\lambda_2(\tilde L_{\text{sym}})}$$

| 사실 | 의미 |
|------|------|
| **Connected**: $\dim \ker(\tilde L) = 1$ | 모든 노드 같은 limit |
| **Disconnected**: $\dim \ker = k$ | $k$ component 별 collapse |
| **Spectral gap $\tilde\lambda_2$** | 수렴 속도 결정 |
| **Convergence**: $O((1-\tilde\lambda_2)^L)$ | exponentially fast |
| **Limit vector**: $\sqrt{\tilde d}$ | degree 의 sqrt 비례 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $K_4$ (complete 4-graph) 의 $P$ matrix 와 spectrum 을 손으로 계산하라. Over-smoothing 속도가 빠른 이유 분석.

<details>
<summary>해설</summary>

$K_4$: 모든 node $d = 3$, $\tilde d = 4$. $\tilde A = J$ (all ones).

$P = (1/\sqrt 4) J (1/\sqrt 4) = J/4$.

$P$ eigenvalues:
- $J$ eigenvalues: $\{4, 0, 0, 0\}$
- $J/4$ eigenvalues: $\{1, 0, 0, 0\}$

Spectral gap: $1 - |\mu_2| = 1 - 0 = 1$ (max possible).

**의미**: 한 layer 후 즉시 over-smoothed. $P^1 = J/4$ — 모든 row 같음 (uniform 평균).

이는 **dense / well-connected graph** 의 빠른 collapse 의 극단 예.

**일반**: $K_n$: spectrum $\{1, 0, \ldots, 0\}$. 한 step 으로 random walk stationary 도달.

**대비**: $C_n$ (cycle): spectrum $\cos(2\pi k/n)$ 분포 — small gap, slow.

따라서 graph topology 가 over-smoothing 속도를 결정. $\square$

</details>

**문제 2** (심화): Real GCN 의 weight matrix $W^{(l)}$ 와 ReLU 가 over-smoothing 분석에 어떻게 영향을 주는가? Oono-Suzuki 2020 의 결과 review.

<details>
<summary>해설</summary>

**Linear analysis (정리 2.2)**: $H^{(L)} = P^L H^{(0)} \prod_l W^{(l)}$. Spectrum dynamics 가 $P^L$ 에 의해 결정.

**ReLU 의 영향**:

$H^{(l+1)} = \sigma(P H^{(l)} W^{(l)})$. ReLU $\sigma(x) = \max(x, 0)$.

**Oono-Suzuki 2020 정리**: ReLU 가 있어도 over-smoothing 발생. 정확히는 :
$$
\text{rank}(H^{(l)}) \leq \text{rank}(H^{(l-1)})
$$

(ReLU 가 rank 늘리지 못함)

따라서 $H^{(L)}$ 의 rank 가 layer 마다 단조 감소 — eventually rank-1 (over-smoothing 의 형태). 

**Weight matrix $W^{(l)}$ 의 영향**:

$\|W\|$ 가 작으면 (small-norm weight) → information 압축 → over-smoothing 가속.
$\|W\|$ 가 크면 → numerical issue, 단 information bottleneck 약함.

**Practical mitigation**:
- **Initialization**: Xavier/He init 가 평균 norm $\approx 1$
- **Weight regularization**: $L_2$ penalty 가 too small weight 방지
- **BatchNorm**: rescale 으로 dynamics 안정

**결론**: 비선형 GCN 도 본질적 over-smoothing 가짐 — linear 분석이 여전히 valuable. Real-world 학습에서는 어떻게 $\|W\|$ 가 학습되는지 관찰해야.

**Empirical (Oono-Suzuki 2020)**: $L = 50$ 레이어 GCN 에서도 동일 dynamics — weight 와 nonlinearity 가 약 1.2~1.5x slow factor 만 더할 뿐, 본질적 collapse.

</details>

**문제 3** (논문 비평): GCNII (Chen 2020) 가 initial residual 으로 깊은 GNN 가능하게 한 것을 spectral 관점에서 설명하라.

<details>
<summary>해설</summary>

**GCNII propagation**:
$$
H^{(l+1)} = \sigma\left( ((1 - \alpha) P + \alpha I_n) H^{(l)} ((1 - \beta_l) I + \beta_l W^{(l)}) \right)
$$

- $\alpha$: initial residual coefficient (보통 0.1)
- $\beta_l = \log(1/l + 1)$: identity mapping coefficient (decreasing)

**Spectral 분석**:

Effective propagation: $\hat P = (1 - \alpha) P + \alpha I$.

Spectrum: $\hat \mu_k = (1 - \alpha) \mu_k + \alpha$.

- $\hat \mu_1 = (1 - \alpha) + \alpha = 1$ (보존)
- $\hat \mu_2 = (1 - \alpha) \mu_2 + \alpha < 1$ — but **$\hat \mu_2$ 가 $\mu_2$ 보다 1 에 더 가까움**

**Key insight**: Spectral gap $1 - |\hat \mu_2| = (1 - \alpha)(1 - |\mu_2|) = (1 - \alpha) \cdot \gamma$.

- 원본 GCN: gap = $\gamma$
- GCNII: gap = $(1 - \alpha) \gamma$ — $\alpha = 0.1$ 시 gap 의 90%

따라서 **수렴 속도가 약간 느려짐** — over-smoothing 지연.

**더 중요한 효과**: $\alpha I$ teleport 이 initial feature $H^{(0)}$ 를 보존:

$H^{(L)} = \hat P^L H^{(0)} (\text{weights})$

$\hat P^L$ 가 collapse 해도, "effective receptive field" 가 천천히 확장. 

$\alpha = 0$ 시 GCN, $\alpha = 1$ 시 MLP. 0.1 이 sweet spot — graph 정보 + initial 정보 balance.

**Identity mapping $\beta_l$**: Each layer 의 weight 가 identity 에 가까워야 (작은 perturbation). 이것이 ResNet identity mapping 의 graph 일반화.

**결과**: GCNII 64-layer 가 vanilla GCN 16-layer 와 동등 또는 우월. Cora 84.4%, Citeseer 73.4%, Pubmed 80.4% (Chen 2020 reported).

**GCNII vs APPNP (Ch5-05)**:
- GCNII: layer-wise 에서 initial residual + identity mapping (parameterized)
- APPNP: closed-form PPR — non-parametric propagation + 단일 MLP

Conceptual 같지만 implementation 다름. APPNP 는 더 simple, GCNII 는 더 expressive.

</details>

---

<div align="center">

[◀ 이전](./01-phenomenon.md) | [📚 README](../README.md) | [다음 ▶](./03-dropedge-pairnorm.md)

</div>
