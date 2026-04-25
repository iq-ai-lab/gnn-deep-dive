# 03. Normalized Laplacian — Symmetric and Random Walk

## 🎯 핵심 질문

- Symmetric normalized Laplacian $L_{\text{sym}} = D^{-1/2} L D^{-1/2}$ 와 random walk Laplacian $L_{\text{rw}} = D^{-1} L$ 의 정의와 차이는?
- 두 normalized Laplacian의 고유값이 왜 $[0, 2]$ 에 있는가?
- $L_{\text{sym}}$ 과 $L_{\text{rw}}$ 의 고유값은 일치하지만 고유벡터가 다른 이유는?
- Bipartite graph에서 왜 $\lambda_{\max} = 2$ 가 정확히 달성되는가?
- GCN이 왜 $L_{\text{sym}}$ 기반인지, GraphSAGE가 왜 $L_{\text{rw}}$ 기반인지?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

Unnormalized Laplacian $L = D - A$ 는 이론적으로 깔끔하지만 실전에서 두 가지 한계가 있습니다:

1. **고유값 scale이 그래프에 의존** — $\lambda_{\max}(L) \leq 2 \Delta$ ($\Delta$ = max degree). Hub node가 있으면 $\lambda_{\max}$ 가 매우 커짐 → polynomial filter (ChebNet) 의 stability 문제.
2. **Degree-bias** — 고차수 노드의 영향력이 과대. Random walk 관점에서 stationary distribution이 uniform이 아닌 $\pi \propto d$.

**Normalization은 두 문제를 모두 해결**합니다. Spectral GCN의 모든 결과는 $L_{\text{sym}}$ 또는 $L_{\text{rw}}$ 위에서 전개되며, GCN 자체도 $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} = I - L_{\text{sym}}^{(\text{aug})}$ 형태입니다. 이 문서에서는 **두 normalized Laplacian의 정의·성질·관계**를 정리합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [02-unnormalized-laplacian.md](./02-unnormalized-laplacian.md) — $L$ 의 PSD, kernel
- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Similarity transform, congruence, generalized eigenvalue
- 확률론: Stochastic matrix, stationary distribution

---

## 📖 직관적 이해

### 두 가지 정규화 전략

$L = D - A$ 에서 degree-bias를 제거하는 두 방법:

1. **Symmetric normalization**: $D^{-1/2}$ 를 양쪽에 곱해 symmetry 유지.
   $$L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}$$

2. **Random walk normalization**: 왼쪽에만 $D^{-1}$ 곱해 row-stochastic 의미.
   $$L_{\text{rw}} = D^{-1} L = I - D^{-1} A$$

   여기서 $P := D^{-1} A$ 는 **transition matrix** — $P_{ij}$ = $i$ 에서 $j$ 로 갈 확률.

### Symmetric vs Stochastic의 trade-off

- $L_{\text{sym}}$: symmetric → 실수 고유값 + 직교 고유벡터 (spectral 분석에 좋음). 단, 직접적인 확률 해석 없음.
- $L_{\text{rw}}$: stochastic interpretation 명확. 단, 비대칭 (실수 고유값이지만 고유벡터 직교 X).

### 고유값 $[0, 2]$ Bound의 직관

$L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}$. $D^{-1/2} A D^{-1/2}$ 의 고유값을 $\mu$ 라 하면 $L_{\text{sym}}$ 고유값은 $1 - \mu$. $\|D^{-1/2} A D^{-1/2}\| \leq 1$ 임을 보이면 $\mu \in [-1, 1]$, 따라서 $1 - \mu \in [0, 2]$.

### Bipartite와 $\lambda_{\max} = 2$

Bipartite graph에서 $A$ 의 spectrum이 $\{\pm \mu\}$ 짝으로 나타남 (홀수 cycle 없음). $\mu = -1$ ($D^{-1/2}AD^{-1/2}$ 의 minimal eigenvalue 도달) 가능 → $L_{\text{sym}}$ 의 maximal eigenvalue $= 2$.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Symmetric Normalized Laplacian

Connected graph (no isolated nodes, $d_i > 0$) 에서:
$$
L_{\text{sym}} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}
$$

Symmetric (정의에서 직접 확인). Real eigenvalues, orthonormal eigenvectors.

### 정의 3.2 — Random Walk Laplacian

$$
L_{\text{rw}} = D^{-1} L = I - D^{-1} A = I - P
$$

여기서 $P = D^{-1} A$ 는 **random walk transition matrix** (row-stochastic).

비대칭이지만 real eigenvalues (similar to $L_{\text{sym}}$).

### 정의 3.3 — Normalized Adjacency

종종 다음 두 표기를 사용:
- $\hat{A} := D^{-1/2} A D^{-1/2}$ (symmetric normalized adj)
- $P = D^{-1} A$ (row-stochastic transition)

각각 $L_{\text{sym}} = I - \hat{A}$, $L_{\text{rw}} = I - P$.

---

## 🔬 정리와 증명

### 정리 3.1 — $L_{\text{sym}}$ 과 $L_{\text{rw}}$ 의 고유값 일치

**Theorem**: $\lambda$ 가 $L_{\text{sym}}$ 의 고유값 $\Leftrightarrow$ $\lambda$ 가 $L_{\text{rw}}$ 의 고유값.

**증명**: $L_{\text{rw}} = D^{-1/2} L_{\text{sym}} D^{1/2}$ (similarity transform).

확인: $D^{-1/2} L_{\text{sym}} D^{1/2} = D^{-1/2} (D^{-1/2} L D^{-1/2}) D^{1/2} = D^{-1} L = L_{\text{rw}}$. ✓

Similar matrices share eigenvalues. $\square$

**고유벡터 관계**: $L_{\text{sym}} v = \lambda v$ 이면 $L_{\text{rw}} (D^{-1/2} v) = \lambda (D^{-1/2} v)$. 따라서 $u = D^{-1/2} v$ 가 $L_{\text{rw}}$ 고유벡터.

### 정리 3.2 — 고유값 Bound $\lambda \in [0, 2]$

**Theorem**: $L_{\text{sym}}$ (그리고 $L_{\text{rw}}$) 의 고유값 $\lambda \in [0, 2]$.

**증명** ($L_{\text{sym}}$ 에 대해, $L_{\text{rw}}$ 는 정리 3.1에서 따름):

$y = D^{1/2} x$ 로 치환:
$$
\frac{x^T L_{\text{sym}} x}{x^T x} = \frac{x^T D^{-1/2} L D^{-1/2} x}{x^T x}
$$

$y = D^{-1/2} x$ 로 두면 $x = D^{1/2} y$:
$$
= \frac{y^T L y}{y^T D y} = \frac{\frac{1}{2} \sum_{(i,j) \in E} (y_i - y_j)^2}{\sum_i d_i y_i^2}
$$

(분자는 $L$ 의 quadratic form, 정리 2.1)

**하한** $\geq 0$: 분자 $\geq 0$, 분모 $> 0$. ✓

**상한** $\leq 2$: $(y_i - y_j)^2 \leq 2(y_i^2 + y_j^2)$ (AM-QM). 따라서:
$$
\frac{1}{2} \sum_{(i,j) \in E} (y_i - y_j)^2 \leq \sum_{(i,j) \in E} (y_i^2 + y_j^2) = \sum_i d_i y_i^2
$$

(각 노드 $i$ 가 $d_i$ 개 엣지에 속하므로 $y_i^2$ 가 $d_i$ 번 등장)

따라서 Rayleigh quotient $\leq 2$. $\lambda_{\max}(L_{\text{sym}}) \leq 2$. $\square$

### 정리 3.3 — $\lambda_{\max} = 2$ Iff Bipartite

**Theorem**: Connected graph $G$ 에서 $\lambda_{\max}(L_{\text{sym}}) = 2$ $\Leftrightarrow$ $G$ 는 bipartite.

**증명 sketch**:

**($\Leftarrow$)** $G = (V_1 \cup V_2, E)$ bipartite. 정의 $y_i = +1$ for $i \in V_1$, $y_i = -1$ for $i \in V_2$. 모든 엣지 $(i, j)$ 가 $V_1, V_2$ 사이이므로 $(y_i - y_j)^2 = 4$:
$$
\frac{\frac{1}{2} \sum_E 4}{\sum_i d_i \cdot 1} = \frac{2m}{2m} = 2
$$

이 vector에서 Rayleigh quotient = 2. 그런데 Rayleigh max = 2 가 max bound. → equal $\square$.

**($\Rightarrow$)** $\lambda_{\max} = 2$ 달성 vector $y$ 는 모든 엣지에서 $(y_i - y_j)^2 = 2(y_i^2 + y_j^2) \Leftrightarrow y_i = -y_j$. 이는 노드를 $\{y > 0, y < 0\}$ 로 split했을 때 모든 엣지가 양 집합 사이 → bipartite. $\square$

### 정리 3.4 — Random Walk Stationary Distribution

**Theorem**: $L_{\text{rw}} = I - P$ 의 nullspace는 $\pi^T = (d_1, d_2, \ldots, d_n) / (2m)$ 의 left eigenvector를 포함 (좌eigenvector).

**증명**: $\pi^T P = \pi^T$ 확인:
$$
(\pi^T P)_j = \sum_i \pi_i P_{ij} = \sum_i \frac{d_i}{2m} \cdot \frac{A_{ij}}{d_i} = \frac{1}{2m} \sum_i A_{ij} = \frac{d_j}{2m} = \pi_j
$$

따라서 $\pi^T (I - P) = 0$, i.e. $\pi$ 가 $L_{\text{rw}}$ 의 left null vector. $\square$

**의미**: Random walk stationary $\pi_i \propto d_i$ — 고차수 노드일수록 자주 방문. PageRank 의 base case (Ch1-06).

### 정리 3.5 — Normalized 와 Unnormalized 의 관계

$\lambda$ ($L_{\text{sym}}$ 고유값) 와 $\lambda'$ ($L$ 고유값) 사이의 정확한 관계는 **존재하지 않음** (둘 다 graph 의존). 단:
- 둘 다 $\lambda_1 = 0$, $\dim \ker$ = #components
- 일반적으로 $L_{\text{sym}}$ 는 well-conditioned ($\lambda \in [0, 2]$), $L$ 은 그래프마다 scale 다름

---

## 💻 NumPy 구현 검증

### 실험 1 — 두 Normalized Laplacian 구성

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

G = nx.karate_club_graph()
n = G.number_of_nodes()
A = nx.adjacency_matrix(G).toarray().astype(float)
deg = A.sum(1)
D = np.diag(deg)
D_inv_sqrt = np.diag(1 / np.sqrt(deg))
D_inv = np.diag(1 / deg)

L = D - A
L_sym = D_inv_sqrt @ L @ D_inv_sqrt
L_rw = D_inv @ L

# 검증: L_sym = I - D^{-1/2} A D^{-1/2}
A_hat = D_inv_sqrt @ A @ D_inv_sqrt
print(f'L_sym == I - D^(-1/2) A D^(-1/2)? {np.allclose(L_sym, np.eye(n) - A_hat)}')

# Symmetry
print(f'L_sym symmetric? {np.allclose(L_sym, L_sym.T)}')  # True
print(f'L_rw symmetric? {np.allclose(L_rw, L_rw.T)}')      # False
```

### 실험 2 — 고유값 일치 확인

```python
eig_sym = np.linalg.eigvalsh(L_sym)   # symmetric → eigh
eig_rw = np.sort(np.linalg.eigvals(L_rw).real)   # 비대칭 → eig

print(f'L_sym eigenvalues (first 5): {eig_sym[:5]}')
print(f'L_rw  eigenvalues (first 5): {eig_rw[:5]}')
print(f'Equal (within tolerance)? {np.allclose(eig_sym, eig_rw, atol=1e-6)}')

print(f'\nλ_max(L_sym) = {eig_sym[-1]:.6f}  (≤ 2)')
print(f'λ_max(L_rw)  = {eig_rw[-1]:.6f}  (≤ 2)')
```

### 실험 3 — Bipartite Graph의 $\lambda_{\max} = 2$

```python
G_bip = nx.complete_bipartite_graph(3, 4)   # K_{3,4}
A_b = nx.adjacency_matrix(G_bip).toarray().astype(float)
deg_b = A_b.sum(1)
D_b_inv_sqrt = np.diag(1 / np.sqrt(deg_b))
L_sym_b = np.eye(7) - D_b_inv_sqrt @ A_b @ D_b_inv_sqrt
eig_b = np.linalg.eigvalsh(L_sym_b)
print(f'Bipartite K_{{3,4}} L_sym eigenvalues:')
print(eig_b.round(6))
print(f'λ_max = {eig_b[-1]:.6f}  (= 2 exactly for bipartite)')
```

**출력**:
```
Bipartite K_{3,4} L_sym eigenvalues:
[-0.        1.        1.        1.        1.        1.        2.      ]
λ_max = 2.000000  (= 2 exactly for bipartite)
```

### 실험 4 — Random Walk Stationary

```python
P = D_inv @ A
# 좌 stationary: π^T P = π^T
pi_target = deg / deg.sum()
pi_check = pi_target @ P
print(f'π^T P 와 π^T 일치? {np.allclose(pi_check, pi_target)}')   # True

# Power iteration으로 stationary 찾기
v = np.ones(n) / n
for _ in range(1000):
    v = v @ P
print(f'Power iteration stationary == d/(2m)? {np.allclose(v, pi_target)}')
```

### 실험 5 — 고유벡터 관계 ($v$ vs $u = D^{-1/2} v$)

```python
eigvals_sym, eigvecs_sym = np.linalg.eigh(L_sym)
# 두 번째 작은 고유값의 eigenvector (Fiedler 유사)
v = eigvecs_sym[:, 1]
u = D_inv_sqrt @ v   # L_rw eigenvector

# L_rw u = λ u 검증
lhs = L_rw @ u
rhs = eigvals_sym[1] * u
print(f'L_rw u = λ u? {np.allclose(lhs, rhs)}')  # True
```

---

## 🔗 실전 활용

### 1. GCN의 $\tilde{L}_{\text{sym}}$

GCN propagation $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$ 는 self-loop 추가 후의 normalized adjacency. 등가로 $I - L_{\text{sym}}^{(\text{aug})}$ 형태이며, $L_{\text{sym}}^{(\text{aug})}$ 는 $\tilde{A} = A + I$ 의 normalized Laplacian. Self-loop 덕에 $\lambda_{\max} < 2$ 보장 (bipartite도 $\tilde{A}$ 에선 bipartite 아님). Ch2-03 참조.

### 2. GraphSAGE의 mean aggregator $\approx L_{\text{rw}}$

$h_i^{(l+1)} = \sigma(W \cdot \text{mean}_{j \in N(i)} h_j^{(l)})$ 의 mean = $\frac{1}{d_i} \sum_j A_{ij} h_j$ — random walk Laplacian의 적용 형태.

### 3. Spectral Clustering의 두 변종

- **Normalized cut** (Shi-Malik 2000): $L_{\text{rw}}$ 의 generalized eigenvalue problem $L u = \lambda D u$
- **Normalized cut** (Ng-Jordan-Weiss 2001): $L_{\text{sym}}$ 고유벡터를 row-normalize 후 k-means

두 방법 모두 $L_{\text{sym}}$/$L_{\text{rw}}$ 사용으로 degree-bias 회피.

### 4. Personalized PageRank

$\pi_v = \alpha (I - (1-\alpha) P)^{-1} e_v$ 의 closed-form은 $L_{\text{rw}}$ 의 resolvent. APPNP (Klicpera 2019) — Ch5-05.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| 모든 노드 $d_i > 0$ | Isolated node 시 $D^{-1}$ 정의 불가 → self-loop 추가 또는 isolated 제거 |
| Undirected | Directed graph는 PageRank Laplacian 등 일반화 |
| Connected (1개 component) | Disconnected 시 component-wise normalization |
| Self-loop 없음 가정 | GCN은 $\tilde{A} = A + I$ 사용 — augmented Laplacian (Ch2-03) |
| Real-valued weight | Signed graph는 부호 있는 Laplacian (Kunegis 2010) |

---

## 📌 핵심 정리

$$\boxed{L_{\text{sym}} = I - D^{-1/2} A D^{-1/2}, \quad L_{\text{rw}} = I - D^{-1} A = I - P}$$

$$\boxed{\lambda(L_{\text{sym}}) = \lambda(L_{\text{rw}}) \in [0, 2]}$$

| 성질 | $L_{\text{sym}}$ | $L_{\text{rw}}$ |
|------|------------------|----------------|
| Symmetric | ✓ | ✗ |
| 고유값 | real, $[0, 2]$ | real, $[0, 2]$ (similar) |
| 고유벡터 | orthonormal | $D^{-1/2} v$ |
| $\lambda_1$ eigenvector | $D^{1/2} \mathbb{1}$ | $\mathbb{1}$ |
| Stationary 직접 보임 | ✗ | $\pi \propto d$ (left null) |
| GNN 사용 | GCN, ChebNet | GraphSAGE mean, APPNP |
| $\lambda_{\max} = 2$ 조건 | bipartite | bipartite |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 4-cycle $C_4$ 의 $L_{\text{sym}}$ 을 구하고 모든 고유값을 손으로 계산하라. Bipartite임을 확인하라.

<details>
<summary>해설</summary>

$C_4$: 모든 노드 $d = 2$, $D = 2I$, $D^{-1/2} = (1/\sqrt 2) I$.

$$
A = \begin{pmatrix} 0 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 0 & 1 & 0 \end{pmatrix}, \quad
\hat{A} = \frac{1}{2} A
$$

$L_{\text{sym}} = I - \hat{A}$ 의 고유값 = $1 - \mu(\hat{A})$.

$A$ 의 고유값 (cycle $C_n$): $2 \cos(2\pi k/n)$ for $k = 0, 1, \ldots, n-1$. $n=4$: $\{2, 0, -2, 0\}$.

$\hat{A}$ 의 고유값: $\{1, 0, -1, 0\}$.

$L_{\text{sym}}$ 고유값: $\{0, 1, 2, 1\}$ → 정렬 $\{0, 1, 1, 2\}$.

$\lambda_{\max} = 2$ → bipartite (실제로 $C_4$ 는 even cycle, bipartite). $\square$

</details>

**문제 2** (심화): $L_{\text{rw}}$ 가 비대칭임에도 실수 고유값을 가짐을 정리 3.1로부터 어떻게 결론낼 수 있는가? 또한 right eigenvector $v$ 와 left eigenvector $w$ 의 직교성 (다른 고유값 사이) 을 증명하라.

<details>
<summary>해설</summary>

**실수 고유값**: $L_{\text{rw}} = D^{-1/2} L_{\text{sym}} D^{1/2}$ similarity transform. Similar matrices share eigenvalues. $L_{\text{sym}}$ symmetric → real eigenvalues → $L_{\text{rw}}$ 도 real eigenvalues. ✓

**Bi-orthogonality**: $L_{\text{sym}} v_i = \lambda_i v_i$ orthonormal $\{v_i\}$ basis. $L_{\text{rw}}$ right eigenvector $u_i = D^{-1/2} v_i$, left eigenvector $w_i = D^{1/2} v_i$ (확인: $w_i^T L_{\text{rw}} = (D^{1/2} v_i)^T D^{-1/2} L_{\text{sym}} D^{1/2} = v_i^T L_{\text{sym}} D^{1/2} = \lambda_i v_i^T D^{1/2} = \lambda_i w_i^T$).

직교성: $w_i^T u_j = (D^{1/2} v_i)^T (D^{-1/2} v_j) = v_i^T v_j = \delta_{ij}$. $\square$

이 bi-orthogonality는 $L_{\text{rw}}$ 의 spectral 분해에 사용.

</details>

**문제 3** (논문 비평): GCN (Kipf 2017) 은 $L_{\text{sym}}$, GraphSAGE (Hamilton 2017) 은 effectively $L_{\text{rw}}$ (mean aggregator). 두 선택의 trade-off는 무엇인가? Inductive learning 관점에서 어느 쪽이 유리한가?

<details>
<summary>해설</summary>

**$L_{\text{sym}}$ (GCN) 의 장점**:
- Symmetric → 직교 eigenbasis, spectral 분석 용이
- Bipartite 시 $\lambda_{\max} = 2$ — polynomial filter (ChebNet) stability 명확
- Renormalization trick으로 $\lambda_{\max} < 2$ 보장

**$L_{\text{sym}}$ (GCN) 의 단점**:
- $D^{-1/2} A D^{-1/2}$ — 양 끝 노드 모두 normalize, 직접 확률 해석 X
- Inductive: 새 노드 추가 시 $D^{-1/2}$ 가 변함 → 재계산 필요

**$L_{\text{rw}}$ (GraphSAGE mean) 의 장점**:
- Mean aggregator: 자연스럽게 inductive (새 노드의 이웃 mean 계산만)
- 확률 해석: 한 step random walk
- Sampling 친화적: $|N(i)|$ 부분 sampling 시 mean 보존

**$L_{\text{rw}}$ (GraphSAGE mean) 의 단점**:
- 비대칭 → 이론 분석 약간 까다로움
- Mean은 multiset injectivity 약함 → GIN보다 표현력 ↓ (Ch4)

**Inductive 관점**: GraphSAGE의 mean aggregator + sampling이 새 노드에 자연스럽게 적용. GCN은 in principle inductive지만, $D^{-1/2}$ 가 graph-wide 정보 의존 → 실전에서 transductive 가정 (whole graph 보면서 학습).

따라서 large-scale + new-node 시나리오에서 GraphSAGE / random-walk 기반이 우위.

</details>

---

<div align="center">

[◀ 이전](./02-unnormalized-laplacian.md) | [📚 README](../README.md) | [다음 ▶](./04-spectral-theory.md)

</div>
