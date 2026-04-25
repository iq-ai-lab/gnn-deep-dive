# 01. 그래프의 수학적 정의와 기본 표현

## 🎯 핵심 질문

- 그래프 $G = (V, E)$ 를 컴퓨터에서 다룰 수 있는 행렬 표현으로 어떻게 변환하는가?
- Adjacency matrix $A$, degree matrix $D$, incidence matrix $B$ 는 각각 무엇을 인코딩하며 어떤 관계가 있는가?
- Directed / undirected / weighted / multi-graph의 행렬 표현은 어떻게 다른가? Symmetry는 언제 보존되는가?
- 그래프의 sparsity는 GNN의 계산 비용을 어떻게 결정하는가? Graph density $\rho$ 의 의미는?
- $A^k$ 의 $(i, j)$ 성분이 왜 길이 $k$ 의 walk 수와 같은가? 이 사실이 message passing과 어떻게 연결되는가?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

GNN의 모든 연산은 결국 **행렬 곱셈**으로 환원됩니다. GCN의 propagation $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H W$, GAT의 attention $\alpha_{ij}$ 계산, GraphSAGE의 sampling — 모두 adjacency matrix $A$ 를 사전에 정의해야 가능합니다. 그러나 많은 실무자가 $A$ 와 $D$ 의 관계, $A$ 가 symmetric일 조건, weighted graph에서 $D$ 의 정의 등을 정확히 알지 못한 채 PyG의 `Data` 객체를 사용합니다. 이는 다음의 문제를 야기합니다:

1. **Directed graph에서의 오류** — Directed $A$ 는 비대칭이므로 $L_{\text{sym}}$ 가 정의되지 않음. PyG는 자동으로 symmetric화 (`to_undirected`) 하지만 그 의미를 모르면 정보 손실 위험.
2. **Self-loop 처리의 미묘함** — GCN은 $\tilde{A} = A + I$ 로 self-loop 추가하지만 weighted graph에서 self-loop weight 선택 (1 vs degree-aware) 이 결과를 바꿈.
3. **Density에 따른 sampling 전략** — Sparse graph (Cora $\rho \approx 4 \times 10^{-4}$) 와 dense graph (분자 $\rho \approx 0.3$) 에서 적절한 GNN 아키텍처가 다름.

이 문서에서는 그래프의 행렬 표현을 **엄밀히 정의**하고, GNN 연산의 기반을 다집니다.

---

## 📐 수학적 선행 조건

- [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive): Matrix multiplication, rank, symmetric matrices
- 집합론: $|V|$, multiset $\{\!\{ \cdot \}\!\}$ 표기
- 조합론: walk, path, cycle의 차이

---

## 📖 직관적 이해

### 그래프의 두 가지 시각

그래프는 두 가지 관점으로 볼 수 있습니다:

1. **조합론적 객체**: 노드 집합 $V$ 와 엣지 집합 $E \subseteq V \times V$
2. **선형대수적 객체**: 행렬 $A$ 와 그 위에서 정의되는 연산자

GNN은 **두 관점을 잇는 다리**입니다. 노드 feature를 vector로 두고 $A$ 를 곱하면, "이웃 노드 정보를 모은다"는 조합론적 직관이 행렬 곱셈이라는 선형대수 연산이 됩니다.

### Adjacency Matrix의 직관

$A_{ij} = 1$ 이면 노드 $i$ 와 $j$ 가 연결됨, $0$ 이면 아님. Undirected graph에서는 $A = A^T$ (symmetric).

```
     1 ─── 2
     │     │
     3 ─── 4

     A =  [0 1 1 0]
          [1 0 0 1]
          [1 0 0 1]
          [0 1 1 0]
```

행 $A_i$ 는 노드 $i$ 의 **이웃 indicator vector**. $\sum_j A_{ij} = d_i$ (노드 $i$ 의 degree).

### Walks와 $A^k$

$A^k$ 의 $(i, j)$ 성분이 길이 $k$ 의 walk 수와 같다는 사실은 message passing의 수학적 뿌리입니다. $k=2$ 일 때:

$$
(A^2)_{ij} = \sum_k A_{ik} A_{kj}
$$

이는 "$i \to k \to j$ 경로가 가능한 $k$ 의 수"입니다. GCN 한 layer가 1-hop neighbor 정보를 모으듯, $A^k$ 는 $k$-hop reachability를 인코딩합니다.

### Degree Matrix와 정규화

$D = \text{diag}(d_1, d_2, \ldots, d_n)$ 는 degree를 대각에 배치한 행렬. **$D^{-1} A$** 는 row-stochastic (각 행 합이 1) — random walk transition probability. **$D^{-1/2} A D^{-1/2}$** 는 symmetric 정규화 — GCN의 핵심.

### Incidence Matrix와 엣지 관점

$B \in \mathbb{R}^{n \times m}$ 은 **노드-엣지 관계** 인코딩. Undirected graph에서 엣지 $e = (i, j)$ 의 column은:

$$
B_{:, e} = e_i - e_j \quad \text{(arbitrary orientation)}
$$

$B$ 는 graph gradient 연산자로 해석됩니다 — 엣지 위에서 노드 값의 차이를 출력합니다.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Graph

**Graph** $G = (V, E)$ 는 다음으로 구성됩니다:
- **Vertex set** $V$, $|V| = n$
- **Edge set** $E \subseteq V \times V$, $|E| = m$

$G$ 가 **undirected**이면 $(i, j) \in E \Leftrightarrow (j, i) \in E$. **Simple** graph는 self-loop ($(i, i) \in E$) 와 multi-edge가 없음.

### 정의 1.2 — Adjacency Matrix

$$
A_{ij} = \begin{cases} 1 & (i, j) \in E \\ 0 & \text{otherwise} \end{cases}
$$

**Weighted graph**의 경우 $A_{ij} = w_{ij} \in \mathbb{R}_{\geq 0}$. Undirected graph에서 $A = A^T$.

### 정의 1.3 — Degree

노드 $i$ 의 **degree** $d_i$:
$$
d_i = \sum_{j} A_{ij}
$$

Weighted graph에서는 weighted degree (sum of incident edge weights). **Degree matrix** $D = \text{diag}(d_1, \ldots, d_n)$.

### 정의 1.4 — Incidence Matrix

각 엣지에 임의 방향을 부여하면, undirected graph의 incidence matrix $B \in \mathbb{R}^{n \times m}$ 의 엣지 $e = (u \to v)$ column:

$$
B_{i, e} = \begin{cases} +1 & i = u \\ -1 & i = v \\ 0 & \text{otherwise} \end{cases}
$$

(가중치 있을 시 $\sqrt{w_e}$ 곱)

### 정의 1.5 — Walk, Path, Cycle

- **Walk** of length $k$: 노드 시퀀스 $v_0, v_1, \ldots, v_k$ with $(v_{l-1}, v_l) \in E$. 반복 허용.
- **Path**: 노드 반복 없는 walk.
- **Cycle**: $v_0 = v_k$ 인 walk.

### 정의 1.6 — Connected Components

Undirected graph $G$ 는 **connected**이면 임의의 두 노드 사이에 path 존재. **Connected component**는 maximal connected subgraph. Disconnected graph의 component 수는 GNN이 그래프 단위로 정보를 모을 때 핵심 통계량.

### 정의 1.7 — Graph Density

$$
\rho = \frac{2m}{n(n-1)} \quad \text{(undirected)}
$$

$\rho \to 0$: sparse graph (Cora, Reddit, social network), $\rho \to 1$: dense graph (complete graph, 분자).

---

## 🔬 정리와 증명

### 정리 1.1 — Adjacency의 거듭제곱과 Walk Counting

**Theorem**: $(A^k)_{ij}$ 는 $i$ 에서 $j$ 로 가는 길이 $k$ 의 walk 수와 같다.

**증명** (induction on $k$):

**Base** $k = 1$: $A_{ij} = 1$ iff edge $(i,j)$, 정확히 1-step walk 수.

**Inductive step**: $(A^{k-1})_{il}$ 가 $i \to l$ 의 길이 $k-1$ walk 수라고 가정. 그러면:

$$
(A^k)_{ij} = \sum_l (A^{k-1})_{il} A_{lj}
$$

각 항은 "길이 $k-1$ walk $i \to l$" $\times$ "edge $l \to j$" 수, 즉 "$i \to l \to j$ 의 길이 $k$ walk 수의 $l$ 별 분해". 합하면 전체 길이 $k$ walk 수. $\square$

### 정리 1.2 — Incidence와 Adjacency의 관계

**Theorem**: Undirected graph에서 $B B^T = D - A = L$ (Laplacian).

**증명**:

$$
(B B^T)_{ij} = \sum_e B_{ie} B_{je}
$$

$i = j$ 일 때: 각 incident edge $e$ 에서 $B_{ie}^2 = 1$, 합하면 $d_i$.

$i \neq j$ 일 때: 엣지 $e = (i, j)$ 만 양쪽에 nonzero, $B_{ie} B_{je} = (+1)(-1) = -1$ (방향 부여 무관). 따라서 $-A_{ij}$.

종합: $(BB^T)_{ij} = d_i \delta_{ij} - A_{ij} = (D - A)_{ij}$. $\square$

이는 다음 문서 (02) 에서 본격적으로 다룰 Laplacian의 첫 등장입니다.

### 정리 1.3 — Trace와 Triangle Counting

**Theorem**: Undirected simple graph에서 triangle 수 $T = \text{tr}(A^3) / 6$.

**증명**: $(A^3)_{ii}$ 는 $i$ 에서 시작하여 $i$ 로 돌아오는 길이 3 walk 수 = $i$ 를 포함하는 triangle 수 $\times 2$ (두 방향). $\text{tr}(A^3) = \sum_i (A^3)_{ii} = $ 각 triangle을 3 vertex $\times$ 2 direction $= 6$ 번 카운트. 나누면 정확. $\square$

### 정리 1.4 — Sparsity와 GNN 비용

**Observation**: GNN propagation $A H$ 의 비용은 dense matmul $O(n^2 d)$ 가 아니라 sparse matmul $O(m d)$. 따라서:
- Sparse graph ($m = O(n)$): $O(n d)$ — 선형
- Dense graph ($m = O(n^2)$): $O(n^2 d)$ — MLP과 비슷

이 사실은 GraphSAGE·Cluster-GCN·GraphSAINT의 sampling 전략에 직접적인 동기를 제공합니다.

---

## 💻 NumPy/PyG 구현 검증

### 실험 1 — Adjacency, Degree, Incidence 구성

```python
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 작은 그래프: 1-2, 1-3, 2-4, 3-4
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
n = 4
G = nx.Graph()
G.add_nodes_from(range(n))
G.add_edges_from(edges)

# Adjacency matrix
A = nx.adjacency_matrix(G).toarray().astype(float)
print('Adjacency A:')
print(A)
print('A == A.T:', np.allclose(A, A.T))   # True (undirected)

# Degree matrix
deg = A.sum(axis=1)
D = np.diag(deg)
print(f'Degrees: {deg}')   # [2 2 2 2]

# Incidence matrix
B = nx.incidence_matrix(G, oriented=True).toarray()
print('Incidence B (oriented):')
print(B)
print('B shape:', B.shape)   # (4, 4) for 4 nodes, 4 edges
```

### 실험 2 — $L = BB^T = D - A$ 검증

```python
L_from_B = B @ B.T
L_from_DA = D - A
print('BB^T:')
print(L_from_B)
print('D - A:')
print(L_from_DA)
print('Equal?', np.allclose(L_from_B, L_from_DA))   # True
```

**출력**:
```
BB^T:
[[ 2. -1. -1.  0.]
 [-1.  2.  0. -1.]
 [-1.  0.  2. -1.]
 [ 0. -1. -1.  2.]]
Equal? True
```

### 실험 3 — $A^k$ 와 Walk Counting

```python
# A^2: 길이 2 walk 수
A2 = A @ A
print('A^2:')
print(A2)
# (0,3) 성분: 0 → 1 → 3 과 0 → 2 → 3, 즉 2 walks → A2[0,3] = 2 예상
print(f'Walks of length 2 from node 0 to node 3: {int(A2[0, 3])}')

# A^3: 길이 3 walk → triangle
A3 = A @ A @ A
triangles = int(np.trace(A3) / 6)
print(f'Number of triangles: {triangles}')   # 0 (4-cycle 만 있음)

# Triangle 있는 그래프로 재실험
G2 = nx.Graph([(0,1), (1,2), (0,2), (2,3)])
A_g2 = nx.adjacency_matrix(G2).toarray().astype(float)
print(f'Triangles in K3+pendant: {int(np.trace(A_g2 @ A_g2 @ A_g2) / 6)}')   # 1
```

### 실험 4 — Real-world Graph의 Sparsity

```python
G_karate = nx.karate_club_graph()
n_k = G_karate.number_of_nodes()
m_k = G_karate.number_of_edges()
rho = 2 * m_k / (n_k * (n_k - 1))
print(f'Karate Club: n={n_k}, m={m_k}, density ρ={rho:.4f}')

# Cora dataset (PyG)
try:
    from torch_geometric.datasets import Planetoid
    cora = Planetoid(root='./data/Cora', name='Cora')[0]
    n_c, m_c = cora.num_nodes, cora.num_edges // 2   # undirected, /2
    rho_c = 2 * m_c / (n_c * (n_c - 1))
    print(f'Cora: n={n_c}, m={m_c}, density ρ={rho_c:.6f}')   # ~4e-4
except ImportError:
    print('PyG not installed; skipping Cora')
```

### 실험 5 — Directed vs Undirected

```python
# Directed graph
DG = nx.DiGraph([(0, 1), (1, 2), (2, 0), (1, 3)])
A_dir = nx.adjacency_matrix(DG).toarray().astype(float)
print('Directed adjacency:')
print(A_dir)
print('Symmetric?', np.allclose(A_dir, A_dir.T))   # False

# To undirected (PyG `to_undirected` equivalent)
A_undir = (A_dir + A_dir.T).clip(0, 1)
print('Symmetrized:')
print(A_undir)
```

---

## 🔗 실전 활용

### 1. PyG의 `edge_index` 표현

PyG는 dense $A$ 대신 **`edge_index ∈ ℝ^{2 × 2m}`** (COO format) 사용. 이는 sparse storage로 메모리 효율적이며 message passing이 자동 sparse matmul로 처리됩니다. 단, edge feature를 다룰 때는 별도의 `edge_attr` 필요.

### 2. Self-loop의 의미

GCN은 $\tilde{A} = A + I$ 로 self-loop 추가 — 노드 자신의 feature도 aggregation에 포함. 이 작은 trick이 큰 차이를 만드는데, self-loop 없으면 노드 자신 정보가 한 layer 후 사라집니다 (mean aggregator의 경우 특히).

### 3. Heterogeneous graph

Knowledge graph (entity + relation) 에서는 단일 $A$ 가 아닌 **relation별 $A_r$** 을 사용 (R-GCN, Ch3-05). 각 $A_r$ 의 sparsity가 다르며, 모든 relation을 한 번에 처리하는 trick이 필요합니다.

### 4. Bipartite graph

추천 시스템 (user-item) 의 bipartite graph는 $A = \begin{pmatrix} 0 & R \\ R^T & 0 \end{pmatrix}$ 형태. SVD가 LightGCN의 핵심 연산.

---

## ⚖️ 가정과 한계

| 가정 | 한계 및 대응 |
|------|-------------|
| Static graph (시간 불변) | Temporal GNN, dynamic graph 필요 (CTDG, DTDG) |
| Discrete edge (있음/없음) | Soft edge (continuous weight, attention) 필요 시 GAT |
| Single edge type | Heterogeneous graph (R-GCN, HAN) — Ch3-05 |
| 작은 size 가정 | 대규모 그래프는 sampling (GraphSAGE, Cluster-GCN, GraphSAINT) — Ch5-04, Ch7-04 |
| Structural information만 | Node feature $X$, edge feature $E$ 추가는 GNN의 입력으로 처리 |
| Connected 가정 | Disconnected graph는 component별 처리 또는 virtual super-node 추가 |

---

## 📌 핵심 정리

$$\boxed{A_{ij} = \mathbb{1}[(i,j) \in E], \quad d_i = \sum_j A_{ij}, \quad D = \text{diag}(d_i)}$$

$$\boxed{L = D - A = B B^T \quad \text{(Laplacian의 첫 등장)}}$$

| 개념 | 정의 | 차원 |
|------|------|------|
| **Adjacency matrix** $A$ | $A_{ij} = 1$ if $(i,j) \in E$ | $n \times n$ |
| **Degree matrix** $D$ | $D = \text{diag}(d_i)$, $d_i = \sum_j A_{ij}$ | $n \times n$ |
| **Incidence matrix** $B$ | 엣지 endpoint $\pm 1$ 인코딩 | $n \times m$ |
| **Walk count** | $(A^k)_{ij}$ = 길이 $k$ walk 수 | scalar |
| **Triangle count** | $\text{tr}(A^3) / 6$ | scalar |
| **Density** | $\rho = 2m / (n(n-1))$ | scalar in $[0, 1]$ |
| **Symmetry** | Undirected $\Rightarrow A = A^T$ | property |
| **Sparsity** | $\text{nnz}(A) = 2m$ for undirected | storage |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 다음 그래프의 $A$, $D$, $B$ 를 손으로 작성하라.

```
   1
  / \
 2   3
 |
 4
```

<details>
<summary>해설</summary>

엣지: $(1,2), (1,3), (2,4)$. $n = 4$, $m = 3$.

$$
A = \begin{pmatrix} 0 & 1 & 1 & 0 \\ 1 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}, \quad
D = \text{diag}(2, 2, 1, 1)
$$

엣지 방향 $(1 \to 2), (1 \to 3), (2 \to 4)$ 로 부여 시:

$$
B = \begin{pmatrix} +1 & +1 & 0 \\ -1 & 0 & +1 \\ 0 & -1 & 0 \\ 0 & 0 & -1 \end{pmatrix}
$$

검증: $B B^T = D - A$ 확인. $\square$

</details>

**문제 2** (심화): $G$ 가 connected이면 $A$ 의 spectral radius $\rho(A) \geq \bar{d}$ ($\bar{d}$ = 평균 degree) 임을 증명하라.

<details>
<summary>해설</summary>

$\mathbb{1} \in \mathbb{R}^n$ (모두 1인 vector) 에 대해 $A \mathbb{1}$ 의 $i$ 성분 $= \sum_j A_{ij} = d_i$. 따라서:

$$
\frac{\mathbb{1}^T A \mathbb{1}}{\mathbb{1}^T \mathbb{1}} = \frac{\sum_i d_i}{n} = \frac{2m}{n} = \bar{d}
$$

Rayleigh quotient bound (symmetric $A$) 에 의해:

$$
\lambda_{\max}(A) = \max_{x \neq 0} \frac{x^T A x}{x^T x} \geq \bar{d}
$$

$\rho(A) = |\lambda_{\max}(A)| \geq \lambda_{\max}(A) \geq \bar{d}$. $\square$

연습: $d$-regular graph에서 등호 성립 (equality $\Leftrightarrow$ $\mathbb{1}$ 이 eigenvector $\Leftrightarrow$ regular).

</details>

**문제 3** (논문 비평): PyG의 `Data` 객체는 `edge_index ∈ ℝ^{2 × 2m}` (COO) 와 `edge_attr ∈ ℝ^{2m × d_e}` 를 사용한다. 왜 dense $A$ 대신 이런 표현을 쓰며, 이는 message passing 구현에 어떤 영향을 주는가? Self-loop 추가 (`add_self_loops`) 가 왜 별도 함수인지도 답하라.

<details>
<summary>해설</summary>

**Sparse 표현의 이유**:
- 메모리: dense $A \in \mathbb{R}^{n \times n}$ 는 $O(n^2)$, COO는 $O(m)$. Cora는 $n^2 \approx 7.3 \times 10^6$ vs $2m \approx 10^4$.
- GPU 효율: PyG의 `MessagePassing` 은 `scatter_add` 로 sparse matmul 구현, 이는 GPU에서 highly parallelizable.
- Mini-batch: 여러 작은 그래프를 block-diagonal $A$ 로 합치는 것이 sparse에서 자연스러움.

**Message passing에의 영향**:
- 표준 형식: 각 엣지 $(i, j)$ 에 대해 source/target index lookup → message 계산 → target에 scatter.
- Vectorization: edge-wise 연산 후 node-wise aggregation, batched GPU operations.

**Self-loop가 별도 함수인 이유**:
- Self-loop 의미가 모델별로 다름: GCN은 필수 ($\tilde{A} = A + I$), GAT는 선택적, GraphSAGE는 자체 weight $W_{\text{self}}$ 로 따로 처리.
- Weighted graph에서 self-loop weight가 1인지 $d_i$ 인지 모델 가정에 따라 다름.
- Multiple message passing 시 self-loop가 중복 추가되는 것을 방지.

따라서 데이터 구조 (edge_index) 와 모델 가정 (self-loop 여부) 를 분리하는 것이 SW 설계상 자연스럽습니다.

</details>

---

<div align="center">

[◀ 이전](../README.md) | [📚 README](../README.md) | [다음 ▶](./02-unnormalized-laplacian.md)

</div>
