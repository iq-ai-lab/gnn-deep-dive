# 05. Position-Aware GNN

## 🎯 핵심 질문

- WL 한계를 우회하는 positional encoding (PE) 의 일반 원리는?
- P-GNN (You 2019) 이 어떻게 random anchor set 으로 position 을 인코딩하는가?
- Laplacian Positional Encoding (LapPE, Dwivedi 2020): $\sqrt{\lambda_k} u_k$ 의 의미와 sign-flip 모호성 처리?
- Random-walk PE: $[P, P^2, \ldots, P^K]$ 의 diagonal 이 어떻게 noise-robust position 을 capture 하는가?
- 각 PE 방법이 1-WL 한계를 어떻게 우회하는가?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

Ch4-04 의 k-WL 은 표현력 우월하지만 $O(n^k)$ 비용으로 비실용. **Position-aware GNN** 은 효율을 유지하면서 1-WL 한계를 우회하는 대안:

1. **Symmetric graph 구분** — CSL 같은 1-WL fail 그래프
2. **Substructure counting** — Triangle, cycle 정보를 implicit 으로
3. **Long-range dependency** — Position feature 가 k-hop 보다 멀리 봄
4. **Modern GNN 의 표준 components** — Graphormer 의 spatial encoding 기반

이 문서에서는 P-GNN, LapPE, Random-walk PE 의 메커니즘과 trade-off 를 정리합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-wl-test.md](./01-wl-test.md), [Ch1-04](../ch1-graph-laplacian/04-spectral-theory.md), [Ch1-06](../ch1-graph-laplacian/06-random-walk-pagerank.md)
- 통계: 확률 분포의 random sampling
- 선형대수: Spectral decomposition, eigenvector ambiguity

---

## 📖 직관적 이해

### Position 의 의미

WL 한계의 본질: 노드들이 "structural role" 만 의존, "absolute position" 무시. Symmetric graph (모든 노드 같은 role) 에서 모든 노드 동등 → 구분 불가.

**Position 추가**: 각 노드에 graph 내에서의 "위치" 정보 부여. 같은 role 이라도 다른 위치 → 다른 representation.

### 세 가지 접근

1. **P-GNN (You 2019)**: Random anchor set 까지의 거리 → "다른 노드와의 관계" 로 위치 정의
2. **Laplacian Positional Encoding (Dwivedi 2020)**: Spectral decomposition $u_k$ — graph Fourier basis 가 자동 위치 정보
3. **Random-walk PE (Dwivedi 2022)**: $P^k$ 의 return probability — random walk 동역학으로 위치

각각 다른 원리이지만 모두 noise-robust 하고 deterministic.

### LapPE 의 직관

$L_{\text{sym}}$ 의 eigenvector $u_k$ 를 "graph 위 sinusoid" 로 보기 (Ch1-05). 노드 $i$ 의 좌표 $(u_1(i), u_2(i), \ldots, u_K(i))$ — 그래프 내 위치.

같은 structural role 의 노드들도 다른 spectral 좌표 → 구분 가능.

**문제**: Eigenvector $u_k$ 의 sign 이 모호 ($u_k$ 와 $-u_k$ 둘 다 valid). 후속에서 처리.

### P-GNN 의 직관

랜덤하게 anchor set $S$ 선택, 노드 $i$ 의 representation 에 SP distance $d(i, s)$ for $s \in S$ 추가.

**예**: Karate Club 에서 anchor = {0, 33}. 노드 1 의 PE = (d(1, 0), d(1, 33)) — 두 club 까지의 거리. 같은 구조적 역할의 노드도 다른 PE.

### Random-walk PE 의 직관

$P^k_{ii}$ = return probability — $i$ 가 $k$-step 안에 자기로 돌아오는 확률. 그래프 내 노드의 "rotation symmetry" 정보 인코딩.

$[P^1_{ii}, P^2_{ii}, \ldots, P^K_{ii}]$ 가 노드 $i$ 의 fingerprint.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Positional Encoding

함수 $\text{PE}: V \to \mathbb R^{d_{\text{PE}}}$ 가 그래프 의 각 노드에 vector 를 부여.

**Permutation equivariance** (좋은 PE):
$$
\pi: V_1 \to V_2 \text{ isomorphism} \Rightarrow \text{PE}_{G_1}(v) = \text{PE}_{G_2}(\pi(v))
$$

(isomorphic graph 의 corresponding 노드는 같은 PE)

### 정의 5.2 — P-GNN (Position-aware GNN, You 2019)

랜덤 anchor set $\{S_1, \ldots, S_K\}$ ($|S_k|$ random sizes, e.g., $|S_k| \sim \log n$).

노드 $i$ 의 anchor-based feature:
$$
z_i^{(k)} = \min_{s \in S_k} d(i, s)
$$

(SP distance to nearest anchor in $S_k$)

PE: $\text{PE}(i) = [z_i^{(1)}, \ldots, z_i^{(K)}]$.

**Update rule**: 표준 GNN message passing + PE 를 input feature 에 concat.

### 정의 5.3 — Laplacian PE (Dwivedi 2020)

$L_{\text{sym}} = U \Lambda U^T$. $U_{[K]}$ = first $K$ nontrivial eigenvectors (smallest $\lambda_k$).

$$
\text{PE}_{\text{Lap}}(i) = (u_1(i), u_2(i), \ldots, u_K(i)) \in \mathbb R^K
$$

또는 weighted: $\sqrt{\lambda_k} u_k(i)$.

### 정의 5.4 — Sign-Flip Ambiguity

$u_k$ 와 $-u_k$ 모두 같은 eigenvector. 학습/inference 마다 다른 sign 가능. 처리 방법:

1. **Random sign flip during training**: $u_k \leftarrow \pm u_k$ randomly
2. **Sign-equivariant model**: $\phi(u_k)$ 가 $\phi(\pm u_k)$ 에 invariant (e.g., absolute value, $u_k^2$)
3. **SignNet (Lim 2022)**: $\phi(\{u_k, -u_k\})$ 를 양쪽 모두 처리 후 결합

### 정의 5.5 — Random-Walk Positional Encoding (RWPE, Dwivedi 2022)

$P = D^{-1} A$ (random walk transition).

$$
\text{PE}_{\text{RW}}(i) = (P^1_{ii}, P^2_{ii}, \ldots, P^K_{ii})
$$

(노드 $i$ 의 $k$-step return probability)

또는 $\sum_j P^k_{ij}$ 등 row sum 계열.

### 정의 5.6 — PE Augmented GNN

PE 를 GNN input feature 에 concat:
$$
\tilde X_i = [X_i; \text{PE}(i)]
$$

또는 별도 PE encoder MLP 후 concat:
$$
\tilde X_i = [X_i; \text{MLP}_{\text{PE}}(\text{PE}(i))]
$$

---

## 🔬 정리와 결과

### 정리 5.1 — P-GNN 의 표현력 우위

**Theorem (You 2019)**: P-GNN 가 1-WL 동등 그래프 일부 (CSL 등) 에서 더 strict 표현력.

**증명 sketch**: Random anchor set 이 graph 의 random partition. Symmetric graph 도 anchor 의 random sample 이 다르면 다른 PE → 다른 node representation.

단점: random sampling → variance, 같은 graph 에서 다른 sample 마다 다른 결과. 평균 (multiple samples) 으로 stable 화.

### 정리 5.2 — LapPE 의 1-WL 우회

**Theorem**: $K$ 가 충분히 크면 LapPE-augmented GNN 가 1-WL 보다 strict 강함.

**증명 idea**: $L$ 의 eigenvector 가 "graph spectrum" 인코딩. 두 1-WL 동등 graph 도 다른 spectrum (cospectral 가 아니라면) → 다른 LapPE → 다른 representation.

**실증**: CSL 에서 LapPE+GIN 100% accuracy (random guess 50% from pure GIN).

### 정리 5.3 — Sign-Equivariance 의 충분성

**Theorem (Lim 2022)**: Sign-equivariant function $\phi$ 적용 후 LapPE 가 sign 모호성 회피하면서 표현력 보존.

$\phi(u_k) = \phi(-u_k)$ 보장하는 architecture (e.g., $\phi(u) = \psi(u^2)$ for some MLP $\psi$, or $\phi(u) + \phi(-u)$). 이는 SignNet, BasisNet (Lim 2022) 에서 제안.

### 정리 5.4 — RWPE 의 invariance 와 noise-robustness

**Theorem**: $P^k_{ii}$ 는 graph isomorphism 에 invariant. 또한:
- Sign 모호성 X (real number)
- Noise-robust (eigenvalue degeneracy 영향 작음)

따라서 LapPE 보다 stable.

**한계**: 표현력 약간 제한적 — $P^k$ 의 spectrum 만 encode, 자세한 spectral 정보 (eigenvector phase) 손실.

### 정리 5.5 — PE 의 각 방법 비교

| Method | 표현력 | Cost | 안정성 |
|--------|--------|------|--------|
| **P-GNN** | > 1-WL (probabilistic) | $O(n m)$ for SPD | Variance |
| **LapPE** | > 1-WL | $O(n^3)$ eigendecomp | Sign / basis ambiguity |
| **RWPE** | > 1-WL | $O(K m)$ | Stable |
| **k-FGNN** | k-WL | $O(n^k)$ | Stable |
| **None (1-WL)** | 1-WL | $O(m)$ | Stable |

RWPE 가 sweet spot (효율 + 안정).

---

## 💻 구현

### 실험 1 — P-GNN Anchor-based PE

```python
import numpy as np
import networkx as nx
import torch

def shortest_path_dist(G, source, target):
    try:
        return nx.shortest_path_length(G, source, target)
    except nx.NetworkXNoPath:
        return -1   # disconnected

def p_gnn_pe(G, num_anchors=8, anchor_size_dist='log'):
    n = G.number_of_nodes()
    K = num_anchors
    
    # Anchor sets: 다양한 size
    anchor_sets = []
    for k in range(K):
        size = 2**(k % int(np.log2(n)) + 1)   # exponentially varying
        if size > n:
            size = n
        anchor_sets.append(np.random.choice(n, size, replace=False).tolist())
    
    # PE matrix [n, K]
    pe = np.zeros((n, K))
    for k, anchors in enumerate(anchor_sets):
        for i in range(n):
            min_d = float('inf')
            for a in anchors:
                d = shortest_path_dist(G, i, a)
                if d >= 0 and d < min_d:
                    min_d = d
            pe[i, k] = min_d if min_d != float('inf') else n
    
    return torch.tensor(pe, dtype=torch.float32)

# 예시
G = nx.karate_club_graph()
pe_pgnn = p_gnn_pe(G, num_anchors=8)
print(f'P-GNN PE shape: {pe_pgnn.shape}')
print(f'PE for node 0: {pe_pgnn[0]}')
print(f'PE for node 33: {pe_pgnn[33]}')
```

### 실험 2 — Laplacian Positional Encoding

```python
def laplacian_pe(G, K=8):
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    deg = A.sum(1)
    D_inv_sqrt = np.diag(1 / np.sqrt(deg + 1e-6))
    L_sym = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt
    
    eigvals, U = np.linalg.eigh(L_sym)
    # Skip first (lambda_1 = 0)
    pe = U[:, 1:K+1]
    # Optional: weighted by sqrt(lambda)
    pe_weighted = pe * np.sqrt(eigvals[1:K+1])[None, :]
    return torch.tensor(pe, dtype=torch.float32), torch.tensor(pe_weighted, dtype=torch.float32)

pe_lap, pe_lap_weighted = laplacian_pe(G, K=8)
print(f'LapPE shape: {pe_lap.shape}')
print(f'PE for node 0: {pe_lap[0]}')
```

### 실험 3 — Sign Flip Augmentation

```python
def random_sign_flip(pe):
    """Randomly flip signs of each PE column."""
    K = pe.size(-1)
    signs = torch.where(torch.rand(K) > 0.5, 1.0, -1.0)
    return pe * signs

# 학습 시 매 epoch random sign flip
pe_lap_aug = random_sign_flip(pe_lap)
```

### 실험 4 — Random-Walk PE

```python
def random_walk_pe(G, K=8):
    n = G.number_of_nodes()
    A = nx.adjacency_matrix(G).toarray().astype(float)
    deg = A.sum(1)
    P = np.diag(1/(deg + 1e-6)) @ A
    
    pe = np.zeros((n, K))
    P_k = np.eye(n)
    for k in range(K):
        P_k = P_k @ P
        pe[:, k] = np.diag(P_k)   # return probability
    
    return torch.tensor(pe, dtype=torch.float32)

pe_rw = random_walk_pe(G, K=8)
print(f'RWPE shape: {pe_rw.shape}')
print(f'PE for node 0: {pe_rw[0]}')
```

### 실험 5 — CSL 분류: LapPE+GIN vs Plain GIN

```python
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

def csl(n, skip):
    G = nx.cycle_graph(n)
    for i in range(n):
        G.add_edge(i, (i + skip) % n)
    return G

class GINLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_in, d_out), nn.ReLU(),
                                 nn.Linear(d_out, d_out))
    def forward(self, x, edge_index):
        src, dst = edge_index
        return self.mlp(x + scatter_add(x[src], dst, dim=0, dim_size=x.size(0)))

class GIN_LapPE(nn.Module):
    def __init__(self, d_x, d_pe, d_hid, d_out, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(d_x + d_pe, d_hid)
        self.layers = nn.ModuleList([GINLayer(d_hid, d_hid) for _ in range(num_layers)])
        self.cls = nn.Linear(d_hid * num_layers, d_out)
    def forward(self, x, pe, edge_index):
        h = self.input_proj(torch.cat([x, pe], dim=-1))
        outs = []
        for layer in self.layers:
            h = F.relu(layer(h, edge_index))
            outs.append(h.sum(0))
        return self.cls(torch.cat(outs))

# CSL(8, 2) vs CSL(8, 3) - 1-WL 동등
def graph_to_input(G, K_pe=4):
    n = G.number_of_nodes()
    edges = np.array(list(G.edges())).T
    edge_index = torch.tensor(np.concatenate([edges, edges[::-1]], axis=1), dtype=torch.long)
    x = torch.ones(n, 1)   # 동일 input feature
    _, pe_lap = laplacian_pe(G, K=K_pe)
    return x, pe_lap, edge_index

x1, pe1, ei1 = graph_to_input(csl(8, 2), K_pe=4)
x2, pe2, ei2 = graph_to_input(csl(8, 3), K_pe=4)

torch.manual_seed(0)
model = GIN_LapPE(d_x=1, d_pe=4, d_hid=16, d_out=2, num_layers=3)
model.eval()
with torch.no_grad():
    z1 = model(x1, pe1, ei1)
    z2 = model(x2, pe2, ei2)

print(f'GIN+LapPE diff CSL(8,2) vs CSL(8,3): {(z1 - z2).norm().item():.4f}')
# 큰 값 (구분 가능) — pure GIN 은 0 근처
```

---

## 🔗 실전 활용

### 1. Graphormer 의 Position Encoding (Ch7-01)

Graphormer 가 LapPE 와 유사한 spatial encoding 사용. Centrality (degree) + spatial (SP distance) + edge encoding 모두 PE 의 일종.

### 2. Long Range Graph Benchmark (LRGB)

LRGB 의 PEPTIDES, COCO 등 large graph benchmark 에서 RWPE 가 standard. 단순 message passing 으로 부족, PE augmented 가 SOTA.

### 3. PyG 의 Transform

```python
from torch_geometric.transforms import AddLaplacianEigenvectorPE, AddRandomWalkPE

transform = AddLaplacianEigenvectorPE(k=8)   # LapPE
transform = AddRandomWalkPE(walk_length=8)   # RWPE
data = transform(data)
# data.laplacian_eigenvector_pe 또는 data.random_walk_pe 추가
```

### 4. SignNet / BasisNet

Lim 2022 의 sign/basis-equivariant network — eigenvector 모호성 처리하면서 표현력 보존. State-of-the-art on heterophily benchmarks.

### 5. Diffusion-based PE

최근 trend: diffusion model 의 latent code 를 PE 로 사용 (Yang 2023 Diffusion-PE). 매우 expressive but 계산 비쌈.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Eigenvector well-defined (no degeneracy) | Repeated eigenvalue 시 basis 모호 |
| Connected graph | Disconnected 시 component-wise PE |
| Sign 처리 (LapPE) | 학습 instability 가능 (random flip) |
| Anchor sampling (P-GNN) | High variance — multiple samples 평균 |
| RWPE 의 spectral 정보 손실 | Diagonal 만 — eigenvector phase 무시 |
| Static PE | Dynamic graph 시 PE 재계산 |
| $K$ 차원 선택 | 너무 작으면 표현력 부족, 너무 크면 noise |

---

## 📌 핵심 정리

$$\boxed{\text{LapPE: } u_k(i), \quad \text{RWPE: } P^k_{ii}, \quad \text{P-GNN: } \min_{s \in S_k} d(i, s)}$$

| Method | 표현력 | 효율 | 안정성 | Sign issue |
|--------|--------|------|--------|------------|
| P-GNN | > 1-WL | $O(nm)$ | Variance | None |
| LapPE | > 1-WL | $O(n^3)$ | Stable | Sign flip |
| RWPE | > 1-WL | $O(Km)$ | Stable | None |
| SignNet/BasisNet | LapPE 보강 | $O(n^3)$ | Stable | Resolved |

핵심: PE 가 1-WL 한계 우회 + k-WL 보다 효율. Modern GNN 의 표준 component.

---

## 🤔 생각해볼 문제

**문제 1** (기초): $C_4$ (4-cycle) 의 LapPE (K=2) 를 손으로 계산하라.

<details>
<summary>해설</summary>

$C_4$: $L_{\text{sym}}$ 고유값 $\{0, 1, 1, 2\}$ (Ch1-04 문제 1).

Eigenvectors (Ch2-04 문제 1):
- $u_1 = (1,1,1,1)/2$ ($\lambda=0$)
- $u_2, u_3$: $\lambda = 1$ degenerate. Pick $u_2 = (1, 0, -1, 0)/\sqrt 2$, $u_3 = (0, 1, 0, -1)/\sqrt 2$
- $u_4 = (1, -1, 1, -1)/2$ ($\lambda = 2$)

LapPE (skip $\lambda_1 = 0$, K=2):

PE for node 0: $(u_2(0), u_3(0)) = (1/\sqrt 2, 0)$
PE for node 1: $(0, 1/\sqrt 2)$
PE for node 2: $(-1/\sqrt 2, 0)$
PE for node 3: $(0, -1/\sqrt 2)$

**관찰**: 4 개 노드의 PE 가 모두 다름 — 1-WL 가 모든 노드 동등으로 보지만 LapPE 는 위치 구분.

**문제**: $\lambda_2 = \lambda_3 = 1$ (degenerate). $u_2, u_3$ 의 선택이 임의 — basis ambiguity. 
다른 basis 선택: $\tilde u_2 = (u_2 + u_3) / \sqrt 2 = (1, 1, -1, -1)/2$, $\tilde u_3 = (u_2 - u_3)/\sqrt 2$. 

이 경우 PE 가 완전히 다름 — basis ambiguity 의 직접 예시.

이를 처리하기 위해 BasisNet (Lim 2022) 가 필요.

</details>

**문제 2** (심화): RWPE 의 $K$ (walk length) 를 어떻게 선택하는가? Graph diameter 와의 관계는?

<details>
<summary>해설</summary>

**$K$ 선택의 trade-off**:

- **작은 $K$**: PE 차원 작음, computation 빠름. 단 short-range info 만.
- **큰 $K$**: 더 풍부한 PE. 단 $P^k$ 의 수렴 — large $k$ 에서 모든 entry $\to \pi_i$ (stationary), 정보 X.

**Graph diameter** $D$ = max shortest-path 거리. Random walk 가 $D$ step 이후 점차 stationary 에 수렴.

**Heuristic**: $K \approx D / 2$ ~ $D$. 정확한 mixing time 은 $1/(1 - \mu_2)$ 인데, expander graph 시 $K = O(\log n)$ 충분.

**Empirical**:
- Cora ($n \sim 3000$, $D \sim 19$): $K = 16$ 표준
- Reddit ($n \sim 230k$, $D \sim 8$): $K = 8$
- 분자 (small graph, $D \sim 5-10$): $K = 4-8$

**또 다른 관점**: $K$-th power $P^K$ 가 "$K$-hop information mass distribution". $K$ 가 작으면 local structural fingerprint, 큰 $K$ 는 global topology + community structure.

**Adaptive $K$**: PEPTIDES 등 long-range benchmark 에서 $K = 24$ 또는 더 크게. Task-specific tuning.

따라서 $K$ 는 graph 의 mixing time 과 task 의 long-range dependency 에 의해 결정.

</details>

**문제 3** (논문 비평): Random sign flip + LapPE 의 표현력 손실 문제와, SignNet (Lim 2022) 의 해결책을 비교 분석하라.

<details>
<summary>해설</summary>

**Random sign flip 의 문제**:

LapPE 의 $u_k$ 와 $-u_k$ 가 같은 eigenvector — 학습 시 random sign 으로 학습 → 모델이 sign 에 invariant 한 함수 학습 강제.

**한계**:

1. **Information loss**: Sign 자체가 정보를 가질 수 있음 (e.g., Fiedler vector 의 부호 = bisection direction). Random flip 시 이 정보 손실.

2. **Variance**: 매 epoch 다른 sign → noisy gradient → 학습 instability, slow convergence.

3. **Inference 의 ambiguity**: Test time 에 어떤 sign 을 선택할지 — average over multiple flips 가 표준이지만 비싼.

4. **Repeated eigenvalue 의 basis ambiguity**: Sign 만 문제가 아니라 basis 회전도 ambiguity. Random flip 으로 처리 X.

**SignNet (Lim 2022)**:

각 eigenvector $u_k$ 를 모델 $\phi$ 에 input 할 때 **두 sign 모두** 처리:
$$
\phi_{\text{SignNet}}(u_k) = \rho(\phi(u_k) + \phi(-u_k))
$$

(또는 set encoding $\rho(\{\phi(u_k), \phi(-u_k)\})$ — order-invariant)

**장점**:
- **Deterministic**: Random sampling 없음
- **Sign-equivariant**: $\phi_{\text{SignNet}}(u_k) = \phi_{\text{SignNet}}(-u_k)$ 자동 보장
- **Information preservation**: Sign 정보 implicit 으로 인코딩 (대신 sign 자체는 학습 안 됨)

**단점**:
- 계산 2배 ($u_k$ 와 $-u_k$ 둘 다 forward)
- Repeated eigenvalue (basis ambiguity) 는 별도 처리 — BasisNet (같은 paper) 가 확장.

**BasisNet**:

Eigenspace (multiple eigenvector with same eigenvalue) 의 basis ambiguity 처리. $\text{O}(d_{\text{eigenspace}})$ orthogonal group 에 대한 invariance.

**Empirical**:
- Heterophilic graph (CSL, EXP) 에서 SignNet+GIN > LapPE+GIN > GIN
- 분자 (ZINC) 에서 +1~2% 향상

**현대 사용**: Modern Graph Transformer (GraphGPS, GRIT) 는 SignNet/BasisNet 을 표준 component.

따라서 sign 처리는 LapPE 의 핵심 challenge — random flip 의 단순함 vs SignNet 의 정확성. Production setting 에서 SignNet 권장.

</details>

---

<div align="center">

[◀ 이전](./04-k-wl.md) | [📚 README](../README.md) | [다음 ▶](../ch5-over-smoothing/01-phenomenon.md)

</div>
