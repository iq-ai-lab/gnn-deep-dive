# 04. Higher-Order GNN — k-WL and Beyond

## 🎯 핵심 질문

- k-WL 은 1-WL 의 어떤 일반화이고, k-tuple 을 super-node 로 보는 의미는?
- k-WL 의 위계 1-WL ⊊ 2-WL ⊊ 3-WL ⊊ … 가 strict 인가?
- k-FGNN (Maron 2019) 이 어떻게 k-WL 표현력의 GNN 으로 구현되는가?
- 3-WL 이 strongly regular graph 를 구분 가능한가?
- k-WL 의 $O(n^k)$ 비용이 실전에서 $k = 2, 3$ 도 어려운 이유는?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

Ch4-02, Ch4-03 에서 message passing GNN 이 1-WL 상한임을 보았습니다. 이를 넘어서려면 **더 강한 graph isomorphism test** 인 k-WL 이 필요:

1. **이론적 강력함**: $k$-WL 이 sufficient $k$ 에서 GI 해결 가능 (Babai 의 quasi-polynomial 결과)
2. **Strongly regular graph 구분**: 3-WL 부터 가능
3. **GNN 의 표현력 확장**: k-FGNN 가 k-WL 표현력 도달

그러나 실전 한계:
- $O(n^k)$ 메모리·계산 — $k = 3, n = 100$ 만으로 $10^6$ 비용
- 작은 그래프에서만 가능

이 문서는 k-WL 의 정의·위계·효율 trade-off 를 정리하고, 후속 position-aware GNN (Ch4-05) 으로의 동기를 제공합니다.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-wl-test.md](./01-wl-test.md), [03-gin-optimality.md](./03-gin-optimality.md)
- 조합론: $k$-tuple, $V^k$
- Tensor algebra: 고차원 tensor 와 contraction

---

## 📖 직관적 이해

### k-Tuple as Super-Node

1-WL: 각 노드 $v \in V$ 가 super-node, 이웃 = $N(v)$.

**k-WL**: 각 $k$-tuple $(v_1, v_2, \ldots, v_k) \in V^k$ 가 super-node. 이웃 = "한 tuple 자리 가 다른 모든 tuple". 즉 $k$ 차 hypergraph 위에서 message passing.

총 super-node 수: $|V|^k = n^k$. 따라서 $O(n^k)$ 비용 — 폭발적.

### k-WL 의 핵심 직관

**더 풍부한 정보**: 각 tuple 이 "여러 노드 사이 관계" 를 인코딩.
- 1-WL: 노드 $v$ 의 1-hop neighborhood 만
- 2-WL: 노드 쌍 $(u, v)$ 의 관계 (edge 있는지, common neighbor 수, 거리, …)
- 3-WL: 노드 triple — triangle, common 이웃, …

이를 통해 1-WL 가 못보는 fine-grained structure 캡처.

### 위계의 strictness

$k$-WL ⊊ $(k+1)$-WL: 일부 graph 가 $(k+1)$-WL 가 구분하지만 $k$-WL 가 못함.

**예**:
- 1-WL ⊊ 2-WL: 일부 cospectral graph (Schwenk pair) — 2-WL 가 구분
- 2-WL ⊊ 3-WL: Strongly regular graph (Paley, CSL) — 3-WL 가 구분 가능
- 일반: $k$-WL ⊊ $(k+1)$-WL strict

### Folklore vs Standard k-WL

- **Folklore k-WL** (k-FWL): 더 강력한 형태, $k$-FWL = $(k+1)$-WL (Cai-Fürer-Immerman 1992).
- **Standard k-WL**: 약간 약함 — 일부 paper 는 standard, 일부 folklore 사용.

여기서는 standard k-WL 사용 (의미상 동일).

---

## ✏️ 엄밀한 정의

### 정의 4.1 — k-WL Algorithm

**Initial coloring** for $k$-tuple $\bar v = (v_1, \ldots, v_k) \in V^k$:
$$
c^{(0)}(\bar v) = \text{atomic type}(\bar v)
$$

(induced subgraph type — $k$ nodes 사이 edge 패턴 + label)

**Iteration**:
$$
c^{(l+1)}(\bar v) = \text{hash}\left(c^{(l)}(\bar v), N_1^{(l)}(\bar v), \ldots, N_k^{(l)}(\bar v)\right)
$$

여기서 $N_j^{(l)}(\bar v)$ = "$j$-번째 자리 만 변형" multiset of color:
$$
N_j^{(l)}(\bar v) = \{\!\{c^{(l)}(v_1, \ldots, v_{j-1}, w, v_{j+1}, \ldots, v_k) : w \in V\}\!\}
$$

### 정의 4.2 — k-WL Equivalence

$$
G_1 \stackrel{k\text{-WL}}{\equiv} G_2 \Leftrightarrow \{\!\{c^*(\bar v) : \bar v \in V_1^k\}\!\} = \{\!\{c^*(\bar w) : \bar w \in V_2^k\}\!\}
$$

(stable color partition multiset equality)

### 정의 4.3 — k-FGNN (Folklore GNN, Maron 2019)

$k$-WL 도달하는 GNN. Tensor representation:
$$
T \in \mathbb R^{n^k \times d}
$$

(각 $k$-tuple 마다 feature)

**Layer**: equivariant linear 변환 + nonlinearity.
- Equivariant: permutation 에 invariant (group $S_n$)
- Linear basis: Maron 2019 가 $k$-tuple 의 equivariant linear map 의 basis 를 enumerate

### 정의 4.4 — 2-IGN (Invariant Graph Network, Maron 2019)

가장 작은 nontrivial case. Tensor $T \in \mathbb R^{n \times n \times d}$ 를 input/output:
- Pair-wise feature
- Equivariant linear: 15 개 basis (constant + Kronecker delta + permutation 등)

표현력: 2-IGN ≥ 1-WL (Maron Theorem 4). 실증적으로 비슷 or 약간 우월.

### 정의 4.5 — 3-WL과 k-FGNN

**3-WL 표현력 도달**: 3-FGNN. Tensor $T \in \mathbb R^{n^3 \times d}$ — 각 triple 마다 feature.

비용: $O(n^3 d^2)$ per layer — $n = 100$ 만 해도 $10^6 d^2$ 매 forward.

---

## 🔬 정리와 결과

### 정리 4.1 — k-WL 위계의 strictness (Cai-Fürer-Immerman 1992)

**Theorem**: 모든 $k \geq 1$ 에 대해, $k$-WL 가 구분 못하지만 $(k+1)$-WL 가 구분하는 graph pair 존재. 즉:
$$
k\text{-WL} \subsetneq (k+1)\text{-WL}
$$

**증명 sketch**: CFI graph construction — 각 $k$ 에 대해 specific graph pair 만들기.

핵심 기술: bit-encoded graph 에서 일부 "twist" — $k$-WL 가 못 본다.

### 정리 4.2 — k-WL 의 Polynomial Time

**Theorem**: $k$-WL 가 $O(n^{k+1} \log n)$ time 안에 종료.

**증명 sketch**: $n^k$ tuple, 각 step 에서 $n^{k-1}$ neighbor multiset 처리 + hash. $O(n)$ iteration 안에 stable. $\square$

따라서 k-WL 자체는 polynomial time, but exponent $k$.

### 정리 4.3 — k-FGNN 표현력 = k-WL

**Theorem (Maron 2019)**: k-FGNN 의 표현력 = k-WL.

즉:
$$
G_1 \stackrel{k\text{-WL}}{\equiv} G_2 \Leftrightarrow \exists \text{ k-FGNN } \phi^* \text{ s.t. } \phi^*(G_1) = \phi^*(G_2)
$$

(GIN 이 1-WL 도달과 같은 의미)

**증명**: k-FGNN 의 equivariant layer 가 k-WL refinement step 을 시뮬레이션 가능 (UAT-like 추론).

### 정리 4.4 — 3-WL 가 Strongly Regular Graph 구분

**Theorem**: 일부 strongly regular graph (Paley, Petersen, …) 가 1-WL, 2-WL 동등이지만 3-WL 가 구분.

**예**: $4 \times 4$ rook graph (16 vertices) vs Shrikhande graph — 둘 다 (16, 6, 2, 2)-strongly regular. 3-WL 가 구분 (Murphy 2019).

이것이 **3-WL 의 실전적 가치** — strongly regular graph 구분 필요한 task (분자 isomorphism, chemical fingerprint).

### 정리 4.5 — k-WL ↔ Counting Logic CL$_k$

**Theorem (Cai-Fürer-Immerman)**: k-WL 가 graph $G_1, G_2$ 를 구분 ⟺ first-order logic with counting quantifiers and $k$ variables (CL$_k$) 가 구분.

이는 graph isomorphism 의 logic-theoretic characterization. GNN 표현력의 정확한 logical statement.

### 정리 4.6 — k-FGNN 의 메모리·계산 비용

**Theorem**: $L$-layer k-FGNN cost:
- Memory: $O(n^k d)$
- Computation per layer: $O(n^{k+1} d)$ or $O(n^k d^2)$ (equivariant basis 따라)
- Total: $O(L n^{k+1} d)$

**Practical**:
- $k=2, n=100$: $10^6 d$ — manageable (GPU)
- $k=3, n=100$: $10^8 d$ — borderline
- $k=3, n=1000$: $10^{12} d$ — unfeasible

---

## 💻 구현

### 실험 1 — 2-WL 의 Pair-wise Coloring

```python
import numpy as np
import networkx as nx
from collections import Counter

def initial_2wl_color(G, u, v):
    """Atomic type of pair (u, v)."""
    n = G.number_of_nodes()
    if u == v:
        return ('self', G.degree(u))
    elif G.has_edge(u, v):
        return ('edge', G.degree(u), G.degree(v))
    else:
        return ('non-edge', G.degree(u), G.degree(v))

def two_wl_step(G, colors):
    """One iteration of 2-WL on pair colors."""
    nodes = list(G.nodes())
    new_colors = {}
    for u in nodes:
        for v in nodes:
            # Neighbor multisets: change u or v to any w
            N1 = tuple(sorted(colors[(w, v)] for w in nodes))
            N2 = tuple(sorted(colors[(u, w)] for w in nodes))
            new_colors[(u, v)] = (colors[(u, v)], N1, N2)
    # Canonicalize
    label_map = {}
    canonical = {}
    next_id = 0
    for k, c in new_colors.items():
        if c not in label_map:
            label_map[c] = next_id
            next_id += 1
        canonical[k] = label_map[c]
    return canonical

def two_wl(G, max_iter=10):
    nodes = list(G.nodes())
    colors = {(u, v): hash(initial_2wl_color(G, u, v)) % 100000 for u in nodes for v in nodes}
    for _ in range(max_iter):
        new_colors = two_wl_step(G, colors)
        if Counter(new_colors.values()) == Counter(colors.values()):
            return colors
        colors = new_colors
    return colors

# Test on small graph
G = nx.cycle_graph(4)
colors = two_wl(G)
print(f'C_4 2-WL pair colors:')
for (u, v), c in sorted(colors.items()):
    if u <= v:
        print(f'  ({u},{v}): color {c}')
```

### 실험 2 — 1-WL vs 2-WL on Cospectral Pair

```python
# 이론상 1-WL 가 못 구분, 2-WL 가 구분하는 graph pair
# (간단한 예 찾기 어려움 — Cai-Fürer-Immerman construction 필요)
# 실증적 차이: rook graph vs Shrikhande (3-WL 만 구분)

def rook_graph(n):
    """n x n rook graph."""
    G = nx.Graph()
    for i in range(n):
        for j in range(n):
            G.add_node((i, j))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if k != j:
                    G.add_edge((i, j), (i, k))
                if k != i:
                    G.add_edge((i, j), (k, j))
    return G

# 4x4 rook graph
G_rook = rook_graph(4)
print(f'Rook 4x4: n={G_rook.number_of_nodes()}, m={G_rook.number_of_edges()}')
print(f'All deg = {set(dict(G_rook.degree()).values())}')   # all 6
```

### 실험 3 — 2-IGN Layer (Maron 2019, Simplified)

```python
import torch
import torch.nn as nn

class TwoIGN(nn.Module):
    """Simplified 2-IGN: input/output tensor T ∈ R^{n×n×d}."""
    def __init__(self, d_in, d_out, num_basis=15):
        super().__init__()
        # 15 equivariant basis maps for 2x2 IGN (Maron 2019 Appendix)
        # 여기선 단순화: 4개 representative basis
        # 1: identity (t -> t)
        # 2: transpose (t -> t.T)
        # 3: diagonal extraction
        # 4: row sum
        self.W = nn.Parameter(torch.randn(num_basis, d_in, d_out) / d_in**0.5)
        self.num_basis = num_basis
    
    def forward(self, T):
        """T: [n, n, d_in]"""
        n, _, d = T.shape
        outs = []
        # Basis 1: T 자체
        outs.append(T)
        # Basis 2: transpose
        outs.append(T.transpose(0, 1))
        # Basis 3: diagonal에서 broadcast
        diag = torch.diagonal(T, dim1=0, dim2=1).T   # [n, d]
        outs.append(diag.unsqueeze(1).expand(n, n, d))
        outs.append(diag.unsqueeze(0).expand(n, n, d))
        # Basis 4: row/col sum
        outs.append(T.sum(0).unsqueeze(0).expand(n, n, d))
        outs.append(T.sum(1).unsqueeze(1).expand(n, n, d))
        # Total mean
        outs.append(T.mean(dim=(0, 1)).expand(n, n, d))
        # ... (15까지 확장 가능)
        
        # Apply weights to first 7 basis (단순화)
        result = torch.zeros(n, n, self.W.size(-1))
        for i, basis in enumerate(outs[:min(self.num_basis, 7)]):
            result = result + basis @ self.W[i]
        return result

# Toy test
n, d = 5, 4
T = torch.randn(n, n, d)
layer = TwoIGN(d_in=d, d_out=8, num_basis=7)
out = layer(T)
print(f'2-IGN: input {T.shape} → output {out.shape}')
```

### 실험 4 — k-WL 의 Computational Cost 시각화

```python
import matplotlib.pyplot as plt

ns = [10, 50, 100, 500, 1000]
costs = {
    'k=1 (GIN)': [n for n in ns],
    'k=2 (2-IGN)': [n**2 for n in ns],
    'k=3 (3-FGNN)': [n**3 for n in ns],
    'k=4': [n**4 for n in ns],
}

fig, ax = plt.subplots(figsize=(8, 6))
for k, c in costs.items():
    ax.loglog(ns, c, 'o-', label=k)
ax.set_xlabel('n (number of nodes)')
ax.set_ylabel('Memory/Compute (log scale)')
ax.set_title('k-WL/k-FGNN cost scaling')
ax.legend(); ax.grid()
plt.show()
```

### 실험 5 — Strongly Regular Graph 의 어려움

```python
# Paley graph P(13): strongly regular (13, 6, 2, 3)
def paley_graph(q):
    QR = set((i * i) % q for i in range(1, q))
    G = nx.Graph()
    G.add_nodes_from(range(q))
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in QR or (i - j) % q in QR:
                G.add_edge(i, j)
    return G

P13 = paley_graph(13)
print(f'P(13) is (13, 6, λ, μ)-srg, all deg = {set(dict(P13.degree()).values())}')

# Cospectral non-isomorphic graph 찾기 어려움 — Paley(q) for prime q ≡ 1 mod 4 isomorphic 가능
# 더 정확한 test: Paley(q1) vs Paley(q2) — 보통 다른 size, 의미 X

# CSL family 가 더 명확
def csl(n, skip):
    G = nx.cycle_graph(n)
    for i in range(n):
        G.add_edge(i, (i + skip) % n)
    return G

# CSL(10, 2) vs CSL(10, 3): 1-WL 동등, 둘 다 4-regular
# 이런 graph pair 가 k-WL 의 보강이 필요한 motivation
```

---

## 🔗 실전 활용

### 1. 실전 GNN 모델

- **k-GNN (Morris 2019)**: Local k-WL 의 sparse 버전 — node $k$-tuple 만 (induced subgraph). 효율 ↑.
- **k-FGNN (Maron 2019)**: 표준 k-WL 표현력. PyG 가 일부 구현.
- **PPGN (Provably Powerful GN, Maron 2019)**: Matrix multiplication 으로 2-WL 보다 강함 (local 2-FWL).

### 2. CSL Benchmark

CSL graph classification: 4-regular, 1-WL fail. **3-WL or PPGN** 만 정확히 분류 가능.

GIN (1-WL): random guess.
PPGN: 100% accuracy.

### 3. Subgraph-based GNN

대안 approach: 각 노드의 ego-graph (k-hop subgraph) 를 별도 GNN 으로 처리 — Bevilacqua 2022 (ESAN), Zhao 2022 (NestedGNN). 표현력 ↑ but cost ↓ (compared to k-FGNN).

### 4. Cycle Detection

분자 chemistry 의 ring counting: Subgraph GNN 이 GIN 보다 우월. 일반 message passing 의 substructure counting 한계 (Chen 2020).

### 5. Practical Limit

$k = 2$ 까지 (PPGN, 2-IGN) 가 실전 한계. $k = 3$ 은 toy graph (n < 50) 에서만.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Polynomial time $O(n^{k+1})$ | $k = 3$ 도 large graph 에서 비실용 |
| Memory $O(n^k)$ | $k = 2$ 도 $n = 1000$ 시 100M tensor |
| Equivariant linear basis enumeration | k 커지면 basis 수 폭발 (Maron 2019: 2-IGN 15, 3-IGN 280+) |
| 표현력 = k-WL ceiling | Real-world task 에서 marginal 향상 |
| Permutation symmetric input | 일부 task 는 sub-permutation 만 |
| Implementation complexity | 표준 PyG/DGL 의 first-class support 부족 |

---

## 📌 핵심 정리

$$\boxed{1\text{-WL} \subsetneq 2\text{-WL} \subsetneq 3\text{-WL} \subsetneq \ldots}$$

$$\boxed{\text{k-FGNN expressive power} = \text{k-WL}}$$

| k | 표현력 | 비용 | 실전 |
|---|--------|------|------|
| 1 | GCN/GIN level | $O(n)$ | 표준 |
| 2 | 2-IGN, PPGN | $O(n^2)$ | 가능 (small graph) |
| 3 | 3-FGNN | $O(n^3)$ | borderline (toy) |
| 4+ | 이론적 | $O(n^4)$ | 비실용 |

**Trade-off**: 표현력 ↑ vs 비용 폭발.

**대안**: Position-aware (Ch4-05), Subgraph GNN — 효율과 표현력 의 다른 axis.

---

## 🤔 생각해볼 문제

**문제 1** (기초): $k = 2$ 와 $k = 1$ (Standard) WL 의 차이를 simple example 으로 비교하라.

<details>
<summary>해설</summary>

**1-WL**: 노드 $v$ 의 color 가 자신과 이웃 multiset 으로 결정.

**2-WL**: 노드 쌍 $(u, v)$ 의 color 가 (u 의 자리 변경, v 의 자리 변경 시) multiset 으로 결정. 더 풍부한 정보.

**Example: $C_4$ (4-cycle)**:

1-WL: 모든 노드 동등 → 1 cell.

2-WL: 노드 쌍 종류:
- $(v, v)$: self
- $(v, u)$ adjacent ($u \in N(v)$, distance 1)
- $(v, u)$ distance 2 (반대편)

3 cell. 1-WL 보다 더 풍부한 분할 — 거리 정보 capture.

**$K_4$ (complete) 와 $C_4$ 의 2-WL 차이**:

- $K_4$: 모든 nontrivial pair $(u, v)$ adjacent, distance 1. 2 cell (self, edge).
- $C_4$: edge pair, non-edge pair (distance 2). 3 cell (self, edge, non-edge).

2-WL color multiset 다름 → 구분 가능.

1-WL 도 $K_4$ ($d = 3$) vs $C_4$ ($d = 2$) 구분 (degree 차이) — 두 다 가능 in this case.

**더 정확한 차이**: 같은 degree distribution + 다른 structure (CSL pair) — 1-WL 못함, 2-WL 일부 가능. $\square$

</details>

**문제 2** (심화): 3-WL 가 strongly regular graph 를 구분하는 메커니즘을 분석하라. (Hint: 3-tuple 의 induced subgraph type)

<details>
<summary>해설</summary>

**Strongly regular graph** $(n, k, \lambda, \mu)$:
- 모든 노드 $k$-regular
- 인접 두 노드의 공통 이웃 수 $\lambda$
- 비인접 두 노드의 공통 이웃 수 $\mu$

**1-WL 한계**: 모든 노드 같은 색 (k-regular). Stable 후 1 cell.

**2-WL 한계**: 노드 쌍의 atomic type — edge / non-edge / self. 이는 $\lambda, \mu$ 정보 보유 (2-WL refinement 후 spread).

하지만 **두 (n, k, λ, μ) 동등 strongly regular graph 가 두 다 isomorphic 아닌 경우** — 같은 parameter 가지지만 다름. 2-WL color partition 이 같음 (parameter 만 의존).

**3-WL의 우위**:

3-tuple $(u, v, w)$ 의 atomic type 은 induced subgraph $G[\{u, v, w\}]$ 의 isomorphism class. 가능한 triple:
- 3 isolated nodes (no edges among 3)
- Path of 3 nodes (1 edge)
- "V" shape (2 edges sharing one node)
- Triangle (3 edges)

**Strongly regular** 가 (n, k, λ, μ) 만으로 결정되는 모든 통계는 1-WL, 2-WL 가 capture. 그러나 **triple 의 분포** 는 더 미세 — 두 SR graph 가 같은 (n, k, λ, μ) 가지지만 다른 triple distribution 가능.

**Example: 4×4 rook vs Shrikhande**:

- Rook 4×4: $4 \times 4$ grid 의 row/column 연결. (16, 6, 2, 2)-srg.
- Shrikhande: 다른 (16, 6, 2, 2)-srg.

같은 parameter 지만 isomorphic 아님 — triple 의 induced subgraph distribution 다름. 3-WL 가 이를 capture.

**증명 sketch**: Triangle 수 vs 3-path 수의 비율이 두 graph 에서 다름. 이는 atomic 3-tuple coloring 에서 직접 추출.

따라서 3-WL ⊋ 2-WL 의 strict 우위가 SR graph family 에서 명확. $\square$

</details>

**문제 3** (논문 비평): k-FGNN 의 $O(n^k)$ 비용이 실전 비실용. 대안 (subgraph GNN, position-aware) 이 어떻게 표현력 vs 효율 trade-off 를 다르게 다루는가?

<details>
<summary>해설</summary>

**k-FGNN 의 trade-off**:
- 표현력: Provably k-WL
- 비용: $O(n^k)$ — exponential in $k$

**대안 1: Subgraph GNN (Bevilacqua 2022 ESAN, Zhao 2022 NestedGNN)**:

각 노드의 ego-subgraph (k-hop) 를 별도 GNN 으로 처리, output 을 결합:
$$
h_v = \text{Combine}(\{\text{GNN}(\text{Ego}_k(u)) : u \in V\})
$$

비용: $O(n \cdot |\text{Ego}_k|)$ — 평균 $|\text{Ego}_k| = O(\bar d^k)$, sparse graph 시 manageable.

표현력: 1-WL 보다 strict 강 (substructure counting 가능). 일부 task 에서 3-WL level.

**대안 2: Position-aware GNN (Ch4-05)**:

Random anchor distance, Laplacian eigenvector 등 graph-specific positional feature 추가:
- LapPE: $\sqrt{\lambda_k} u_k$ 를 노드 PE 로
- P-GNN: random anchor set 까지 SP distance

비용: $O(n^2 d)$ for full LapPE (eigendecomposition), $O(n m)$ for sparse.

표현력: 1-WL 우회 (position 정보 추가). Symmetric graph 구분 가능. 단 정확한 k-WL 도달 X.

**대안 3: ID-GNN (You 2021)**:

각 노드에 unique ID injection. 표현력 ↑↑ (CSL 도 구분), 단 permutation invariance 깨짐.

**비교**:

| 방법 | 표현력 | 비용 | 안정성 |
|------|--------|------|--------|
| k-FGNN | k-WL | $O(n^k)$ | 결정적 |
| Subgraph GNN | > 1-WL, ≈ 3-WL | $O(n \bar d^k)$ | 결정적 |
| Position-aware | > 1-WL | $O(n m)$ | 결정적 |
| Random ID | 매우 강 | $O(n m)$ | Stochastic (variance) |

**현대 추세**:

- **Graphormer (Ch7-01)**: Position + dense attention + structural encoding. 효율 + 표현력 trade-off 의 sweet spot.
- **GPS (Rampášek 2022)**: Local message passing + global attention 의 hybrid.

따라서 k-FGNN 의 직접적 사용은 toy benchmark 에 한정, 실전 GNN 의 표현력 향상은 다른 axis (subgraph, position, attention) 로.

이는 **deep learning 의 일반 패턴** — exact 표현력 ceiling 보다 inductive bias + 효율 + 일반화 의 종합이 중요. Ch7-04 의 GNN 미래에서 더 자세히.

</details>

---

<div align="center">

[◀ 이전](./03-gin-optimality.md) | [📚 README](../README.md) | [다음 ▶](./05-positional-encoding.md)

</div>
