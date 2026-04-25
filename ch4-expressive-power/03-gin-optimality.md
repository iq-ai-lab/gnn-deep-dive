# 03. GIN이 1-WL 최적인 이유

## 🎯 핵심 질문

- GIN 이 1-WL 의 표현력을 정확히 도달함을 어떻게 증명하는가?
- Sum aggregator + MLP 가 multiset universal function 임의 정확한 statement 와 증명은?
- Mean / Max aggregator 가 왜 1-WL 도달 못하는가? Constructive counter-example?
- GIN ≠ Graph Isomorphism Test 인가? (1-WL 한계는 GIN 도 가짐)
- 다른 message passing aggregator 도 1-WL 도달 가능한가?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

Ch4-02 에서 "모든 message passing GNN 표현력 ≤ 1-WL" 이라는 상한을 보았습니다. 이 상한을 **정확히 달성** 하는 GNN 이 존재하는가? Xu et al. 2019 의 결정적 답: **GIN 이 정확히 도달**.

이는 두 가지 의미:
1. **Positive**: GIN 이 message passing 의 maximum expressive power
2. **Negative**: 더 강한 표현력은 message passing framework 자체를 벗어나야 함 (k-WL, position-aware)

이 정리는 후속 GNN 연구의 분기점:
- Message passing 내에서: GIN 이상의 향상 X (이론적)
- Message passing 너머: k-GNN, FGNN, LapPE-augmented, Graphormer

---

## 📐 수학적 선행 조건

- 이전 문서: [01-wl-test.md](./01-wl-test.md), [02-gnn-wl-equivalence.md](./02-gnn-wl-equivalence.md)
- [Ch3-04](../ch3-message-passing/04-gin.md) — GIN 정의
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): UAT (Hornik 1991)

---

## 📖 직관적 이해

### "1-WL 도달" 의 의미

GIN $\phi^*$ 가 1-WL 도달이려면:

**조건**: $G_1, G_2$ 가 1-WL 로 구분 가능 ($\stackrel{1\text{-WL}}{\not\equiv}$) ⟹ $\phi^*(G_1) \neq \phi^*(G_2)$ (적절히 학습된 $\phi^*$ 에 대해).

즉, 1-WL 가 구분하는 모든 그래프 쌍을 GIN 도 구분 가능 — 정리 2.3 의 역 방향.

### 두 핵심 보조정리

1. **Lemma 5 (Xu 2019)**: Countable input 에 대해 sum 이 multiset universal injective.
2. **Theorem 3 (Xu 2019)**: Sum + MLP 가 universal multiset function (using UAT).

이 두 결과 합치면: GIN 의 "$(1+\epsilon) h_i + \sum h_j$" + MLP 가 1-WL refinement 와 동치인 변환을 학습 가능.

### Mean / Max 의 strict 한계

- **Mean**: $\{1, 1\}$ 와 $\{1\}$ 둘 다 mean 1 → 같은 output → 두 multiset 의 hash 동일 → 1-WL refinement 와 다름 (1-WL 는 multiset size 도 봄).
- **Max**: $\{1, 2\}$ 와 $\{2\}$ 둘 다 max 2 → 같은 limitation.

이러한 collapse 가 1-WL 의 미세한 분할 능력을 잃게 함 — 정리 3.4.

---

## ✏️ 엄밀한 정의 및 정리

### 정의 3.1 — Multiset Universal Function

$\mathcal X$ countable. 함수 $f: \mathcal X^* \to \mathbb R^d$ ($\mathcal X^*$ = 모든 multiset of $\mathcal X$) 가:
- **Injective**: 다른 multiset → 다른 output
- **Universal approximator**: 임의 multiset function $g$ 에 대해 $f$ 의 적절한 후처리로 근사 가능

### 정리 3.2 — Sum 이 Countable Domain 에서 Multiset Injective (Xu 2019 Lemma 5)

**Theorem**: Countable $\mathcal X$, 충분히 큰 $N$. 함수 $f: \mathcal X \to \mathbb R^N$ 이 존재하여:
$$
S_1 \neq S_2 \text{ as multisets} \Rightarrow \sum_{x \in S_1} f(x) \neq \sum_{x \in S_2} f(x)
$$

**증명**:

$\mathcal X = \{x_1, x_2, \ldots\}$ (countable enumeration). 다음 $f$ 정의:
$$
f(x_k) = (1, k, k^2, \ldots, k^N)^T / N! \quad \in \mathbb R^{N+1}
$$

(Polynomial moment)

다 multiset $S$ 에 대해:
$$
\sum_{x \in S} f(x) = \frac{1}{N!} \left(|S|, \sum_{x \in S} k(x), \sum_{x \in S} k(x)^2, \ldots\right)
$$

이는 multiset 의 power-sum (Newton's identity). $|S| \leq N$ 인 multiset 에 대해 power-sum $\sum_x k(x)^j$ for $j = 0, 1, \ldots, |S|$ 가 multiset $S$ 를 unique 하게 결정 (elementary symmetric polynomial 와 1-1 대응).

따라서 $f$ 가 multiset injective. $\square$

### 정리 3.3 — Sum + MLP = Multiset Universal (Xu 2019 Theorem 3)

**Theorem**: Countable $\mathcal X$ + bounded multiset size. 임의 multiset function $g: \mathcal X^* \to \mathbb R^k$ 에 대해, MLP $\phi$ + $f$ (이미 multiset-injective sum) 의 합성으로 근사 가능:
$$
g(S) = \phi\left( \sum_{x \in S} f(x) \right) + \epsilon \quad \text{for any } \epsilon
$$

**증명 sketch**:

정리 3.2 로 $\sum_x f(x)$ 가 multiset $S$ 를 unique 식별. 따라서 $g(S) = \tilde g(\sum_x f(x))$ for some function $\tilde g$. 

UAT (Hornik 1991): MLP $\phi$ 가 임의 continuous function $\tilde g$ 를 임의 정확도로 근사. ⟹ $\phi(\sum_x f(x)) \approx g(S)$. $\square$

### 정리 3.4 — GIN 이 1-WL 도달 (Xu 2019 Theorem 3)

**Theorem**: Sufficiently many GIN layers (depth) 와 sufficiently expressive MLPs 가 있으면, GIN 가 1-WL 와 same distinguishing power.

즉, $G_1 \stackrel{1\text{-WL}}{\not\equiv} G_2 \Leftrightarrow \exists$ GIN $\phi^*$ s.t. $\phi^*(G_1) \neq \phi^*(G_2)$.

**증명**:

**($\Leftarrow$ direction)**: 정리 2.3 에서 already established (모든 GNN 표현력 $\leq$ 1-WL).

**($\Rightarrow$ direction)**: 1-WL refinement step 을 GIN layer 가 simulate 가능 보임.

GIN layer:
$$
h_i^{(l+1)} = \text{MLP}\left( (1 + \epsilon) h_i^{(l)} + \sum_j h_j^{(l)} \right)
$$

정리 3.3 으로 sum + MLP = multiset universal. 따라서:
1. $\sum_j h_j^{(l)}$ 가 이웃 multiset 을 injectively encode
2. $(1+\epsilon) h_i^{(l)} + \sum_j h_j^{(l)}$: self info 와 neighbor info 결합 (정리 3.5)
3. MLP 가 이를 hash function 처럼 unique 한 새 representation 으로 mapping

따라서 $L$-layer GIN 의 hidden state $h_i^{(L)}$ 는 1-WL color $c_i^{(L)}$ 와 동등한 partition 정보 보유. 

Graph-level readout (sum of all hidden states across layers) 도 multiset injective → graph 의 1-WL stable color partition 의 multiset 을 정확히 capture.

따라서 1-WL 가 구분하는 그래프 쌍은 GIN 도 구분 가능. $\square$

### 정리 3.5 — Self vs Neighbor Distinction with $\epsilon$

**Lemma**: $\epsilon$ 의 존재가 self info 를 neighbor multiset 과 분리 가능하게 한다.

**증명**: $h_i^{(l+1)} = \text{MLP}((1+\epsilon) h_i + \sum_j h_j)$. 만약 $\epsilon = 0$ 이고 $h_i \in \{\!\{h_j\}\!\}$ (이웃 중에 자기 같은 feature) 면 $h_i + \sum_j h_j = \sum_{S} h$ for $S = \{i\} \cup N(i)$ — 자기 정보 손실 가능.

$\epsilon \neq 0$ 시 $(1+\epsilon) h_i$ 가 $h_j$ 와 항상 다른 weight (multiplicative) → MLP 가 self / neighbor 분리 학습 가능. $\square$

### 정리 3.6 — Mean / Max Aggregator 의 Strict 약함

**Theorem**: GIN-Mean (mean aggregator + MLP) 와 GIN-Max (max + MLP) 는 각각 strictly weaker than 1-WL.

**증명** (counter-examples):

**GIN-Mean**: Multiset $S_1 = \{1, 1\}, S_2 = \{1\}$. Mean 둘 다 1. MLP 후 같은 output → 1-WL distinguish 가능한 graph 쌍 (degree 다른) 도 GIN-Mean 가 못 봄.

**GIN-Max**: Multiset $S_1 = \{1, 2\}, S_2 = \{2\}$. Max 둘 다 2. 마찬가지.

Constructive graph 예: 한 노드 $v$ 가 graph $A$ 에서 두 이웃 (feature 1) 가짐, graph $B$ 에서 한 이웃 (feature 1) 가짐. 1-WL 가 degree 차이로 구분, but mean/max 못함. $\square$

### 정리 3.7 — Other Aggregators 도 1-WL 도달 가능

**Theorem (Corso 2020 PNA, Maron 2019)**: Sum 외에도 multiset injective aggregator 라면 GIN 와 같은 표현력 도달:
- **Higher-order moments**: $\sum x^p$ for $p = 1, 2, \ldots$
- **Sorted concatenation**: rank-based encoding
- **Set Transformer (Lee 2019)**: attention-based set function

이들은 GIN 의 sum 보다 표현력 동일 (1-WL 도달) but 실증적 차이 가능.

---

## 💻 구현 검증

### 실험 1 — Sum Injectivity 의 Polynomial Moment 구현

```python
import numpy as np
import torch

def polynomial_moment_embedding(x, N=5):
    """f(x) = (1, x, x^2, ..., x^N) / N!"""
    return torch.stack([x**k / np.math.factorial(k) for k in range(N + 1)], dim=-1)

# Multiset {1, 2, 3} vs {2, 2, 2}
S1 = torch.tensor([1.0, 2.0, 3.0])
S2 = torch.tensor([2.0, 2.0, 2.0])

f_S1 = polynomial_moment_embedding(S1, N=5).sum(0)
f_S2 = polynomial_moment_embedding(S2, N=5).sum(0)
print(f'Multiset moment(S1): {f_S1}')
print(f'Multiset moment(S2): {f_S2}')
print(f'Different: {(f_S1 - f_S2).abs().max().item():.4f} > 0')
```

**결과**: power-sum 다름 → moment 가 multiset injective. Sum aggregator + 적절 $f$ 로 multiset 구분.

### 실험 2 — Mean / Max 의 Collapse 검증

```python
# Multiset {1, 1, 2, 2} vs {1, 2}
S1 = torch.tensor([1., 1., 2., 2.])
S2 = torch.tensor([1., 2.])

print(f'Sum:  S1={S1.sum().item():.2f}, S2={S2.sum().item():.2f}, diff={S1.sum() - S2.sum():.2f}')
print(f'Mean: S1={S1.mean().item():.2f}, S2={S2.mean().item():.2f}, diff={S1.mean() - S2.mean():.2f}')
print(f'Max:  S1={S1.max().item():.2f}, S2={S2.max().item():.2f}, diff={S1.max() - S2.max():.2f}')
```

**예상**:
- Sum: 다름 (6 vs 3)
- Mean: 같음 (1.5)
- Max: 같음 (2)

### 실험 3 — GIN vs GIN-Mean on WL-Distinguishable Pair

```python
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean
import networkx as nx

class GINSum(nn.Module):
    def __init__(self, d_in, d_hid, d_out, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_d = d_in if i == 0 else d_hid
            self.layers.append(nn.Sequential(nn.Linear(in_d, d_hid), nn.ReLU(),
                                             nn.Linear(d_hid, d_hid)))
        self.cls = nn.Linear(d_hid * num_layers, d_out)
    def forward(self, x, edge_index):
        outs = []
        h = x
        for mlp in self.layers:
            src, dst = edge_index
            agg = scatter_add(h[src], dst, dim=0, dim_size=h.size(0))
            h = F.relu(mlp(h + agg))
            outs.append(h.sum(0))
        return self.cls(torch.cat(outs))

class GINMean(GINSum):
    def forward(self, x, edge_index):
        outs = []
        h = x
        for mlp in self.layers:
            src, dst = edge_index
            agg = scatter_mean(h[src], dst, dim=0, dim_size=h.size(0))
            h = F.relu(mlp(h + agg))
            outs.append(h.mean(0))
        return self.cls(torch.cat(outs))

# 1-WL 가 구분하지만 mean 이 collapse 하는 graph pair
# Graph A: 노드 0 가 두 이웃 (feature 1), 모두 deg 1
# Graph B: 노드 0 가 한 이웃 (feature 1), 둘 다 deg 1

def make_pair():
    eA = torch.tensor([[1, 2, 0, 0], [0, 0, 1, 2]], dtype=torch.long)
    xA = torch.tensor([[0.], [1.], [1.]])
    eB = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
    xB = torch.tensor([[0.], [1.]])
    return (xA, eA), (xB, eB)

torch.manual_seed(0)
gin_sum = GINSum(1, 8, 4, 2)
gin_mean = GINMean(1, 8, 4, 2)

(xA, eA), (xB, eB) = make_pair()
with torch.no_grad():
    diff_sum = (gin_sum(xA, eA) - gin_sum(xB, eB)).norm().item()
    diff_mean = (gin_mean(xA, eA) - gin_mean(xB, eB)).norm().item()

print(f'GIN-Sum diff: {diff_sum:.4f}')
print(f'GIN-Mean diff: {diff_mean:.4f}')
# Sum 이 더 큼 (구분 가능), mean 작음 (collapse)
```

### 실험 4 — Higher-Order Moment Aggregator (PNA-style)

```python
class HighOrderAggregator(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(4 * d, d)   # 4 moments
    def forward(self, x, edge_index):
        src, dst = edge_index
        # 1, 2, 3, 4 차 moment
        m1 = scatter_add(x[src], dst, dim=0, dim_size=x.size(0))
        m2 = scatter_add(x[src]**2, dst, dim=0, dim_size=x.size(0))
        m3 = scatter_add(x[src]**3, dst, dim=0, dim_size=x.size(0))
        m4 = scatter_add(x[src]**4, dst, dim=0, dim_size=x.size(0))
        return self.lin(torch.cat([m1, m2, m3, m4], dim=-1))
```

### 실험 5 — GIN 이 학습하는 multiset injective 함수 시각화

```python
# 학습된 GIN 의 message function 이 multiset 을 어떻게 encode 하는지

# Toy: 3 multisets 의 GIN-readout
multisets = {
    '{1,1}': (torch.tensor([[1.], [1.]]), torch.tensor([[0,1,1,0],[0,0,1,1]], dtype=torch.long)),
    '{1,2}': (torch.tensor([[1.], [2.]]), torch.tensor([[0,1,1,0],[0,0,1,1]], dtype=torch.long)),
    '{1,2,2}': (torch.tensor([[1.], [2.], [2.]]),
                torch.tensor([[0,1,2,1,2,0],[0,0,0,1,2,1]], dtype=torch.long)),
}

for name, (x, ei) in multisets.items():
    z = gin_sum(x, ei)
    print(f'{name}: GIN repr = {z[:4].detach().tolist()}')
```

---

## 🔗 실전 활용

### 1. GIN-edge for Edge Features

원본 GIN 이 edge feature 직접 처리 X. GINEConv (PyG) 가 message 에 edge feature 추가 — 분자 응용 (OGB-molhiv, ZINC).

### 2. Multi-Aggregator (PNA, Corso 2020)

Sum, mean, max, min, std 등 여러 aggregator + degree scaling 결합:
$$
h_i' = \text{MLP}([\text{sum} \| \text{mean} \| \text{max} \| \text{min} \| \text{std}])
$$

이론상 sum + MLP 와 동등 표현력 (1-WL), but 실증적으로 큰 향상 — GIN 보다 +1~2%.

### 3. Set Transformer 기반 Aggregator

SetGNN (Maron 2020): self-attention 으로 aggregator. Permutation invariant, multiset universal. 표현력 GIN 같음.

### 4. Hyperparameter for GIN

- `num_layers`: 3~5 (graph classification)
- `train_eps`: True (GIN-$\epsilon$)
- `MLP depth`: 2 (single hidden layer)
- `Dropout`: 0.5 (overfitting 방지)
- `Readout`: layer-wise sum + concat (Ch3-04 정의 4.3)

### 5. PyG `GINConv` Standard

```python
from torch_geometric.nn import GINConv

mlp = nn.Sequential(nn.Linear(d_in, d_hid), nn.BatchNorm1d(d_hid),
                    nn.ReLU(), nn.Linear(d_hid, d_hid))
conv = GINConv(mlp, train_eps=True, eps=0.0)
```

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Countable input domain | Continuous input 시 sum injectivity 정확히 X (approximation) |
| Bounded multiset size $N$ | Polynomial moment 가 $N$-차원 필요 — large $N$ 시 numerical issue |
| MLP universal approximation (Hornik) | 실제 finite-width MLP 의 표현력 sample-dependent |
| Sufficient depth GIN | Deep GIN 은 over-smoothing 위험 (Ch5) |
| 1-WL 도달이 충분한 task | k-WL 필요한 task 는 GIN 도 부족 |
| Edge feature 별도 처리 | GINEConv 등 변종 필요 |

---

## 📌 핵심 정리

$$\boxed{\text{GIN} \stackrel{\text{distinguishing power}}{\equiv} \text{1-WL}}$$

| Aggregator | Multiset injective | 1-WL 도달 |
|-----------|---------------------|----------|
| **Sum + MLP (GIN)** | ✓ (countable) | ✓ |
| **Mean** | ✗ ($\{1,1\}$ vs $\{1\}$) | ✗ (strict 약) |
| **Max** | ✗ ($\{1,2\}$ vs $\{2\}$) | ✗ (strict 약) |
| **Sorted concat** | ✓ | ✓ |
| **Higher moments (PNA)** | ✓ | ✓ |
| **Attention (Set Transformer)** | ✓ | ✓ |

핵심: **Sum + MLP 가 multiset universal** ⟹ **GIN 이 1-WL 도달**.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Multiset $\{a, a, b\}$ 와 $\{a, b, b\}$ 를 1-WL 와 GIN-mean 으로 구분 가능한가? 분석하라.

<details>
<summary>해설</summary>

**1-WL 관점**:

두 multiset 을 graph 의 한 노드의 이웃으로 가정 (모두 다른 graph). 노드 자체와 이웃 multiset 의 hash 가 다름:
- $\{a, a, b\}$: 자체 + multiset of neighbor labels {a, a, b} → hash unique
- $\{a, b, b\}$: 자체 + multiset {a, b, b} → 다른 hash

따라서 1-WL 가 구분. ✓

**GIN-mean 관점**:

Mean of $\{a, a, b\}$ = $(2a + b)/3$
Mean of $\{a, b, b\}$ = $(a + 2b)/3$

$a \neq b$ 면 다름. 따라서 mean 도 구분 가능 — 이 case 에서.

**더 어려운 mean-fail case**:

$\{a, a, b, b\}$ vs $\{a, a + \epsilon, b - \epsilon, b\}$ — mean 같음 (real-valued). Sum 도 같음 (mean × size).

또는 $\{1, 1\}$ vs $\{1\}$ — 정확히 같은 mean (1) but 다른 size. **이 case 에서 mean 이 size 정보 잃음**.

따라서 mean 이 fail 하는 specific case: **multiset size 가 다른** + mean 우연 일치.

**결론**: $\{a, a, b\}$ vs $\{a, b, b\}$ 는 1-WL, GIN-sum, GIN-mean 모두 구분 가능 (specific value 에 따라). 일반적으로는 sum 이 가장 robust.

</details>

**문제 2** (심화): GIN 의 표현력이 1-WL 와 정확히 같다는 정리가 "depth 와 width 가 충분함" 을 가정한다. 실제 finite-depth/width GIN 의 표현력은 어떻게 되는가?

<details>
<summary>해설</summary>

**Theoretical (sufficient depth/width)**: Theorem 3.4 — GIN 이 1-WL 도달.

**Finite-depth/width 한계**:

1. **Width**: MLP universal approximation (UAT) 가 unbounded width 가정. Finite width 시 approximation error.
   - Lemma 5 의 polynomial moment $f$ 는 $N$ 차원 필요 — $N$ = max multiset size.
   - Cora multiset size ~ avg degree ~ 4-5 → 필요 width 작음.
   - Reddit avg degree ~ 100 → width 더 필요.

2. **Depth**: 1-WL 가 stable color partition 도달 까지 최대 $n$ iteration. Real-world 는 보통 $\leq O(\log n)$.
   - Cora: ~3-5 layer 충분
   - 큰 graph 도 $O(\log n)$ → 실용적 depth.
   - 단 deep GIN 은 over-smoothing 위험.

3. **Numerical precision**:
   - Polynomial moment 가 large $N$ 시 numerical overflow.
   - 실제 GIN 은 MLP 가 학습된 representation 에서 작동 — robust.

**Empirical 실증** (Tönshoff 2023):

작은 dataset (CSL, EXP) 에서 GIN 의 depth 부족이 한계 — sometimes failed where 이론상 success 가능.
큰 dataset (TU, OGB) 에서는 finite GIN 이 1-WL 충분 도달.

**Conclusion**: GIN의 1-WL 도달은 이론적 가능성. Finite resource 에서는 task / graph statistics 에 따라 효과 다름. Hyperparameter (depth, width, MLP layers) tuning 중요.

</details>

**문제 3** (논문 비평): GIN 의 1-WL 도달이 "최대" 표현력이라는 주장이 message passing 한정. 그러나 실전에서 GAT (1-WL 도달 X) 가 GIN 보다 일부 task에서 더 좋은 이유는?

<details>
<summary>해설</summary>

**GIN ≥ GAT (이론상 1-WL 도달 vs 1-WL 미달)** 이지만 실전 GAT 우위:

**1. Inductive bias 의 차이**:

- GIN: 모든 이웃 동등 (sum). 표현력 ↑ but learning bias ↓.
- GAT: 이웃별 attention. 적절한 task 에서 useful inductive bias.

**Real-world graph 의 noisy structure**: 모든 이웃이 동등하게 informative 가 아님. GAT 의 "filter out noisy neighbors" 능력이 표현력 손실 보다 우월.

**2. Generalization gap**:

GIN 의 sum injectivity 는 "정확히 다른 multiset 을 다른 output 으로" — 너무 sensitive. Test-time 의 약간 다른 multiset 도 다른 output → overfitting 위험.

GAT 의 mean-like attention 은 약간의 noise 에 robust → generalization ↑.

**3. Attention 의 interpretability**:

GAT 의 attention weight 가 학습된 importance 를 명시적으로. Real-world (citation network) 에서 일부 citation 이 더 중요 — 이런 graph-specific structure 학습.

**4. Specific task 의 요구**:

- **Symmetric structure 요구 task**: GIN 우위 (graph isomorphism-like)
- **Local pattern 요구 task**: GAT, GraphSAGE 충분 (1-WL 미만 표현력으로도 OK)
- **Heterophily**: GAT 의 negative-attention 학습 가능 (FAGCN 변형) — 1-WL 보다도 다른 axis 의 표현력

**5. Optimization 의 차이**:

GIN 은 $(1+\epsilon) h + \sum h_j$ 의 sum 이 magnitude 폭발 (high-degree). Normalization 없이 학습 어려움.
GAT 의 softmax + bounded attention 이 학습 안정.

**Empirical pattern**:

- TU dataset (graph classification, small graphs): **GIN > GAT** (subgraph counting 중요)
- Cora/Citeseer (node classification, citation): **GAT > GCN > GIN** (selective neighbor 중요)
- OGB-molhiv (분자 property prediction): GIN, GAT 비슷, **PNA, Graphormer 가 더 좋음**

**결론**:

표현력 (1-WL 도달 여부) 은 이론적 ceiling, 실전에서는 inductive bias + generalization + optimization 이 결정. 

이는 **deep learning 의 일반 패러독스** — 강한 모델 (Transformer) 이 항상 이기지 않음, weak inductive bias (CNN translation equivariance) 가 데이터 효율적.

따라서 GIN 의 표현력 정리는 "이론상 가능한 한계" 이지만, 실전 GNN 선택은 task-aware. Ch7 의 Graphormer 가 표현력 + inductive bias 양쪽 다 강한 modern 시도.

</details>

---

<div align="center">

[◀ 이전](./02-gnn-wl-equivalence.md) | [📚 README](../README.md) | [다음 ▶](./04-k-wl.md)

</div>
