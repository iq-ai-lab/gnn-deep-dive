# 03. Graph Attention Network (Velickovic 2018)

## 🎯 핵심 질문

- GAT 의 attention coefficient $\alpha_{ij}$ 가 어떻게 계산되며 GCN 의 fixed weight 와 무엇이 다른가?
- $\alpha_{ij} = \text{softmax}_j(\text{LeakyReLU}(a^T [Wh_i \| Wh_j]))$ 의 각 부분의 의미는?
- Multi-head attention 이 GAT 에서 어떻게 작동하고 단일 head 보다 우위인가?
- GAT 와 Transformer 의 self-attention 은 어떤 관계인가?
- Cora 에서 GAT attention weight 가 무엇을 학습하는가?

---

## 🔍 왜 이 기법이 그래프 학습에 필수인가

GCN 의 한계 중 하나: 모든 이웃을 **fixed weight** ($1/\sqrt{\tilde d_i \tilde d_j}$) 로 동일하게 다룬다는 것. 그러나 실제 그래프에서는 **이웃 간 중요도 차이** 가 큼:

- Citation network: 모든 인용이 같은 영향력 X
- Social network: 친한 친구 vs 일반 follower
- Molecule: 강한 bond vs 약한 bond

GAT (Velickovic et al. 2018 — **"Graph Attention Networks"**) 가 이 문제를 해결: **이웃별 weight 를 데이터로부터 학습** (attention mechanism). 또한 GAT 는 **Transformer 의 graph 일반화** 의 첫 시도이며, 후속 Graph Transformer (Graphormer) 의 직접적 전신.

---

## 📐 수학적 선행 조건

- 이전 문서: [01-mpnn-framework.md](./01-mpnn-framework.md) — MPNN, attention as aggregator
- [Transformer Deep Dive](https://github.com/iq-ai-lab/transformer-deep-dive): Self-attention, multi-head
- [Neural Network Theory Deep Dive](https://github.com/iq-ai-lab/neural-network-theory-deep-dive): Softmax

---

## 📖 직관적 이해

### Attention 의 직관

자연어 처리에서: "the cat sat on the mat" 의 "cat" 이 "mat" 보다 "sat" 에 더 attention. 학습된 weight $\alpha_{ij}$.

GAT 에서: 노드 $i$ 가 이웃 $j_1, j_2, \ldots$ 중 어떤 노드에 더 attention 할지 학습. 실제 graph structure (edge 만) 에서 출발하지만, **attention 은 sparse Transformer** — 이웃에 대한 self-attention.

### 3단계 계산

1. **Linear transform**: $z_i = W h_i$ (GAT 모든 노드에 같은 $W$)
2. **Attention score**: $e_{ij} = \text{LeakyReLU}(a^T [z_i \| z_j])$
3. **Softmax 정규화**: $\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in N(i)} \exp(e_{ik})}$
4. **Aggregation**: $h_i' = \sigma(\sum_j \alpha_{ij} z_j)$

### Multi-Head 의 의미

다양한 attention pattern 을 동시 학습:
$$
h_i' = \big\|_{k=1}^K \sigma\left( \sum_{j \in N(i)} \alpha_{ij}^{(k)} W^{(k)} h_j \right)
$$

(concat 또는 mean — 마지막 layer 는 mean)

각 head 가 다른 "관계" 학습 (예: 헤드 1=근접 friend, 헤드 2=common interest).

### Transformer 와의 비교

Transformer self-attention:
$$
\text{Attn}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d}) V
$$

GAT:
$$
\alpha_{ij} \propto \exp(a^T [W h_i \| W h_j])
$$

**유사**: softmax 정규화, query-key 매칭.

**차이**:
- Transformer: dot-product, $QK^T$
- GAT: additive (concat + $a$)
- Transformer: fully-connected (모든 token 쌍)
- GAT: sparse (graph edge 만)

GAT 가 sparse Transformer-on-graph. Graphormer (Ch7-01) 가 fully-connected 로 일반화.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — GAT Single-Head Attention

각 노드 pair $(i, j)$ 에 대해 ($j \in N(i) \cup \{i\}$, with self-loop):

**Linear transform**: $z_i = W h_i$, $z_j = W h_j$ (shared $W \in \mathbb R^{d_{\text{out}} \times d_{\text{in}}}$).

**Attention score** (additive, learnable $a \in \mathbb R^{2 d_{\text{out}}}$):
$$
e_{ij} = \text{LeakyReLU}(a^T [z_i \| z_j])
$$

(LeakyReLU slope $\alpha_{\text{leak}} = 0.2$ 표준)

**Softmax over neighborhood**:
$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in N(i) \cup \{i\}} \exp(e_{ik})}
$$

**Aggregation**:
$$
h_i' = \sigma\left( \sum_{j \in N(i) \cup \{i\}} \alpha_{ij} z_j \right)
$$

### 정의 3.2 — Multi-Head GAT

$K$ heads, 각 head 별 $\{W^{(k)}, a^{(k)}\}$:

**Hidden layer (concat)**:
$$
h_i' = \big\|_{k=1}^K \sigma\left( \sum_j \alpha_{ij}^{(k)} W^{(k)} h_j \right) \in \mathbb R^{K \cdot d_{\text{out}}}
$$

**Output layer (mean)**:
$$
h_i' = \sigma\left( \frac{1}{K} \sum_{k=1}^K \sum_j \alpha_{ij}^{(k)} W^{(k)} h_j \right) \in \mathbb R^{d_{\text{out}}}
$$

### 정의 3.3 — GATv2 (Brody 2022)

GAT 의 attention 이 "static" (특정 query 에 모든 key 가 같은 ranking) 한계 있음. GATv2:
$$
e_{ij} = a^T \text{LeakyReLU}(W [h_i \| h_j])
$$

(LeakyReLU 안에 $W$, 밖에 $a$ — order 변경)

이는 query-dependent ranking 가능 → 표현력 ↑.

---

## 🔬 정리와 결과

### 정리 3.1 — GAT as MPNN Instantiation

GAT 은 MPNN form 으로:
- $M(h_i, h_j) = \alpha_{ij}(h, j) W h_j$ (input-dependent edge function)
- $\bigoplus = $ sum (with attention weight)
- $U = \sigma$

**증명**: 정의 직접 비교. $\square$

단, $\alpha_{ij}$ 가 **set-wise softmax** — 단순 edge-wise function 이 아님 (Ch3-01 정리 1.5 의 한계).

### 정리 3.2 — Computational Cost

**Theorem**: GAT 한 layer 비용 (single head):
- Linear $W h$: $O(n d_{\text{in}} d_{\text{out}})$
- Attention $e_{ij}$: $O(m d_{\text{out}})$ (각 edge)
- Softmax + aggregation: $O(m d_{\text{out}})$

총 $O(m d_{\text{in}} d_{\text{out}} + n d_{\text{in}} d_{\text{out}})$ — sparse graph 에서 GCN 과 동일 order.

Multi-head $K$: $O(K m d_{\text{in}} d_{\text{out}})$.

### 정리 3.3 — Inductive Capability

GAT 의 학습 파라미터 $W, a$ 는 graph 와 독립 (정확히 GraphSAGE 와 같은 의미) → inductive learning 가능.

특히 PPI inductive benchmark 에서 GAT 이 GraphSAGE 보다 우월 (Velickovic 2018).

### 정리 3.4 — GAT 의 표현력 (제한적)

**Static attention 한계 (Brody 2022)**: 표준 GAT 는 ranking-static — given $Q = h_i$, $\arg\max_j \alpha_{ij}$ 가 $h_i$ 와 무관 (고정). 이는 query-dependent ranking 표현력의 부재.

**예시**: 그래프 dictionary lookup task — node $i$ 의 query 에 따라 다른 key 가 매칭되어야 하는데 GAT 는 못함.

GATv2 는 이를 해결.

### 정리 3.5 — Multi-Head Diversity

**Empirical**: 학습된 attention head 들이 다른 pattern 을 보임 — 일부는 local cluster 강조, 일부는 long-range dependency.

이는 Transformer 의 multi-head diversity 와 유사 — explicit regularization 없이도 자연스럽게 분화.

### 정리 3.6 — GAT 와 GAT 의 sparse Transformer interpretation

**Interpretation**: 만약 graph 가 fully-connected ($A = J - I$) + GAT attention 이 적용되면:
- Sparse mask 사라짐 → full pairwise softmax
- 표준 Transformer self-attention 과 거의 동일 (additive vs dot-product 차이만)

따라서 **GAT 는 sparse self-attention on graph** — Transformer 의 graph 일반화.

---

## 💻 구현

### 실험 1 — GAT Single-Head Layer

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_add

class GATLayer(nn.Module):
    def __init__(self, d_in, d_out, leak_slope=0.2):
        super().__init__()
        self.W = nn.Linear(d_in, d_out, bias=False)
        self.a = nn.Parameter(torch.randn(2 * d_out))
        self.leak = nn.LeakyReLU(leak_slope)
    
    def forward(self, x, edge_index):
        z = self.W(x)   # [n, d_out]
        src, dst = edge_index
        # 각 edge 의 attention score
        cat = torch.cat([z[src], z[dst]], dim=-1)   # [m, 2 d_out]
        e = self.leak((cat * self.a).sum(dim=-1))   # [m]
        # Softmax over each destination node's incoming edges
        alpha = scatter_softmax(e, dst)
        # Weighted aggregation
        out = scatter_add(alpha.unsqueeze(-1) * z[src], dst, dim=0, dim_size=x.size(0))
        return F.elu(out)
```

### 실험 2 — Multi-Head GAT

```python
class MultiHeadGAT(nn.Module):
    def __init__(self, d_in, d_out, K=8, concat=True):
        super().__init__()
        self.K = K
        self.concat = concat
        self.heads = nn.ModuleList([GATLayer(d_in, d_out) for _ in range(K)])
    
    def forward(self, x, edge_index):
        head_outs = [h(x, edge_index) for h in self.heads]
        if self.concat:
            return torch.cat(head_outs, dim=-1)   # [n, K * d_out]
        else:
            return torch.stack(head_outs).mean(0)   # [n, d_out]

# GAT 표준 (2-layer)
class GAT(nn.Module):
    def __init__(self, d_in, d_hid, d_out, K1=8, K2=1):
        super().__init__()
        self.gat1 = MultiHeadGAT(d_in, d_hid, K=K1, concat=True)   # concat
        self.gat2 = MultiHeadGAT(d_hid * K1, d_out, K=K2, concat=False)   # mean (output)
    
    def forward(self, x, edge_index):
        h = F.elu(self.gat1(x, edge_index))
        return self.gat2(h, edge_index)
```

### 실험 3 — Karate Club 학습

```python
import networkx as nx
import numpy as np

G = nx.karate_club_graph()
n = G.number_of_nodes()
edges = np.array(list(G.edges())).T
ei = torch.tensor(np.concatenate([edges, edges[::-1]], axis=1), dtype=torch.long)

# Self-loop 추가
self_loops = torch.arange(n).unsqueeze(0).repeat(2, 1)
ei = torch.cat([ei, self_loops], dim=1)

X = torch.eye(n)
labels = torch.tensor([G.nodes[i]['club'] == 'Officer' for i in range(n)], dtype=torch.long)
train_mask = torch.zeros(n, dtype=torch.bool); train_mask[[0, 33]] = True

model = GAT(d_in=n, d_hid=4, d_out=2, K1=4, K2=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=5e-4)

for epoch in range(200):
    model.train(); optimizer.zero_grad()
    out = model(X, ei)
    loss = F.cross_entropy(out[train_mask], labels[train_mask])
    loss.backward(); optimizer.step()

acc = (model(X, ei).argmax(-1) == labels).float().mean().item()
print(f'GAT accuracy on Karate Club: {acc:.2%}')
```

### 실험 4 — Attention Weight 시각화

```python
import matplotlib.pyplot as plt

# 첫 layer 의 첫 head 의 attention extract
def get_attention(model, x, edge_index):
    head = model.gat1.heads[0]
    z = head.W(x)
    src, dst = edge_index
    cat = torch.cat([z[src], z[dst]], dim=-1)
    e = head.leak((cat * head.a).sum(dim=-1))
    alpha = scatter_softmax(e, dst)
    return alpha, src, dst

alpha, src, dst = get_attention(model, X, ei)

# Edge thickness 로 attention 시각화
pos = nx.spring_layout(G, seed=42)
edge_widths = []
for s, d, a in zip(src.tolist(), dst.tolist(), alpha.detach().tolist()):
    if s != d and (s, d) in G.edges() or (d, s) in G.edges():
        edge_widths.append(a * 10)

fig, ax = plt.subplots(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color=labels, cmap='RdBu',
        ax=ax, node_size=200)
ax.set_title('GAT learned attention (edge thickness ∝ α)')
plt.show()
```

### 실험 5 — Static Attention 한계 (GATv2 비교)

```python
class GATv2Layer(nn.Module):
    """GATv2: linear before attention"""
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W = nn.Linear(2 * d_in, d_out)
        self.a = nn.Parameter(torch.randn(d_out))
        self.leak = nn.LeakyReLU(0.2)
    
    def forward(self, x, edge_index):
        src, dst = edge_index
        cat = torch.cat([x[src], x[dst]], dim=-1)   # [m, 2 d_in]
        z = self.leak(self.W(cat))                  # [m, d_out]
        e = (z * self.a).sum(-1)                    # [m]
        alpha = scatter_softmax(e, dst)
        # 다른 message: x[src] 또는 별도 transform
        msg = z   # GATv2 에선 transformed cat 자체가 message
        return scatter_add(alpha.unsqueeze(-1) * msg, dst, dim=0, dim_size=x.size(0))
```

---

## 🔗 실전 활용

### 1. Cora·Citeseer·Pubmed Standard

GAT (8 heads, 8 hidden) 의 표준 결과:
- Cora: 83.0% (vs GCN 81.5%)
- Citeseer: 72.5% (vs GCN 70.3%)
- Pubmed: 79.0% (vs GCN 79.0%)

Attention 이 작은 그래프에서 marginal 이득.

### 2. Inductive PPI

PPI (Protein-Protein Interaction): 24 separate graphs. GAT 의 inductive 우위:
- GraphSAGE: 0.612 micro-F1
- GAT: 0.973 micro-F1

이는 inductive setting 에서 GAT 의 큰 우위.

### 3. PyG GATConv / GATv2Conv

```python
from torch_geometric.nn import GATConv, GATv2Conv

gat = GATConv(in_channels=16, out_channels=8, heads=8, concat=True)
out = gat(x, edge_index, return_attention_weights=True)
# out: (output, (edge_index_with_self_loops, attention))
```

### 4. GraphTransformer 와의 융합

Dwivedi & Bresson (2021) "A Generalization of Transformer Networks to Graphs" — GAT + Transformer 의 본격 통합. 후속 Graphormer (Ch7-01), GraphGPS, SAN 등 전체 graph transformer 의 ancestor.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Static attention (GAT) | Query-dependent ranking 못함 → GATv2 |
| Local attention only (sparse) | Long-range dependency 약함 → Graphormer (dense) |
| Single attention scoring (additive) | Dot-product, MLP scoring 등 다양화 가능 |
| Self-loop 추가 가정 | 경우에 따라 self exclusion 도 필요 |
| Memory $O(m K d^2)$ | Large $K, d$ 시 메모리 폭발 |
| Softmax sharpness | Temperature tuning 필요 (entropy collapse) |

---

## 📌 핵심 정리

$$\boxed{e_{ij} = \text{LeakyReLU}(a^T [W h_i \| W h_j]), \quad \alpha_{ij} = \text{softmax}_j(e_{ij})}$$

$$\boxed{h_i' = \big\|_{k=1}^K \sigma\left(\sum_{j \in N(i) \cup \{i\}} \alpha_{ij}^{(k)} W^{(k)} h_j\right)}$$

| 항목 | GAT |
|------|-----|
| **Aggregator** | Attention-weighted sum |
| **Attention scoring** | Additive (concat + LeakyReLU + linear) |
| **Multi-head** | $K$ heads, concat (hidden) / mean (output) |
| **Inductive** | ✓ |
| **Cost** | $O(K m d^2)$ |
| **Transformer 관계** | Sparse self-attention on graph |
| **GATv2 개선** | Linear before attention → query-dependent |
| **현대 후속** | Graphormer (dense + structural bias, Ch7-01) |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $K_4$ (complete graph 4 nodes) 에서 모든 노드 feature 가 동일 ($h_i = c$ for all $i$) 일 때, GAT attention $\alpha_{ij}$ 가 무엇인지 계산하라.

<details>
<summary>해설</summary>

$h_i = c$ (모두 같음) 이면 $W h_i = Wc$ 도 같음. 모든 $e_{ij} = \text{LeakyReLU}(a^T [Wc \| Wc])$ 도 같음.

따라서 모든 $j \in N(i) \cup \{i\}$ 에 대해 $e_{ij}$ 동일 → softmax 결과 uniform:
$$
\alpha_{ij} = \frac{1}{|N(i) \cup \{i\}|} = \frac{1}{4}
$$

(4-node complete + self-loop)

**의미**: GAT 의 attention 이 input feature 에 의존 — 같은 feature 면 fixed weight (≈ GCN-like). Diverse feature 일 때 attention 의 효과가 나타남.

특히 학습 초기 (random init feature) 또는 over-smoothed regime 에서 GAT ≈ GCN. 이는 over-smoothing 시 GAT 의 attention 우위가 사라진다는 관찰의 이론적 근거.

</details>

**문제 2** (심화): GATv2 가 GAT 의 "static attention" 한계를 어떻게 우회하는지 식 차이를 통해 설명하고, 표현력 위계 GAT $\subset$ GATv2 $\subset$ Transformer 를 논하라.

<details>
<summary>해설</summary>

**GAT**:
$$
e_{ij} = \text{LeakyReLU}(a_1^T (W h_i) + a_2^T (W h_j))
$$

(여기서 $a = [a_1; a_2]$ split)

**GATv2**:
$$
e_{ij} = a^T \text{LeakyReLU}(W [h_i \| h_j])
$$

**Static 한계 (GAT)**:

GAT 은 분해 가능: $e_{ij} = f(h_i) + g(h_j)$ (LeakyReLU 의 piecewise linearity 무시 시). 따라서:
$$
\arg\max_j e_{ij} = \arg\max_j g(h_j)
$$

— **$h_i$ 와 무관**. 즉 어떤 query node $i$ 에 대해서도 같은 ranking → "global key importance" 만 학습.

**GATv2** 는 LeakyReLU 안에 $W [h_i \| h_j]$ — $h_i$ 와 $h_j$ 가 nonlinearity 로 entangle. 따라서 query-dependent ranking 가능.

**위계**:

- GAT: ranking $\arg\max_j$ 가 query-independent
- GATv2: query-dependent ranking 가능
- Transformer: dot-product $h_i^T h_j$ — fully bilinear, multiplicative interaction

GAT $\subset$ GATv2 (additive) $\subset$ Transformer (multiplicative).

**경험적**: Brody 2022 의 dictionary lookup task 에서 GAT 실패, GATv2 성공. Real-world (Cora 등) 에서는 GAT, GATv2 차이 크지 않음 — task 의 ranking-flexibility 요구 정도가 결정.

</details>

**문제 3** (논문 비평): GAT 의 multi-head attention 이 다양한 pattern 을 학습한다는 주장이 있지만, 일부 연구는 head redundancy 를 보고했다. Multi-head 가 단일 head 보다 우월한 이론적 근거와 실증적 한계를 비교하라.

<details>
<summary>해설</summary>

**Multi-head 의 이론적 동기**:

1. **Diverse attention pattern**: 각 head 가 다른 subspace projection ($W^{(k)}$) → 다른 feature 강조. 이론상 풍부한 표현.

2. **Mixture of experts**: 각 head 를 sub-aggregator 로 보면 모델 ensemble 효과.

3. **Increased capacity**: $K$ heads = $K$ independent linear projection.

4. **Variance reduction**: Output mean (last layer) 이 stochastic 평균 → 안정.

**실증적 한계**:

1. **Head redundancy** (Voita 2019, on Transformer): 학습 후 일부 head 가 거의 동일한 attention pattern. Pruning 시 성능 거의 무손실.

2. **Pruning 가능**: Michel 2019 — Transformer head 의 60%+ 제거해도 성능 유지. GAT 도 비슷.

3. **Initialization sensitivity**: Random init 가 head diversity 결정. Bad init 시 모든 head 가 같은 pattern.

4. **Task-dependent**: 작은 그래프 (Cora) 에서는 multi-head 효과 marginal. 큰 그래프 (Reddit) 에서 더 큰 차이.

**중간적 결론**:

Multi-head 의 이론적 효과는 명확하지만 실증에서 head pruning 가능 — 즉 most heads are redundant in well-trained models. 이는 original Transformer 에서도 동일 — over-parameterization 가 학습 안정성 위한 일종의 regularization.

**대안**: 

1. **Sparsemax / structured sparse attention**: 명시적으로 다른 head 끼리 다른 pattern 강제.
2. **Mixture-of-experts (MoE)**: Sparse activation — 각 input 에 대해 일부 head 만 활성.
3. **Diversity loss**: Inter-head attention 차이를 explicit term 으로 추가.

따라서 GAT multi-head 는 실용적이지만 이론적 정당화 보다 empirical hyperparameter 의 성격. Modern GNN (Graphormer Ch7-01) 도 이 관행 계승.

</details>

---

<div align="center">

[◀ 이전](./02-graphsage.md) | [📚 README](../README.md) | [다음 ▶](./04-gin.md)

</div>
