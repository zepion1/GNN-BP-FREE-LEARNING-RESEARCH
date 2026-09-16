# ForwardGNN study notes

**Paper:** Forward Learning of Graph Neural Networks — Park, Wang, Simoulin, Yang, Yang, Rossi, Trivedi, Ahmed (Meta AI, Adobe Research, U. Michigan, Intel Labs). ICLR 2024. arXiv:2403.11004. Code: https://github.com/facebookresearch/forwardgnn

**Notes compiled:** September 2026

---

## Table of contents

1. [Background: GNNs, backpropagation, Forward-Forward](#1-background)
2. [The problem ForwardGNN solves](#2-the-problem)
3. [Adapting FF to graphs (Section 3.1)](#3-adapting-ff-to-graphs)
4. [Single-forward learning (Section 3.2)](#4-single-forward-learning-sf)
5. [Top-down signals (Section 3.3)](#5-top-down-signals)
6. [Link prediction (Section 3.4)](#6-link-prediction)
7. [Experiments and results](#7-experiments-and-results)
8. [Extensibility and takeaways](#8-extensibility-and-takeaways)
9. [Deep dive: the mathematics of Forward-Forward](#9-deep-dive-forward-forward-mathematics)
10. [Deep dive: the single-forward algorithm step by step](#10-deep-dive-single-forward-algorithm)
11. [Deep dive: unlabeled nodes in single-forward](#11-deep-dive-unlabeled-nodes)
12. [Quick-reference glossary](#12-glossary)

---

## 1. Background

### 1.1 Graph neural networks (GNNs)

A graph G = (V, E) has nodes (papers, products, people) and edges (citations, co-purchases, follows). Each node i has a feature vector **xᵢ** and possibly a class label.

A GNN computes a node embedding **hᵢ** by repeated **message passing**. One layer, for each node i (Eq. 1):

1. **Message:** for each neighbor j, compute ψ(hᵢ, hⱼ, e₍ⱼ,ᵢ₎).
2. **Aggregate:** combine all incoming messages with a permutation-invariant ⊕ (mean, max, sum).
3. **Update:** γ(hᵢ, aggregated messages) → new h⁽ˡ⁾ᵢ.

h⁽⁰⁾ᵢ = xᵢ. After L layers, a node's embedding reflects its L-hop neighborhood.

Architectures used in the paper (differ only in ψ, ⊕, γ):

| Model | Aggregation |
|---|---|
| GCN | Normalized weighted average, self-loops, symmetric normalization |
| GraphSAGE (SAGE) | Mean aggregator, neighbors transformed separately from self |
| GAT | Learned attention weights over neighbors (4 heads, LeakyReLU slope 0.2) |

Tasks:
- **Node classification** — predict a node's label.
- **Link prediction** — predict whether an edge exists between two nodes.

### 1.2 Backpropagation (BP) and its constraints

Forward pass computes outputs and stores activations; backward pass propagates error via the chain rule; gradient descent updates parameters.

The paper's three objections (biologically implausible and practically limiting):
1. Forward activations must be **stored** for the backward pass → memory grows with depth.
2. Each layer's update depends on **non-local** signals from all downstream layers.
3. Updates happen only **after** the full forward pass, in reverse order → limits parallelism.

### 1.3 The Forward-Forward algorithm (FF, Hinton 2022)

- Replaces forward + backward with **two forward passes**; trains layers **greedily one at a time**, bottom to top, freezing each after training.
- **Positive data:** image with correct one-hot label overlaid. **Negative data:** same image with a random wrong label.
- **Goodness** (Eq. 2): G = ‖a‖² = Σᵢ aᵢ² (sum of squared ReLU activations).
- **Probability of positive** (Eq. 3): p = σ(G − θ), θ a threshold (2.0 in this paper).
- **Normalization:** output is scaled to unit length before being passed up, so the next layer must use direction, not length.

Related forward-only methods (baselines in the paper):
- **CaFo** — network weights random and frozen; only per-layer classifiers trained.
- **SymBa** — FF variant with a loss that directly maximizes the goodness gap.
- **PEPITA** — pre-FF; uses output error to modulate the input for a second pass.
- **GFF** — FF for graph *classification* only; cannot do node classification or link prediction.

---

## 2. The problem

GNNs inherit BP's constraints. FF doesn't transfer to graphs because:
1. Negative data generation is **task-specific** and undefined for graphs.
2. Generating negatives is **not scalable** (memory and time grow with the number of negatives).
3. Forward-only learning had never been studied for node classification or link prediction.

**ForwardGNN contributes:**
- Two graph-specific ways to adapt FF (label appending, virtual nodes).
- A **single-forward** method needing no negative samples ("no longer forward-forward").
- **Top-down signals** from upper layers into lower layers without backprop.
- Extension to link prediction.
- Experiments on 3 GNNs × 5 datasets.

---

## 3. Adapting FF to graphs

### 3.1 Option A — Extend node features with labels (FF-LA)

Eq. (4): x′ᵢ = [xᵢ ‖ Label-To-Feat(i)]
- Positive: one-hot of the correct label. Negative: one-hot of a random wrong label.
- Unlabeled nodes get a **uniform distribution** over classes.
- Graph structure unchanged.

### 3.2 Option B — Extend graph structure with virtual nodes (FF-VN)

Eq. (5): add K virtual nodes (one per class). Link labeled real node i to virtual node c when Label-RE(i) = Label-VR(c).
- Label-VR(c) always returns the virtual node's true class.
- Label-RE(i) returns correct label → G^pos′, or random wrong label → G^neg′.
- In the positive graph, each virtual node aggregates only from nodes of its class → **class representative**.
- From a real node's viewpoint, positive vs. negative differ only in which virtual node it's linked to.

### 3.3 Training (Algorithm 1)

Layer by layer: forward on positive and negative inputs → FF loss (Eq. 3) → update that layer with Adam → repeat until convergence → keep best-validation weights → compute outputs → **detach** → feed to next layer. K−1 negatives per node (all wrong classes).

### 3.4 Inference

For each candidate label ℓ, attach ℓ (feature or virtual link), run all layers, **sum goodness over layers**, pick the max. Requires K forward passes.

---

## 4. Single-forward learning (SF)

### 4.1 Motivation

FF-style methods inject labels at the input, so positive and negative inputs must be built and run separately → high compute and memory.

### 4.2 Key idea

Use the **positively augmented graph G^pos′** for a single forward pass. Message passing turns each virtual node into a class representative c⁽ˡ⁾ₖ **in the same pass** that produces real-node embeddings h⁽ˡ⁾ᵢ. No separate prototype architecture.

### 4.3 Contrastive loss (Eq. 6)

```
L(h, i, ℓ) = −log [ exp(C(h, c_Label(i)) / τ) / Σₖ exp(C(h, cₖ) / τ) ]
L⁽ˡ⁾ = (1/|V_labeled|) Σ_{i ∈ V_labeled} L(h⁽ˡ⁾ᵢ, i, ℓ)
```

- C(·,·) = similarity (dot product). τ = temperature (1.0).
- Softmax over the K class representatives; the K−1 wrong classes are the negatives — obtained for free.
- Same-class nodes pulled together, different-class nodes pushed apart.

### 4.4 Training (Algorithm 2)

Same greedy bottom-to-top scheme, one forward per step, Eq. (6) loss, detach before passing up.

### 4.5 Inference

Softmax in Eq. (6) gives a class distribution per layer; **average across layers**, take the argmax. One forward pass.

---

## 5. Top-down signals

Greedy layer-wise training means layer ℓ never sees layers ℓ+1…L. Training is reframed as happening over time steps t; layer ℓ at step t uses upper-layer outputs from step t−1.

### 5.1 Top-to-Input (Eq. 7)

h⁽ˡ⁻¹⁾′ᵢ,ₜ = Merge(h⁽ˡ⁻¹⁾ᵢ,ₜ, {h⁽ˡ⁺¹⁾ᵢ,ₜ₋₁, …, h⁽ᴸ⁾ᵢ,ₜ₋₁})

Concatenate lower-layer output with upper-layer output(s). Experiments use only the immediate upper layer. Topmost layer's signal = context vector (one-hot label for training nodes, uniform for others).

### 5.2 Top-to-Loss (Eq. 8)

Layer input unchanged; the layer's **output** is merged with upper-layer outputs before the loss and at inference. Affects only the loss, not what's passed up.

### 5.3 Properties

- **No gradient crosses layers** — still forward-only.
- All layers trained concurrently (Alg. 3), synchronous or asynchronous (even/odd alternating) updates.
- Loses the "aggregate once, reuse" speedup because inputs change each step.

---

## 6. Link prediction

Standard link prediction is already FF-shaped: score real (positive) edges above sampled nonexistent (negative) edges.

Eq. (9): pᵢⱼ = LinkProb(EdgeEmb(hᵢ, hⱼ)) — experiments use dot product + sigmoid.

- Each layer trained **locally** (Alg. 4); positive and negative edge embeddings come from **one** forward pass.
- Loss variants: BCE (**ForwardGNN-CE**), FF goodness loss (**ForwardGNN-FF**), SymBa loss (**ForwardGNN-SymBa**).
- Top-down signals can be added the same way.
- **Inference:** average link probabilities over layers.

---

## 7. Experiments and results

### 7.1 Setup

| Dataset | Nodes | Edges | Features | Classes | Domain |
|---|---|---|---|---|---|
| PubMed | 19,717 | 88,648 | 500 | 3 | Citation |
| CiteSeer | 4,230 | 10,674 | 602 | 6 | Citation |
| CoraML | 2,995 | 16,316 | 2,879 | 7 | Citation |
| Amazon (Photo) | 7,650 | 238,162 | 745 | 8 | Co-purchase |
| GitHub | 37,700 | 578,006 | 128 | 2 | Followership |

- Splits 64/16/20 train/val/test, 5 random splits, mean ± std.
- Node classification: accuracy. Link prediction: ROC-AUC, equal positive/negative edges.
- 128 hidden units, Adam lr 0.001, weight decay 0.0005, ≤1000 epochs, early stopping patience 100.
- Memory = GPU memory for parameters, activations, gradients, optimizer moments. NVIDIA H100.

### 7.2 SF vs. BP (Section 4.2)

**Node classification**
- SF matches BP in many cases, beats it in several, across all three GNNs.
- BP memory grows with depth — up to **18×** at 4 layers. SF memory is **flat** regardless of depth.
- BP unstable at shallow depth on some datasets (e.g. Amazon GCN 2-layer 64.9 ± 16.1 vs. SF 93.7 ± 0.4).
- SF-GAT weak at 1 layer (60–67%), recovers with depth; SF-Top-To-Input fixes 1-layer GAT (~90%).

**Link prediction**
- SF **mostly beats BP**; BP degrades as layers are added, SF improves or holds.
- Same flat-memory pattern.

### 7.3 Forward-only methods compared (Section 4.3)

**Node classification (2-layer models)**
- Best accuracy always from SF-family or FF-family.
- FF methods (FF-VN, FF-LA, FF-SymBa) **unstable** (huge std devs) and need **much more memory** (e.g. FF-VN-GAT Amazon 1241 MB vs. SF-GAT 178 MB).
- CaFo: least memory, much worse accuracy (GNN never trained).
- PEPITA: essentially fails (15–40%).
- **Top-down helps** (Table 1, averaged over GNNs and depths):

| Dataset | SF | SF-TopDown |
|---|---|---|
| CiteSeer | 91.14 | 93.58 |
| CoraML | 84.95 | 86.01 |
| PubMed | 81.90 | 82.21 |
| Amazon | 89.69 | 92.09 |
| GitHub | 83.16 | 83.68 |

- Top-to-Input is the better path; Top-to-Loss can collapse with depth (GCN Amazon 93.8 → 47.4 at 4 layers).

**Link prediction**
- All ForwardGNN variants beat CaFo by a large margin.
- **ForwardGNN-CE best overall** (accuracy + stability). SymBa often high-variance.

### 7.4 Training time (Appendix F.3)

- Greedy forward methods train GCN/SAGE **up to 9× faster** than BP at 4 layers: a layer's input is fixed during its training, so neighborhood aggregation is computed **once** and reused.
- Doesn't apply to GAT (attention changes aggregation) or top-down variants; there forward learning can be up to ~1.6× slower than BP.
- SF faster than FF variants (no multiple passes).

### 7.5 Limited training data (Appendix F.4)

Planetoid Cora/CiteSeer/PubMed (60–140 training nodes): BP mostly wins at 2 layers — the contrastive objective **needs more labels** — but the gap closes with depth, and SF wins some 4-layer cases. BP relative memory up to ~21×.

### 7.6 Edge directionality for virtual nodes (Appendix F.5)

Default: bidirectional real↔virtual. One-way (real → virtual only):
- FF and SF-Top-To-Loss: bidirectional better.
- SF: unidirectional sometimes better (PubMed, GitHub).
- SF-Top-To-Input: no difference.

---

## 8. Extensibility and takeaways

- **Graph classification:** replace GFF's multiple passes with one pass if graph-level class representatives can be built (pooling + virtual nodes), then reuse Eq. (6).
- **Graph clustering:** no labels → virtual nodes don't apply; swap contrastive objective for self-supervised signals (homophily, community structure).
- **Related layer-wise GNN methods** (You et al. 2020, Wang et al. 2020) still use BP inside blocks and lack top-down signals.

**Bottom line:** SF trains GNNs layer by layer with a contrastive loss against virtual-node class representatives — no backprop, no negative samples. Matches or beats BP on node classification, beats it on link prediction, constant memory with depth, often faster. Costs: needs more labeled data; GAT/top-down variants lose the aggregation-caching speedup.

---

## 9. Deep dive: Forward-Forward mathematics

Step-by-step derivation of what one FF layer computes (explored with an interactive goodness/sigmoid widget).

**Step 1 — One layer, one forward pass.** Input vector → weight matrix → ReLU → activation vector a = (a₁, …, aₙ), all ≥ 0.

**Step 2 — Goodness = squared length.** G = ‖a‖² = Σᵢ aᵢ². Squaring makes a single very active neuron count disproportionately; differentiable everywhere.
- Example: a = (1.2, 0.8, 1.5, 0.4) → G = 1.44 + 0.64 + 2.25 + 0.16 = 4.49.

**Step 3 — Goodness → probability.** p(positive) = σ(G − θ), σ(z) = 1/(1+e⁻ᶻ). θ fixed (2.0). G ≫ θ → p → 1; G ≪ θ → p → 0.

**Step 4 — Local loss (binary cross-entropy).**
```
L = −log σ(G_pos − θ) − log(1 − σ(G_neg − θ))
  = softplus(θ − G_pos) + softplus(G_neg − θ)
```
First term pushes G_pos above θ; second pushes G_neg below θ. Gradient descent on *this layer's weights only*.

**Step 5 — Why "forward-forward".** G_pos and G_neg require running the same layer on two inputs (correct vs. wrong label attached) → two forward passes. Gradient is local; never flows to lower layers.

**Step 6 — Normalize before passing up.** Send a / ‖a‖ upward; the next layer can't cheat by reading input length and must find new evidence in the direction.

**Step 7 — Inference.** For each candidate label ℓ: attach ℓ, run all layers, score(ℓ) = Σ_layers G⁽ˡ⁾. Predict argmax. K passes for K classes.

**Observations**
- θ sets the *scale* the network must produce, not a boundary between classes — both inputs can sit above θ; the loss just reports how badly the negative is misplaced.
- The loss saturates once dots are far on the correct side → motivation for SymBa's gap-widening loss.

---

## 10. Deep dive: single-forward algorithm

Step-by-step, mapped onto the augmented-graph / contrastive-loss picture (explored with an interactive graph + softmax widget).

**Step 1 — Build G′ once, before training.** Eq. (5) with correct labels only. Add K virtual nodes; link every *labeled* real node to its true class's virtual node. Unlabeled nodes keep only real edges. Done once; no negative graph.

**Step 2 — One GNN layer forward on G′.** Standard message passing (Eq. 1). Outputs in the same pass:
- h⁽ˡ⁾ᵢ for each real node i.
- c⁽ˡ⁾ₖ for each virtual node k — a **class representative**, since virtual node k only receives messages from class-k nodes. No prototype network, no extra parameters. Links are bidirectional by default, so real nodes also hear back from their class node.

**Step 3 — Score every (node, class) pair.** C(hᵢ, cₖ) for k = 1…K via dot product.

**Step 4 — Softmax with temperature (Eq. 6).**
```
p(k | i) = exp(C(hᵢ, cₖ)/τ) / Σⱼ exp(C(hᵢ, cⱼ)/τ)
L(i)     = −log p(Label(i) | i)
L⁽ˡ⁾     = mean over labeled nodes
```
InfoNCE / supervised-contrastive form. The K−1 wrong classes are the negatives — vectors already in the output of one pass, not extra inputs. τ low → near one-hot; τ high → near uniform (paper: τ = 1).

**Step 5 — Update only this layer.** Gradient of L⁽ˡ⁾ w.r.t. layer ℓ's weights (Adam, lr 0.001). Gradient flows into both hᵢ and cₖ — nodes move toward prototypes and prototypes toward nodes simultaneously.

**Step 6 — Repeat, freeze, move up (Alg. 2).** Loop steps 2–5 until validation stops improving (patience 100), restore best weights, compute outputs, `.detach()`, feed to layer ℓ+1. Layer ℓ never touched again. Fixed input during training → GCN/SAGE aggregate once and reuse (source of the up-to-9× speedup).

**Step 7 — Inference: one pass, all classes.** Softmax from Step 4 gives a class distribution per layer; average over L layers, argmax. No label-attaching, no K passes.

**SF vs. FF in one line each**
- FF injects labels at the *input*; SF injects labels into the *structure* once.
- FF: positive + negative passes; SF: one pass.
- FF: binary "real vs. fake" on scalar goodness; SF: K-way classification on vector similarity.
- FF inference: sum scalar goodness over K trials; SF: average a probability vector from one trial.

**Subtlety:** virtual nodes aggregate from many real nodes and broadcast back, which can homogenize real-node embeddings. Appendix F.5: one-directional links (real → virtual) sometimes help SF (PubMed, GitHub).

---

## 11. Deep dive: unlabeled nodes

What happens to test/validation nodes in SF.

**During training**
- Label links exist only for i ∈ V_labeled (the 64% training split). Test/validation nodes keep real edges but have no virtual-node link.
- Message passing runs over the **whole graph**: unlabeled nodes receive messages (embeddings are computed for them) and **send** messages to labeled neighbors. So labeled embeddings — and hence class representatives cₖ — are shaped partly by unlabeled neighbors' features (standard transductive semi-supervised setup).
- Unlabeled nodes never touch virtual nodes, so they don't directly influence any cₖ, and they're excluded from the loss sum in Eq. (6). No gradient signal from them.
- Their embeddings are still computed, detached, and passed up, so every layer sees the full node set.
- Validation nodes: used only for early stopping and selecting best per-layer weights; loss on them is never backpropagated.

**During inference**
- Same graph G′, same single pass. Training nodes remain linked to their class nodes, so cₖ are still meaningful prototypes.
- Test node: each layer computes softmax over C(h⁽ˡ⁾ᵢ, c⁽ˡ⁾ₖ)/τ → K-way distribution; average over layers, argmax.
- The test node's label is never attached — unlike FF-VN, which needs K passes (one per candidate link).

**Edge cases**
- FF variants need a placeholder because the label is part of the input: FF-LA appends a uniform vector; FF-VN gives no virtual link. SF sidesteps this since labels only enter through edges.
- SF-TopDown (top-to-input): topmost layer's context vector = one-hot for training nodes, uniform for everyone else.

**Why it matters**
Class representatives are built only from labeled nodes, so SF quality depends on label count. This explains Appendix F.4: on Planetoid (60–140 training nodes) BP beats SF at shallow depth because prototypes are noisy; the gap narrows with depth as prototypes absorb wider unlabeled neighborhoods.

---

## 12. Glossary

| Term | Meaning |
|---|---|
| BP | Backpropagation |
| FF | Forward-Forward algorithm (Hinton 2022) |
| FF-LA | FF for graphs via label appending to node features (Sec. 3.1) |
| FF-VN | FF for graphs via virtual nodes (Sec. 3.1) |
| SF | Single-forward learning (Sec. 3.2) |
| SF-TopDown / SF-Top-To-Input | SF with upper-layer outputs concatenated into a layer's input (Sec. 3.3) |
| SF-Top-To-Loss | SF with upper-layer outputs merged into a layer's output for the loss only (Sec. 3.3) |
| ForwardGNN-CE / -FF / -SymBa | Link-prediction variants with BCE / FF / SymBa losses (Sec. 3.4) |
| Goodness G | Σ aᵢ², sum of squared ReLU activations (Eq. 2) |
| θ | Goodness threshold in σ(G − θ) (Eq. 3); 2.0 in experiments |
| τ | Temperature in the contrastive softmax (Eq. 6); 1.0 in experiments |
| C(·,·) | Critic / similarity function (dot product) |
| Virtual node | One added node per class, linked to labeled nodes of that class |
| Class representative cₖ | Embedding of virtual node k; acts as a class prototype |
| V_labeled | Set of labeled (training) nodes |
| Merge(·) | Function combining lower-layer and upper-layer signals (Eqs. 7–8) |
| CaFo | Cascaded Forward: frozen network, trained per-layer predictors |
| SymBa | FF variant maximizing goodness gap |
| PEPITA | Error-driven input modulation, pre-FF forward-only method |
| GFF | Graph Forward-Forward, graph classification only |

---

## Equation index

| Eq. | Content |
|---|---|
| (1) | GNN message passing: h⁽ˡ⁾ᵢ = γ(h⁽ˡ⁻¹⁾ᵢ, ⊕ⱼ ψ(h⁽ˡ⁻¹⁾ᵢ, h⁽ˡ⁻¹⁾ⱼ, e₍ⱼ,ᵢ₎)) |
| (2) | Goodness: G = ‖a‖² |
| (3) | p(positive) = σ(G − θ) |
| (4) | Label appending: x′ᵢ = [xᵢ ‖ Label-To-Feat(i)] |
| (5) | Virtual-node augmented graph G′ |
| (6) | SF contrastive loss (softmax over class representatives) |
| (7) | Top-to-Input merge |
| (8) | Top-to-Loss merge |
| (9) | Link probability pᵢⱼ = LinkProb(EdgeEmb(hᵢ, hⱼ)) |

## Algorithm index

| Alg. | Content |
|---|---|
| 1 | FF learning of GNNs for node classification (Sec. 3.1) |
| 2 | Single-forward learning for node classification (Sec. 3.2) |
| 3 | Single-forward with top-down signal paths (Sec. 3.3) |
| 4 | Single-forward learning for link prediction (Sec. 3.4) |