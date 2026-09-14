# BP-Free Learning: Study Guide

A consolidated summary of five documents on backpropagation-free (BP-free) training, split/federated learning, and forward learning of graph neural networks.

**Documents covered**

| # | Document | Type | Date |
|---|---|---|---|
| 1 | *Beyond-Backpropagation Training: Methods, Applications, and Perspectives* (Ye et al.) | Survey paper | Jan 2026 |
| 2 | *Forward Learning of Graph Neural Networks* — ForwardGNN (Park et al., ICLR 2024) | Research paper | Apr 2024 |
| 3 | Work Report, Week of Aug 18 (Anower Zihad) — GNN benchmark | Project report | Aug 2026 |
| 4 | Progress Report — Part 0 Complete, "What Crosses the Cut?" benchmark | Project report | Aug 24, 2026 |
| 5 | Results Report — All Methods Running, "What Crosses the Cut?" benchmark | Project report | Aug 30, 2026 |

---

## Table of Contents

1. [Key Concepts Primer](#1-key-concepts-primer)
2. [The Survey: Beyond-Backpropagation Training](#2-the-survey-beyond-backpropagation-training)
3. [ForwardGNN: Forward Learning of GNNs](#3-forwardgnn-forward-learning-of-gnns)
4. [Work Report (Aug 18): GNN Benchmark](#4-work-report-aug-18-gnn-benchmark)
5. [Progress Report Part 0 (Aug 24): Pipeline Validation](#5-progress-report-part-0-aug-24-pipeline-validation)
6. [Results Report (Aug 30): All Six Methods Running](#6-results-report-aug-30-all-six-methods-running)
7. [How Everything Connects](#7-how-everything-connects)
8. [Open Questions and Next Steps](#8-open-questions-and-next-steps)

---

## 1. Key Concepts Primer

Read this section first if any of the terms below are unfamiliar. Everything later builds on these ideas.

### 1.1 Backpropagation (BP) and why people want to avoid it

**Backpropagation** is the standard way to train neural networks. It works in two passes:

1. **Forward pass** — input flows through the layers to produce an output. All intermediate activations are stored.
2. **Backward pass** — the error between output and target is propagated backward through the network using the chain rule, computing a gradient for every weight. Weights are then adjusted by gradient descent.

BP is extremely effective, but it has three structural problems:

| Problem | What it means | Why it matters |
|---|---|---|
| **Weight transport** | Layer *l* needs the exact transpose of layer *l+1*'s weights to compute its error signal. | Biologically implausible — real neurons cannot access other neurons' weights. |
| **Locking** | *Forward locking*: layer *l* cannot compute until *l−1* finishes. *Backward locking*: layer *l*'s gradient depends on *l+1*'s gradient. | Layers cannot be trained in parallel; limits throughput on distributed hardware. |
| **Memory bottleneck** | All forward-pass activations must be kept until the backward pass. | Memory grows with depth; a major issue on edge devices and large models. |

**BP-free methods** try to sidestep one or more of these by generating learning signals differently or by updating weights at different times.

### 1.2 Split learning and federated learning

- **Federated learning (FL)** — many clients each train a copy of a model on their own data; a server periodically averages ("aggregates") the model weights. Raw data never leaves the client. Communication cost = weights sent up and down.
- **Split learning (SL)** — a single model is cut at some layer; the client runs the bottom part, the server runs the top part. At the **cut point**, the client sends activations forward and (with BP) receives gradients back every training step. Communication cost = activations + gradients crossing the cut.
- **Split-federated learning (SplitFed / SFL)** — combines both: multiple clients each run the bottom of the model and split-train with a server, and client-side weights are also federated-averaged. This is the setting of the "What Crosses the Cut?" benchmark.

**Why BP-free matters here:** with BP, gradients must cross the cut on *every* step. If the client side can be trained locally (e.g., with an auxiliary loss), gradients never need to come back, so cut communication collapses.

### 1.3 Communication accounting: cut vs. weights

The benchmark separates communication into two buckets:

- **Cut communication** — activations and gradients exchanged across the split point.
- **Weight communication** — model weights exchanged for federated aggregation.

Reporting only the total hides *why* a method is expensive. A full-BP method is dominated by cut traffic; a BP-free method may shift the cost toward weight traffic.

### 1.4 IID vs. non-IID data

- **IID** (α = ∞) — every client's data looks like a random sample of the whole dataset.
- **Non-IID** (e.g., α = 0.5) — client data is skewed; some clients see mostly a few classes. α is the concentration parameter of a Dirichlet distribution — smaller α means more skew. Non-IID data is realistic and typically hurts federated methods.

### 1.5 Graph neural networks (GNNs)

A GNN learns node representations by **message passing**: at each layer, every node aggregates information from its neighbors and combines it with its own embedding. After *k* layers a node "sees" its *k*-hop neighborhood. Common architectures:

- **GCN** — normalized mean over neighbors.
- **GraphSAGE** — sample-and-aggregate (mean, max-pool, etc.).
- **GAT** — learned attention weights over neighbors.

Standard tasks: **node classification** (label each node) and **link prediction** (does an edge exist between two nodes?).

In *distributed* GNN training, the graph is partitioned across clients, and the key issue is **how many cross-client edges are preserved** — dropping them loses neighborhood information and hurts accuracy.

### 1.6 The forward-forward (FF) algorithm and single-forward (SF) training

- **Forward-forward (Hinton, 2022)** — replaces the backward pass with a second forward pass. Each layer is trained to give high "goodness" (e.g., sum of squared activations) to *positive* (real) inputs and low goodness to *negative* (corrupted) inputs. Layers train greedily, bottom to top, with purely local objectives. Drawback: you must construct negative samples, which is task-specific and memory-hungry.
- **Single-forward (SF, from ForwardGNN)** — removes the need for negatives. Each layer gets a local contrastive loss against learned class representatives, computed in one forward pass. Layers are still trained one at a time, and once trained, their outputs are detached (no gradient flows between layers).

### 1.7 Metrics used across the reports

| Metric | Meaning | Better |
|---|---|---|
| Accuracy / Macro-F1 | Task performance | Higher |
| ROC-AUC | Link prediction quality | Higher |
| Communication (MB/GB) | Data exchanged between parties during training | Lower |
| Latency (s) | End-to-end wall-clock time | Lower |
| Peak memory (MB) | Maximum GPU/simulator memory during training | Lower |
| **R** (operational rank) | Rank-based composite across metrics and settings (see §6.3) | Lower |

---

## 2. The Survey: *Beyond-Backpropagation Training*

**Authors:** Ye, Ye, Huang, Tang, Liu (SUSTech, Montclair State, Tsinghua). Preprint, Jan 2026.

**Purpose:** the most comprehensive review to date of BP-free learning, proposing a unified taxonomy and practical guidelines. Previous surveys covered only single method families (e.g., zeroth-order, predictive coding) or single domains (e.g., LLMs).

### 2.1 The two-dimensional taxonomy

**Dimension 1 — How is the learning signal generated?**

```
BP-free signals
├── Gradient-based (retain some gradient information)
│   ├── Task signal (derived directly from the loss)
│   │   ├── Forward gradient — directional derivative via forward-mode AD
│   │   ├── Zeroth-order (ZO) optimization — estimate gradient from loss differences under random perturbations (e.g., MeZO)
│   │   └── Proxy gradient — replace the weight transpose with random/learned feedback
│   │       (Feedback Alignment, Direct FA, DNI synthetic gradients, NoProp)
│   └── Synthetic signal (auxiliary objectives, not the task loss)
│       ├── Predictive coding (PC) — each layer predicts the layer below; reconstruction error drives learning
│       ├── Target propagation (TP) — learned inverse mappings give each layer a local target
│       ├── Equilibrium propagation (EP) — energy-based; contrast free vs. nudged equilibrium states
│       └── Activation-statistics signals — forward-forward and its variants (goodness of positive vs. negative data)
└── Gradient-free (no gradient at all)
    ├── Hebbian rules — "neurons that fire together wire together" (contrastive HL, SoftHebb, PEPITA, burst-dependent plasticity)
    └── Evolutionary rules — population search (evolution strategies, PSO, simulated annealing)
```

**Dimension 2 — When are parameters updated?** (ordered from most to least locked)

| Regime | Description | Examples |
|---|---|---|
| Sequential backward | Error for layer *l* derived from layer *l+1* — still fully locked | FA, Z-IL |
| Parallel backward | A global signal (output error or target) is broadcast to all layers | DFA, DRTP, Kickback |
| Post-forward | No backward pass, but updates wait for a full forward evaluation | PEPITA, EP, PC, MeZO, evolution strategies |
| Sequential forward | Each layer updates immediately as its forward computation completes | SoftHebb, forward-forward, SigProp, forward-mode AD, DTP |
| Parallel forward | Layers fully decoupled and asynchronous | DNI, replay buffers, ADMM, NoProp |

### 2.2 Application domains

| Domain | Key BP-free contributions |
|---|---|
| **Reinforcement learning** | EP-BP hybrid actor-critic; RTRRL (online eligibility traces); MOHQA (Hebbian side-module for long time lags) |
| **Continual learning** | Task-specific orthogonal feedback matrices; Hebbian gating; sleep-like replay with EP (SRC); spiking forward-forward |
| **Federated learning** | TIFeD (integer-only DFA on microcontrollers); FedFwd (layer-wise FF, low peak memory); LAFD (DFA with local privacy) |
| **LLMs** | Black-box prompt tuning via evolution strategies (BBT, BBTv2, Clip-Tuning); zeroth-order fine-tuning with inference-level memory (MeZO, ZO-AdaMU, SubZero); MemPEPITA and FoRo for on-device adaptation |
| **Novel hardware** | Neuromorphic (EqSpike), photonic (DFA co-processors, ODFA), quantum (QEP) — all avoid BP's need for stored activations and precise arithmetic |

### 2.3 Comparative analysis (Tables 2–3 of the survey)

Methods are graded on **biological plausibility**, **locking removal**, and **memory efficiency**, and compared to BP on accuracy.

- **Bio-plausible and still accurate:** FF, PFF, SoftHebb reach accuracy comparable to gradient methods.
- **Unlocked and near-BP:** NoProp, PC, DNI enable parallel/layer-wise updates at little cost.
- **Memory-efficient and near-BP:** FF, PFF, PredSim, SID avoid activation storage entirely.
- **Biggest gaps vs. BP:** fixed-feedback methods (DFA up to −20 pts), ASAP, DRTP, and ZO methods on from-scratch training (DeepZero −5 pts).

### 2.4 Strategic guidelines (who should use what)

| Goal | Recommendation |
|---|---|
| Fine-tune a large pre-trained model on a memory-limited device | Zeroth-order optimization (MeZO / ZO-AdaMU) + PEFT such as LoRA; FwdLLM is the current best system |
| Maximize throughput on parallel/distributed hardware | Local greedy methods (DGL, DNI) or NoProp — layers train independently |
| High accuracy *and* biological plausibility | Learned-feedback methods (Kolen-Pollack, weight mirror, ASAP, PFA) — near-BP on ImageNet; EP for theoretical grounding |
| Neuromorphic, event-driven hardware | SoftHebb, predictive coding (NGC, ANGC), spiking EP (EqSpike) |

### 2.5 Open problems

1. **Theory** — few convergence or generalization guarantees; is BP-free learning PAC-learnable?
2. **Scalability** — most results are on small networks; auxiliary components (DNI estimators, NoProp diffusion steps) and ZO variance all grow with depth/size.
3. **Balancing plausibility, performance, memory** — no unified way to compare these; a multi-objective trade-off surface is proposed.
4. **Architecture mismatch** — BP-free rules are tested on ANNs designed for BP; may need spikes, dendrites, neuromodulation to shine.
5. **Hardware** — neuromorphic computing awaits an "AlexNet moment."

---

## 3. ForwardGNN: Forward Learning of GNNs

**Authors:** Park, Wang, Simoulin, Yang, Yang (Meta AI), Rossi (Adobe), Trivedi (Michigan), Ahmed (Intel). ICLR 2024. Code: github.com/facebookresearch/forwardgnn.

**Goal:** train GNNs for node classification and link prediction *without backpropagation*, avoiding BP's memory and locality constraints, and improving on the forward-forward algorithm's need for negative samples.

### 3.1 Three ideas, building on each other

**(a) Forward-forward adapted to graphs (Sec. 3.1)**

Two ways to build positive/negative inputs for FF on graph data:

- **FF-LA (label appending)** — concatenate a one-hot label to each node's features; correct label = positive, random wrong label = negative. Unlabeled nodes get a uniform label vector.
- **FF-VN (virtual nodes)** — add one virtual node per class; connect each labeled real node to the correct class's virtual node (positive graph) or a wrong one (negative graph). Virtual nodes aggregate from their class members, so they become *class representatives*.

Each layer is trained to give high goodness (‖ReLU activation‖²) to positives and low to negatives. Inference: compute goodness under every candidate label, sum across layers, pick the max.

**(b) Single-Forward — SF (Sec. 3.2) — the main contribution**

Use *only* the positively augmented graph. Because the virtual nodes are class representatives, each layer can be trained with a local **contrastive loss**: pull a node's embedding toward its own class's virtual-node embedding and push it away from the other classes'. This needs **one forward pass and no negative samples** — hence "no longer forward-forward."

Training loop (Alg. 2):
```
for each layer, bottom to top:
    repeat until converged:
        H_out = layer.forward(G_pos, H_in)
        loss = contrastive(H_out, class-representative embeddings, labels)
        update this layer only
    H_in = best_layer.forward(G_pos, H_in).detach()   # no gradient crosses layers
```
Inference: average each layer's softmax over classes, pick the max.

**(c) Top-down signals (Sec. 3.3)**

Greedy layer-wise training only lets a layer see bottom-up information. ForwardGNN adds feedback from *previous-time-step* outputs of upper layers, without any gradient flow:

- **Top-To-Input** — concatenate the upper layer's previous output to the current layer's input.
- **Top-To-Loss** — merge the upper layer's previous output into the current layer's output *only* for loss computation.

**(d) Link prediction (Sec. 3.4)** — already contrasts existing vs. non-existent edges, so it maps naturally onto local per-layer training with a BCE loss on edge scores (dot product + sigmoid). Final prediction = average of per-layer link probabilities.

### 3.2 Experimental setup

- **Models:** GCN, GraphSAGE, GAT (128 hidden units, Adam, early stopping)
- **Datasets:** PubMed, CiteSeer, CoraML (citation), Amazon (co-purchase), GitHub (followership)
- **Depths:** 1–4 layers; 5 random 64/16/20 splits
- **Baselines:** BP, FF-SymBa, CaFo (frozen layers + local predictors), PEPITA

### 3.3 Results

**Node classification (SF vs. BP)**
- SF matches BP in most settings and beats it in several.
- SF's GPU memory is **constant** with depth; BP's grows **up to 18×** at four layers.
- Top-To-Input signals improve SF on every dataset:

| Method | CiteSeer | CoraML | PubMed | Amazon | GitHub |
|---|---|---|---|---|---|
| SF | 91.14 | 84.95 | 81.90 | 89.69 | 83.16 |
| SF-TopDown | **93.58** | **86.01** | **82.21** | **92.09** | **83.68** |

**Link prediction (SF vs. BP)**
- SF (ForwardGNN-CE) mostly **outperforms** BP. BP's ROC-AUC *declines* with depth (e.g., CiteSeer GCN: 90.0 → 80.1 from 1 to 4 layers) while SF's holds or improves (89.3 → 93.6).

**Among forward methods**
- FF variants (FF-VN, FF-LA, FF-SymBa) can reach high accuracy but are unstable and use far more memory (multiple negatives).
- CaFo is cheapest but weak (GNN layers frozen). PEPITA fails on graphs.
- ForwardGNN-CE is the best link-prediction variant; SymBa is high-variance.

**Training time (Appendix F.3)**
- SF trains GCN/SAGE up to **~9× faster** than BP: neighborhood aggregation is computed once per layer and reused across epochs because the layer's input is fixed during its training. Not applicable to GAT (attention changes aggregation) or top-down variants.

**Other appendix findings**
- With very sparse labels (Planetoid splits), BP wins at 2 layers but the gap closes by 3–4 layers.
- Unidirectional real→virtual edges help SF on PubMed/GitHub but hurt FF.

### 3.4 Takeaways

1. BP-free, single-pass layer-wise training can match or beat BP on core graph tasks.
2. Memory is flat in depth; training is often faster.
3. Virtual nodes + contrastive loss remove FF's dependence on hand-built negatives.
4. Cheap top-down feedback recovers some of the benefit BP gets from global signals.

---

## 4. Work Report (Aug 18): GNN Benchmark

**Author:** Anower Zihad. Compares distributed GNN training paradigms under SF vs. BP.

### 4.1 Setup

- **5 paradigms:** Central, FedSage, SplitFed, PipeGCN-style, SplitGNN-style
- **2 training modes:** Single-Forward (SF, layer-wise, no cross-layer BP) and standard BP
- **10 configurations** × 5 runs, Cora_ML, 4-layer GraphSAGE, 5 clients/workers
- Metrics: accuracy, Macro-F1, training time, logical communication, peak memory, edges retained

### 4.2 Results

| Configuration | Accuracy | Notes |
|---|---|---|
| Central-SF | **87.68%** | Best overall |
| PipeGCN-style-SF | 87.25% | Preserves all edges; highest communication |
| SplitGNN-style-SF | 85.74% | Very little communication — best balance |
| FedSage / SplitFed (SF or BP) | ~68–72% | Partitioning keeps only ~20% of edges |

- **SF beat BP in all five paradigms** at four layers.
- BP was often faster in wall-clock time (trains the whole network at once), but SF is cheaper in split settings because it sends detached embeddings once instead of repeatedly exchanging activation gradients.

### 4.3 Trade-offs and conclusion

- Preserving cross-client edges → better accuracy.
- FedSage/SplitFed lose edges → lose accuracy.
- PipeGCN keeps accuracy but pays the most communication.
- SplitGNN-SF is the sweet spot.

**Conclusion:** BP-free SF is competitive and can outperform BP in deeper GNNs, but in distributed settings the real bottleneck is *retaining cross-client neighborhood information without excessive communication*, not just removing BP.

---

## 5. Progress Report Part 0 (Aug 24): Pipeline Validation

**Project:** "What Crosses the Cut?" — a BP-free split-learning benchmark.

**Purpose of Part 0:** a cheap "smoke test" on MNIST to verify the shared framework runs different methods on equal footing and measures all four metrics consistently, *before* spending real compute on CIFAR.

### 5.1 What was built

- Shared framework confirmed end-to-end on a standard method and dataset.
- **Communication split into cut vs. weights** (plus total) — the central measurement improvement.
- MNIST added as a test dataset, using the same model as the main experiments.
- End-to-end latency and peak memory added; all four metrics logged automatically to a results file.

### 5.2 Results (CPU-only validation runs)

| Measurement | SplitFedv2 (baseline) | CSE-FSL (BP-free) |
|---|---|---|
| Communication — cut (MB) | 1875.0 | 192.0 |
| Communication — weights (MB) | 20.9 | 85.1 |
| Communication — total (MB) | 1895.9 | 277.1 |
| Cut : weights share | ~99 : 1 | ~69 : 31 |
| Test accuracy | 98.8% | 98.4% |
| Latency (s) | 538.5 | 350.6 |
| Peak memory (MB) | 1060.8 | 1113.4 |

**Interpretation**
- CSE-FSL uses ~7× less total communication — it exchanges full gradients only periodically rather than every step.
- The cut/weights breakdown reveals *why*: the baseline is almost all cut traffic; CSE-FSL shifts a big share to weight traffic (it aggregates an extra model each round). A single combined number would hide this.
- Latency dropped in step; accuracy comparable (expected on MNIST).

### 5.3 What this proves and what's next

- The shared framework runs multiple methods with no special-casing — the fair basis the benchmark depends on.
- All four metrics measure uniformly and discriminate between methods.
- A pre-existing bug in a default model config was found and documented.
- **Next:** short CIFAR-10 runs, then full training with multiple seeds and heterogeneity settings.

---

## 6. Results Report (Aug 30): All Six Methods Running

**Setup:** six methods, GPU, MNIST, IID (α=∞) and non-IID (α=0.5), 4 rounds, seed 200. Still pipeline-verification runs, not final numbers.

### 6.1 Main table

| Method | IID acc % | Non-IID acc % | Comm (GB) | R |
|---|---|---|---|---|
| SplitFedv1 | 99.08 | 83.86 | 3.87 | 4.00 |
| SplitFedv2 | 98.88 | 89.74 | 3.87 | 4.00 |
| CSE-FSL | 98.85 | 94.83 | 1.22 | 2.75 |
| FSL-SAGE | 98.83 | 96.84 | 0.67 | **2.25** |
| HO-SFL | 52.38 | 57.14 | 0.16 | 3.00 |
| MU-SplitFed* | 11.28 | 11.83 | 3.57 | 5.00 |

\*Third-party (HO-SFL repo) implementation, not the original authors' code.

### 6.2 What the numbers show

- **Efficiency methods work as advertised.** FSL-SAGE matches full-BP accuracy (~98.8% IID) with ~5.8× less communication and ranks best overall. CSE-FSL is close behind.
- **Non-IID separates the methods.** Baselines drop sharply (SplitFedv1 → 83.9%); FSL-SAGE and CSE-FSL hold up best (96.8%, 94.8%) — the BP-free auxiliary-loss methods are more robust to heterogeneity *and* cheaper.
- **HO-SFL is a genuine trade-off point:** least communication (0.16 GB) but only ~52–57% in 4 rounds; the curve is still rising.
- **MU-SplitFed is flat at chance (~11% = 1/10)** across all rounds — accuracy and loss don't move. This is the signature of an implementation/config problem, not slow convergence. Flagged, not dropped.

### 6.3 How R (operational rank) is computed

R combines metrics on different scales (accuracy in %, communication in GB) fairly by ranking instead of averaging raw values.

1. Within each setting, rank the methods on each metric separately (accuracy: higher = rank 1; communication: lower = rank 1). Ties share the average rank.
2. Average the metric ranks → per-setting rank.
3. Average the per-setting ranks across IID and non-IID → R.
4. Lower R wins.

**Worked example — FSL-SAGE:**

| Setting | Acc rank | Comm rank | Per-setting avg |
|---|---|---|---|
| IID | 4 (98.83%) | 2 (0.67 GB) | (4+2)/2 = 3.00 |
| α=0.5 | 1 (96.84%) | 2 (0.67 GB) | (1+2)/2 = 1.50 |
| **R** | | | (3.00 + 1.50)/2 = **2.25** |

FSL-SAGE is only 4th on IID accuracy but consistently strong on *both* axes in *both* settings, which is why it wins. In the full benchmark, R extends to memory and latency. R is provisional while MU-SplitFed is unresolved — its worst-place ranks nudge every other method's R.

### 6.4 The Pareto view

Plotting accuracy vs. communication (up-and-left is better): FSL-SAGE and CSE-FSL dominate the efficient region; full-BP baselines pay far more communication for similar or lower accuracy; HO-SFL and MU-SplitFed sit in the low-cost/low-accuracy corner.

### 6.5 Next steps

- Run MU-SplitFed standalone for many rounds; if still flat, obtain authors' code or drop it.
- Confirm HO-SFL with more rounds.
- Scale to CIFAR-10 / CIFAR-100, then add STL-10 (higher-resolution natural images) and MedMNIST (medical domain shift).

---

## 7. How Everything Connects

```
Survey (Ye et al.)          ← the map of the whole BP-free field
        │
        ├── Forward-forward / activation-statistics signals
        │       └── ForwardGNN (Park et al.)   ← SF training for graphs
        │               └── Work Report Aug 18   ← SF vs. BP across distributed GNN paradigms
        │
        └── Federated / split learning applications
                └── "What Crosses the Cut?" benchmark
                        ├── Part 0 (Aug 24)  ← measurement pipeline validated (cut vs. weights)
                        └── Results (Aug 30) ← six methods, IID/non-IID, R ranking
```

**Consistent findings across all documents**

1. **BP-free local training matches BP accuracy** in the regimes tested — MNIST split learning, Cora_ML GNNs, and the five graphs in ForwardGNN.
2. **Communication drops 5–7×** when gradients no longer have to cross the cut (FSL-SAGE, CSE-FSL) or when embeddings are sent once, detached (SF in split GNNs).
3. **Robustness to non-IID data** is better for the auxiliary-loss BP-free methods than for full-BP baselines.
4. **Memory is flat in depth** with layer-wise forward training (ForwardGNN), versus up to 18× growth with BP.
5. **Measurement design matters** — splitting communication into cut vs. weights, and ranking rather than averaging raw metrics, are what make the comparisons interpretable.

---

## 8. Open Questions and Next Steps

**From the survey (field-level)**
- Theory: convergence and generalization guarantees for BP-free methods.
- Scalability to deep/large models.
- Whether biologically inspired rules need non-standard (spiking, neuromorphic) architectures to reach their potential.

**From the projects (practical)**
- Resolve the MU-SplitFed implementation issue.
- Confirm HO-SFL's slow convergence with longer runs.
- Move from MNIST to CIFAR-10/100, STL-10, MedMNIST with full training and multiple seeds.
- Extend R to include memory and latency.
- For distributed GNNs: find ways to preserve cross-client neighborhood information without paying PipeGCN-level communication.

---

## Quick Glossary

| Term | One-line definition |
|---|---|
| **BP** | Backpropagation — gradient computation via the chain rule, backward through the network |
| **BP-free** | Any training method that avoids the global backward pass |
| **FF** | Forward-forward — two forward passes (positive/negative) with a per-layer goodness objective |
| **SF** | Single-forward — one forward pass, per-layer contrastive loss against class representatives |
| **DFA / FA** | (Direct) feedback alignment — replace weight transposes with fixed random matrices |
| **ZO** | Zeroth-order optimization — estimate gradients from loss values under random perturbations |
| **PC / TP / EP** | Predictive coding / target propagation / equilibrium propagation — synthetic local signals |
| **FL / SL / SFL** | Federated / split / split-federated learning |
| **Cut** | The layer boundary where a model is split between client and server |
| **IID / non-IID** | Client data is / is not representative of the whole dataset; α controls skew |
| **R** | Operational rank — rank-based composite score, lower is better |
| **GNN / GCN / GraphSAGE / GAT** | Graph neural network families differing in how neighbors are aggregated |
| **Virtual node** | A synthetic per-class node added to a graph to act as a class representative |
| **Goodness** | FF's per-layer score, typically the squared L2 norm of activations |