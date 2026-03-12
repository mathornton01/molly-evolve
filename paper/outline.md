# Molly Evolution: Biologically-Inspired Gene Conversion for Continual Learning in Large Language Models

## Paper Outline

---

### Abstract
- Problem: catastrophic forgetting in sequential fine-tuning of LLMs
- Inspiration: Amazon Molly fish gene conversion (asexual vertebrate that maintains genetic diversity through gene conversion between homologous chromosomes)
- Method: dual-strand genome encoding of neural network weights with Bayesian gene scoring for selective repair
- Key results: outperforms LoRA and QLoRA on multi-domain continual learning (GM perplexity), competitive speed (1.22x LoRA)
- Contribution: a new paradigm for continual learning that treats neural network weights as a genome

---

### 1. Introduction

**1.1 The catastrophic forgetting problem**
- Sequential fine-tuning degrades prior capabilities
- Growing need for models that continually acquire domains
- Current solutions: regularization (EWC), replay, parameter-efficient methods (LoRA)

**1.2 Biological inspiration**
- Amazon Molly (*Poecilia formosa*) — all-female, asexual vertebrate
- Maintains genetic health through gene conversion: homologous recombination between chromosome copies repairs deleterious mutations
- Key insight: the fish doesn't need sexual recombination — it uses its own backup copy to detect and repair damage
- Analogy to neural networks: fine-tuning introduces "mutations" (weight changes); some are beneficial (new knowledge), some are deleterious (forgetting)

**1.3 Our contribution**
- Dual-strand encoding: quantized primary/complement weight copies
- Biologically meaningful gene boundaries (attention heads, MLP components)
- Bayesian gene scoring via gradient approximation (O(N_objectives) not O(N_genes × N_objectives))
- Gene conversion decisions: repair deleterious genes, fix beneficial ones
- Open-source package with C++/CUDA acceleration

---

### 2. Related Work

**2.1 Continual learning**
- Elastic Weight Consolidation (EWC) — Kirkpatrick et al. 2017
- Progressive Neural Networks — Rusu et al. 2016
- Experience Replay methods
- PackNet, SupSup, etc.

**2.2 Parameter-efficient fine-tuning**
- LoRA — Hu et al. 2021
- QLoRA — Dettmers et al. 2023
- Adapter methods — Houlsby et al. 2019
- The O(1/N) dilution problem with sequential LoRA merging

**2.3 Biological metaphors in ML**
- Neuroevolution (NEAT, etc.)
- Genetic algorithms for hyperparameter optimization
- DNA-inspired encoding for neural networks
- How our approach differs: we model a specific biological mechanism (gene conversion) at the weight level

---

### 3. Method

**3.1 Dual-strand genome encoding**
- Symmetric quantization to int16 (primary and complement strands)
- Scale factor per gene segment
- Memory overhead analysis: 4 extra bytes per parameter (int16 × 2)

**3.2 Gene boundary definition**
- Component-level: attention QKV, output projection, MLP up/down, LayerNorm (75 genes for GPT-2)
- Head-level: per-attention-head QKV slices (207 genes for GPT-2)
- The SlicedTransformerGene abstraction
- Why head-level works better: finer granularity → more precise repair

**3.3 Evolution cycle**
- Snapshot: mark current weights as healthy reference
- Fine-tune: standard training on new domain
- Score: evaluate each gene's impact on all objectives
- Convert: repair damaged genes, fix beneficial ones

**3.4 Bayesian gene scoring**
- Gradient approximation: δ_loss_g ≈ ∇L · (w_complement - w_primary)
- Replaces O(N_genes × N_objectives) chimeric forward passes with O(N_objectives) forward+backward passes
- Per-gene posterior probabilities:
  - P(deleterious | previous objectives): probability that repairing this gene from complement would reduce loss on prior domains
  - P(beneficial | current objective): probability that the gene's current (trained) state helps the new domain
- Multi-objective scoring: max P(deleterious) across ALL previous domains

**3.5 Gene conversion decisions**
- Purifying selection: if P(del) - α·P(ben) > threshold → repair from complement
- Adaptive selection: if P(del) < 0.3 → fix (copy primary to complement, accept adaptation)
- Threshold and alpha as hyperparameters

---

### 4. Implementation

**4.1 Software architecture**
- Python package: molly-evolution
- C++/CUDA extension: batched gene scoring kernel, fused repair
- Streaming mode: O(1) GPU memory for deltas (vs O(N) precomputed)
- Multi-GPU support: objective parallelism across GPUs
- FSDP integration for large models

**4.2 Optimization details**
- AMP (mixed precision) for forward/backward passes
- Vectorized dot products on GPU tensors
- Warp-level reductions in CUDA scoring kernel
- Comparison: 343s → 9.2s (37x speedup from initial to optimized)

---

### 5. Experiments

**5.1 Experimental setup**
- Base model: GPT-2 (124M), GPT-2 XL (1.5B), [LLaMA 7B if RunPod results ready]
- Domains: General → Code → Legal → Medical → Science → Finance
- Metrics: per-domain perplexity, geometric mean perplexity
- Baselines: LoRA (rank 8), QLoRA (4-bit + rank 8), EWC, L2 regularization, Weight Averaging

**5.2 Multi-domain continual learning**
- 6-domain sequential fine-tuning
- Perplexity trajectory across domains for each method
- Gene conversion vs LoRA scaling behavior (O(1/N) dilution)

**5.3 Granularity ablation**
- Component-level (75 genes) vs Head-level (207 genes)
- Impact on repair precision and final perplexity

**5.4 Speed benchmarks**
- Wall-clock time per evolution cycle
- Breakdown: snapshot, train, score, repair
- Comparison against LoRA training time
- Scaling to larger models (projections)

**5.5 Scoring method comparison**
- Chimeric evaluation (original, slow) vs Gradient approximation (fast)
- Accuracy of gradient approximation
- Speedup: 45x scoring, negligible quality loss

---

### 6. Results

**6.1 Quality comparison**
[TABLE: Method × Domain perplexity matrix + GM]
- Gene conversion: GM = 8.26 (best)
- LoRA: GM = ~13 (O(1/N) dilution visible)
- QLoRA: GM = ~14
- EWC: GM = ~12
- Weight averaging: GM = ~9 (strong baseline)

**6.2 Scaling behavior**
[FIGURE: GM perplexity vs number of domains for each method]
- Gene conversion: stable or improving
- LoRA: degrading linearly
- Key insight: gene conversion becomes MORE advantageous as domains increase

**6.3 Speed comparison**
[TABLE: Method × Time breakdown]
- Gene conversion: 9.2s total (1.22x LoRA)
- LoRA: 7.6s total
- Gene conversion overhead is in scoring (amortized across objectives)

**6.4 Memory analysis**
[TABLE: Method × GPU memory]
- Streaming mode memory vs precomputed
- Scaling projections for 7B, 13B, 70B models

---

### 7. Discussion

**7.1 When does gene conversion help?**
- Multi-domain scenarios (>2 domains)
- When prior domain preservation is critical
- When catastrophic forgetting is the primary concern vs. speed

**7.2 The biological analogy**
- Why the Amazon Molly metaphor is apt:
  - No recombination partner needed (self-repair)
  - Complement strand = "genomic immune system"
  - Gene-level, not weight-level, is the right granularity
- Limitations of the analogy

**7.3 Limitations**
- 16-bit quantization introduces small reconstruction error
- Gradient approximation is first-order (could miss nonlinear interactions)
- Head-level granularity is GPT-2 specific (need architecture-agnostic mapping)
- Not yet tested at >7B scale

**7.4 Future work**
- Self-repairing codes (error-correcting genome encoding)
- Architecture-agnostic gene boundary detection
- Online/streaming gene conversion (during training, not after)
- Biological analogy extensions: epigenetics, gene regulation

---

### 8. Conclusion
- Gene conversion provides a new paradigm for continual learning
- Biologically inspired but computationally practical
- Outperforms LoRA/QLoRA with near-equal speed
- Open-source implementation: pip install molly-evolution

---

### Appendices

**A. Hyperparameter sensitivity**
- Threshold (0.3–0.7), alpha (0.1–0.5)
- Quantization bits (8 vs 16)

**B. Full benchmark tables**
- All domain × method perplexity numbers
- Per-domain timing breakdowns

**C. Architecture details**
- GPT-2 gene map (full 207-gene listing)
- C++/CUDA kernel implementation details

---

## Section Writing Plan

Write sections in this order:
1. Section 3 (Method) — core technical contribution
2. Section 5 (Experiments) — setup and methodology
3. Section 6 (Results) — fill in with actual numbers from benchmarks
4. Section 1 (Introduction) — motivate the work
5. Section 2 (Related Work) — position against prior art
6. Section 4 (Implementation) — practical details
7. Section 7 (Discussion) — analysis and future directions
8. Abstract and Conclusion — summarize

Each section should be 1-2 pages (conference format). Target venue: NeurIPS, ICML, or AAAI.
Total: ~8-10 pages + appendices.
