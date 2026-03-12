# System Architecture — Conceptual Design

## Overview

molly-evolve is a framework for LLM self-evolution using a Bayesian analog of the Amazon Molly's gene conversion mechanism. The system maintains a probabilistic model over LLM weight space and applies selective "conversion" events to repair or improve the model without full retraining.

---

## Core Components

### 1. Dual-Weight Representation ("Homologous Chromosomes")
- Maintain two versions of the model: a **stable genome** (S) and an **active genome** (A)
- S = last known high-fitness checkpoint
- A = current working model (accumulates mutations via fine-tuning / gradient descent)
- Conceptually: S and A are the two ancestral haploid genomes of the molly

### 2. Bayesian Mutation Tracker
- Track weight-level "mutations" as a probability distribution over parameter space
- At each step, maintain a posterior P(w | D) where D is performance data
- Identify regions of A that have diverged significantly from S (= deleterious mutations)
- Identify regions where A outperforms S (= beneficial mutations)
- Inspired by: Bayesian model comparison, variational inference

### 3. Gene Conversion Operator
- For each candidate region (layer, attention head, MLP block):
  - If P(deleterious | region) > threshold_bad → copy corresponding region from S → A (repair)
  - If P(beneficial | region) > threshold_good → copy region from A → S (fixation)
- Conversion is probabilistic — sampled, not deterministic
- Rate parameter β controls conversion frequency (Bayesian hyperparameter)

### 4. Fitness Evaluation ("Natural Selection")
- Evaluate both genomes on a held-out task distribution
- Fitness = f(loss, task diversity, out-of-distribution generalization)
- Bayesian posterior update after each evaluation cycle

### 5. Clonal Reproduction ("Checkpointing")
- At intervals, create exact copies of A
- Apply gene conversion operator to copies independently
- Run parallel evaluation — select fittest clone to continue
- Discard unfit clones (analogous to clonal selection)

---

## Algorithmic Sketch

```
Initialize:
  S = pretrained LLM weights
  A = copy of S
  β = initial conversion rate
  P(w) = uninformative prior over weight space

Loop:
  1. Fine-tune A on task batch (introduce "mutations")
  2. Evaluate fitness(A) and fitness(S) on held-out set
  3. Update posterior P(w | fitness data)
  4. For each layer l in model:
       sample conversion_event ~ Bernoulli(β)
       if conversion_event:
         if P(deleterious | layer l of A) > θ_bad:
           A[l] ← S[l]  # repair
         elif P(beneficial | layer l of A) > θ_good:
           S[l] ← A[l]  # fix beneficial mutation
  5. If fitness(A) > fitness(S): update S ← A
  6. Adjust β via Bayesian hyperparameter update
  7. Optional: spawn clonal variants, select fittest
```

---

## HPC / GPU Considerations

- Step 1 (fine-tuning) is standard — handled by PyTorch + DeepSpeed/FSDP
- Step 4 (gene conversion operator) requires:
  - Fast layer-wise weight copying between GPU buffers
  - CUTLASS could accelerate custom GEMM ops if we need to transform weight subspaces (not just copy)
  - At minimum: CUDA memory ops, manageable without CUTLASS initially
- Scale target: 7B–13B parameter models on a 4–8 GPU node

---

## Open Design Questions

1. **Granularity of conversion:** full layers vs. attention heads vs. individual weight matrices?
2. **Prior choice:** what prior distribution over LLM weights is appropriate?
3. **Fitness function design:** single task or diverse task distribution?
4. **Conversion rate schedule:** should β decay over time (stabilization) or be adaptive?
5. **Mosaic detection:** can we borrow MosaicForecast's read-based phasing concept to detect weight-level "mosaic" patterns between S and A?

---

## Next Steps

- [ ] Formalize the Bayesian update rule mathematically
- [ ] Prototype on a small model (GPT-2 scale) before scaling
- [ ] Implement basic fitness evaluator
- [ ] Implement gene conversion operator as a PyTorch module
- [ ] Benchmark weight-copy speed at different granularities
