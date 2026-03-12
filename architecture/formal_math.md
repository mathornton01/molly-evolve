# Formal Mathematical Framework

## 1. Definitions and Setup

### 1.1 Parameter Space

Let **θ** ∈ ℝ^d be the full parameter vector of an LLM with d parameters.

Partition θ into L **genes** (the unit of conversion). Each gene gᵢ is a contiguous
block of parameters — a layer, an attention head, or an MLP block:

    θ = (g₁, g₂, ..., g_L)    where gᵢ ∈ ℝ^(dᵢ)  and  Σ dᵢ = d

### 1.2 Dual Genome ("Homologous Chromosomes")

Maintain two complete parameter vectors:

    S = (s₁, s₂, ..., s_L)    — the **stable genome** (last known high-fitness state)
    A = (a₁, a₂, ..., a_L)    — the **active genome** (currently training, accumulating mutations)

At initialization: A = S = θ_pretrained

### 1.3 Fitness Function

Define fitness as expected negative loss over a held-out task distribution D:

    F(θ) = -𝔼_{x ~ D}[ ℒ(θ, x) ]

Higher fitness = better model. We estimate F by sampling batches from D.

---

## 2. The Mutation Process (Training)

Standard gradient descent introduces "mutations" to A at each step t:

    A^(t+1) = A^(t) - η ∇ℒ(A^(t), x_batch)

S remains unchanged during this phase. After T mutation steps, A has diverged
from S. The divergence at gene i is:

    δᵢ = aᵢ - sᵢ    (the "mutation vector" for gene i)

---

## 3. Bayesian Mutation Scoring

This is the core innovation. For each gene i, we want to infer:

    P(δᵢ is beneficial | observed fitness data)
    P(δᵢ is deleterious | observed fitness data)

### 3.1 Per-Gene Fitness Effect

Define the **mutation effect** for gene i as the change in fitness caused by
swapping gene i between genomes. Construct a chimeric model where gene i
comes from A but everything else comes from S:

    θ_chimera(i) = (s₁, ..., s_{i-1}, aᵢ, s_{i+1}, ..., s_L)

The mutation effect is:

    Δᵢ = F(θ_chimera(i)) - F(S)

If Δᵢ > 0: gene i's mutation is beneficial (A's version is better than S's)
If Δᵢ < 0: gene i's mutation is deleterious (A's version is worse)
If Δᵢ ≈ 0: neutral mutation

### 3.2 Prior

Place a zero-mean Gaussian prior on each mutation effect:

    Δᵢ ~ N(0, σ²_prior)

This encodes the assumption that most mutations are neutral, with some
probability of being beneficial or deleterious. σ²_prior is a hyperparameter
controlling how much mutation effect we expect a priori.

### 3.3 Likelihood

After evaluating the chimeric model on a batch of K examples from D, we get
a noisy estimate of Δᵢ:

    Δ̂ᵢ = F̂(θ_chimera(i)) - F̂(S)

Model the noise as Gaussian:

    Δ̂ᵢ | Δᵢ ~ N(Δᵢ, σ²_noise / K)

where σ²_noise is the variance of per-example fitness.

### 3.4 Posterior

By conjugacy (Gaussian-Gaussian), the posterior is:

    Δᵢ | Δ̂ᵢ ~ N(μ_post, σ²_post)

where:

    σ²_post = 1 / (1/σ²_prior + K/σ²_noise)
    μ_post  = σ²_post × (Δ̂ᵢ × K / σ²_noise)

The posterior mean μ_post is a **shrinkage estimate** — it pulls the observed
effect toward zero (the prior mean), with the degree of shrinkage controlled
by how much data we have (K) relative to how uncertain our prior is.

### 3.5 Decision Probabilities

    P(beneficial | data) = P(Δᵢ > 0 | Δ̂ᵢ) = 1 - Φ(-μ_post / σ_post)
    P(deleterious | data) = P(Δᵢ < 0 | Δ̂ᵢ) = Φ(-μ_post / σ_post)

where Φ is the standard normal CDF.

---

## 4. Gene Conversion Operator

For each gene i, at each conversion cycle:

### 4.1 Conversion Sampling

Sample whether a conversion event occurs:

    cᵢ ~ Bernoulli(β)

where β ∈ (0,1) is the **conversion rate** — a global hyperparameter analogous
to the biological gene conversion rate. If cᵢ = 0, gene i is left unchanged.

### 4.2 Conversion Direction

If cᵢ = 1 (conversion event occurs):

    If P(deleterious | data) > θ_repair:
        aᵢ ← sᵢ                          # REPAIR: overwrite bad mutation

    Else if P(beneficial | data) > θ_fix:
        sᵢ ← aᵢ                          # FIXATION: preserve good mutation

    Else:
        no action                          # insufficient evidence either way

θ_repair and θ_fix are **decision thresholds** (e.g., 0.8 and 0.9 respectively).
Setting θ_fix > θ_repair makes fixation harder than repair — matching the
biological asymmetry where beneficial mutations are rarer than neutral repairs.

### 4.3 Full Operator

The complete gene conversion operator G acts on the dual genome (S, A):

    G(S, A) = (S', A')

where for each gene i:

    (s'ᵢ, a'ᵢ) = { (sᵢ, sᵢ)   if repair     (S overwrites A)
                   { (aᵢ, aᵢ)   if fixation   (A overwrites S)
                   { (sᵢ, aᵢ)   otherwise     (no change)

---

## 5. Conversion Rate Adaptation

The conversion rate β should not be fixed. In biology, gene conversion rate
varies across the genome and likely responds to selective pressure.

### 5.1 Bayesian Update on β

Place a Beta prior on β:

    β ~ Beta(α₀, β₀)

After observing n conversion events in N genes at one cycle:

    β | data ~ Beta(α₀ + n, β₀ + N - n)

But we also want β to respond to fitness: if fitness is improving, reduce β
(fewer repairs needed). If fitness is declining, increase β (more repair needed).

### 5.2 Fitness-Adaptive Rate

    β^(t+1) = β^(t) × exp(-γ × ΔF)

where ΔF = F(A^(t)) - F(A^(t-1)) and γ is a sensitivity parameter.

When fitness drops (ΔF < 0): β increases → more conversion events → more repair
When fitness rises (ΔF > 0): β decreases → fewer interventions → let training run

---

## 6. Full Algorithm

```
INITIALIZE:
    S ← θ_pretrained
    A ← θ_pretrained
    β ← β₀ (initial conversion rate, e.g. 0.1)
    σ²_prior ← initial prior variance

FOR cycle = 1, 2, ...:

    # --- MUTATION PHASE ---
    FOR step = 1 to T:
        A ← A - η ∇ℒ(A, x_batch)          # standard training

    # --- FITNESS EVALUATION ---
    F_S ← evaluate(S, D_heldout)
    F_A ← evaluate(A, D_heldout)

    # --- BAYESIAN SCORING ---
    FOR each gene i = 1 to L:
        Construct θ_chimera(i)
        Evaluate Δ̂ᵢ = F̂(θ_chimera(i)) - F̂(S)
        Compute posterior P(Δᵢ | Δ̂ᵢ)
        Compute P(beneficial | data), P(deleterious | data)

    # --- GENE CONVERSION ---
    FOR each gene i = 1 to L:
        Sample cᵢ ~ Bernoulli(β)
        IF cᵢ = 1:
            Apply conversion rule (Section 4.2)

    # --- RATE ADAPTATION ---
    Update β based on fitness trajectory (Section 5.2)

    # --- OPTIONAL: CLONAL SELECTION ---
    Spawn K copies of A
    Apply independent gene conversion to each
    Evaluate fitness of each clone
    A ← fittest clone
```

---

## 7. Computational Cost Analysis

The **bottleneck** is Section 3 — Bayesian Scoring requires L chimeric evaluations
per conversion cycle. For a model with L genes:

    Cost per cycle = T × (cost of one training step)     # mutation phase
                   + (L + 2) × (cost of one eval pass)   # fitness + chimeras
                   + L × (cost of weight copy)            # conversion

For a 7B-parameter model with L = 64 (per-layer genes):
    - 64 chimeric evaluations per cycle — expensive but parallelizable
    - Each chimeric eval is a forward pass only (no backprop) — ~2x cheaper than training

### 7.1 Approximation: Grouped Scoring

Instead of evaluating each gene independently, group genes and evaluate
groups simultaneously. This reduces chimeric evaluations from L to L/k
where k is the group size. Trade-off: coarser resolution.

### 7.2 Approximation: Weight-Space Distance

Skip chimeric evaluation entirely. Instead, use weight-space divergence as
a proxy for fitness effect:

    δ̂ᵢ = ||aᵢ - sᵢ||₂ / ||sᵢ||₂    (relative divergence)

High divergence → more likely to be a large (possibly deleterious) mutation.
This is much cheaper but less principled — it doesn't distinguish beneficial
from deleterious divergence.

---

## 8. Connection to Existing Methods

| Method | Relationship to molly-evolve |
|---|---|
| Evolutionary strategies (ES) | ES perturbs all params randomly; we do targeted repair |
| Population-based training (PBT) | PBT evolves hyperparams; we evolve weight subspaces |
| Model soups / weight averaging | Soups average all weights; we selectively copy regions |
| Lottery ticket hypothesis | LTH prunes globally; we repair locally |
| Elastic weight consolidation (EWC) | EWC penalizes divergence from prior; we actively revert it |

The key distinction: **molly-evolve is the only approach that maintains two
genomes with probabilistic, bidirectional, local conversion.** All other
methods are either global (soups, ES), unidirectional (EWC), or operate on
hyperparameters rather than weights (PBT).

---

## 9. Future Direction: Self-Repairing Weight Encoding

(Noted from initial project discussion — to be formalized later)

Rather than applying gene conversion as an external operator, encode the
weights themselves in a redundant representation (analogous to DNA's
double-helix / two-copy system) such that the conversion operator is
intrinsic to the weight representation.

Candidate approaches:
- Error-correcting codes over quantized weights
- Dual-weight layers with built-in parity constraints
- Autoencoder-based weight compression where the decoder acts as a repair function

This would make gene conversion a **property of the architecture** rather
than an external training procedure — a much stronger analogy to biology.
