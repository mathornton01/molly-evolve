# Roadmap

## Completed
- [x] Biological research (Amazon Molly gene conversion mechanism)
- [x] Formal mathematical framework (Bayesian conjugate model)
- [x] Prototype on small MLP (+8.8% worst-task improvement)
- [x] GPT-2 scale experiment (+24.2% geometric mean improvement)
- [x] Transformer gene mapping (75 component-level genes)

## In Progress
- [ ] Iterative multi-domain evolution loop (Direction 1)

## Planned Directions

### Direction 2: Benchmark against existing continual learning methods
Compare gene conversion against:
- EWC (Elastic Weight Consolidation) — Fisher-information-weighted regularization
- LoRA — low-rank adapters that freeze base weights
- Model merging / TIES-Merging — weight averaging across fine-tuned models
- Progressive Neural Networks — lateral connections to frozen columns
Goal: show gene conversion is competitive or superior on multi-domain retention

### Direction 3: Self-repairing weight encoding
Make gene conversion intrinsic to the architecture rather than an external process:
- Error-correcting codes over quantized weights (Hamming, Reed-Solomon)
- Dual-weight layers with built-in parity constraints
- Autoencoder-based weight compression where decoder acts as repair function
Goal: weights that carry their own reconstruction information

### Direction 4: Scale to larger models
- GPT-2-medium (355M) — verify results hold at 3x scale
- 1B+ parameter model — test on modern architectures (Llama, Mistral)
- Head-level gene granularity (207 genes) for finer control
- CUTLASS integration for custom weight-space GEMM operations
Goal: demonstrate practical viability at production scale
