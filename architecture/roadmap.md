# Roadmap

## Completed
- [x] Biological research (Amazon Molly gene conversion mechanism)
- [x] Formal mathematical framework (Bayesian conjugate model)
- [x] Prototype on small MLP (+8.8% worst-task improvement)
- [x] GPT-2 scale experiment (+24.2% geometric mean improvement)
- [x] Transformer gene mapping (75 component-level genes)
- [x] Iterative multi-domain evolution loop (3 domains, beats pretrained by 17%)
- [x] Benchmark against continual learning methods (Direction 2)
  - Beats LoRA by 40% (GM 8.37 vs 13.85)
  - Beats QLoRA by 41% (GM 8.37 vs 14.14)
  - Beats EWC, L2 Reg, Weight Averaging
  - Only 1.9x slower than LoRA (15s vs 8s)
- [x] Head-level gene granularity (207 genes via SlicedTransformerGene)
  - 29.7% improvement over component-level (75 genes)
- [x] Multi-objective Bayesian scoring (protects all prior domains)
- [x] Gradient-based fast scoring (45x speedup over chimeric evaluation)

## Planned Directions

### Direction 3: Self-repairing weight encoding
Make gene conversion intrinsic to the architecture rather than an external process:
- Error-correcting codes over quantized weights (Hamming, Reed-Solomon)
- Dual-weight layers with built-in parity constraints
- Autoencoder-based weight compression where decoder acts as repair function
Goal: weights that carry their own reconstruction information

### Direction 4: Scale to larger models
- GPT-2-medium (355M) — verify results hold at 3x scale
- 1B+ parameter model — test on modern architectures (Llama, Mistral)
- CUTLASS integration for custom weight-space GEMM operations
Goal: demonstrate practical viability at production scale
