# molly-evolve

A research project building a **Bayesian genetic algorithm for LLM self-evolution**, inspired by the genome repair mechanism of the Amazon Molly (*Poecilia formosa*).

## Quick start — as easy as LoRA

Fine-tune any HuggingFace causal LM with MOLLI's gene-conversion protection
against catastrophic forgetting in **five lines**:

```python
from molly_evolution import MolliTrainer

trainer = MolliTrainer.from_pretrained("gpt2")
trainer.fit(train_texts=open("corpus.txt").read().split("\n"))
trainer.save_pretrained("./my-molli-model")
```

Reload later for inference:

```python
from molly_evolution import MolliTrainer
trainer = MolliTrainer.from_pretrained("./my-molli-model")
print(trainer.generate("Once upon a time"))
```

### Side-by-side with LoRA (PEFT)

```python
# ── LoRA ──                                    # ── MOLLI ──
from peft import LoraConfig, get_peft_model     from molly_evolution import MolliTrainer
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("gpt2")
model = get_peft_model(base,                    trainer = MolliTrainer.from_pretrained("gpt2")
    LoraConfig(r=8, target_modules=[...]))
# ... your training loop ...                    trainer.fit(train_texts=my_texts)
model.save_pretrained("./adapter")              trainer.save_pretrained("./my-molli-model")
```

Same number of lines, same ergonomics — **plus** MOLLI protects previously
learned capabilities via Bayesian gene conversion instead of LoRA's dilution
as adapters accumulate.

### Or from the shell

```bash
molly train --model gpt2 --train-file corpus.txt -o ./my-molli-model
molly generate --model ./my-molli-model --prompt "Once upon a time"
```

See `molly train --help` for the full flag list (epochs, lr, granularity,
protect-file for multi-domain, etc.).

## Biological Inspiration

The Amazon Molly is an all-female fish species that has reproduced entirely asexually for ~100,000 years without the genetic decay theory predicts. A 2026 Nature paper ("Gene conversion empowers natural selection in a clonal fish species") revealed the mechanism: **gene conversion** — a copy-paste DNA repair process where segments of one chromosome are copied to the homologous chromosome, erasing bad mutations and occasionally fixing beneficial ones. This allows natural selection to act on a clonal organism as if recombination were occurring.

**Paper:** https://www.nature.com/articles/s41586-026-10180-9
**Key researchers:** Dr. Manfred Schartl (Univ. Würzburg / Texas State), Dr. Yuan Lu, David Bierbach, Wesley C. Warren, Ronald B. Walter

## Core Idea

Map the biological gene conversion process onto LLM weight space:

| Biology | AI Analogue |
|---|---|
| Gene conversion (copy segment between homologous chromosomes) | Transplant weight subspaces between model checkpoints |
| Purifying selection (remove harmful mutations) | Bayesian posterior update — prune low-performing parameters |
| Clonal reproduction (exact copy + repair) | Model checkpoint + targeted fine-tuning |
| Homologous chromosomes (dual genome) | Dual-weight ensemble / paired model versions |
| Natural selection via fitness | Loss landscape / reward signal as selection pressure |
| Mosaic genetic variants | Low-frequency weight perturbations tracked probabilistically |

## Research Questions

1. Can gene conversion mechanics be formalized as a Bayesian update rule over LLM weight distributions?
2. What is the right "fitness function" for an LLM — loss on a held-out task distribution?
3. How do we represent "homologous chromosomes" in weight space — layer pairs? Attention head pairs?
4. Can we use mosaic variant detection methods (MosaicForecast) as inspiration for identifying which weight regions to "convert"?
5. What is the minimum HPC/GPU requirement to run this at meaningful scale?

## Project Structure

```
molly-evolve/
├── README.md               # This file
├── notes/                  # Research notes and literature summaries
│   ├── biology/            # Amazon Molly and gene conversion literature
│   ├── bayesian/           # Bayesian modeling approaches
│   └── llm/                # LLM architecture references
├── architecture/           # System design documents
├── src/                    # Source code (future)
│   ├── gene_conversion/    # Core algorithm
│   ├── bayesian/           # Bayesian modeling layer
│   └── hpc/                # HPC/CUDA/CUTLASS integration
└── experiments/            # Experiment configs and results
```

## Technology Stack (Planned)

- **Bayesian modeling:** NumPyro or PyMC
- **LLM framework:** HuggingFace Transformers + PyTorch
- **GPU kernels:** NVIDIA CUTLASS (custom GEMM for weight-space operations)
- **HPC:** SLURM-compatible job scripts
- **Mosaic variant detection inspiration:** MosaicForecast methodology

## Status

- [x] Repository initialized
- [x] Biological mechanism researched
- [x] Formal mathematical model drafted
- [x] Proof-of-concept implementation
- [x] LoRA-parity high-level API (`MolliTrainer`, `molly train` CLI, save/load)
- [ ] HPC environment setup
