# Transformer Gene Mapping for GPT-2

## The Problem

At GPT-2-small scale (124M parameters), using gene_size=16 gives ~7.7M genes.
Chimeric evaluation (one forward pass per gene) is intractable at that scale.

We need biologically meaningful, coarser gene boundaries that:
1. Map to functional units of the transformer
2. Keep total gene count under ~200 for tractable chimeric evaluation
3. Allow gene conversion to make structurally meaningful repairs

## GPT-2-small Architecture

    Embedding:        wte (50257 x 768)  +  wpe (1024 x 768)
    Transformer:      12 layers, each containing:
                        - LayerNorm 1         (768)
                        - Attention:
                            - c_attn (768 x 2304) = Q,K,V projection
                            - c_proj (768 x 768)  = output projection
                        - LayerNorm 2         (768)
                        - MLP:
                            - c_fc   (768 x 3072) = up-projection
                            - c_proj (3072 x 768) = down-projection
    Final LayerNorm:  ln_f (768)
    LM Head:          (tied to wte)

## Gene Mapping Strategy

### Option A: Layer-level genes (coarsest)

Each transformer layer = 1 gene.  Plus embeddings + final LN.

    Gene count: 12 (layers) + 2 (embeddings) + 1 (final LN) = 15 genes
    Pros: Very fast chimeric evaluation (15 forward passes)
    Cons: Too coarse -- can't distinguish attention vs MLP damage within a layer

### Option B: Component-level genes (recommended)

Each functional component within each layer = 1 gene.

    Per layer (x12):
      - attn_qkv:    c_attn weight + bias              (1 gene)
      - attn_proj:   c_proj weight + bias              (1 gene)
      - attn_ln:     ln_1 weight + bias                (1 gene)
      - mlp_up:      c_fc weight + bias                (1 gene)
      - mlp_down:    c_proj weight + bias              (1 gene)
      - mlp_ln:      ln_2 weight + bias                (1 gene)
      = 6 genes per layer

    Plus:
      - wte (token embeddings)                         (1 gene)
      - wpe (position embeddings)                      (1 gene)
      - ln_f (final layer norm)                        (1 gene)

    Gene count: 12 * 6 + 3 = 75 genes
    Pros: Good balance of granularity and tractability
    Cons: QKV are bundled (can't repair Q independently of K,V)

### Option C: Head-level genes (finest practical)

Split attention by head (12 heads in GPT-2-small).

    Per layer (x12):
      - attn_head_0 .. attn_head_11:  Q,K,V slices     (12 genes)
      - attn_proj:    output projection                  (1 gene)
      - attn_ln:      layer norm                         (1 gene)
      - mlp_up:       up-projection                      (1 gene)
      - mlp_down:     down-projection                    (1 gene)
      - mlp_ln:       layer norm                         (1 gene)
      = 17 genes per layer

    Plus:
      - wte, wpe, ln_f                                  (3 genes)

    Gene count: 12 * 17 + 3 = 207 genes
    Pros: Most biologically faithful (each head = independent functional unit)
    Cons: 207 chimeric evaluations per cycle (still tractable on RTX 4090)

## Recommendation

**Start with Option B (75 genes)** for the initial GPT-2 experiment.
This keeps chimeric evaluation fast (~75 forward passes = seconds on 4090)
while being granular enough to show meaningful selective repair.

If results look promising, move to Option C to test whether head-level
granularity improves the quality of gene conversion decisions.

## Biological Analogy

    Transformer layer     ~  Chromosome
    Component (attn/MLP)  ~  Gene
    Attention head        ~  Exon (functional subunit within a gene)
    Individual weight     ~  Nucleotide / codon

Gene conversion in biology typically operates on gene-sized stretches
(hundreds to thousands of base pairs). Option B maps closest to this:
gene conversion operates on functional components, not individual weights
and not entire chromosomes.

## Task Design for GPT-2

Unlike our prototype (synthetic classification), GPT-2 operates on text.

    Task A: Standard language modeling (WikiText-2 or similar)
    Task B: Fine-tune on a specific domain (e.g., code, medical text, dialogue)

    Forgetting: After Task B fine-tuning, general language modeling degrades
    Gene conversion: Repair general capability while keeping domain expertise

    Evaluation:
      - Task A fitness: perplexity on WikiText-2 held-out set
      - Task B fitness: perplexity on domain held-out set
      - Combined: harmonic mean of inverse perplexities

## Compute Budget (RTX 4090 Laptop, 16GB)

    Model size:             ~500 MB (fp32) or ~250 MB (fp16)
    Dual genome overhead:   ~500 MB (quantized int16 copies)
    Forward pass:           ~2 GB peak (batch_size=8, seq_len=512)
    Training (fine-tune):   ~6-8 GB peak (with gradient checkpointing)
    Total peak:             ~10 GB -- fits in 16 GB VRAM

    Chimeric eval (75 genes): ~75 forward passes x ~0.1s = ~8 seconds
    Full experiment:          ~30 min (train A, train B, score, convert)
