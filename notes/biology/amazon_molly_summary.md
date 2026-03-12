# Amazon Molly — Research Notes

## Species
- *Poecilia formosa* — all-female, obligate asexual (gynogenetic) fish
- Arose from a single hybridization event between *Poecilia mexicana* and *Poecilia latipinna* ~100,000 years ago
- Uses sperm from males of related species only to trigger egg development (no paternal DNA incorporated)

## The Problem It Solves (Muller's Ratchet)
Asexual organisms should accumulate deleterious mutations over time because:
- No recombination = no way to separate bad mutations from good ones
- Each generation the mutation load only increases ("ratchets" upward)
- Theory predicts eventual extinction — yet the Amazon Molly has thrived for 100,000 years

## The Mechanism: Gene Conversion
**Paper:** "Gene conversion empowers natural selection in a clonal fish species"
*Nature*, March 2026
https://www.nature.com/articles/s41586-026-10180-9

**Key researchers:**
- Dr. Manfred Schartl — Univ. Würzburg & Texas State (Xiphophorus Genetic Stock Center)
- Dr. Yuan Lu
- David Bierbach — Leibniz Institute / Humboldt Universität zu Berlin
- Wesley C. Warren, Ronald B. Walter

### How Gene Conversion Works
1. The Amazon Molly carries two diverged haploid genomes (from its two ancestral species)
2. When a deleterious mutation occurs on one chromosome, the repair machinery copies the corresponding "healthy" segment from the homologous chromosome
3. This effectively **overwrites** the bad mutation with the good sequence
4. The reverse can also happen — a beneficial mutation can be copied and "fixed" across both chromosomes
5. This mimics the role of sexual recombination without sex

### Key Findings
- Amazon Molly accumulates mutations *faster* than sexual ancestors (as expected)
- But these mutations do NOT lead to functional decay
- Gene conversion rate is high enough to counteract Muller's Ratchet
- First empirical proof this mechanism works in an asexual vertebrate
- Gene conversion facilitates both **purifying selection** (removing bad mutations) and **adaptive selection** (fixing good mutations)

## Related Earlier Work
- 2021 — "Fixation of allelic gene expression landscapes and expression bias pattern shape the transcriptome of the clonal Amazon molly" — *Genome Research*
  - Lu et al., DOI: https://genome.cshlp.org/content/early/2021/02/05/gr.268870.120.abstract
  - Found that either one or the other ancestral genome dominates expression in different genomic regions
  - Suggests the two ancestral genomes function as semi-independent units — important for our "homologous chromosomes = dual weight ensemble" analogy

- 2018 — "Clonal polymorphism and high heterozygosity in the celibate genome of the Amazon molly"
  - *Nature Ecology & Evolution*
  - https://www.nature.com/articles/s41559-018-0473-y
  - Showed unexpectedly high genetic diversity across clone lines — consistent with active gene conversion

## Relevance to MosaicSeq / Mosaic Variant Detection
The Amazon Molly genome exhibits **mosaic patterns** — different genomic regions using different ancestral alleles. This is analogous to mosaic mutations in somatic cells (low-frequency variants present in some cells but not others).

Tools like **MosaicForecast** (detecting low-frequency mosaic variants without matched controls) could inspire our method for detecting which "weight regions" in an LLM are candidates for gene conversion.

**MosaicForecast:** https://pmc.ncbi.nlm.nih.gov/articles/PMC7065972/

## Open Questions for the Project
- What is the exact molecular rate of gene conversion in P. formosa? (needed to calibrate Bayesian priors)
- Is conversion biased toward one ancestral genome over the other?
- Are there hotspots — genomic regions with higher conversion frequency? (analogous to which LLM layers to target)
- Does gene conversion occur during meiosis-like events, or continuously?
