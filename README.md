# Causal Structured Stochasticity

**Empirical studies of how temporal structure in external stochasticity influences iterative learning and autoregressive language-model dynamics.**

This repository contains reproducible computational records for two related projects under the broader **Structured Stochasticity** research program.

## Project I — Stochastic Decoding and Causal Activation Patching

The Phase II experiment asks whether temporal organization in an externally supplied stochastic source can alter an autoregressive language model even when the source has the same one-step marginal distribution.

The experiment compares matched IID and autoregressive (AR) stochastic sources, holding model, prompt, seed, marginal source distribution, and sampling procedure fixed while changing only temporal dependence in the external random process.

### Causal intervention

For paired trajectories generated with common Gaussian innovations, residual-stream activations from a donor trajectory are transplanted into a recipient trajectory at GPT-2 small residual layers 8, 9, and 10. The primary endpoint is **recovery**, measuring how far the recipient next-token distribution moves toward the donor distribution after activation patching.

### Final experimental design

- GPT-2 small
- 20 prompts × 10 seeds
- AR coefficients: 0.25, 0.50, 0.75, 0.90
- Layers: 8, 9, 10
- Positions: 5, 10, 15, 20, 25, 30, 35, 39
- Both AR → IID and IID → AR interventions
- 38,400 raw patch measurements / 4,800 step-aggregated pair-level observations

### Main result

Mean recovery is approximately 0.932–0.935 at layer 8, 0.967–0.969 at layer 9, and 0.986–0.987 at layer 10, with closely matched effects in both intervention directions.

These results support a mediated causal picture:

`external stochastic structure → sampled token trajectory → residual representation → next-token distribution`

The experiment does **not** establish that GPT-2 contains a dedicated representation of the raw stochastic source or of the AR coefficient itself.

## Project II — Structured Stochasticity in Iterative Machine Learning

Proposal II asks whether the statistical structure of a stochastic data-order process affects optimization dynamics after relevant marginal sampling properties are controlled.

The core experiment uses a matched **structured-versus-echo-null** design: the structured condition and null contain the exact same underlying mini-batch multiset, while the null randomly permutes batch order. Dataset, batch contents, number of updates, optimizer, learning rate, initialization, and training horizon are therefore held fixed within each pair; temporal organization is the manipulated property.

The experimental program progressed from synthetic logistic regression through a class-balanced latent-feature design, a nonlinear MLP, a handwritten-digits benchmark, and an actual MNIST confirmation.

### Frozen primary result

The primary endpoint is lag-1 cosine similarity between consecutive gradient vectors. In the MNIST confirmation, positive temporal ordering produced a dose-response slope of **+0.182 [0.148, 0.216]**, while negative temporal ordering produced **−0.106 [−0.132, −0.080]**. Both effects survived Benjamini–Hochberg correction, and the corresponding within-epoch effects were essentially identical in magnitude.

The bounded conclusion is that temporal structure in stochastic data ordering can systematically alter the optimization trajectory of SGD even when mini-batch class composition and the underlying batch multiset are held fixed. The study does **not** establish that structured stochasticity generally improves optimization or that the observed relationship transfers unchanged across architectures, optimizers, datasets, or batch sizes.

### Proposal II report

- `docs/Proposal_II_Professional_Writeup.md` — frozen professional analysis report
- `results/table1_reproduced.csv` — reproduced Phase I/II activation-patching statistics

## Reproduction — Project I

### Lightweight analysis reproduction — no GPU required

The central Project I statistics can be reproduced from the released Phase II patch-level NPZ without rerunning GPT-2 generation. The intended analysis notebook is:

```text
notebooks/reproduce_table1_from_npz.ipynb
```

It reconstructs the 4,800 pair-level observations, mean recovery by direction and layer, prompt-cluster bootstrap 95% CIs, and the recovery-vs-layer figure.

### Full experiment

The original generation + activation-patching pipeline requires a PyTorch environment capable of loading GPT-2 small and is substantially more computationally expensive.

```bash
pip install -r requirements.txt
python src/phaseII_full_paired_v2.py --prompts prompts/phaseII_prompts_20.txt --out data/phaseII_full_20prompts.npz
```

Analysis helper:

```bash
python src/analyze_phaseII_full_v2.py --input data/phaseII_full_20prompts.npz --out results/pair_level_analysis.csv
```

## Repository contents

```text
src/phaseII_full_paired_v2.py
src/analyze_phaseII_full_v2.py
data/phaseII_full_20prompts.npz
notebooks/reproduce_table1_from_npz.ipynb
figures/recovery_vs_layer_bootstrap.png
results/table1_reproduced.csv
prompts/phaseII_prompts_20.txt
requirements.txt
docs/Structured_Stochasticity_Causal_Manuscript_Neal_Nanda.pdf
docs/Proposal_II_Professional_Writeup.md
```

## Interpretation and scope

The broader research question is whether stochastic-process structure can become a meaningful part of computation in sequential learning and inference systems. The two projects test this in different settings: external sampling during autoregressive generation and data ordering during iterative optimization. Claims are deliberately bounded to the demonstrated experiments; future work should focus on independent replication, mechanism localization, and transfer across models and training regimes.

## License

MIT. See `LICENSE`.
