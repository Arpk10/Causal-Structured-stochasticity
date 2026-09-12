# Causal Structured Stochasticity

**Mechanistic study of how temporal structure in external stochasticity influences autoregressive language-model dynamics using causal activation patching.**

This repository contains the reproducible computational record for the Phase II experiment accompanying the manuscript **When Randomness Becomes Part of the Computation: Temporal Structure in Stochastic Decoding and Causal Activation Patching**.

## Research question

Can temporal organization in an externally supplied stochastic source alter an autoregressive language model even when the source has the same one-step marginal distribution?

The experiment compares matched IID and autoregressive (AR) stochastic sources, holding model, prompt, seed, marginal source distribution, and sampling procedure fixed while changing only temporal dependence in the external random process.

## Causal intervention

For paired trajectories generated with common Gaussian innovations, residual-stream activations from a donor trajectory are transplanted into a recipient trajectory at GPT-2 small residual layers 8, 9, and 10. The primary endpoint is **recovery**, measuring how far the recipient next-token distribution moves toward the donor distribution after activation patching.

## Final experimental design

- GPT-2 small
- 20 prompts × 10 seeds
- AR coefficients: 0.25, 0.50, 0.75, 0.90
- Layers: 8, 9, 10
- Positions: 5, 10, 15, 20, 25, 30, 35, 39
- Both AR → IID and IID → AR interventions
- 38,400 raw patch measurements / 4,800 step-aggregated pair-level observations

## Main result

Mean recovery is approximately 0.932–0.935 at layer 8, 0.967–0.969 at layer 9, and 0.986–0.987 at layer 10, with closely matched effects in both intervention directions.

The independent unit for inference is the prompt; confidence intervals use prompt-cluster bootstrap resampling.

These results support a mediated causal picture:

`external stochastic structure → sampled token trajectory → residual representation → next-token distribution`

The experiment does **not** establish that GPT-2 contains a dedicated representation of the raw stochastic source or of the AR coefficient itself.

## Reproduction

### Lightweight analysis reproduction — no GPU required

The central reported statistics can be reproduced from the released Phase II patch-level NPZ without rerunning GPT-2 generation. The intended analysis notebook is:

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
```

## Scope and interpretation

The principal claim is causal and trajectory-level: external temporal structure produces divergent autoregressive trajectories, and late residual-stream states on those trajectories causally influence subsequent token distributions. The study does not claim that correlated randomness improves language quality, that larger AR correlation necessarily produces a larger effect, or that the observed mechanism generalizes to every language model.

## License

MIT. See `LICENSE`.
