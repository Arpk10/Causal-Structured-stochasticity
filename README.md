# Causal Structured Stochasticity

**Mechanistic study of how temporal structure in external stochasticity influences autoregressive language-model dynamics using causal activation patching.**

This repository contains the reproducible code for the Phase II causal activation-patching experiment accompanying the manuscript *When Randomness Becomes Part of the Computation: Temporal Structure in Stochastic Decoding and Causal Activation Patching*.

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

These results support a mediated causal picture:

`external stochastic structure → sampled token trajectory → residual representation → next-token distribution`

The experiment does **not** establish that GPT-2 contains a dedicated representation of the raw stochastic source or of the AR coefficient itself.

## Repository contents

```text
src/phaseII_full_paired_v2.py      # Full paired generation + activation patching
src/analyze_phaseII_full_v2.py     # Analysis helpers
requirements.txt                   # Python dependencies
prompts/phaseII_prompts_20.txt     # Prompt-set record
```

## Reproduction

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run the generation / patching script:

```bash
python src/phaseII_full_paired_v2.py
```

Analysis can be run with:

```bash
python src/analyze_phaseII_full_v2.py
```

The scripts assume a PyTorch/Transformers environment capable of loading GPT-2 small. Generation can be computationally nontrivial; see the source comments for configuration details.

## Scope and interpretation

The principal claim is causal and trajectory-level: external temporal structure produces divergent autoregressive trajectories, and late residual-stream states on those trajectories causally influence subsequent token distributions. The study does not claim that correlated randomness improves language quality, that larger AR correlation necessarily produces a larger effect, or that the observed mechanism generalizes to every language model.

## Manuscript

The associated manuscript and statistical analysis are maintained separately from this code repository. The repository is intended to provide a clean, inspectable computational record of the core experiment.
