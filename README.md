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


## Token-history confounding diagnostics

The Phase II generator records, for every donor-recipient intervention:

- whether generated token histories agree strictly before the patched position;
- the first generated-token position at which paired trajectories diverge (`-1` if they never diverge);
- the number of positions elapsed since first divergence at the intervention point.

The analysis script reports patch metrics separately for matched and divergent histories, and stratifies divergent-history measurements by divergence age. These are diagnostic strata, not independent samples: repeated positions and layers remain nested within the same prompt/seed pair.

**Interpretation constraint:** a donor-state patch after token-history divergence can transfer information about the donor's different linguistic context. A large patching effect in that stratum alone does not identify a representation of the external stochastic source. The matched-history stratum is informative but may be sparse because distinct sampling uniforms can select the same token only by chance. An empty or small matched-history stratum is not evidence of no source effect.

The new fields are produced by the updated generator. Older NPZ files do not contain them and cannot support this stratified analysis; regenerate the experiment to obtain these diagnostics.


## Matched-prefix negative control

GPT-2 receives sampled token IDs, not the external uniforms directly. Consequently, if two paired trajectories have identical token prefixes before position t, their deterministic pre-sampling hidden states and logits at t should agree up to numerical precision, even if the external source histories differ. At the first token position where the trajectories choose different tokens, the pre-sampling state is still computed from the shared prefix and should therefore remain equal; divergence in model state can only arise at later positions after the different token has entered the context.

The generator now records per-layer pre-patching state distances and pre-sampling logit/JS distances for each paired position, and flags the first token-divergence position. The analysis reports these quantities at first divergence and by history-match status. This is a negative-control/invariance check, not evidence that the model internally represents the random source. Nonzero differences at the first divergence position would indicate a pipeline, pairing, precision, or nondeterminism issue to investigate.

This control is deliberately distinct from activation patching: it tests whether source histories can alter GPT-2's state before they have altered its token context. Since the source is not an explicit model input, a direct state effect with identical token prefixes is not expected under this setup.


## Component-localization experiment: residual vs attention vs MLP

The Phase II generator now captures final-token outputs from each GPT-2 block's attention and MLP submodules in addition to the block residual output. At each selected intervention position and layer, it performs three separate donor-to-recipient patches:

1. `residual`: replace the block output at the final token position;
2. `attention`: replace the attention-module output before the block's residual addition;
3. `mlp`: replace the MLP-module output before the block's residual addition.

The resulting rows are labelled by `patch_component`; analysis aggregation keeps the three intervention types separate. The metadata records the component list and hook location.

These interventions are not additive decompositions: attention and MLP outputs interact through the block computation and downstream nonlinearities. A larger component-specific effect is evidence that the intervention at that component is more influential under this patching setup, not proof that the component uniquely stores source-history information. Interpret results with matched-history status and divergence age, and use prompt/seed-paired inference rather than treating layers, positions, or component patches as independent observations.

Run the updated generator and analysis with:

```bash
python src/phaseII_full_paired_v2.py --prompts prompts/phaseII_prompts_20.txt --out data/phaseII_components.npz --meta results/phaseII_components_metadata.json
python src/analyze_phaseII_full_v2.py --input data/phaseII_components.npz --out results/phaseII_components_pair_level.csv
```

The component capture adds memory and compute overhead. For a first smoke test on constrained hardware, use `--max-prompts 2 --seeds 0 1 --rhos 0.5 --layers 8 9 10 --steps 5 10 15` before running the full design.
