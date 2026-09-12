# Structured Stochasticity in Iterative Machine Learning

*Proposal II — Empirical Study of Temporal Structure in SGD*

**Frozen analysis report • September 2026**

## Abstract

Stochastic optimization is usually characterized by the marginal distribution of the random choices used to construct training updates. This study asks whether the temporal organization of those stochastic choices can itself alter optimization dynamics after relevant marginal properties are controlled. We study stochastic gradient descent (SGD) under matched data-order interventions, using an echo-null control that preserves the same underlying mini-batch multiset while destroying its temporal organization. The experimental program progressed from a controlled synthetic logistic model to a class-balanced latent-feature design, a nonlinear multilayer perceptron, a handwritten-digit benchmark, and finally an actual MNIST confirmation. The clearest result is that temporal structure produces dose-dependent changes in consecutive-gradient similarity: positive ordering increases gradient persistence, whereas negative ordering decreases it. On MNIST, the estimated gradient-cosine dose-response was +0.182 [0.148, 0.216] for positive structure and −0.106 [−0.132, −0.080] for negative structure, with both effects remaining significant after Benjamini–Hochberg correction. The effect persisted when epoch-boundary transitions were excluded. Gradient-norm autocorrelation was less consistent and is therefore treated as secondary. The results support a bounded claim: the temporal structure of stochastic algorithmic inputs can influence optimization trajectories even when conventional batch-composition marginals are held fixed.

## 1. Introduction

The motivating question is whether stochasticity in an iterative learning algorithm should be treated only through its marginal properties or also through its temporal structure. The original proposal identifies SGD as the natural test case because the stochastic data-selection process enters a sequential feedback loop: data selection produces a gradient, the gradient updates the parameters, and the new parameters change the next gradient. The resulting structure is θ_t → θ_{t+1} through g_t, with the ordering of future stochastic inputs potentially affecting the trajectory.

The proposal explicitly distinguishes this question from a simple comparison of random shuffling and non-random shuffling. The intended novelty is to manipulate the statistical structure of the stochastic process while preserving relevant marginal sampling properties, and then measure mechanistic changes in the optimization trajectory. The proposal also specifies that final accuracy alone is insufficient: gradient norms, lag-1 gradient autocorrelation, consecutive-gradient cosine similarity, parameter updates, convergence, and run-to-run variability are candidate observables.

## 2. Research Question and Hypothesis

**Primary research question:** Does the statistical structure of the stochastic data-order process affect optimization dynamics after marginal sampling properties are controlled?

**Null hypothesis (marginal-only view):** once dataset composition and relevant marginal sampling properties are fixed, the temporal organization of the stochastic data stream does not systematically affect optimization dynamics.

**Structured-stochasticity hypothesis:** temporal dependence in the data-order process changes the downstream optimization trajectory, producing systematic differences in gradient and update dynamics even when the underlying collection of mini-batches is held fixed.

## 3. Experimental Design

### 3.1 Core intervention

The central control is a matched structured-versus-echo-null comparison. A structured training sequence is constructed from a fixed collection of mini-batches. The echo-null condition uses the exact same mini-batch multiset but randomly permutes the order of those batches. Thus, within each pair, the dataset, batch contents, number of updates, optimizer, learning rate, initialization, and training horizon are held fixed. The principal manipulated property is temporal organization.

### 3.2 Removal of the class-order confound

Early pilots used class-pure batches and demonstrated a strong effect, but that design could be explained by class clustering or class switching. The decisive redesign used class-balanced mini-batches. In the later experiments, every batch contained equal numbers from the relevant classes, while temporal structure was induced through a label-independent latent feature score. This made the ordering manipulation substantially more specific to stochastic sequence structure.

### 3.3 Dose-response design

Rather than treating structure as a binary intervention, the study used multiple structure strengths from approximately 0 to 1. Positive conditions cluster similar latent batch types in time; negative conditions impose anti-persistent alternation. The realized sequence autocorrelation was measured directly. Echo-null sequences were expected to remove this temporal structure while preserving the same batch collection.

### 3.4 Progression of models

The experimental program was deliberately staged. It began with a synthetic logistic regression system for pipeline validation, progressed to a class-balanced latent-feature synthetic design, then to a small nonlinear MLP, then to the scikit-learn handwritten-digits benchmark, and finally to an actual MNIST confirmation. This progression was intended to separate mechanistic validation from benchmark-scale confirmation.

## 4. Measurements and Statistical Analysis

The frozen primary mechanistic endpoint is lag-1 cosine similarity between consecutive gradient vectors. A second primary presentation is the corresponding within-epoch measure, which excludes the transition from the final batch of one epoch to the first batch of the next. Gradient-norm autocorrelation is retained as a secondary mechanistic endpoint. Final loss and test accuracy are treated as secondary optimization outcomes rather than the main evidence for the mechanism.

Inference was performed at the paired-seed level. Dose-response slopes were estimated from structured-minus-echo-null effects across structure strength, and uncertainty was quantified by bootstrap resampling across seeds. Paired sign-flip/randomization logic was used for the structured-versus-echo comparison. Multiple primary mechanistic tests were adjusted using the Benjamini–Hochberg procedure. The state-to-next-gradient coupling metric proposed in the original plan was evaluated but did not show a reliable effect in the MNIST confirmation and is therefore excluded from the frozen primary analysis.

## 5. Results

### 5.1 Synthetic and nonlinear validation

The controlled synthetic experiments established the core phenomenon before benchmark scaling. In the matched echo-null experiment, positive structure produced substantially more positive consecutive-gradient similarity than its echo-null counterpart, while negative structure produced substantially more negative similarity. The subsequent dose-response experiment showed approximately monotonic increases and decreases in the paired gradient-cosine effect as structure strength increased. Importantly, this pattern survived the redesign in which every mini-batch was class-balanced and temporal structure was tied to a label-independent latent feature.

The effect remained detectable after moving from logistic regression to a small nonlinear MLP, although it became smaller and noisier. This motivated the benchmark confirmation rather than treating the nonlinear pilot as sufficient evidence.

### 5.2 MNIST confirmation

The final benchmark experiment used actual MNIST data with class-balanced mini-batches, a nonlinear 784→64→10 MLP, and the same matched structured-versus-echo-null logic. The realized ordering manipulation was strong: at maximum strength, positive sequences had approximately +0.98 lag-1 structure autocorrelation, while negative sequences had approximately −0.72; echo-null sequences were near zero.

The primary MNIST result was a clear, opposite-signed dose-response in consecutive-gradient cosine similarity. For positive temporal structure, the estimated slope was +0.182 with a bootstrap 95% interval of [0.148, 0.216]. For negative temporal structure, the estimated slope was −0.106 with a bootstrap 95% interval of [−0.132, −0.080]. The corresponding within-epoch effects were essentially identical in magnitude (+0.183 and −0.106), showing that the signal was not dependent on the small number of epoch-boundary transitions.

Gradient-norm autocorrelation was less robust. The overall dose-response estimates were not consistently different from zero, although within-epoch norm autocorrelation showed a positive response under positive ordering. Because this endpoint did not reproduce the same clean sign-symmetric pattern as gradient cosine, it is retained as secondary rather than used to define the main conclusion.

The MNIST gradient-cosine findings remained significant after Benjamini–Hochberg correction across the primary mechanistic tests. The study therefore provides convergent evidence that temporal organization of the stochastic data stream changes the temporal geometry of SGD gradients under controlled marginal batch composition.

### 5.3 Optimization outcomes

Endpoint performance effects were more complicated than the mechanistic gradient effect. In the MNIST experiment, stronger positive structure was associated with higher test loss and lower test accuracy, despite producing more persistent consecutive gradients. These results are treated as secondary and model-specific rather than as evidence that temporal structure universally improves or harms generalization.

## 6. Interpretation

The most defensible interpretation is not that one particular ordering scheme is a better optimizer. Instead, the experiments show that the temporal organization of stochastic algorithmic inputs can become dynamically relevant. When the same underlying collection of mini-batches is presented in different temporal arrangements, the gradients generated by SGD can acquire systematically different temporal correlation structures.

This is consistent with the sequential feedback structure of SGD. A batch affects the current gradient; the resulting update changes the parameter state; that changed state then determines how subsequent batches are translated into gradients. Consequently, temporal dependence in the stochastic input can interact with the optimizer's evolving state rather than averaging away as mere sampling noise.

## 7. What the Study Does Not Establish

- It does not establish that structured stochasticity generally improves optimization.
- It does not establish that positive correlation is universally beneficial or negative correlation universally harmful.
- It does not establish a universal relationship between gradient persistence and final generalization.
- It does not establish that the effect transfers unchanged to every architecture, optimizer, dataset, or batch size.
- It does not establish the originally broader Random Forest hypothesis; that remains a secondary, untested extension.
- It does not identify a single causal micro-mechanism beyond the demonstrated relationship between ordering structure and gradient trajectory statistics.

## 8. Limitations

- The experiments use relatively small models and controlled training horizons compared with modern large-scale training.
- The benchmark confirmation uses one principal architecture and optimizer configuration.
- The temporal intervention is based on a constructed latent-feature ordering rather than a naturally occurring data-order process.
- Some secondary endpoints were inconsistent across model classes and should not be overgeneralized.
- The final MNIST analysis should be accompanied by an explicit machine-readable accounting of the exact number of paired seeds contributing to each inferential quantity before external publication.

## 9. Frozen Primary Conclusion

The study supports the following bounded conclusion: temporal structure in stochastic data ordering can systematically alter the optimization trajectory of SGD even when mini-batch class composition and the underlying batch multiset are held fixed. In the actual MNIST confirmation, positive temporal ordering increased consecutive-gradient persistence while negative temporal ordering decreased it, with a dose-dependent and within-epoch-reproducible effect that survived multiplicity correction. The result supports treating stochastic-process structure as a potentially meaningful dimension of iterative optimization, rather than assuming that marginal sampling properties fully characterize the stochastic input.

## 10. Relation to the Original Proposal

This outcome directly addresses the original Proposal II question. The proposal called for controlled stochastic interventions, preservation of relevant marginal properties, trajectory-level mechanistic measurements, systematic structure-strength relationships, and explicit treatment of null results. The final analysis follows that logic and narrows the evidence to the endpoint that reproduced most cleanly: lag-1 gradient cosine similarity.

The original proposal also positioned Random Forests as a secondary target rather than forcing a single mechanism across all learning algorithms. That distinction should be preserved in the final research program: SGD provides the primary evidence for structured stochasticity in iterative optimization, while any Random Forest study should be framed as a separate test of structured randomness in ensemble construction.

## 11. Recommended Next Research

1. Independent replication on additional datasets and architectures while keeping the frozen primary endpoint.
2. Test whether the effect depends on learning rate, batch size, optimizer momentum, or training stage.
3. Compare structured ordering against conventional random reshuffling and low-discrepancy sequences under the same matching framework.
4. Examine whether gradient persistence predicts convergence speed, stability, or generalization across a broader hyperparameter grid.
5. Only after the SGD result is independently replicated, run the proposed Random Forest extension using matched bootstrap samples and manipulated feature-selection randomness.

## 12. Reproducibility and Analysis Freeze

The primary endpoint hierarchy is now frozen: (1) lag-1 gradient cosine similarity, (2) within-epoch lag-1 gradient cosine similarity, and (3) gradient-norm autocorrelation as a secondary endpoint. The state-to-next-gradient coupling metric is not a primary endpoint. The matched echo-null design, paired-seed analysis, bootstrap uncertainty, and Benjamini–Hochberg correction constitute the frozen inferential framework. Future experiments should be treated as replications or extensions rather than used to retrospectively redefine the primary outcome.

## Appendix A. Conceptual Data-Flow

Stochastic ordering structure → mini-batch sequence → gradient g_t → parameter update θ_{t+1} → subsequent gradient g_{t+1}. The empirical target is whether changing the first component, while preserving the relevant marginal batch properties, changes the temporal relationship between g_t and g_{t+1}.

## Appendix B. Source Basis

The research question, proposed controls, candidate mechanistic metrics, SGD-first framing, Random Forest secondary status, and success criteria are derived from the original Proposal II document. The results sections summarize the executed synthetic, digits, and actual-MNIST experiments and the frozen Phase G inference outputs produced during this research session.
