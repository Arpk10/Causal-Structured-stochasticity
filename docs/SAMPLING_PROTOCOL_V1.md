# Sampling protocol v1

Both research repositories define top-p nucleus sampling identically.

Given normalized probabilities sorted in descending order, retain the smallest prefix whose cumulative mass is greater than or equal to top_p. Exact equality is included and the next token is excluded. Renormalize the retained probabilities before inverse-CDF sampling. The inverse CDF uses left-sided search (the first cumulative value greater than or equal to u); the distinction matters only at exact CDF boundaries for continuous uniforms, but is fixed for deterministic reproducibility.

The original model probability of the selected token and its CDF position in the renormalized nucleus distribution are distinct quantities and must not be conflated.

Valid threshold: 0 < top_p <= 1. Input probabilities must be finite, non-negative, one-dimensional, non-empty, and normalized to one.

Results produced with a different nucleus mask or inverse-CDF boundary convention must be regenerated before pooling with protocol-v1 results.
