import argparse
import numpy as np
import pandas as pd


METRICS = [
    "recovery",
    "intervention_kl",
    "js_to_donor",
    "logit_direction_cosine",
    "donor_token_prob_gain",
]


def main(a):
    data = np.load(a.input, allow_pickle=True)
    df = pd.DataFrame(list(data["rows"]))

    if "patch_component" not in df.columns:
        df["patch_component"] = "residual"
    keys = ["pair_id", "prompt_idx", "seed", "rho", "direction", "layer", "patch_component"]
    agg = (
        df.groupby(keys)
        .agg(
            recovery=("recovery", "mean"),
            intervention_kl=("intervention_kl", "mean"),
            js_to_donor=("js_to_donor", "mean"),
            logit_cosine=("logit_direction_cosine", "mean"),
            donor_token_gain=("donor_token_prob_gain", "mean"),
            history_match_rate=("history_match", "mean"),
            n=("t", "count"),
        )
        .reset_index()
    )
    agg.to_csv(a.out, index=False)

    print("\n=== Pair-level means ===")
    print(
        agg.groupby(["patch_component", "direction", "layer"])[
            ["recovery", "intervention_kl", "js_to_donor", "logit_cosine", "donor_token_gain"]
        ].mean().to_string()
    )
    print("\n=== Recovery by rho and layer ===")
    print(agg.groupby(["direction", "rho", "layer"])["recovery"].mean().to_string())
    print("\n=== Intervention KL by rho and layer ===")
    print(agg.groupby(["direction", "rho", "layer"])["intervention_kl"].mean().to_string())

    print("\n=== History-match diagnostics ===")
    print(agg.groupby(["direction", "rho"])["history_match_rate"].mean().to_string())

    required = {"first_divergence_step", "divergence_age", "history_match"}
    if required.issubset(df.columns):
        print("\n=== Patch effects conditional on token-history status ===")
        history = (
            df.groupby(["patch_component", "history_match", "direction", "rho", "layer"])[METRICS]
            .agg(["mean", "count"])
        )
        print(history.to_string())

        divergent = df.loc[~df["history_match"]].copy()
        if not divergent.empty:
            print("\n=== Patch effects by time since first token divergence ===")
            age = (
                divergent.groupby(["patch_component", "divergence_age", "direction", "rho", "layer"])[METRICS]
                .agg(["mean", "count"])
            )
            print(age.to_string())

        print("\n=== First-divergence distribution ===")
        divergence = (
            df[["pair_id", "rho", "direction", "patch_component", "first_divergence_step"]]
            .drop_duplicates()
            .groupby(["patch_component", "direction", "rho"])["first_divergence_step"]
            .agg(["count", "mean", "min", "max"])
        )
        print(divergence.to_string())
        print("\n=== Pre-sampling state invariance at first divergence ===")
        first = df.loc[df["is_first_divergence_step"]]
        if not first.empty:
            print(
                first.groupby(["patch_component", "direction", "rho", "layer"])[
                    ["prepatch_state_l2", "prepatch_state_max_abs",
                     "prepatch_logit_max_abs", "prepatch_js"]
                ].agg(["mean", "max", "count"]).to_string()
            )
        else:
            print("No first-divergence positions occur among selected intervention steps.")

        print("\n=== Pre-sampling state differences by history status ===")
        print(
            df.groupby(["history_match", "direction", "rho", "layer"])[
                ["prepatch_state_l2", "prepatch_logit_max_abs", "prepatch_js"]
            ].agg(["mean", "count"]).to_string()
        )
    else:
        print(
            "\nNOTE: input predates divergence diagnostics; "
            "rerun Phase II generation to obtain history-stratified results."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    main(parser.parse_args())
