import argparse
import numpy as np
import pandas as pd

def main(a):
    d=np.load(a.input,allow_pickle=True)
    df=pd.DataFrame(list(d["rows"]))
    keys=["pair_id","prompt_idx","seed","rho","direction","layer"]
    agg=df.groupby(keys).agg(recovery=("recovery","mean"),intervention_kl=("intervention_kl","mean"),js_to_donor=("js_to_donor","mean"),logit_cosine=("logit_direction_cosine","mean"),donor_token_gain=("donor_token_prob_gain","mean"),history_match_rate=("history_match","mean"),n=("t","count")).reset_index()
    agg.to_csv(a.out,index=False)
    print("\n=== Pair-level means ===")
    print(agg.groupby(["direction","layer"])[["recovery","intervention_kl","js_to_donor","logit_cosine","donor_token_gain"]].mean().to_string())
    print("\n=== Recovery by rho and layer ===")
    print(agg.groupby(["direction","rho","layer"])["recovery"].mean().to_string())
    print("\n=== Intervention KL by rho and layer ===")
    print(agg.groupby(["direction","rho","layer"])["intervention_kl"].mean().to_string())
    print("\n=== History-match diagnostics ===")
    print(agg.groupby(["direction","rho"])["history_match_rate"].mean().to_string())

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--input",required=True); p.add_argument("--out",required=True); main(p.parse_args())
