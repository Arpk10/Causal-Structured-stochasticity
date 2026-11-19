import argparse, math, json
from pathlib import Path
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def ar1_u(eps, rho):
    if rho == 0: z = eps.copy()
    else:
        z = np.empty_like(eps); z[0] = eps[0]
        for t in range(1, len(eps)): z[t] = rho*z[t-1] + math.sqrt(1-rho*rho)*eps[t]
    return 0.5*(1 + np.vectorize(math.erf)(z/np.sqrt(2)))

def trajectory_divergence(donor_tokens, recipient_tokens, step):
    """Describe pre-intervention token-history agreement and divergence age."""
    donor_tokens = np.asarray(donor_tokens)
    recipient_tokens = np.asarray(recipient_tokens)
    if donor_tokens.shape != recipient_tokens.shape:
        raise ValueError("paired token trajectories must have identical shapes.")
    if not 0 <= step <= len(donor_tokens):
        raise ValueError("step must be within the trajectory.")
    mismatches = np.flatnonzero(donor_tokens != recipient_tokens)
    first = int(mismatches[0]) if len(mismatches) else -1
    same_history = bool(np.array_equal(donor_tokens[:step], recipient_tokens[:step]))
    age = int(step - first) if first >= 0 and first < step else 0
    return same_history, first, age


def paired_state_diagnostics(donor_state, recipient_state, donor_logits, recipient_logits):
    """Quantify pre-sampling differences for a paired token-prefix position."""
    donor_state = np.asarray(donor_state, dtype=np.float64)
    recipient_state = np.asarray(recipient_state, dtype=np.float64)
    donor_logits = np.asarray(donor_logits, dtype=np.float64)
    recipient_logits = np.asarray(recipient_logits, dtype=np.float64)
    if donor_state.shape != recipient_state.shape:
        raise ValueError("paired states must have identical shapes.")
    if donor_logits.shape != recipient_logits.shape:
        raise ValueError("paired logits must have identical shapes.")
    state_delta = donor_state - recipient_state
    state_l2 = float(np.linalg.norm(state_delta))
    state_max_abs = float(np.max(np.abs(state_delta))) if state_delta.size else 0.0
    logit_delta = donor_logits - recipient_logits
    logit_max_abs = float(np.max(np.abs(logit_delta))) if logit_delta.size else 0.0

    def softmax(x):
        x = x - np.max(x)
        ex = np.exp(x)
        return ex / ex.sum()

    p = softmax(donor_logits)
    q = softmax(recipient_logits)
    midpoint = 0.5 * (p + q)
    eps = 1e-15
    js = 0.5 * np.sum(p * np.log((p + eps) / (midpoint + eps)))
    js += 0.5 * np.sum(q * np.log((q + eps) / (midpoint + eps)))
    return {
        "prepatch_state_l2": state_l2,
        "prepatch_state_max_abs": state_max_abs,
        "prepatch_logit_max_abs": logit_max_abs,
        "prepatch_js": float(js),
    }


def top_p_set(probs, top_p):
    """Return the minimal descending-probability prefix with mass >= top_p.

    Exact threshold equality is included, and no additional token is retained.
    This matches temporal-sampling-correlation's protocol v1.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1 or probs.size == 0:
        raise ValueError("probs must be a non-empty one-dimensional array.")
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("probs must contain finite non-negative values.")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1].")
    total = probs.sum()
    if not np.isclose(total, 1.0, rtol=1e-7, atol=1e-9):
        raise ValueError("probs must be normalized to sum to one.")

    order = np.argsort(-probs, kind="stable")
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs, dtype=np.float64)
    last = int(np.searchsorted(cumulative, top_p, side="left"))
    last = min(last, len(order) - 1)
    idx = order[: last + 1]
    q = probs[idx].copy()
    q /= q.sum()
    return idx, q

@torch.no_grad()
def forward(model, ids):
    out=model(input_ids=ids, use_cache=False, output_hidden_states=True)
    return [h[0,-1,:].float() for h in out.hidden_states[1:]], out.logits[0,-1,:].float()


@torch.no_grad()
def forward_with_components(model, ids, capture_layers):
    """Capture final-position attention and MLP outputs only at selected 1-based layers."""
    capture_layers = tuple(sorted(set(int(x) for x in capture_layers)))
    if not capture_layers or any(x < 1 or x > len(model.transformer.h) for x in capture_layers):
        raise ValueError("capture_layers must contain valid 1-based GPT-2 block indices.")
    captured = {"attn": {}, "mlp": {}}
    handles = []

    def make_hook(kind, index):
        def hook(module, inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captured[kind][index] = value[0, -1, :].detach().float()
        return hook

    for index, block in enumerate(model.transformer.h):
        if index + 1 in capture_layers:
            handles.append(block.attn.register_forward_hook(make_hook("attn", index)))
            handles.append(block.mlp.register_forward_hook(make_hook("mlp", index)))
    try:
        out = model(input_ids=ids, use_cache=False, output_hidden_states=True)
    finally:
        for handle in handles:
            handle.remove()

    residual = [h[0, -1, :].detach().float() for h in out.hidden_states[1:]]
    attention = [captured["attn"][i - 1] for i in capture_layers]
    mlp = [captured["mlp"][i - 1] for i in capture_layers]
    return residual, attention, mlp, out.logits[0, -1, :].detach().float()

def dist_metrics(recipient_logits, donor_logits, patched_logits, donor_token):
    pr=torch.softmax(recipient_logits.double(),dim=-1); pd=torch.softmax(donor_logits.double(),dim=-1); pp=torch.softmax(patched_logits.double(),dim=-1); eps=1e-12
    kl_patch_rec=torch.sum(pp*(torch.log(pp+eps)-torch.log(pr+eps))).item(); kl_rec_donor=torch.sum(pr*(torch.log(pr+eps)-torch.log(pd+eps))).item(); kl_patch_donor=torch.sum(pp*(torch.log(pp+eps)-torch.log(pd+eps))).item()
    recovery=float(1-kl_patch_donor/kl_rec_donor) if kl_rec_donor>1e-12 else float("nan")
    m=(pp+pd)/2; js=.5*torch.sum(pp*(torch.log(pp+eps)-torch.log(m+eps)))+.5*torch.sum(pd*(torch.log(pd+eps)-torch.log(m+eps)))
    a=patched_logits-recipient_logits; b=donor_logits-recipient_logits; cos=torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)+eps)
    return {"intervention_kl":kl_patch_rec,"kl_recipient_to_donor":kl_rec_donor,"kl_patched_to_donor":kl_patch_donor,"recovery":recovery,"js_to_donor":float(js.item()),"logit_direction_cosine":float(cos.item()),"donor_token_prob_recipient":float(pr[donor_token].item()),"donor_token_prob_patched":float(pp[donor_token].item()),"donor_token_prob_donor":float(pd[donor_token].item()),"donor_token_prob_gain":float((pp[donor_token]-pr[donor_token]).item())}

@torch.no_grad()
def patched_logits(model, ids, layer_idx, donor_state, component="residual"):
    if component == "residual":
        module = model.transformer.h[layer_idx-1]
    elif component == "attention":
        module = model.transformer.h[layer_idx-1].attn
    elif component == "mlp":
        module = model.transformer.h[layer_idx-1].mlp
    else:
        raise ValueError("component must be residual, attention, or mlp")

    def hook(module, inp, out):
        if isinstance(out, tuple):
            x=out[0].clone()
            x[:, -1, :] = donor_state.to(x.device, dtype=x.dtype)
            return (x,) + out[1:]
        x=out.clone()
        x[:, -1, :] = donor_state.to(x.device, dtype=x.dtype)
        return x

    handle=module.register_forward_hook(hook)
    try:
        return model(input_ids=ids, use_cache=False).logits[0,-1,:].float()
    finally:
        handle.remove()

def run(a):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print("device=",device)
    if device.type=="cuda": print("gpu=",torch.cuda.get_device_name(0))
    tok=GPT2TokenizerFast.from_pretrained(a.model); model=GPT2LMHeadModel.from_pretrained(a.model).to(device).eval()
    prompts=[x.strip() for x in Path(a.prompts).read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.max_prompts: prompts=prompts[:a.max_prompts]
    rows=[]
    for pi,prompt in enumerate(prompts):
        base=tok(prompt,return_tensors="pt")["input_ids"].to(device)
        for seed in a.seeds:
            pair_id=f"p{pi:03d}_s{seed:03d}"; rng=np.random.default_rng(seed); eps=rng.normal(size=a.tokens); trajectories={}
            for name,rho in [("IID",0.0)]+[(f"AR{rho:g}",rho) for rho in a.rhos if rho!=0]:
                u=ar1_u(eps,rho); ids=base.clone(); states=[]; attention_states=[]; mlp_states=[]; logits_list=[]; tokens=[]
                for t in range(a.tokens):
                    hs, attn, mlp, logits=forward_with_components(model,ids,a.layers); probs=torch.softmax(logits.double()/a.temperature,dim=-1).cpu().numpy(); idx,q=top_p_set(probs,a.top_p); k=np.searchsorted(np.cumsum(q),u[t],side="left"); y=int(idx[min(k,len(idx)-1)])
                    states.append(torch.stack(hs).cpu().numpy())
                    attention_states.append(torch.stack(attn).cpu().numpy())
                    mlp_states.append(torch.stack(mlp).cpu().numpy())
                    logits_list.append(logits.cpu().numpy()); tokens.append(y); ids=torch.cat([ids,torch.tensor([[y]],device=device)],dim=1)
                trajectories[name]={"rho":rho,"u":u,"states":np.stack(states),"attention":np.stack(attention_states),"mlp":np.stack(mlp_states),"component_layer_indices":list(a.layers),"logits":np.stack(logits_list),"tokens":np.array(tokens)}
            iid=trajectories["IID"]
            for rho in a.rhos:
                if rho==0: continue
                ar=trajectories[f"AR{rho:g}"]
                for direction,donor,recipient in [("AR_to_IID",ar,iid),("IID_to_AR",iid,ar)]:
                    mismatch_steps = np.flatnonzero(donor["tokens"] != recipient["tokens"])
                    first_divergence_for_pair = int(mismatch_steps[0]) if len(mismatch_steps) else -1
                    eval_steps = set(a.steps)
                    if 0 <= first_divergence_for_pair < a.tokens:
                        eval_steps.add(first_divergence_for_pair)
                    for t in sorted(eval_steps):
                        history_same, first_divergence, divergence_age = trajectory_divergence(
                            donor["tokens"], recipient["tokens"], t
                        )
                        for layer in a.layers:
                            state_diag = paired_state_diagnostics(
                                donor["states"][t, layer-1],
                                recipient["states"][t, layer-1],
                                donor["logits"][t],
                                recipient["logits"][t],
                            )
                            ids=base.clone(); toks=recipient["tokens"][:t]
                            if len(toks): ids=torch.cat([ids,torch.tensor(toks[None,:],device=device)],dim=1)
                            for component, key in [("residual","states"),("attention","attention"),("mlp","mlp")]:
                                component_layer_slot = a.layers.index(layer)
                                state_slot = layer - 1 if component == "residual" else component_layer_slot
                                donor_state=torch.tensor(donor[key][t,state_slot],device=device)
                                patch=patched_logits(model,ids,layer,donor_state,component=component)
                                m=dist_metrics(torch.tensor(recipient["logits"][t],device=device),torch.tensor(donor["logits"][t],device=device),patch,int(donor["tokens"][t]))
                                rows.append({"pair_id":pair_id,"prompt_idx":pi,"seed":seed,"rho":rho,"layer":layer,"t":t,"direction":direction,"patch_component":component,"history_match":bool(history_same),"first_divergence_step":first_divergence,"divergence_age":divergence_age,"is_first_divergence_step":bool(first_divergence == t),"donor_token":int(donor["tokens"][t]),"recipient_token":int(recipient["tokens"][t]),**state_diag,**m})
    np.savez_compressed(a.out,rows=np.array(rows,dtype=object)); meta={"model":a.model,"tokens":a.tokens,"temperature":a.temperature,"top_p":a.top_p,"rhos":a.rhos,"layers":a.layers,"steps":a.steps,"patch_components":["residual","attention","mlp"],"component_hook_location":"module_output_at_final_token_position","explicit_pairing":True,"common_gaussian_innovations":True}; Path(a.meta).write_text(json.dumps(meta,indent=2)); print("saved",a.out,"measurements=",len(rows))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--prompts",required=True); p.add_argument("--out",required=True); p.add_argument("--meta",default="phaseII_full_metadata.json"); p.add_argument("--model",default="gpt2"); p.add_argument("--tokens",type=int,default=40); p.add_argument("--temperature",type=float,default=1.0); p.add_argument("--top-p",type=float,default=.9); p.add_argument("--rhos",nargs="+",type=float,default=[.25,.5,.75,.9]); p.add_argument("--layers",nargs="+",type=int,default=[8,9,10]); p.add_argument("--steps",nargs="+",type=int,default=[5,10,15,20,25,30,35,39]); p.add_argument("--seeds",nargs="+",type=int,default=list(range(10))); p.add_argument("--max-prompts",type=int,default=None); run(p.parse_args())
