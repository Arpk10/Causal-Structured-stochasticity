import argparse, math, json
from pathlib import Path
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

def ar1_u(eps, rho):
    if rho == 0:
        z = eps.copy()
    else:
        z = np.empty_like(eps)
        z[0] = eps[0]
        for t in range(1, len(eps)):
            z[t] = rho*z[t-1] + math.sqrt(1-rho*rho)*eps[t]
    return 0.5*(1 + np.vectorize(math.erf)(z/np.sqrt(2)))

def top_p_set(probs, top_p):
    order=np.argsort(probs)[::-1]
    sp=probs[order]; c=np.cumsum(sp)
    keep=c<=top_p
    if not np.any(keep): keep[0]=True
    else:
        j=np.argmax(c>top_p)
        if c[j]>top_p: keep[j]=True
    idx=order[keep]
    q=probs[idx].astype(np.float64); q/=q.sum()
    return idx,q

@torch.no_grad()
def forward(model, ids):
    out=model(input_ids=ids, use_cache=False, output_hidden_states=True)
    hs=[h[0,-1,:].float() for h in out.hidden_states[1:]]
    logits=out.logits[0,-1,:].float()
    return hs,logits

def dist_metrics(recipient_logits, donor_logits, patched_logits, donor_token):
    pr=torch.softmax(recipient_logits.double(),dim=-1); pd=torch.softmax(donor_logits.double(),dim=-1); pp=torch.softmax(patched_logits.double(),dim=-1); eps=1e-12
    kl_patch_rec=torch.sum(pp*(torch.log(pp+eps)-torch.log(pr+eps))).item(); kl_rec_donor=torch.sum(pr*(torch.log(pr+eps)-torch.log(pd+eps))).item(); kl_patch_donor=torch.sum(pp*(torch.log(pp+eps)-torch.log(pd+eps))).item()
    recovery=float(1-kl_patch_donor/kl_rec_donor) if kl_rec_donor>1e-12 else float("nan")
    m=(pp+pd)/2; js=.5*torch.sum(pp*(torch.log(pp+eps)-torch.log(m+eps)))+.5*torch.sum(pd*(torch.log(pd+eps)-torch.log(m+eps)))
    a=patched_logits-recipient_logits; b=donor_logits-recipient_logits; cos=torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)+eps)
    return {"intervention_kl":kl_patch_rec,"kl_recipient_to_donor":kl_rec_donor,"kl_patched_to_donor":kl_patch_donor,"recovery":recovery,"js_to_donor":float(js.item()),"logit_direction_cosine":float(cos.item()),"donor_token_prob_recipient":float(pr[donor_token].item()),"donor_token_prob_patched":float(pp[donor_token].item()),"donor_token_prob_donor":float(pd[donor_token].item()),"donor_token_prob_gain":float((pp[donor_token]-pr[donor_token]).item())}

@torch.no_grad()
def patched_logits(model, ids, layer_idx, donor_state):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            x=out[0].clone(); x[:, -1, :] = donor_state.to(x.device, dtype=x.dtype); return (x,) + out[1:]
        x=out.clone(); x[:, -1, :] = donor_state.to(x.device, dtype=x.dtype); return x
    h=model.transformer.h[layer_idx-1].register_forward_hook(hook)
    try:
        out=model(input_ids=ids, use_cache=False); return out.logits[0,-1,:].float()
    finally: h.remove()

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
                u=ar1_u(eps,rho); ids=base.clone(); states=[]; logits_list=[]; tokens=[]
                for t in range(a.tokens):
                    hs,logits=forward(model,ids); probs=torch.softmax(logits.double()/a.temperature,dim=-1).cpu().numpy(); idx,q=top_p_set(probs,a.top_p); k=np.searchsorted(np.cumsum(q),u[t],side="right"); y=int(idx[min(k,len(idx)-1)])
                    states.append(torch.stack(hs).cpu().numpy()); logits_list.append(logits.cpu().numpy()); tokens.append(y); ids=torch.cat([ids,torch.tensor([[y]],device=device)],dim=1)
                trajectories[name]={"rho":rho,"u":u,"states":np.stack(states),"logits":np.stack(logits_list),"tokens":np.array(tokens)}; print(f"generated {pair_id} {name}")
            iid=trajectories["IID"]
            for rho in a.rhos:
                if rho==0: continue
                ar=trajectories[f"AR{rho:g}"]
                for direction,donor,recipient in [("AR_to_IID",ar,iid),("IID_to_AR",iid,ar)]:
                    for t in a.steps:
                        history_same=np.array_equal(donor["tokens"][:t],recipient["tokens"][:t])
                        for layer in a.layers:
                            donor_state=torch.tensor(donor["states"][t,layer-1],device=device); ids=base.clone(); toks=recipient["tokens"][:t]
                            if len(toks): ids=torch.cat([ids,torch.tensor(toks[None,:],device=device)],dim=1)
                            patch=patched_logits(model,ids,layer,donor_state); m=dist_metrics(torch.tensor(recipient["logits"][t],device=device),torch.tensor(donor["logits"][t],device=device),patch,int(donor["tokens"][t]))
                            rows.append({"pair_id":pair_id,"prompt_idx":pi,"seed":seed,"rho":rho,"layer":layer,"t":t,"direction":direction,"history_match":bool(history_same),"donor_token":int(donor["tokens"][t]),"recipient_token":int(recipient["tokens"][t]),**m})
    np.savez_compressed(a.out,rows=np.array(rows,dtype=object)); meta={"model":a.model,"tokens":a.tokens,"temperature":a.temperature,"top_p":a.top_p,"rhos":a.rhos,"layers":a.layers,"steps":a.steps,"explicit_pairing":True,"common_gaussian_innovations":True}; Path(a.meta).write_text(json.dumps(meta,indent=2)); print("saved",a.out,"measurements=",len(rows))

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--prompts",required=True); p.add_argument("--out",required=True); p.add_argument("--meta",default="phaseII_full_metadata.json"); p.add_argument("--model",default="gpt2"); p.add_argument("--tokens",type=int,default=40); p.add_argument("--temperature",type=float,default=1.0); p.add_argument("--top-p",nargs="?",type=float,default=.9); p.add_argument("--rhos",nargs="+",type=float,default=[.25,.5,.75,.9]); p.add_argument("--layers",nargs="+",type=int,default=[8,9,10]); p.add_argument("--steps",nargs="+",type=int,default=[5,10,15,20,25,30,35,39]); p.add_argument("--seeds",nargs="+",type=int,default=list(range(10))); p.add_argument("--max-prompts",type=int,default=None); run(p.parse_args())
