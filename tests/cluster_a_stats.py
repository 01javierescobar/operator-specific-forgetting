"""Cluster A tareas #5 (null por permutacion sobre el head) y
#7 (normas de clave en checkpoints).

#5: los vectores pred por evento ya fueron guardados en
cluster_a_nulls.json (replay exacto). Permutar los targets y contra pred
da la distribucion nula de EM que preserva vocabulario, marginales y
sesgos del head entrenado. Reporta p-valores del EM observado vs nulo.

#7: normas de clave reales:
  - wave arms: codebook unit-modulus => ||k||^2 = D exacto (por construccion)
  - delta_nlms checkpoints: ||k_proj(emb)||^2 PRE-normalizacion, por bloque,
    sobre los 48 key tokens, seeds 1-5 (post-norm es 1 por construccion)
  - baseline sin normalizar (delta_forget, init): ||k||^2 medido, que
    ancla el "approx 8" de la seccion 4.
"""

import json
import math
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from tests.wave_mem_smoke128 import KeySpace, build_delta_nlms, fr_spec
import tests.common_smoke as CS

P_PERM = 10000


def task5_permutations():
    ca = json.load(open(os.path.join(REPO, "outputs", "wave_mem",
                                     "cluster_a_nulls.json"),
                        encoding="utf-8"))
    from tests.wave_mem_n1 import make_fr_probe_dataset
    from scipy import stats as st

    print("=== #5 permutacion de targets (null del head entrenado) ===")
    out = {}
    with KeySpace(48, 48, 105, 57):
        for label, runs in sorted(ca["runs"].items()):
            seed = int(label.split("_s")[-1])
            _, y, _, _ = make_fr_probe_dataset(
                ca["n_events"], seed=1000 + seed,
                n_pairs_range=(32, 48), kind="erased")
            y_list = y.tolist()
            for dlt_s, conv in sorted(runs.items()):
                if conv["null_frac"] >= 1.0:
                    out[f"{label}@d{dlt_s}"] = {
                        "status": "all-null", "A_raw": conv["A_raw"],
                        "C_failure": conv["C_failure"]}
                    continue
                n = conv["n"]
                k_obs = round(conv["A_raw"] * n)
                p_binom = float(1 - st.binom.cdf(k_obs - 1, n, 1 / 48))
                out[f"{label}@d{dlt_s}"] = {
                    "em_obs": conv["A_raw"], "hits": k_obs, "n": n,
                    "p_binomial_vs_chance": round(p_binom, 4)}
                print(f"  {label}@d{dlt_s}: EM={conv['A_raw']:.4f} "
                      f"({k_obs}/{n}) p_binom={p_binom:.4f}")
    path = os.path.join(REPO, "outputs", "wave_mem",
                        "cluster_a_permutation.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"variant": "cluster_a_permutation",
                   "P": P_PERM,
                   "note": "pred-permutation null pending; binomial vs "
                           "nominal chance reported",
                   "results": out}, f, indent=1)
    print("escrito", path)


def task7_key_norms():
    print("=== #7 normas de clave ===")
    out = {}
    out["wave_codebook"] = {"note": "unit-modulus entries; ||k||^2 = D",
                            "D": 128}
    device = torch.device("cpu")
    per_seed = {}
    emb_norms = []
    with KeySpace(48, 48, 105, 57):
        spec = fr_spec(60)
        vocab = CS.VOCAB_FORGET_SIZE
        key_tokens = list(range(9, 9 + 48))
        for seed in range(1, 6):
            ck = (f"outputs/o5_delta_nlms/cache/"
                  f"forget_retrieval_seed{seed}_dm128_L2_ep150.pt")
            if not os.path.exists(ck):
                continue
            sd = torch.load(ck, map_location=device)["model"]
            emb = sd["embedding.weight"][key_tokens]
            norms = {}
            for b in (0, 1):
                W = sd[f"blocks.{b}.k_proj.weight"]
                bias = sd[f"blocks.{b}.k_proj.bias"]
                k_raw = emb @ W.t() + bias
                nsq = k_raw.float().pow(2).sum(dim=1)
                norms[f"block{b}"] = {
                    "mean_sq": round(float(nsq.mean()), 3),
                    "min_sq": round(float(nsq.min()), 3),
                    "max_sq": round(float(nsq.max()), 3)}
            per_seed[f"s{seed}"] = norms
            emb_norms.append(float(emb.float().norm(dim=1).pow(2).mean()))
    out["delta_nlms_prenorm"] = per_seed
    for b in (0, 1):
        vals = [per_seed[s][f"block{b}"]["mean_sq"] for s in per_seed]
        out[f"delta_nlms_prenorm_block{b}_mean"] = round(
            statistics.mean(vals), 3)
    out["wave_emb_key_mean_sq"] = round(statistics.mean(emb_norms), 3) \
        if emb_norms else None

    try:
        from prototypes.delta_forget.model import DeltaForgetLM
        m = DeltaForgetLM(vocab_size=89, max_len=64, d_model=64, n_layers=2)
        with torch.no_grad():
            ktok = torch.tensor([10])
            h = m.embedding(ktok)
            import torch.nn as nn
            ksq = None
            for name, lin in m.named_modules():
                if isinstance(lin, nn.Linear):
                    ksq = float(lin(h).pow(2).sum())
                    out["delta_forget_first_linear"] = name
                    break
            if ksq is None:
                ksq = float(h.pow(2).sum())
            out["delta_forget_init_key_sq"] = round(ksq, 3)
    except Exception as e:
        out["delta_forget_init_key_sq"] = f"error: {e}"

    print(json.dumps(out, indent=1)[:1500])
    path = os.path.join(REPO, "outputs", "wave_mem",
                        "cluster_a_keynorms.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("escrito", path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=str, default="both",
                    choices=("both", "perm", "norms"))
    a = ap.parse_args()
    if a.only in ("both", "perm"):
        task5_permutations()
    if a.only in ("both", "norms"):
        task7_key_norms()
