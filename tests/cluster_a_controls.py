"""Cluster A tarea #6: convenciones de nulos para EM_erased y selectividad.

El O05/TOST uso la convencion A (EM sin guarda sobre todos los eventos del
probe erased en delta=0; los eventos con residual nulo quedaron puntuados
por argmax sobre ruido). Este modulo cuantifica como cambian EM_erased y
la selectividad resultante bajo tres convenciones:
  A_raw      : EM sobre todos los eventos [la usada en O05]
  B_excluded : eventos con ratio <= eps_null excluidos del promedio
  C_failure  : eventos nulos cuentan como fallo

Ejecuta el replay externo exacto (replay_wave) sobre los checkpoints D=128
de wave_complex y wave_re (clave cell, donde viven los nulos) para delta=0
y toda la grilla. El brazo delta_nlms no tiene guardas (su erase es
sustractivo; EM directo) y queda fuera de esta descomposicion.

Salida: outputs/wave_mem/cluster_a_nulls.json
Uso: python tests/cluster_a_controls.py --device cpu [--smoke]
"""

import argparse
import json
import os
import statistics
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from tests.wave_mem_smoke128 import KeySpace, EPS_NULL, build_wave, fr_spec
from tests.o04b_run import replay_wave
from tests.wave_mem_n1 import make_fr_probe_dataset
import tests.common_smoke as CS


def conventions(correct, ratios, eps):
    """correct: lista bool por evento; ratios: norma residual relativa."""
    n = len(correct)
    nonnull = [c for c, r in zip(correct, ratios) if r > eps]
    null_frac = 1.0 - len(nonnull) / max(n, 1)
    raw = sum(correct) / max(n, 1)
    excluded = (sum(nonnull) / len(nonnull)) if nonnull else None
    failure = sum(nonnull) / max(n, 1)
    return {"A_raw": round(raw, 4),
            "B_excluded": (round(excluded, 4)
                           if excluded is not None else None),
            "C_failure": round(failure, 4),
            "null_frac": round(null_frac, 4), "n": n}


def run_seed(model, seed, device, deltas, n_events, eps):
    x, y, seqs, v_erased = make_fr_probe_dataset(
        n_events, seed=1000 + seed, n_pairs_range=(32, 48), kind="erased")
    res = {}
    for dlt in deltas:
        em, fuga, ratios, pred = replay_wave(
            model, x.to(device), y, seqs, v_erased, "clave", float(dlt),
            device, per_event=True, return_pred=True)
        correct = [int(p) == int(t) for p, t in zip(pred.tolist(),
                                                    y.tolist())]
        conv = conventions(correct, ratios, eps)
        conv["fuga"] = round(fuga[0], 4)
        conv["em_unguarded"] = round(em, 4)
        res[str(dlt)] = conv
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seeds", type=str, default="1,2,3,4,5")
    ap.add_argument("--deltas", type=str,
                    default="0.0,0.1,0.2,0.3,0.4,0.5")
    ap.add_argument("--n_events", type=int, default=320)
    ap.add_argument("--arms", type=str, default="wave_complex,wave_re")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", type=str,
                    default="outputs/wave_mem/cluster_a_nulls.json")
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    deltas = [float(d) for d in args.deltas.split(",")]
    arms = args.arms.split(",")

    out = {"variant": "cluster_a_nulls", "eps_null": EPS_NULL,
           "n_events": args.n_events,
           "conventions": {
               "A_raw": "EM sobre todos los eventos (convencion O05/TOST)",
               "B_excluded": "eventos con ratio<=eps excluidos",
               "C_failure": "eventos nulos cuentan como fallo"},
           "runs": {}}

    with KeySpace(48, 48, 105, 57):
        spec = fr_spec(60)
        for arm in arms:
            chan = "complex" if arm == "wave_complex" else "re"
            build = build_wave(chan)
            for seed in seeds:
                ck = (f"outputs/o5_{arm}/cache/"
                      f"forget_retrieval_seed{seed}"
                      f"_dm128_L2_ep{args.epochs}.pt")
                if not os.path.exists(ck):
                    print(f"  falta {ck} - skip", flush=True)
                    continue
                model = build(vocab_size=CS.VOCAB_FORGET_SIZE,
                              max_len=spec.max_seq_len, d_model=128,
                              n_layers=2)
                model.load_state_dict(
                    torch.load(ck, map_location=device)["model"])
                model.to(device).eval()
                label = f"{arm}_s{seed}"
                print(f"=== {label} ===", flush=True)
                res = {}
                for dlt in deltas:
                    res[str(dlt)] = run_seed(model, seed, device,
                                             [dlt], args.n_events,
                                             EPS_NULL)[str(dlt)]
                    c = res[str(dlt)]
                    print(f"  d={dlt}: A={c['A_raw']} B={c['B_excluded']} "
                          f"C={c['C_failure']} null_frac={c['null_frac']}",
                          flush=True)
                out["runs"][label] = res

    # resumen por convencion (promedio entre seeds, estado del arte: clave)
    summary = {}
    for dlt_str in res.keys() if res else []:
        pass
    for arm in arms:
        for dlt_s in map(str, deltas):
            rows = [out["runs"][f"{arm}_s{s}"][dlt_s]
                    for s in seeds
                    if f"{arm}_s{s}" in out["runs"] and dlt_s
                    in out["runs"][f"{arm}_s{s}"]]
            if not rows:
                continue
            summary[f"{arm}@{dlt_s}"] = {
                conv: round(statistics.mean(
                    r[conv] for r in rows if r[conv] is not None), 4)
                for conv in ("A_raw", "C_failure")}
    out["summary_mean_over_seeds"] = summary

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"escrito {args.out}", flush=True)


if __name__ == "__main__":
    main()
