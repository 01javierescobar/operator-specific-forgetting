"""Cluster A tarea #8: robustez de la celda nula fuera del gauge de fase.

La ley clave/complex = 0 vale para perturbacion de FASE GLOBAL (misma
direccion). Este experimento mide fuga y EM cuando el erase key cambia de
DIRECCION: (a) fase por-coordenada U(-pp,pp), (b) ruido complejo aditivo
de desvio pp. Si la celda deja de ser nula, el 'exact erasure' queda
delimitado al caso global-phase (respuesta a R1-M1 / R3).

Salida: outputs/wave_mem/key_perturbation_robustness.json
Uso: python tests/key_perturbation_robustness.py [--device cpu] [--smoke]
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

VARIANTS = [
    ("global_phase", 0.0),
    ("global_phase", 0.3),
    ("coord_phase", 0.1),
    ("coord_phase", 0.3),
    ("coord_phase", 0.5),
    ("additive", 0.05),
    ("additive", 0.15),
    ("additive", 0.30),
]


def run_variant(model, seed, device, pert, pp, n_events):
    x, y, seqs, v_erased = make_fr_probe_dataset(
        n_events, seed=1000 + seed, n_pairs_range=(32, 48), kind="erased")
    res = {}
    for cell in ("clave", "estado"):
        em, fuga, ratios, pred = replay_wave(
            model, x.to(device), y, seqs, v_erased, cell, float(pp if False else 0.0),
            device, per_event=True, return_pred=True,
            key_perturb=pert if pert != "global_phase" else None,
            pparam=pp)
        correct = [int(p) == int(t) for p, t in zip(pred.tolist(),
                                                    y.tolist())]
        nonnull = [c for c, r in zip(correct, ratios.tolist()) if r > EPS_NULL]
        res[cell] = {
            "fuga": round(fuga[0], 5),
            "em_raw": round(em, 4),
            "null_frac": round(1.0 - len(nonnull) / max(len(correct), 1), 4),
            "em_nonnull": (round(sum(nonnull) / len(nonnull), 4)
                           if nonnull else None),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--n_events", type=int, default=320)
    ap.add_argument("--arms", type=str, default="wave_complex,wave_re")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--out", type=str,
                    default="outputs/wave_mem/"
                            "key_perturbation_robustness.json")
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(",")]
    arms = args.arms.split(",")

    out = {"variant": "key_perturbation_robustness",
           "n_events": args.n_events, "seeds": seeds,
           "note": ("perturbacion aplicada al erase key (celda clave) y "
                    "comparada contra estado sin perturbar; global_phase "
                    "es el caso del paper"),
           "runs": {}}

    with KeySpace(48, 48, 105, 57):
        spec = fr_spec(args.epochs)
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
                for pert, pp in VARIANTS:
                    r = run_variant(model, seed, device, pert, pp,
                                    args.n_events)
                    res[f"{pert}@{pp}"] = r
                    print(f"  {pert}@{pp}: clave fuga={r['clave']['fuga']:.5f} "
                          f"em={r['clave']['em_raw']:.4f} "
                          f"null={r['clave']['null_frac']:.2f} | "
                          f"estado fuga={r['estado']['fuga']:.4f}")
                out["runs"][label] = res

    # resumen: fuga clave promedio entre seeds por variante/canal
    summary = {}
    for pert, pp in VARIANTS:
        key = f"{pert}@{pp}"
        for arm in arms:
            rows = [out["runs"][f"{arm}_s{s}"][key]["clave"]
                    for s in seeds if f"{arm}_s{s}" in out["runs"]]
            if not rows:
                continue
            summary[f"{arm}/{key}"] = {
                "clave_fuga_mean": round(statistics.mean(
                    r["fuga"] for r in rows), 5),
                "clave_em_mean": round(statistics.mean(
                    r["em_raw"] for r in rows), 4),
                "clave_null_frac_mean": round(statistics.mean(
                    r["null_frac"] for r in rows), 3)}
    out["summary"] = summary
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("escrito", args.out, flush=True)


if __name__ == "__main__":
    main()
