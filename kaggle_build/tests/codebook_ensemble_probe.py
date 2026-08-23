"""Probe multi-codebook (round-2 review M4):
Las leyes cerradas son esperanzas sobre el ensamble de claves aleatorias;
los experimentos entrenados usan UN solo codebook (seed 0). Este script
repite el probe sintetico 2x2xdelta sobre K codebooks independientes y
reporta la dispersion entre codebooks de L por celda/canal/delta.

Sin entrenamiento: emulacion exacta del operador (mismo esquema que
tests/o04_laws_verify.py). Salida: outputs/wave_mem/codebook_ensemble.json
"""

import json
import math
import os
import statistics

import torch

D = 64
N = 20
GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
K_CODEBOOKS = 8

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def make_synth(seed, D=D, n=N):
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(n, D, generator=g) * 2.0 * math.pi
    w = torch.complex(torch.cos(theta), torch.sin(theta))
    v = torch.randn(n, D, generator=g)
    return w, v


def ph(dlt):
    return torch.complex(torch.tensor(math.cos(dlt)), torch.tensor(math.sin(dlt)))


def fuga(Ma, wj, vj, channel):
    r = torch.einsum("d,de->e", torch.conj(wj), Ma) / D
    if channel == "re":
        r = torch.complex(r.real, torch.zeros_like(r.real))
    return float((r.abs().pow(2).sum() / vj.pow(2).sum()).item())


def run_codebook(cb_seed):
    w, v = make_synth(cb_seed)
    n = w.shape[0]
    c = (n - 1) / D
    out = {"clave": {}, "estado": {}}
    for dlt in GRID:
        p = ph(dlt)
        for cell in ("clave", "estado"):
            acc = {"complex": 0.0, "re": 0.0}
            den = {"complex": 0.0, "re": 0.0}
            for j in range(n):
                wj = w[j].clone()
                vj = v[j]
                wd = wj * p
                if cell == "clave":
                    M = torch.einsum("nd,ne->de", w, v.to(torch.complex64))
                    est_key = wd
                    sub_key = wd if True else wj
                else:
                    w_drift = w.clone()
                    w_drift[j] = wd
                    M = torch.einsum("nd,ne->de", w_drift, v.to(torch.complex64))
                    est_key = wd
                    sub_key = wj
                for ch in ("complex", "re"):
                    wh = torch.einsum("d,de->e", torch.conj(est_key), M) / D
                    if ch == "re":
                        wh = torch.complex(wh.real, torch.zeros_like(wh.real))
                    Mp = M - sub_key.unsqueeze(-1) * wh.unsqueeze(0)
                    L = fuga(Mp, wj, vj, ch)
                    acc[ch] += L * vj.pow(2).sum().item()
                    den[ch] += vj.pow(2).sum().item()
            out[cell][str(dlt)] = {
                "complex": acc["complex"] / den["complex"],
                "re": acc["re"] / den["re"],
                "c": c,
            }
    return out


def laws(cell, ch, dlt, c):
    s = math.sin(dlt)
    if cell == "clave" and ch == "complex":
        return 0.0
    if cell == "estado" and ch == "complex":
        return 2 * (1 - math.cos(dlt)) * (1 + c)
    if cell == "clave" and ch == "re":
        return s ** 4 + c * (1 - math.cos(2 * dlt)) / 4
    return (1 - math.cos(dlt)) ** 2 + c * (1 - cosh(dlt))


def cosh(dlt):
    return math.cos(dlt)


def main():
    per_cb = []
    for k in range(K_CODEBOOKS):
        res = run_codebook(k)
        per_cb.append(res)
        mx = 0.0
        for cell in ("clave", "estado"):
            for dl in map(str, GRID):
                for ch in ("complex", "re"):
                    c = res[cell][dl]["c"]
                    e = abs(res[cell][dl][ch] - laws(cell, ch, float(dl), c))
                    mx = max(mx, e)
        print(f"codebook {k}: max |L - ley| = {mx:.5f}")

    summary = {}
    for cell in ("clave", "estado"):
        for dl in map(str, GRID):
            for ch in ("complex", "re"):
                vals = [r[cell][dl][ch] for r in per_cb]
                c = per_cb[0][cell][dl]["c"]
                pred = laws(cell, ch, float(dl), c)
                summary[f"{cell}/{ch}@{dl}"] = {
                    "pred": round(pred, 5),
                    "mean": round(statistics.mean(vals), 5),
                    "sd": round(statistics.pstdev(vals), 5),
                    "min": round(min(vals), 5),
                    "max": round(max(vals), 5),
                }
                print(f"{cell}/{ch}@{dl}: pred={pred:.4f} mean={statistics.mean(vals):.4f} sd={statistics.pstdev(vals):.4f} [{min(vals):.4f},{max(vals):.4f}]")

    out_path = os.path.join(ROOT, "outputs", "wave_mem", "codebook_ensemble.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "variant": "codebook_ensemble",
            "d_model": D,
            "n_pairs": N,
            "grid": list(GRID),
            "n_codebooks": K_CODEBOOKS,
            "per_codebook": per_cb,
            "summary": summary,
        }, f, indent=1)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
