"""Grilla de calibracion k-sigma para bandas del Paper I.

Mide sd(L) del ensamble (claves + valores iid) en una grilla (c, delta)
para D=64 y D=128, y produce kappa_hat(c) = sd(L)*D/A(delta) por celda/canal.
Salida: outputs/wave_mem/varL_calibration.json
"""

import json
import math
import os
import statistics

import torch

D_LIST = (64, 128)
C_TARGETS = (0.0781, 0.1563, 0.2344, 0.2969, 0.3750, 0.4688, 0.6094, 0.7813, 1.0313)
DELTA_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
K = 48

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def draw(seed, D, n):
    g = torch.Generator().manual_seed(seed)
    th = torch.rand(n, D, generator=g) * 2 * math.pi
    w = torch.complex(torch.cos(th), torch.sin(th))
    v = torch.randn(n, D, generator=g)
    return w, v


def ph(x):
    return torch.complex(torch.tensor(math.cos(x)), torch.tensor(math.sin(x)))


def A_factor(cell, ch, dlt):
    if cell == "estado" and ch == "complex":
        return 2 * (1 - math.cos(dlt))
    if cell == "clave" and ch == "re":
        return math.sin(dlt) ** 4 / max(1e-9, 1) + 0.0  # slope handled via c below
    return None


def run_point(D, n, seed_base):
    w_list = [draw(seed_base + k, D, n) for k in range(K)]
    acc = {}
    for ci in range(K):
        w, v = w_list[ci]
        for dlt in DELTA_GRID:
            p = ph(dlt)
            for cell in ("clave", "estado"):
                num = {}
                den = {}
                for j in range(n):
                    wj = w[j].clone()
                    vj = v[j]
                    wd = wj * p
                    if cell == "clave":
                        M = torch.einsum("nd,ne->de", w, v.to(torch.complex64))
                        ek, sk = wd, wd
                    else:
                        wdr = w.clone()
                        wdr[j] = wd
                        M = torch.einsum("nd,ne->de", wdr, v.to(torch.complex64))
                        ek, sk = wd, wj
                    for ch in ("complex", "re"):
                        wh = torch.einsum("d,de->e", torch.conj(ek), M) / D
                        if ch == "re":
                            wh = torch.complex(wh.real, torch.zeros_like(wh.real))
                        Mp = M - sk.unsqueeze(-1) * wh.unsqueeze(0)
                        r = torch.einsum("d,de->e", torch.conj(wj), Mp) / D
                        if ch == "re":
                            r = r.real
                        L = float((r.abs().pow(2).sum() / vj.pow(2).sum()).item())
                        key = (cell, ch, dlt)
                        num[key] = num.get(key, 0.0) + L * vj.pow(2).sum().item()
                        den[key] = den.get(key, 0.0) + vj.pow(2).sum().item()
                for key in num:
                    acc.setdefault(key, []).append(num[key] / den[key])
    return acc


def main():
    out = {"grid": {}, "kappa": {}}
    for D in D_LIST:
        for c_t in C_TARGETS:
            n = round(c_t * D) + 1
            c = (n - 1) / D
            seed_base = 70000 + int(c_t * 10000) + D
            acc = run_point(D, n, seed_base)
            for (cell, ch, dlt), vals in sorted(acc.items()):
                if len(vals) < K:
                    continue
                sd = statistics.pstdev(vals)
                mean = statistics.mean(vals)
                out["grid"][f"D{D}/n{n}/c{c:.4f}/{cell}/{ch}/d{dlt}"] = {
                    "mean": round(mean, 5), "sd": round(sd, 6)}
                if cell == "estado" and ch == "complex" and dlt > 0:
                    A = 2 * (1 - math.cos(dlt))
                    out["kappa"][f"D{D}/c{c:.4f}/d{dlt}"] = round(sd * D / A, 4)
            print(f"D={D} c={c:.4f} done")
    out["meta"] = {"K": K, "delta_grid": list(DELTA_GRID),
                   "c_targets": list(C_TARGETS), "note": "kappa = sd(L)*D/(2(1-cos d)) estado/complex"}
    path = os.path.join(ROOT, "outputs", "wave_mem", "varL_calibration.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)

    ks = [v for k, v in out["kappa"].items() if k.startswith("D64")]
    print("kappa D64 summary:", "min=%.3f max=%.3f" % (min(ks), max(ks)))
    ks128 = [v for k, v in out["kappa"].items() if k.startswith("D128")]
    print("kappa D128 summary:", "min=%.3f max=%.3f" % (min(ks128), max(ks128)))


if __name__ == "__main__":
    main()
