"""MC: sigma de L entre codebooks a D in {32,64,128} con c fijado ~0.297.
Guia para la derivacion analitica de Var(L): la teoria debe reproducir
estas desviaciones estandar (y su escalado con D)."""

import math
import statistics

import torch

GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
K = 48
C_TARGET = 19 / 64


def make_synth(seed, D, n):
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(n, D, generator=g) * 2.0 * math.pi
    w = torch.complex(torch.cos(theta), torch.sin(theta))
    v = torch.randn(n, D, generator=g)
    return w, v


def ph(dlt):
    return torch.complex(torch.tensor(math.cos(dlt)), torch.tensor(math.sin(dlt)))


def fuga(Ma, wj, vj, channel, D):
    r = torch.einsum("d,de->e", torch.conj(wj), Ma) / D
    if channel == "re":
        r = torch.complex(r.real, torch.zeros_like(r.real))
    return float((r.abs().pow(2).sum() / vj.pow(2).sum()).item())


def run_cb(seed, D, n):
    w, v = make_synth(seed, D, n)
    out = {}
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
                    L = fuga(Mp, wj, vj, ch, D)
                    acc[ch] += L * vj.pow(2).sum().item()
                    den[ch] += vj.pow(2).sum().item()
            out[(cell, ch, dlt)] = (acc["complex"] / den["complex"], acc["re"] / den["re"])
    # fix: dict overwritten per delta; store properly below instead
    return out


def run_cb_full(seed, D, n):
    w, v = make_synth(seed, D, n)
    out = {}
    for dlt in GRID:
        p = ph(dlt)
        for cell in ("clave", "estado"):
            acc = {"complex": [], "re": []}
            den = {"complex": 0.0, "re": 0.0}
            num = {"complex": 0.0, "re": 0.0}
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
                    L = fuga(Mp, wj, vj, ch, D)
                    num[ch] += L * vj.pow(2).sum().item()
                    den[ch] += vj.pow(2).sum().item()
            acc["complex"] = num["complex"] / den["complex"]
            acc["re"] = num["re"] / den["re"]
            out[(cell, dlt)] = acc
    return out


def main():
    for D in (32, 64, 128):
        n = round(C_TARGET * D) + 1
        c = (n - 1) / D
        samples = {("clave", "complex"): {dl: [] for dl in GRID},
                   ("estado", "complex"): {dl: [] for dl in GRID},
                   ("clave", "re"): {dl: [] for dl in GRID},
                   ("estado", "re"): {dl: [] for dl in GRID}}
        for k in range(K):
            res = run_cb_full(10000 + k, D, n)
            for key in samples:
                for dl in GRID:
                    ch = key[1]
                    samples[key][dl].append(res[(key[0], dl)][ch])
        print(f"=== D={D} n={n} c={c:.4f} K={K} ===")
        for (cell, ch), by_dl in samples.items():
            row = []
            for dl in GRID:
                vals = by_dl[dl]
                row.append(f"d={dl}: sd={statistics.pstdev(vals):.5f}")
            print(f"  {cell}/{ch}: " + " | ".join(row))
        print()


if __name__ == "__main__":
    main()
