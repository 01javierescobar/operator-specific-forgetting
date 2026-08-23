"""Barrido en n a D fijo: discrimina el escalado de sd(L) con n
(constante vs sqrt(n) vs n) para fijar la estructura de Var(Q)."""

import math
import statistics

import torch

GRID = (0.0, 0.3, 0.5)
K = 48
D = 64
NS = (10, 20, 40, 80)


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


def run_cb_full(seed, D, n):
    w, v = make_synth(seed, D, n)
    out = {}
    for dlt in GRID:
        p = ph(dlt)
        for cell in ("clave", "estado"):
            num = {"complex": 0.0, "re": 0.0}
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
                    num[ch] += L * vj.pow(2).sum().item()
                    den[ch] += vj.pow(2).sum().item()
            out[(cell, dlt)] = {"complex": num["complex"] / den["complex"], "re": num["re"] / den["re"]}
    return out


def main():
    print(f"D={D} fijo, barrido en n, K={K} codebooks")
    for n in NS:
        samples = {(cell, dl): [] for cell in ("clave", "estado") for dl in GRID}
        c = (n - 1) / D
        for k in range(K):
            res = run_cb_full(20000 + k, D, n)
            for (cell, dl) in list(samples.keys()):
                samples[(cell, dl)].append(res[(cell, dl)]["complex"])
        print(f"--- n={n} c={c:.4f} ---")
        for cell in ("clave", "estado"):
            parts = []
            for dl in GRID:
                vals = samples[(cell, dl)]
                parts.append(f"d={dl}: L={statistics.mean(vals):.4f} sd={statistics.pstdev(vals):.5f}")
            print(f"  {cell}/complex: " + " | ".join(parts))


if __name__ == "__main__":
    main()
