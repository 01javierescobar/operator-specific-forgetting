"""Descomposicion de varianza: claves vs valores.
A: claves re-sorteadas, valores fijos (una sola tirada).
B: valores re-sorteados, claves fijas.
C: ambos (baseline). Si A domina -> derivar condicionando en valores.
Si B domina -> los momentos gaussianos de v mandan."""

import math
import statistics

import torch

GRID = (0.0, 0.3, 0.5)
K = 48
D = 64
N = 20


def ph(dlt):
    return torch.complex(torch.tensor(math.cos(dlt)), torch.tensor(math.sin(dlt)))


def draw(seed, D, n):
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(n, D, generator=g) * 2.0 * math.pi
    w = torch.complex(torch.cos(theta), torch.sin(theta))
    v = torch.randn(n, D, generator=g)
    return w, v


def fuga(Ma, wj, vj, channel):
    r = torch.einsum("d,de->e", torch.conj(wj), Ma) / D
    if channel == "re":
        r = torch.complex(r.real, torch.zeros_like(r.real))
    return float((r.abs().pow(2).sum() / vj.pow(2).sum()).item())


def probe(w, v):
    out = {}
    for dlt in GRID:
        p = ph(dlt)
        for cell in ("clave", "estado"):
            num = {"complex": 0.0, "re": 0.0}
            den = {"complex": 0.0, "re": 0.0}
            for j in range(N):
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
                    L = fuga(Mp, wj, vj, ch)
                    num[ch] += L * vj.pow(2).sum().item()
                    den[ch] += vj.pow(2).sum().item()
            out[(cell, ch, dlt)] = num["complex"] / den["complex"] if False else (num["complex"] / den["complex"], num["re"] / den["re"])
    return out


def collect(mode):
    samples = {}
    _, v_fixed = draw(999999, D, N)
    w_fixed, _ = draw(888888, D, N)
    for k in range(K):
        wk, vk = draw(30000 + k, D, N)
        if mode == "A":
            w, v = wk, v_fixed
        elif mode == "B":
            w, v = w_fixed, vk
        else:
            w, v = wk, vk
        res = probe(w, v)
        for key, (lc, lr) in res.items():
            samples.setdefault(key, {"complex": [], "re": []})
            samples[key]["complex"].append(lc)
            samples[key]["re"].append(lr)
    return samples


if __name__ == "__main__":
    for mode, label in (("A", "solo claves varian"), ("B", "solo valores varian"), ("C", "ambos")):
        smp = collect(mode)
        print(f"=== {mode}: {label} ===")
        for cell in ("clave", "estado"):
            for dl in GRID:
                lc = smp[(cell, "re", dl)]["complex"]
                lr = smp[(cell, "re", dl)]["re"]
                print(f"  {cell}@d={dl}: complex mean={statistics.mean(lc):.4f} sd={statistics.pstdev(lc):.5f} | "
                      f"re mean={statistics.mean(lr):.4f} sd={statistics.pstdev(lr):.5f}")
