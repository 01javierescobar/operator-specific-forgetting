"""Descomposicion delta-method de Var(L) para estado/complex y clave/re @ d=0.5:
sd^2(L) ~ [Var(Q) + E[L]^2 Var(Ssum) - 2 E[L] Cov(Q, Ssum)] / Sbar^2.
Mide cada pieza empiricamente para priorizar el calculo cerrado."""

import math
import statistics

import torch

D = 64
N = 20
K = 48


def draw(seed):
    g = torch.Generator().manual_seed(seed)
    th = torch.rand(N, D, generator=g) * 2 * math.pi
    w = torch.complex(torch.cos(th), torch.sin(th))
    v = torch.randn(N, D, generator=g)
    return w, v


def ph(x):
    return torch.complex(torch.tensor(math.cos(x)), torch.tensor(math.sin(x)))


def probe_q(w, v, cell, dlt, ch):
    p = ph(dlt)
    num = 0.0
    den = 0.0
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
        wh = torch.einsum("d,de->e", torch.conj(ek), M) / D
        if ch == "re":
            wh = torch.complex(wh.real, torch.zeros_like(wh.real))
        Mp = M - sk.unsqueeze(-1) * wh.unsqueeze(0)
        r = torch.einsum("d,de->e", torch.conj(wj), Mp) / D
        if ch == "re":
            r = r.real
        Lj = float((r.abs().pow(2).sum() / vj.pow(2).sum()).item())
        num += Lj * vj.pow(2).sum().item()
        den += vj.pow(2).sum().item()
    return num, den


for cell, ch, dlt in (("estado", "complex", 0.5), ("clave", "re", 0.5)):
    Qs = []
    Ss = []
    for k in range(K):
        w, v = draw(30000 + k)
        q, s = probe_q(w, v, cell, dlt, ch)
        Qs.append(q)
        Ss.append(s)
    Ls = [q / s for q, s in zip(Qs, Ss)]
    Qm = statistics.mean(Qs)
    Sm = statistics.mean(Ss)
    var_Q = statistics.pvariance(Qs)
    var_S = statistics.pvariance(Ss)
    covQS = statistics.mean([(q - Qm) * (s - Sm) for q, s in zip(Qs, Ss)])
    Lm = statistics.mean(Ls)
    sdL = statistics.pstdev(Ls)
    contrib_Q = math.sqrt(var_Q) / Sm
    contrib_S = abs(Lm) * math.sqrt(var_S) / Sm
    contrib_cov = abs(2 * Lm * covQS) / Sm ** 2
    pred_quad = math.sqrt(var_Q / Sm ** 2 + (Lm ** 2) * var_S / Sm ** 2 - 2 * Lm * covQS / Sm ** 2)
    print(f"=== {cell}/{ch} @ delta={dlt} ===")
    print(f"  sd(L) medido          = {sdL:.5f}")
    print(f"  sd(Q)/Sbar            = {contrib_Q:.5f}")
    print(f"  E[L]*sd(S)/Sbar       = {contrib_S:.5f}")
    print(f"  |2EL*cov|/Sbar^2      = {contrib_cov:.5f}")
    print(f"  prediccion cuadratica = {pred_quad:.5f}")
    print(f"  E[L]={Lm:.4f}  Qmean={Qm:.1f}  Sbar={Sm:.1f}")
