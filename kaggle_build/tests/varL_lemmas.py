"""Lemas L3/L4: media y varianza de Y_j por item, y matriz de covarianza
Cov(Y_j, Y_i) para estado/complex @ delta=0.5. Objetivo: identificar la
estructura de Var(Q_x) = sum_j Var(Y_j) + sum_{j!=i} Cov(Y_j,Y_i)."""

import math
import statistics

import torch

D = 64


def draw(seed, n):
    g = torch.Generator().manual_seed(seed)
    th = torch.rand(n, D, generator=g) * 2 * math.pi
    w = torch.complex(torch.cos(th), torch.sin(th))
    v = torch.randn(n, D, generator=g)
    return w, v


def ph(x):
    return torch.complex(torch.tensor(math.cos(x)), torch.tensor(math.sin(x)))


def y_vector(w, v, dlt):
    p = ph(dlt)
    ys = []
    for j in range(w.shape[0]):
        wj = w[j].clone()
        vj = v[j]
        wd = wj * p
        wdr = w.clone()
        wdr[j] = wd
        acc_c = 0.0
        for ch in ("complex",):
            M = torch.einsum("nd,ne->de", wdr, v.to(torch.complex64))
            wh = torch.einsum("d,de->e", torch.conj(wd), M) / D
            Mp = M - wj.unsqueeze(-1) * wh.unsqueeze(0)
            r = torch.einsum("d,de->e", torch.conj(wj), Mp) / D
            tgt = (complex(math.cos(dlt) - 1, math.sin(dlt))) * vj.to(torch.complex64)
            rc = r - tgt
            acc_c += float((rc.abs().pow(2).sum() / vj.pow(2).sum()).item())
        ys.append(acc_c)
    return ys


def main():
    for n in (10, 20, 40, 80):
        K = 96
        y_all = []
        for k in range(K):
            w, v = draw(50000 + k, n)
            y_all.append(y_vector(w, v, 0.5))
        # medias y varianzas por posicion promediadas sobre reps y items
        flat = [y for ys in y_all for y in ys]
        m = statistics.mean(flat)
        vr = statistics.pvariance(flat)
        # covarianza entre items distintos promediada sobre pares y reps
        covs = []
        for k in range(K):
            ys = y_all[k]
            mm = statistics.mean(ys)
            for a in range(n):
                for b in range(a + 1, n):
                    covs.append((ys[a] - mm) * (ys[b] - mm))
        cov_avg = statistics.mean(covs)
        # prediccion: Var(Q_x) = n*Var(Y_j) + n*(n-1)*Cov_promedio
        var_q_pred = n * vr + n * (n - 1) * cov_avg
        print(f"n={n}: mean(Y)={m:.5f}  Var(Y_j)={vr:.6f}  sd={math.sqrt(vr):.5f}")
        print(f"       Cov(Y_j,Y_i) promedio={cov_avg:.6f}  (rel a Var: {cov_avg/vr:+.3f})")
        print(f"       sd(Qx)/|g|^2 pred = {math.sqrt(var_q_pred):.4f}; "
              f"objetivo MC: sd(Qx)/|g|^2 = {0.00909 * (20 * 64) if n == 20 else float('nan'):.1f}" if n == 20 else "")
        print(f"       sd(L) implícito = {math.sqrt(var_q_pred) / (n * D) * abs(1 - complex(math.cos(0.5), math.sin(0.5))) ** 2:.5f}")


if __name__ == "__main__":
    main()
