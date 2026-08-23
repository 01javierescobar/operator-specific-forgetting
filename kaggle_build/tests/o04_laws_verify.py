"""Verificacion pre-pre-registro O04 (enmiendas del auditor):
1. Leyes cerradas del diseno 2x2xdelta sobre la EMULACION EXACTA del
   operador entrenado (erase Re-truncado del canal re, lectura Re):
   - complex reread: 0 (proyector invariante)
   - complex represent: (n-1)/D + 2(1-cos d)
   - re reread: ? (derivacion del agente: sin^4 d + (n-1)/D (1-cos2d)/4;
     el auditor infiere 0 plano SIN verificacion numerica)
   - re represent: (n-1)/(2D) + (1-cos d)^2
   Grid d in {0,.1,.2,.3,.4,.5}, D=64 (regimen entrenado), n=24.
2. Condicion de estabilidad LMS del auditor (tabla t=120): ganancia
   |1 - beta||k||^2| vs norma de S. Verifica el teorema "lazo abierto
   incondicionalmente estable vs lazo cerrado con beta||k||^2 < 2".
Salida: outputs/wave_mem/o04_laws_verify.json
"""

import json
import math
import random

import torch

D = 64
N = 24
GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
TRIALS = 20


def make_synth(seed, D=D, n=N):
    rng = random.Random(seed)
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(n, D, generator=g) * 2.0 * math.pi
    w = torch.complex(torch.cos(theta), torch.sin(theta))          # n x D
    v = torch.randn(n, D)                                          # reales
    M = torch.einsum('nd,ne->de', w, v.to(torch.complex64))        # D x D
    return w, v, M


def ph(dlt):
    return torch.complex(torch.tensor(math.cos(dlt)), torch.tensor(math.sin(dlt)))


def erase_reread(M, wd, channel):
    wh = torch.einsum('d,de->e', torch.conj(wd), M) / D
    if channel == 're':
        wh = torch.complex(wh.real, torch.zeros_like(wh.real))
    return M - wd.unsqueeze(-1) * wh.unsqueeze(0)


def erase_represent(M, wd, vj):
    return M - wd.unsqueeze(-1) * vj.unsqueeze(0)


def fuga(Ma, wj, vj, channel):
    r = torch.einsum('d,de->e', torch.conj(wj), Ma) / D
    if channel == 're':
        r = r.real
    return float((r.abs().pow(2).sum() / vj.pow(2).sum().clamp(min=1e-12)).item())


def verify_laws():
    out = {}
    for channel in ('complex', 're'):
        row = {}
        for dlt in GRID:
            acc = {'reread': [], 'represent': []}
            for tr in range(TRIALS):
                w, v, M = make_synth(1000 + int(dlt * 10) * 100 + tr)
                j = N // 2
                wd = ph(dlt) * w[j]
                acc['reread'].append(fuga(erase_reread(M, wd, channel), w[j], v[j], channel))
                acc['represent'].append(fuga(erase_represent(M, wd, v[j]), w[j], v[j], channel))
            c = (N - 1) / D
            row[str(dlt)] = {
                'reread': round(sum(acc['reread']) / TRIALS, 6),
                'represent': round(sum(acc['represent']) / TRIALS, 6),
                'pred_reread_complex': 0.0,
                'pred_represent_complex': round(c + 2 * (1 - math.cos(dlt)), 6),
                'pred_represent_re': round(c / 2 + (1 - math.cos(dlt)) ** 2, 6),
                'pred_reread_re_sin4': round(
                    math.sin(dlt) ** 4 + c * (1 - math.cos(2 * dlt)) / 4, 6),
            }
        out[channel] = row
    return out


def verify_stability():
    """Iteracion del write en S (vector fila), n claves aleatorias ciclando,
    t=120 pasos. Ganancia por direccion = |1 - beta||k||^2|."""
    def run(beta, scale, corrective, t=120, n_keys=8, seed=7):
        rng = random.Random(seed)
        K = [[rng.gauss(0, math.sqrt(scale / 64)) for _ in range(64)]
             for _ in range(n_keys)]
        V = [[rng.gauss(0, 1) for _ in range(64)] for _ in range(n_keys)]
        S = [0.0] * 64
        for t_i in range(t):
            k_i = K[t_i % n_keys]
            v_i = V[t_i % n_keys]
            if corrective:
                kS = sum(k_i[d] * S[d] for d in range(64))
                for d in range(64):
                    S[d] += beta * (v_i[d] - kS) * k_i[d]
            else:
                for d in range(64):
                    S[d] += k_i[d] * v_i[d]
        return sum(x * x for x in S) ** 0.5

    gain = lambda beta, scale: abs(1 - beta * scale)
    return {
        'correctivo_beta1_kk8': {'gain': gain(1, 8), 'normS_t120': run(1, 8, True)},
        'correctivo_beta01_kk8': {'gain': gain(0.1, 8), 'normS_t120': run(0.1, 8, True)},
        'correctivo_beta1_kk15': {'gain': gain(1, 1.5), 'normS_t120': run(1, 1.5, True)},
        'superposicion_beta1': {'gain': 'sin lazo (abierto)', 'normS_t120': run(1, 8, False)},
        'teorema': ('correctivo = lazo cerrado, exige beta||k||^2 < 2; '
                    'superposicion = lazo abierto, estable incondicional'),
    }


def main():
    out = {
        'variant': 'o04_laws_verify', 'D': D, 'n': N, 'grid': list(GRID),
        'trials': TRIALS,
        'leyes': verify_laws(),
        'estabilidad': verify_stability(),
    }
    import os
    os.makedirs('outputs/wave_mem', exist_ok=True)
    with open('outputs/wave_mem/o04_laws_verify.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, default=str)
    for ch in ('complex', 're'):
        print(f'=== canal {ch} ===')
        for dlt, r in out['leyes'][ch].items():
            print(f" d={dlt}: rr={r['reread']} rp={r['represent']} "
                  f"pred_rr_cplx=0 pred_rp_cplx={r['pred_represent_complex']} "
                  f"pred_rp_re={r['pred_represent_re']} "
                  f"pred_rr_re_sin4={r['pred_reread_re_sin4']}")
    print('=== estabilidad ===')
    for k, v in out['estabilidad'].items():
        print(f' {k}: {v}')
    print('Escrito: outputs/wave_mem/o04_laws_verify.json')


if __name__ == '__main__':
    main()