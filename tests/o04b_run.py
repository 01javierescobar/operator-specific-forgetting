"""Runner O04b (reconciliacion 2x2 del auditor): probe de drift
canal {complex, re} x drift {clave, estado} x delta sobre modelos wave
entrenados (sin reentrenar) + verificacion sintetica de las 4 celdas bajo
2 convenciones de medicion. Pre-registro: logs/O04b_test.md.

Inyeccion por celda (explicita, uniforme):
- clave: items escritos limpios; erase en t=pos_forget+1 lee
  conj(e^{id}w).M/D (Re truncado en canal re) y remueve e^{id}w (x) wh.
- estado: SOLO el item olvidado (pending_id == clave olvidada) se escribe
  con phasor e^{id}w (los demas limpios); el erase lee con conj(e^{id}w)
  (el drift se cancela en la estimacion) y remueve con la clave LIMPIA w
  (reloj actual).
- Readout de la query SIEMPRE con clave limpia (Re truncado en canal re).

Convenciones: 'fiel' = Re truncado en wh Y en el readout (lo que ve el
head entrenado; gate d=128); 'auditor' = Re solo en wh (readout complejo;
numeros del auditor 0.289/0.0870).

Salida: outputs/wave_mem/o04b.json
"""

import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from tests.common_smoke import ANSWER_ID, FORGET_ID, QUERY_ID
from tests.wave_mem_n1 import build_wave, make_fr_probe_dataset, v_true_for
from prototypes.wave_mem.model import WaveMemLM, MARKER_MIN

DELTAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
N_EVENTS = 320


def law(cell, channel, dlt, c, conv):
    """Leyes cerradas por celda. conv: 'fiel' | 'auditor'."""
    if channel == 'complex':
        return 0.0 if cell == 'clave' else 2 * (1 - math.cos(dlt)) * (1 + c)
    if conv == 'fiel':
        if cell == 'clave':
            return math.sin(dlt) ** 4 + c * (1 - math.cos(2 * dlt)) / 4
        return (1 - math.cos(dlt)) ** 2 + c * (1 - math.cos(dlt))
    if cell == 'clave':
        return math.sin(dlt) ** 2 + c / 2
    return 2 * (1 - math.cos(dlt)) + c * (3 - 2 * math.cos(dlt)) / 2


def replay_wave(model, x, y, seqs, v_erased, cell, dlt, device, per_event=False,
                return_pred=False, key_perturb=None, pparam=0.0):
    """Replay externo exacto del forward wave (memoria externa, sin
    decode_step del modelo) con inyeccion por celda. Devuelve EM y fuga
    POOLED por capa. Con per_event=True devuelve ademas los ratios de
    potencia residual por evento (capa 0) para la guarda de residual nulo
    (O05, adicion del auditor). Con return_pred=True devuelve tambien el
    vector de predicciones por evento (cluster_a_controls: convenciones
    de nulos). Flags independientes y retrocompatibles."""
    B, T = x.shape
    is_re = model.read_proj == 're'
    D = model.d_model
    cb = model.codebook
    ph = torch.complex(torch.tensor(math.cos(dlt)), torch.tensor(math.sin(dlt)))
    pos_forget = torch.tensor([s.index(FORGET_ID) for s in seqs], dtype=torch.long,
                              device=device)
    pos_qkey = torch.tensor([s.index(QUERY_ID) + 1 for s in seqs], dtype=torch.long,
                            device=device)
    pos_ans = torch.tensor([s.index(ANSWER_ID) for s in seqs], dtype=torch.long,
                           device=device)
    tok_key = torch.tensor([s[int(pos_forget[i]) + 1] for i, s in enumerate(seqs)],
                           dtype=torch.long, device=device)
    M = [torch.zeros(B, D, D, dtype=torch.complex64, device=device)
         for _ in range(model.n_layers)]
    r_persist = [torch.zeros(B, D, dtype=torch.complex64, device=device)
                 for _ in range(model.n_layers)]
    prev = torch.zeros(B, dtype=torch.long, device=device)
    expect = torch.zeros(B, dtype=torch.bool, device=device)
    pending = torch.zeros(B, dtype=torch.long, device=device)
    pred = torch.zeros(B, dtype=torch.long, device=device)
    with torch.no_grad():
        for t in range(T):
            tok = x[:, t]
            non_marker = tok >= MARKER_MIN
            write = expect & non_marker
            erase = (prev == FORGET_ID) & non_marker
            read = (prev == QUERY_ID) & non_marker
            if write.any():
                vts = v_true_for(model, tok)
                wv = cb[pending]
                if cell == 'estado':
                    fg = (pending == tok_key).unsqueeze(-1)
                    wv = torch.where(fg, ph * wv, wv)
                for b in range(model.n_layers):
                    M[b][write] += wv[write].unsqueeze(-1) * vts[b][write].unsqueeze(1)
            if erase.any():
                wd = ph * cb[tok]
                if key_perturb is not None:
                    pp = pparam
                    if key_perturb == 'coord_phase':
                        # fase por-coordenada U(-pp,pp): cambia la direccion
                        noise_ph = torch.exp(1j * ((torch.rand_like(
                            wd[erase].real) * 2 - 1) * pp))
                        wd_eff = wd.clone()
                        wd_eff[erase] = wd[erase] * noise_ph
                    elif key_perturb == 'additive':
                        # ruido complejo gaussiano; amplitud preservada
                        g = (torch.randn_like(wd[erase].real)
                             + 1j * torch.randn_like(wd[erase].real)) \
                            * (pp / math.sqrt(2.0))
                        raw = wd[erase] + g
                        norm_keep = (wd[erase].abs().pow(2).sum(dim=1)
                                     ).sqrt().unsqueeze(-1).clamp(min=1e-12)
                        newn = raw.abs().pow(2).sum(dim=1).sqrt().unsqueeze(-1).clamp(min=1e-12)
                        wd_eff = wd.clone()
                        wd_eff[erase] = raw / newn * norm_keep
                    else:
                        raise ValueError(key_perturb)
                else:
                    wd_eff = wd
                for b in range(model.n_layers):
                    wh = torch.einsum('bd,bdj->bj', torch.conj(wd_eff)[erase], M[b][erase]) / D
                    if is_re:
                        wh = torch.complex(wh.real, torch.zeros_like(wh.real))
                    outer = wd_eff if cell == 'clave' else cb[tok]
                    M[b][erase] -= outer[erase].unsqueeze(-1) * wh.unsqueeze(1)
            if read.any():
                for b in range(model.n_layers):
                    rr = torch.einsum('bd,bdj->bj', torch.conj(cb[tok])[read], M[b][read]) / D
                    if is_re:
                        rr = torch.complex(rr.real, torch.zeros_like(rr.real))
                    rp = r_persist[b].clone()
                    rp[read] = rr
                    r_persist[b] = rp
            if (pos_ans == t).any():
                am = pos_ans == t
                h = model.embedding(tok)
                for b, block in enumerate(model.blocks):
                    rb = r_persist[b]
                    feat = torch.cat([rb.real, rb.imag], dim=-1) if not is_re else rb.real
                    h = h + block.out_proj(feat)
                logits = model.head(model.norm(h))
                p = pred.clone()
                p[am] = logits[am].argmax(dim=-1)
                pred = p
            prev = tok
            expect = non_marker & ~expect
            pending = torch.where(non_marker & ~write, tok, pending)
    em = float((pred == y.to(device)).float().mean().item())
    vts = v_true_for(model, v_erased.to(device))
    fuga = []
    for b in range(model.n_layers):
        r = r_persist[b]
        v = vts[b]
        if is_re:
            r = r.real
            v = v.real
        fuga.append(float(r.abs().pow(2).sum().item() /
                          v.abs().pow(2).sum().clamp(min=1e-12).item()))
    if per_event:
        r0 = r_persist[0]
        v0 = vts[0]
        if is_re:
            r0 = r0.real
            v0 = v0.real
        ratios = (r0.abs().pow(2).sum(dim=1) /
                  v0.abs().pow(2).sum(dim=1).clamp(min=1e-12))
        if return_pred:
            return em, fuga, ratios, pred
        return em, fuga, ratios
    if return_pred:
        return em, fuga, pred
    return em, fuga


def synthetic_cell(channel, cell, dlt, D, n, n_events, seed, conv):
    """Celda sintetica (operador puro, unica capa): fuga pooled."""
    g = torch.Generator().manual_seed(seed)
    th = torch.rand(40, D, generator=g) * 2.0 * math.pi
    cb = torch.complex(torch.cos(th), torch.sin(th))
    ph = complex(math.cos(dlt), math.sin(dlt))
    M = torch.zeros(D, D, dtype=torch.complex64)
    num = den = 0.0
    for _ in range(n_events):
        ks = torch.randperm(40, generator=g)[:n]
        v = torch.randn(n, D, generator=g)
        M.zero_()
        for i in range(n):
            w = cb[int(ks[i].item())]
            vv = torch.complex(v[i], torch.zeros_like(v[i]))
            ww = ph * w if (cell == 'estado' and i == 0) else w
            M += ww.unsqueeze(1) * vv.unsqueeze(0)
        we = cb[int(ks[0].item())]
        wd = ph * we
        proj = (wd.conj() @ M) / D
        if channel == 're':
            proj = torch.complex(proj.real, torch.zeros_like(proj.real))
        outer = wd if cell == 'clave' else we
        M = M - outer.unsqueeze(1) * proj.unsqueeze(0)
        rr = (we.conj() @ M) / D
        if channel == 're' and conv == 'fiel':
            rr = torch.complex(rr.real, torch.zeros_like(rr.real))
        num += rr.abs().pow(2).sum().item()
        den += v[0].abs().pow(2).sum().item()
    return num / max(den, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--n_events', type=int, default=N_EVENTS)
    ap.add_argument('--out', type=str, default='outputs/wave_mem/o04b.json')
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = (1, 2, 3)
    t0 = time.time()

    out = {'variant': 'o04b', 'd_model': 64, 'n_layers': 2,
           'seeds': list(seeds), 'deltas': list(DELTAS),
           'n_events_per_combo': args.n_events,
           'convencion': ('fiel: Re truncado en wh y readout (lo que ve el '
                          'head entrenado). auditor: Re solo en wh (readout '
                          'complejo) - numeros del auditor 0.289/0.0870'),
           'inyeccion': {'clave': 'items limpios; erase lee y remueve con '
                                 'e^{id}w (reloj drifteado en el FORGET)',
                         'estado': 'solo el item olvidado escrito con e^{id}w; '
                                   'erase lee con e^{id}w (drift cancelado) y '
                                   'remueve con w limpia (reloj actual)'},
           'synthetic': {}, 'runs': {}}

    # 1) Verificacion sintetica: 2 grillas x 2 convenciones x 4 celdas x 6 delta
    print('===== sintetico =====', flush=True)
    for (D, n) in ((64, 20), (128, 16)):
        c = (n - 1) / D
        for conv in ('fiel', 'auditor'):
            key = f'D{D}_n{n}_{conv}'
            out['synthetic'][key] = {'c': c, 'cells': {}}
            for channel in ('complex', 're'):
                for cell in ('clave', 'estado'):
                    rows = {}
                    for dlt in DELTAS:
                        fu = synthetic_cell(channel, cell, dlt, D, n,
                                            args.n_events, 777, conv)
                        rows[str(dlt)] = {'fuga': round(fu, 4),
                                          'pred': round(law(cell, channel, dlt, c, conv), 4)}
                    out['synthetic'][key]['cells'][f'{channel}/{cell}'] = rows
                    print(f'  {key} {channel}/{cell}: '
                          f'd=0.5 fuga={rows["0.5"]["fuga"]} pred={rows["0.5"]["pred"]}',
                          flush=True)

    # 2) Probe sobre modelos entrenados (replay externo, 4 celdas x 6 delta x 6 ckpts)
    print('===== probe modelos entrenados (replay externo) =====', flush=True)
    for arm, read_proj in (('wave_complex', 'complex'), ('wave_re', 're')):
        for seed in seeds:
            label = f'{arm}_s{seed}'
            ck = torch.load(f'outputs/n1_{arm}/cache/forget_retrieval_seed{seed}'
                            f'_dm64_L2_ep80.pt', map_location=device)
            model = build_wave(read_proj)(89, 64, 64, 2)
            model.load_state_dict(ck['model'])
            model.to(device).eval()
            rows = {}
            for dlt in DELTAS:
                x, y, seqs, v_erased = make_fr_probe_dataset(
                    args.n_events, seed=2000 + seed, kind='erased',
                    n_pairs_range=(16, 24))
                x = x.to(device)
                em_c, fu_c = replay_wave(model, x, y, seqs, v_erased, 'clave', dlt, device)
                em_e, fu_e = replay_wave(model, x, y, seqs, v_erased, 'estado', dlt, device)
                n_pairs_avg = 20.0
                c = (n_pairs_avg - 1) / 64
                rows[str(dlt)] = {
                    'em_clave': round(em_c, 4), 'em_estado': round(em_e, 4),
                    'fuga_clave': [round(v, 4) for v in fu_c],
                    'fuga_estado': [round(v, 4) for v in fu_e],
                    'pred_clave': round(law('clave', read_proj, dlt, c, 'fiel'), 4),
                    'pred_estado': round(law('estado', read_proj, dlt, c, 'fiel'), 4),
                }
                print(f'  {label} d={dlt}: fu_clave={rows[str(dlt)]["fuga_clave"][0]} '
                      f'fu_estado={rows[str(dlt)]["fuga_estado"][0]} '
                      f'EM_c={em_c:.2f} EM_e={em_e:.2f}', flush=True)
            out['runs'][label] = rows

    out['total_seconds'] = round(time.time() - t0, 1)
    import platform
    import subprocess
    out['env'] = {'python': platform.python_version(), 'torch': torch.__version__}
    try:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                           capture_output=True, text=True, shell=True, timeout=5)
        out['git_hash'] = r.stdout.strip() or None
    except Exception:
        out['git_hash'] = None
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'Escrito: {args.out}  total={out["total_seconds"]}s')


if __name__ == '__main__':
    main()