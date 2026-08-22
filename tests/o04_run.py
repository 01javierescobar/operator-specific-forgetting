"""Runner O04 (decisión del auditor, enmiendas incluidas):
1. Side arm delta_nlms (autopsia): entrena FR (n_pairs 16-24, 3 seeds,
   d=64, 80 ep) con claves normalizadas (NLMS, beta=1). Si entrena
   (EM alto) -> el control TOST (identidad wave_complex vs delta)
   RESUCITA y se reporta; si no -> declaracion "delta FALLIDO en FR"
   hermeticamente documentada (mecanismo: write correctivo beta=1 con
   ||k||=1 tiene ganancia |1-1|=0, estable por construccion).
2. Probe de drift Clase B' sobre modelos entrenados (2x2xdelta):
   canal {complex, re} x erase {reread, represent} x delta {0,.1,.2,.3,
   .4,.5}, fuga_power (>=300 eventos FORGET/brazo/seed) con 4 leyes
   exactas sin parametros libres pre-registradas:
   - reread complex: 0 (plana; proyector invariante a drift)
   - represent complex: (n-1)/D + 2(1-cos d)
   - reread re: sin^4(d) + (n-1)/D (1-cos 2d)/4   [CORREGIDA vs enmienda
     del auditor: la Re-truncacion del erase NO es proyector complejo;
     verificada en tests/o04_laws_verify.py]
   - represent re: (n-1)/(2D) + (1-cos d)^2
   EM = sanity check (sin umbral). Criterio de fallo P1: reread-complex
   fuga > 0.02 a d=0.5 en modelos entrenados -> invarianza no sobrevivio
   al entrenamiento -> Plan B. (reread-re NO usa ese criterio: su ley es
   sin^4, verificada.)
Salida: outputs/wave_mem/o04.json
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from tests.common_smoke import (ANSWER_ID, FORGET_ID, PAD_ID, QUERY_ID,
                                ForgetRetrieveDataset, collate_fn)
from tests.wave_mem_n1 import (build_wave, make_fr_probe_dataset,
                               state_fuga, v_true_for)
from prototypes.wave_mem.model import WaveMemLM
from prototypes.delta_nlms.model import DeltaNlmsLM

DELTAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
N_EVENTS = 320
EPOCHS_FR = 80


def train_delta_nlms(seed, device, out_dir='outputs/o04_delta_nlms'):
    ckpt = f'{out_dir}/cache/forget_retrieval_seed{seed}_dm64_L2_ep{EPOCHS_FR}.pt'
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    if os.path.exists(ckpt):
        ck = torch.load(ckpt, map_location=device)
        print(f'  (resume) delta_nlms s{seed} EM={ck["meta"]["final"]["valid_exact_match"]:.3f}')
        return ck
    model = DeltaNlmsLM(vocab_size=89, max_len=64, d_model=64, n_layers=2)
    col = partial(collate_fn, answer_marker_id=ANSWER_ID, mark_after_marker=False,
                  prefix_answer=False)
    tr = ForgetRetrieveDataset(600, seed=seed, n_pairs_range=(16, 24), n_forget_range=(1, 2))
    va = ForgetRetrieveDataset(150, seed=seed + 1, n_pairs_range=(16, 24), n_forget_range=(1, 2))
    tl = DataLoader(tr, batch_size=32, shuffle=True, collate_fn=col)
    vl = DataLoader(va, batch_size=32, shuffle=False, collate_fn=col)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    S_norm_trace = []
    best = 0.0
    t0 = time.time()
    for ep in range(1, EPOCHS_FR + 1):
        model.train()
        for x, y, tm in tl:
            yt = torch.where(tm, y, torch.full_like(y, PAD_ID))
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x).reshape(-1, 89), yt.reshape(-1), ignore_index=PAD_ID)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            tok = 0
            hits = 0
            tot = 0
            for x, y, tm in vl:
                logits = model(x)
                yt = torch.where(tm, y, torch.full_like(y, PAD_ID))
                val_loss += F.cross_entropy(logits.reshape(-1, 89), yt.reshape(-1),
                                            ignore_index=PAD_ID).item() * (yt != PAD_ID).sum().item()
                tok += (yt != PAD_ID).sum().item()
                pred = logits.argmax(dim=-1)
                hits += (pred[tm] == y[tm]).sum().item()
                tot += tm.sum().item()
            em = hits / max(tot, 1)
            best = max(best, em)
            st = model.init_state(x.size(0), device)
            with torch.no_grad():
                for t in range(x.size(1)):
                    _, st = model.decode_step(x[:, t], st)
            S_norm_trace.append([round(float(s.norm().item()), 1) for s in st.S])
            if ep % 10 == 0 or ep == 1:
                print(f'  delta_nlms s{seed} ep={ep} EM={em:.3f} '
                      f'S_norm={S_norm_trace[-1]}')
    torch.save({'model': model.state_dict(),
                'meta': {'final': {'valid_exact_match': em, 'valid_loss':
                                   val_loss / max(tok, 1)},
                         'best_em': best,
                         'S_norm_trace': S_norm_trace,
                         'total_seconds': round(time.time() - t0, 1)}},
               ckpt)
    return {'model': model.state_dict(),
            'meta': {'final': {'valid_exact_match': em},
                     'best_em': best, 'S_norm_trace': S_norm_trace}}


def probe_em(model, x, y, seqs, device):
    """EM del harness sobre dataset probe (sin intervencion)."""
    B, T = x.shape
    pos_ans = torch.tensor([s.index(ANSWER_ID) for s in seqs], dtype=torch.long)
    st = model.init_state(B, device)
    pred = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        for t in range(T):
            logits_t, st = model.decode_step(x[:, t], st)
            if (pos_ans == t).any():
                am = pos_ans == t
                p = pred.clone()
                p[am] = logits_t[am].argmax(dim=-1)
                pred = p
    return float((pred == y).float().mean().item())


def drift_probe_wave(model, x, y, seqs, v_erased, mode, dlt, device):
    """Probe de drift 2x2xdelta sobre modelos wave entrenados.
    Intervencion en el token FORGET (t = pos_forget+1), skip del erase
    entrenado (prev = clave). Leer/borrar con clave drifteada e^{id}w.
    mode 'reread': erase Re-truncado del modelo con clave drifteada
                   (replica exacta de WaveMemBlock con w_cur -> e^{id}w).
    mode 'represent': erase con v TRUE y clave drifteada: M - e^{id}w v.
    Fuga POOLED (sum|r|^2 / sum|v|^2): robusta a la cola pesada de ||v||^2
    del modelo entrenado (p90/p50 ~ 5.6) que infla la media de ratios.
    """
    B, T = x.shape
    is_re = isinstance(model, WaveMemLM) and model.read_proj == 're'
    pos_forget = torch.tensor([s.index(FORGET_ID) for s in seqs], dtype=torch.long)
    pos_qkey = torch.tensor([s.index(QUERY_ID) + 1 for s in seqs], dtype=torch.long)
    pos_ans = torch.tensor([s.index(ANSWER_ID) for s in seqs], dtype=torch.long)
    tok_key = torch.tensor([s[int(pos_forget[i]) + 1] for i, s in enumerate(seqs)],
                           dtype=torch.long)
    ph = torch.complex(torch.tensor(math.cos(dlt)), torch.tensor(math.sin(dlt)))
    state = model.init_state(B, device)
    readout_b = [torch.zeros(B, model.d_model, dtype=torch.complex64) for _ in range(model.n_layers)]
    pred = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        for t in range(T):
            if (pos_forget == t - 1).any():
                m = pos_forget == t - 1
                idx = m.nonzero(as_tuple=False).squeeze(-1)
                wd = ph * model.codebook[tok_key[idx]]
                for b in range(model.n_layers):
                    M_b = state.M[b]
                    M_new = M_b.clone()
                    if mode == 'reread':
                        wh = torch.einsum('bd,bdj->bj', torch.conj(wd), M_b[idx]) / model.d_model
                        if is_re:
                            wh = torch.complex(wh.real, torch.zeros_like(wh.real))
                        M_new[idx] = M_b[idx] - wd.unsqueeze(-1) * wh.unsqueeze(1)
                    else:
                        vts = v_true_for(model, v_erased[idx])
                        M_new[idx] = M_b[idx] - wd.unsqueeze(-1) * vts[b].unsqueeze(1)
                    state.M[b] = M_new
                prev = state.prev.clone()
                prev[idx] = tok_key[idx]
                state.prev = prev
            logits_t, state = model.decode_step(x[:, t], state)
            if (pos_qkey == t).any():
                qm = pos_qkey == t
                for b in range(model.n_layers):
                    rb = readout_b[b].clone()
                    rb[qm] = state.r[b][qm]
                    readout_b[b] = rb
            if (pos_ans == t).any():
                am = pos_ans == t
                p = pred.clone()
                p[am] = logits_t[am].argmax(dim=-1)
                pred = p
    em = float((pred == y).float().mean().item())
    vts = v_true_for(model, torch.tensor([int(v) for v in v_erased],
                                         dtype=torch.long, device=device))
    fuga = []
    for b in range(model.n_layers):
        r = readout_b[b]
        v = vts[b]
        if is_re:
            r = r.real
            v = v.real
        fuga.append(float(r.abs().pow(2).sum().item() /
                          v.abs().pow(2).sum().clamp(min=1e-12).item()))
    return em, fuga


def laws(dlt, n_pairs, D):
    c = (n_pairs - 1) / D
    return {
        'reread_complex': 0.0,
        'represent_complex': c + 2 * (1 - math.cos(dlt)),
        'reread_re': math.sin(dlt) ** 4 + c * (1 - math.cos(2 * dlt)) / 4,
        'represent_re': c / 2 + (1 - math.cos(dlt)) ** 2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--n_events', type=int, default=N_EVENTS)
    ap.add_argument('--out', type=str, default='outputs/wave_mem/o04.json')
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = (1, 2, 3)
    t0 = time.time()

    out = {'variant': 'o04', 'd_model': 64, 'n_layers': 2,
           'seeds': list(seeds), 'deltas': list(DELTAS),
           'n_events_per_combo': args.n_events,
           'leyes_pre_registradas': laws(0.0, 16, 64),  # referencia
           'nota_correccion': ('reread_re NO es plano (enmienda del auditor '
                               'corregida): el erase Re-truncado con clave '
                               'drifteada deja sin^4(d); verificada en '
                               'tests/o04_laws_verify.py'),
           'runs': {}}

    # 1) Side arm delta_nlms
    print('===== delta_nlms (side arm autopsia) =====', flush=True)
    nlms = {}
    for seed in seeds:
        ck = train_delta_nlms(seed, device)
        nlms[f's{seed}'] = {'final_em': ck['meta']['final']['valid_exact_match'],
                            'best_em': ck['meta'].get('best_em'),
                            'S_norm_first': ck['meta']['S_norm_trace'][0],
                            'S_norm_last': ck['meta']['S_norm_trace'][-1]}
        print(f'  s{seed}: EM={nlms[f"s{seed}"]["final_em"]:.3f} '
              f'S_norm {nlms[f"s{seed}"]["S_norm_first"]} -> '
              f'{nlms[f"s{seed}"]["S_norm_last"]}', flush=True)
    out['delta_nlms'] = nlms
    trained = all(v['final_em'] > 0.5 for v in nlms.values())

    # 2) Probe de drift 2x2xdelta sobre wave entrenado (6 ckpts)
    print('===== probe drift 2x2xdelta (wave entrenado) =====', flush=True)
    for arm, read_proj in (('wave_complex', 'complex'), ('wave_re', 're')):
        for seed in seeds:
            label = f'{arm}_s{seed}'
            ck = torch.load(f'outputs/n1_{arm}/cache/forget_retrieval_seed{seed}'
                            f'_dm64_L2_ep{EPOCHS_FR}.pt', map_location=device)
            model = build_wave(read_proj)(89, 64, 64, 2)
            model.load_state_dict(ck['model'])
            model.to(device).eval()
            rows = {}
            for dlt in DELTAS:
                x, y, seqs, v_erased = make_fr_probe_dataset(
                    args.n_events, seed=2000 + seed, kind='erased',
                    n_pairs_range=(16, 24))
                x = x.to(device)
                em_rr, fu_rr = drift_probe_wave(model, x, y, seqs, v_erased,
                                                'reread', dlt, device)
                em_rp, fu_rp = drift_probe_wave(model, x, y, seqs, v_erased,
                                                'represent', dlt, device)
                n_pairs_avg = 20.0
                pr = laws(dlt, n_pairs_avg, 64)
                rows[str(dlt)] = {
                    'em_reread': round(em_rr, 4), 'em_represent': round(em_rp, 4),
                    'fuga_reread': [round(v, 4) for v in fu_rr],
                    'fuga_represent': [round(v, 4) for v in fu_rp],
                    'pred_reread_complex': pr['reread_complex'],
                    'pred_represent_complex': round(pr['represent_complex'], 4),
                    'pred_reread_re': round(pr['reread_re'], 4),
                    'pred_represent_re': round(pr['represent_re'], 4),
                }
                print(f'  {label} d={dlt}: fuga_rr={rows[str(dlt)]["fuga_reread"][0]} '
                      f'fuga_rp={rows[str(dlt)]["fuga_represent"][0]} '
                      f'EM_rr={em_rr:.2f} EM_rp={em_rp:.2f}', flush=True)
            out['runs'][label] = rows

    # 3) TOST resucitado si delta_nlms entrena: selectividad EM por seed
    if trained:
        print('===== delta_nlms entrenó: TOST del control resucitado =====', flush=True)
        sel_pool = {}
        for seed in seeds:
            model = DeltaNlmsLM(89, 64, 64, 2)
            ck = torch.load(f'outputs/o04_delta_nlms/cache/forget_retrieval_seed{seed}'
                            f'_dm64_L2_ep{EPOCHS_FR}.pt', map_location=device)
            model.load_state_dict(ck['model'])
            model.eval()
            ems = {}
            for kind in ('alive', 'erased'):
                x, y, seqs, v_erased = make_fr_probe_dataset(
                    args.n_events, seed=1000 + seed, kind=kind,
                    n_pairs_range=(16, 24))
                ems[kind] = probe_em(model, x.to(device), y, seqs, device)
            sel_pool[seed] = ems['alive'] - ems['erased']
            print(f'  delta_nlms s{seed}: EM_alive={ems["alive"]:.3f} '
                  f'EM_erased={ems["erased"]:.3f} sel={sel_pool[seed]:.3f}')
        wc = {}
        for seed in seeds:
            ck = torch.load(f'outputs/n1_wave_complex/cache/forget_retrieval_seed{seed}'
                            f'_dm64_L2_ep{EPOCHS_FR}.pt', map_location=device)
            model = build_wave('complex')(89, 64, 64, 2)
            model.load_state_dict(ck['model'])
            model.eval()
            ems = {}
            for kind in ('alive', 'erased'):
                x, y, seqs, v_erased = make_fr_probe_dataset(
                    args.n_events, seed=1000 + seed, kind=kind,
                    n_pairs_range=(16, 24))
                ems[kind] = probe_em(model, x.to(device), y, seqs, device)
            wc[seed] = ems['alive'] - ems['erased']
        diffs = [wc[s] - sel_pool[s] for s in seeds]
        n = len(diffs)
        mu = sum(diffs) / n
        sd = (sum((d - mu) ** 2 for d in diffs) / (n - 1)) ** 0.5
        half = 2.920 * sd / math.sqrt(n)
        out['tost_resucitado'] = {
            'mean_delta': round(mu, 4), 'sd': round(sd, 4),
            'ci90': [round(mu - half, 4), round(mu + half, 4)],
            'equivalence_pass': bool(abs(mu) + half <= 0.02), 'eps': 0.02,
            'selectividad_delta_nlms': sel_pool,
            'selectividad_wave_complex': wc}
        print(f'  TOST: delta={mu:.4f} IC90=[{mu-half:.4f},{mu+half:.4f}] '
              f'pass={abs(mu) + half <= 0.02}')
    else:
        out['tost_resucitado'] = {'status': 'NO resucitado: delta_nlms no '
                                  'entrena FR a d=64 (EM <= 0.5 en las 3 seeds)',
                                  'final_em_por_seed': {k: v['final_em'] for k, v in nlms.items()}}

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