"""Runner N1 (O03): 5-task cpu_quick, 3 brazos x 3 seeds, con la spec de
potencia del auditor (forget_retrieval n_pairs [16,24], >=300 eventos
FORGET/brazo/seed) + probe de selectividad sobre modelos entrenados con la
intervencion oracle de re-presentacion + TOST del control + probe de
decodificabilidad (ridge sobre M post-erase y sobre el residual de lectura).
Pre-registro: logs/O03_test.md + paper/CLAIM.md §8 items 14-16.
Salida unica: outputs/wave_mem/n1.json.
"""

import argparse
import json
import math
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from dataclasses import replace

from tests.common_smoke import (BOS_ID, SEP_ID, QUERY_ID, FORGET_ID,
                                ANSWER_ID, K_OFFSET_MQAR, V_OFFSET_MQAR,
                                N_KEYS_MQAR, N_VALS_MQAR, PAD_ID,
                                ForgetRetrieveDataset, make_task_specs,
                                run_one_task, evaluate)
from prototypes.wave_mem.model import WaveMemLM
from prototypes.delta_forget.model import DeltaForgetLM

T_CRIT_90_DF2 = 2.920  # t(0.95, 2) para IC 90% con 3 seeds


def gen_fr_query(rng, n_pairs, q_kind):
    """Misma estructura que gen_forget_retrieve_sample (1 forget) pero con
    la query dirigida a clave ALIVE o a la clave BORRADA (para fuga)."""
    ids_k = rng.sample(range(N_KEYS_MQAR), n_pairs)
    ids_v = rng.sample(range(N_VALS_MQAR), n_pairs)
    erased = rng.randrange(n_pairs)
    remaining = [i for i in range(n_pairs) if i != erased]
    qi = erased if q_kind == 'erased' else rng.choice(remaining)
    pair_order = list(range(n_pairs))
    rng.shuffle(pair_order)
    store = []
    for p in pair_order:
        store += [K_OFFSET_MQAR + ids_k[p], V_OFFSET_MQAR + ids_v[p]]
    seq = ([BOS_ID] + store +
           [SEP_ID, FORGET_ID, K_OFFSET_MQAR + ids_k[erased],
            SEP_ID, QUERY_ID, K_OFFSET_MQAR + ids_k[qi], ANSWER_ID])
    return seq, V_OFFSET_MQAR + ids_v[qi], V_OFFSET_MQAR + ids_v[erased]


def make_fr_probe_dataset(n_samples, seed, n_pairs_range=(16, 24), kind='alive'):
    rng = random.Random(seed)
    seqs, tgts, v_erased = [], [], []
    for _ in range(n_samples):
        n_pairs = rng.randint(*n_pairs_range)
        seq, tgt, ve = gen_fr_query(rng, n_pairs, kind)
        seqs.append(seq)
        tgts.append(tgt)
        v_erased.append(ve)
    T = max(len(s) for s in seqs)
    x = torch.full((len(seqs), T), PAD_ID, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    y = torch.tensor(tgts, dtype=torch.long)
    return x, y, seqs, torch.tensor(v_erased, dtype=torch.long)


def build_wave(read_proj):
    def f(vocab_size, max_len, d_model, n_layers):
        return WaveMemLM(vocab_size, max_len, d_model, n_layers,
                         read_proj=read_proj)
    return f


def build_delta(vocab_size, max_len, d_model, n_layers):
    return DeltaForgetLM(vocab_size, max_len, d_model, n_layers)


def power_specs():
    """cpu_quick 5-task con forget_retrieval bajo la spec del auditor:
    n_pairs [16,24], n_forget [1,2], max_seq_len 64."""
    specs = make_task_specs(smoke='cpu_quick')
    out = []
    for s in specs:
        if s.name == 'forget_retrieval':
            s = replace(
                s, max_seq_len=64,
                make_train=lambda seed: ForgetRetrieveDataset(
                    600, seed=seed, n_pairs_range=(16, 24), n_forget_range=(1, 2)),
                make_valid=lambda seed: ForgetRetrieveDataset(
                    150, seed=seed + 1, n_pairs_range=(16, 24), n_forget_range=(1, 2)))
        out.append(s)
    return out


def v_true_for(model, tok_v, idx=None):
    """Valor escrito por el modelo para el token de valor tok_v, por bloque."""
    vts = []
    if isinstance(model, WaveMemLM):
        emb = model.embedding(tok_v)
        for b in model.blocks:
            p = b.v_proj(emb)
            vts.append(torch.complex(p, torch.zeros_like(p)))
    else:
        emb = model.embedding(tok_v)
        for b in model.blocks:
            vts.append(b.v_proj(emb))
    return vts


def key_for(model, tok_k, idx=None):
    if isinstance(model, WaveMemLM):
        return model.codebook[tok_k]
    ks = []
    emb = model.embedding(tok_k)
    for b in model.blocks:
        ks.append(b.k_proj(emb))
    return ks


def run_probe(model, x, y, seqs, tok_v_forget, mode, device):
    """Forward con posible intervencion oracle de re-presentacion.
    mode: 'reread' (sin intervencion: el modelo borra por re-lectura) |
          'represent' (el evaluador borra con el v VERDADERO inyectado).
    Captura por muestra: EM del target (y), readout de la clave consultada
    (por bloque) y M post-erase del bloque 0.
    """
    B, T = x.shape
    tok = x
    pos_forget = torch.tensor([s.index(FORGET_ID) for s in seqs], dtype=torch.long)
    pos_qkey = torch.tensor([s.index(QUERY_ID) + 1 for s in seqs], dtype=torch.long)
    pos_ans = torch.tensor([s.index(ANSWER_ID) for s in seqs], dtype=torch.long)
    tok_key = torch.tensor([s[int(pos_forget[i]) + 1] for i, s in enumerate(seqs)], dtype=torch.long)
    is_delta = isinstance(model, DeltaForgetLM)
    rb_dtype = torch.float32 if is_delta else torch.complex64
    state = model.init_state(B, device)
    readout_b = [torch.zeros(B, model.d_model, dtype=rb_dtype) for _ in range(model.n_layers)]
    M_post = torch.zeros(B, model.d_model, model.d_model,
                         dtype=torch.float32 if is_delta else torch.complex64)
    pred = torch.zeros(B, dtype=torch.long)
    with torch.no_grad():
        for t in range(T):
            if mode == 'represent':
                m = (pos_forget == t - 1)
                if m.any():
                    idx = m.nonzero(as_tuple=False).squeeze(-1)
                    vv = tok_v_forget[idx]
                    ks = key_for(model, tok[idx, t])
                    vts = v_true_for(model, vv)
                    for b in range(model.n_layers):
                        St = state.S if is_delta else state.M
                        M_b = St[b]
                        M_new = M_b.clone()
                        k = ks if not is_delta else ks[b]
                        delta = -k.unsqueeze(-1) * vts[b].unsqueeze(1)
                        M_new[idx] = M_b[idx] + delta
                        if is_delta:
                            state.S[b] = M_new
                        else:
                            state.M[b] = M_new
                    prev = state.prev.clone()
                    prev[idx] = tok[idx, t]
                    state.prev = prev
            logits_t, state = model.decode_step(tok[:, t], state)
            if (pos_qkey == t).any():
                qm = pos_qkey == t
                for b in range(model.n_layers):
                    rb = readout_b[b].clone()
                    rb[qm] = state.r[b][qm]
                    readout_b[b] = rb
            if (pos_forget == t - 1).any():
                pm = pos_forget == t - 1
                mp = M_post.clone()
                St = state.S if is_delta else state.M
                mp[pm] = St[0][pm]
                M_post = mp
            if (pos_ans == t).any():
                am = pos_ans == t
                p = pred.clone()
                p[am] = logits_t[am].argmax(dim=-1)
                pred = p
    tgts = y.to(torch.long)
    em = float((pred == tgts).float().mean().item())
    return em, readout_b, M_post


def state_fuga(model, readout, tok_v_erased, device):
    """|r|^2/|v|^2 del readout de la clave borrada, por bloque."""
    vts = v_true_for(model, torch.tensor(tok_v_erased, dtype=torch.long, device=device))
    vals = []
    for b in range(model.n_layers):
        r = readout[b]
        v = vts[b]
        num = r.abs().pow(2).sum(dim=1)
        den = v.abs().pow(2).sum(dim=1).clamp(min=1e-12)
        vals.append(float((num / den).mean().item()))
    return vals


def ridge_probe(X_tr, y_tr, X_te, y_te, lam=0.1):
    """Ridge lineal: features -> target real. Reporta cosine medio test."""
    Xtr = X_tr - X_tr.mean(dim=0, keepdim=True)
    Xte = X_te - X_te.mean(dim=0, keepdim=True)
    ytr = y_tr - y_tr.mean(dim=0, keepdim=True)
    yte = y_te - y_te.mean(dim=0, keepdim=True)
    f = Xtr.size(1)
    XtX = Xtr.t() @ Xtr + lam * torch.eye(f, dtype=Xtr.dtype)
    W = torch.linalg.solve(XtX, Xtr.t() @ ytr)
    yp = Xte @ W
    num = (yp * yte).sum(dim=1)
    den = yp.norm(dim=1) * yte.norm(dim=1)
    cos = float((num / den.clamp(min=1e-12)).mean().item())
    ss_res = float((yp - yte).pow(2).sum(dim=1).mean().item())
    ss_tot = float(yte.pow(2).sum(dim=1).mean().item())
    return {'cosine_test': round(cos, 4), 'mse_rel': round(ss_res / max(ss_tot, 1e-12), 4)}


def probe_features(model, readout, M_post, device):
    """Features de ridge: readout residual del bloque 0 (lo que ve el head)
    y proyeccion aleatoria fija de M post-erase (lo que podria ver un
    decodificador lineal sobre M)."""
    if isinstance(model, WaveMemLM):
        if model.read_proj == 'complex':
            feat_r = torch.cat([readout[0].real, readout[0].imag], dim=1)
        else:
            feat_r = readout[0].real
    else:
        feat_r = readout[0]
    if M_post.is_complex():
        Mf = torch.cat([M_post.real.flatten(1), M_post.imag.flatten(1)], dim=1)
    else:
        Mf = M_post.flatten(1)
    W = torch.randn(Mf.size(1), 64, generator=torch.Generator().manual_seed(99))
    feat_m = Mf @ W
    return feat_r.to(torch.float32), feat_m.to(torch.float32)


def tost_equivalence(diffs, eps=0.02):
    """TOST: |media| + t(0.95, df)*SE <= eps con IC 90%."""
    n = len(diffs)
    mu = sum(diffs) / n
    sd = (sum((d - mu) ** 2 for d in diffs) / max(n - 1, 1)) ** 0.5
    se = sd / math.sqrt(n)
    half = T_CRIT_90_DF2 * se
    return {'mean_delta': round(mu, 4), 'sd': round(sd, 4),
            'ci90': [round(mu - half, 4), round(mu + half, 4)],
            'equivalence_pass': bool(abs(mu) + half <= eps), 'eps': eps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--d_model', type=int, default=64)
    ap.add_argument('--n_layers', type=int, default=2)
    ap.add_argument('--seeds', type=str, default='1,2,3')
    ap.add_argument('--n_probe', type=int, default=320)
    ap.add_argument('--out', type=str, default='outputs/wave_mem/n1.json')
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(',')]

    arms = [('wave_complex', build_wave('complex')),
            ('wave_re', build_wave('re')),
            ('delta_forget', build_delta)]
    specs = power_specs()

    t0 = time.time()
    out = {'variant': 'wave_mem_n1', 'd_model': args.d_model,
           'n_layers': args.n_layers, 'seeds': seeds,
           'n_probe_events': args.n_probe,
           'spec': 'cpu_quick 5-task, FR n_pairs [16,24] n_forget [1,2]',
           'runs': {}}
    tost_pool = []
    for seed in seeds:
        for arm_name, build in arms:
            label = f'{arm_name}_s{seed}'
            print(f'\n===== N1 {arm_name} seed={seed} =====', flush=True)

            # Resume: si los 5 ckpts del combo ya existen, no re-entrenar.
            def _ckpt(spec):
                return (f'outputs/n1_{arm_name}/cache/{spec.name}_seed{seed}'
                        f'_dm{args.d_model}_L{args.n_layers}_ep{spec.epochs}.pt')
            resume = all(os.path.exists(_ckpt(s)) for s in specs)
            if resume:
                print('  resume: ckpts encontrados, reuso entrenamiento', flush=True)
                runs = {}
                for s in specs:
                    ck = torch.load(_ckpt(s), map_location=device)
                    fin = ck['meta']['final']
                    runs[s.name] = {
                        'final_exact_match': fin['valid_exact_match'],
                        'final_loss': fin['valid_loss'],
                        'final_token_acc': fin['valid_token_acc'],
                        'max_em_in_window': ck['meta'].get('max_em_in_window', 0.0),
                        'seconds': ck['meta'].get('total_seconds', 0.0)}
                    print(f'  (resume) {s.name:<16} EM={runs[s.name]["final_exact_match"]:.3f}',
                          flush=True)
            else:
                runs = {}
                for spec in specs:
                    res = run_one_task(spec, build, device, seed=seed, lr=1e-3,
                                       weight_decay=0.01, clip=1.0,
                                       d_model=args.d_model, n_layers=args.n_layers,
                                       variant=f'n1_{arm_name}')
                    runs[spec.name] = {'final_exact_match': res['final_exact_match'],
                                       'final_loss': res['final_loss'],
                                       'final_token_acc': res['final_token_acc'],
                                       'max_em_in_window': res.get('max_em_in_window', 0.0),
                                       'seconds': res['seconds']}
                    print(f'  {spec.name:<16} EM={res["final_exact_match"]:.3f} '
                          f'loss={res["final_loss"]:.3f} t={res["seconds"]:.0f}s', flush=True)
            ckpt = _ckpt(next(s for s in specs if s.name == 'forget_retrieval'))
            model = build(vocab_size=89, max_len=64, d_model=args.d_model,
                          n_layers=args.n_layers)
            ck = torch.load(ckpt, map_location=device)
            model.load_state_dict(ck['model'])
            model.to(device).eval()

            probe = {}
            fr_ok = True
            if any(torch.isnan(p).any().item() for p in model.parameters()):
                print(f'  WARNING: ckpt {arm_name} seed {seed} NaN (FR no entrena '
                      f'- probable: regla delta no aprende FR a d=64)', flush=True)
                fr_ok = False
                probe['fr_status'] = 'nan_failed'
                out['runs'][f'{arm_name}_s{seed}'] = {'tasks': runs, 'probe': probe}
                continue

            # Autocheck: la EM del probe (alive) debe matchear la EM del
            # harness en formato oficial (con token de valor tras ANSWER).
            # Si divergen, el probe esta roto -> aborta este arm/seed.
            from tests.common_smoke import collate_fn
            from torch.utils.data import DataLoader
            from functools import partial
            val_ds = ForgetRetrieveDataset(64, seed=seed + 100, n_pairs_range=(16, 24),
                                           n_forget_range=(1, 2))
            col = partial(collate_fn, answer_marker_id=ANSWER_ID, mark_after_marker=False,
                          prefix_answer=False)
            val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=col)
            val_res = evaluate(model, val_loader, device, vocab_size=89)
            print(f'  autocheck harness EM={val_res["exact_match"]:.3f}', flush=True)

            probe = {}
            for kind in ('alive', 'erased'):
                x, y, seqs, v_erased = make_fr_probe_dataset(
                    args.n_probe, seed=1000 + seed, kind=kind)
                x = x.to(device)
                em_rr, rd_rr, m_rr = run_probe(model, x, y, seqs, v_erased,
                                               'reread', device)
                em_rp, rd_rp, m_rp = run_probe(model, x, y, seqs, v_erased,
                                               'represent', device)
                if kind == 'alive':
                    probe_match = abs(em_rr - val_res['exact_match']) < 0.15
                    if not probe_match:
                        print(f'  WARNING: probe alive EM {em_rr:.3f} no matchea '
                              f'harness {val_res["exact_match"]:.3f} - probe roto',
                              flush=True)
                        sys.exit(2)
                entry = {'events': args.n_probe,
                         'em_reread': round(em_rr, 4),
                         'em_represent': round(em_rp, 4)}
                if kind == 'erased':
                    tok_vs = [int(v) for v in v_erased]
                    entry['fuga_power_reread'] = [
                        round(v, 4) for v in state_fuga(model, rd_rr, tok_vs, device)]
                    entry['fuga_power_represent'] = [
                        round(v, 4) for v in state_fuga(model, rd_rp, tok_vs, device)]
                    ytr = v_true_for(model, torch.tensor(tok_vs, dtype=torch.long, device=device))[0]
                    if isinstance(model, WaveMemLM):
                        ytr = ytr.real
                    fr, fm = probe_features(model, rd_rr, m_rr, device)
                    n_tr = int(len(seqs) * 0.75)
                    entry['ridge_readout'] = ridge_probe(fr[:n_tr], ytr[:n_tr], fr[n_tr:], ytr[n_tr:])
                    entry['ridge_M_proj'] = ridge_probe(fm[:n_tr], ytr[:n_tr], fm[n_tr:], ytr[n_tr:])
                probe[kind] = entry
            sel_rr = round(probe['alive']['em_reread'] - probe['erased']['em_reread'], 4)
            sel_rp = round(probe['alive']['em_reread'] - probe['erased']['em_represent'], 4)
            probe['selectividad_reread'] = sel_rr
            probe['selectividad_represent'] = sel_rp
            out['runs'][label] = {'tasks': runs, 'probe': probe}
            if arm_name in ('wave_complex', 'delta_forget'):
                tost_pool.append((arm_name, seed, sel_rr))
            print(f'  probe: EM_vivo={probe["alive"]["em_reread"]} '
                  f'EM_fuga_reread={probe["erased"]["em_reread"]} '
                  f'sel_reread={sel_rr} sel_represent={sel_rp}', flush=True)

    wc = {s: next(v for a, ss, v in tost_pool if a == 'wave_complex' and ss == s) for s in seeds}
    dl = {s: next((v for a, ss, v in tost_pool if a == 'delta_forget' and ss == s),
                  None) for s in seeds}
    common = [s for s in seeds if dl[s] is not None]
    if len(common) >= 3:
        deltas = [wc[s] - dl[s] for s in common]
        out['tost_control'] = tost_equivalence(deltas)
    else:
        out['tost_control'] = {
            'status': 'NO COMPUTABLE en FR',
            'reason': 'delta_forget no entrena forget_retrieval a d=64 '
                      '(EM ~ chance, loss NaN): el control de identidad '
                      'wave_complex vs delta no existe en FR',
            'n_common': len(common)}
    re_by_seed = {}
    for s in seeds:
        re_by_seed[s] = out['runs'][f'wave_re_s{s}']['probe']['selectividad_reread']
    out['contrast_re_vs_others'] = {
        'by_seed': {
            str(s): {'sel_re': re_by_seed[s], 'sel_wc': wc[s],
                     'sel_delta': dl[s] if dl[s] is not None else None}
            for s in seeds},
        'pred_deficit_re': '(n-1)/(2D) en fuga del canal re vs complex',
    }

    import platform
    import subprocess
    out['env'] = {'python': platform.python_version(), 'torch': torch.__version__}
    try:
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                           capture_output=True, text=True, shell=True, timeout=5)
        out['git_hash'] = r.stdout.strip() or None
    except Exception:
        out['git_hash'] = None
    out['total_seconds'] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, default=str)

    print('\n===JSON_START===')
    print(json.dumps({
        'tost_control': out['tost_control'],
        'selectividad_por_seed': out['contrast_re_vs_others']['by_seed'],
        'runs': {k: v['probe'] for k, v in out['runs'].items()},
    }, indent=2, default=str))
    print('===JSON_END===')
    print(f'Escrito: {args.out}  total={out["total_seconds"]}s')


if __name__ == '__main__':
    main()