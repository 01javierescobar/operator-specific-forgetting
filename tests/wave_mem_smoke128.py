"""Runner O05 (mini-smoke d=128, AUTORIZADO por el auditor 2026-08-19):
3 brazos (wave_complex, wave_re, delta_nlms) x 5 seeds, FR con n_pairs
[32,48] (n/D 0.25-0.38), 150 epocas, probe 2x2 con guarda de residual
nulo (EM_erased solo si ||residual|| > eps), C1 a c emparejada (probe
d=64 con n_pairs (17,24) -> c = 0.3047 = c de d=128) y TOST reportado
tal cual. Pre-registro: logs/O05_test.md (adiciones del auditor).
Salida unica: outputs/wave_mem/o05.json + o05_run.log.
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
print(f'O05 runner: cwd={os.getcwd()} REPO={REPO} path0={sys.path[:3]}', flush=True)

from dataclasses import replace

import tests.common_smoke as CS
from tests.common_smoke import (ANSWER_ID, FORGET_ID, QUERY_ID,
                                ForgetRetrieveDataset, collate_fn, evaluate,
                                run_one_task)
from tests.wave_mem_n1 import build_wave, make_fr_probe_dataset
from tests.o04b_run import law, replay_wave
from prototypes.delta_nlms.model import DeltaNlmsLM

DELTAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
N_EVENTS = 320
EPS_NULL = 1e-4
T_CRIT_90_DF4 = 2.132  # t(0.95, 4) para IC 90% con 5 seeds


class KeySpace:
    """Espacio de claves/valores de O05: n_pairs [32,48] excede los 40
    keys del harness -> 48 keys / 48 vals / vocab 105 (9+48+48) y
    V_OFFSET=57. Parche en runtime de las constantes del modulo y de las
    copias que ya importo wave_mem_n1 (gen_fr_query); el harness fuente
    queda intacto y se restaura al salir. Fuera de este contexto (probe
    C1 d=64) se usa el espacio original (40/40/vocab 89)."""

    def __init__(self, n_keys, n_vals, vocab, v_offset):
        self.cs = CS
        import tests.wave_mem_n1 as N1
        self.n1 = N1
        self.saved = (CS.N_KEYS_MQAR, CS.N_VALS_MQAR,
                      CS.VOCAB_MQAR_SIZE, CS.VOCAB_FORGET_SIZE,
                      CS.V_OFFSET_MQAR, N1.N_KEYS_MQAR, N1.N_VALS_MQAR,
                      N1.V_OFFSET_MQAR)
        self.new = (n_keys, n_vals, vocab, vocab, v_offset,
                    n_keys, n_vals, v_offset)

    def __enter__(self):
        (self.cs.N_KEYS_MQAR, self.cs.N_VALS_MQAR,
         self.cs.VOCAB_MQAR_SIZE, self.cs.VOCAB_FORGET_SIZE,
         self.cs.V_OFFSET_MQAR, self.n1.N_KEYS_MQAR, self.n1.N_VALS_MQAR,
         self.n1.V_OFFSET_MQAR) = self.new
        return self.cs

    def __exit__(self, *a):
        (self.cs.N_KEYS_MQAR, self.cs.N_VALS_MQAR,
         self.cs.VOCAB_MQAR_SIZE, self.cs.VOCAB_FORGET_SIZE,
         self.cs.V_OFFSET_MQAR, self.n1.N_KEYS_MQAR, self.n1.N_VALS_MQAR,
         self.n1.V_OFFSET_MQAR) = self.saved


def build_delta_nlms(vocab_size, max_len, d_model, n_layers):
    return DeltaNlmsLM(vocab_size, max_len, d_model, n_layers)


def fr_spec(epochs, max_seq_len=112, n_train=600, n_valid=128):
    from tests.common_smoke import make_task_specs
    for s in make_task_specs(smoke='smoke'):
        if s.name == 'forget_retrieval':
            s = replace(
                s, max_seq_len=max_seq_len, epochs=epochs,
                make_train=lambda seed, n=n_train: ForgetRetrieveDataset(
                    n, seed=seed, n_pairs_range=(32, 48), n_forget_range=(1, 2)),
                make_valid=lambda seed, n=n_valid: ForgetRetrieveDataset(
                    n, seed=seed + 1, n_pairs_range=(32, 48), n_forget_range=(1, 2)))
            return s
    raise RuntimeError('forget_retrieval spec no encontrada')


def probe_wave_2x2(model, seed, n_events, device, n_pairs_range=(32, 48)):
    """Probe 2x2 completo: 4 celdas x 6 delta, fuga pooled + EM_erased
    con guarda de residual nulo. Autocheck EM_alive vs harness adentro."""
    out = {}
    for kind in ('alive', 'erased'):
        x, y, seqs, v_erased = make_fr_probe_dataset(
            n_events, seed=1000 + seed, n_pairs_range=n_pairs_range, kind=kind)
        x = x.to(device)
        if kind == 'alive':
            em, _ = replay_wave(model, x, y, seqs, v_erased, 'clave', 0.0, device)
            out['em_alive'] = round(em, 4)
        else:
            cells = {}
            for cell in ('clave', 'estado'):
                cells[cell] = {}
                for dlt in DELTAS:
                    em, fuga, ratios = replay_wave(
                        model, x, y, seqs, v_erased, cell, dlt, device,
                        per_event=True)
                    cells[cell][str(dlt)] = {
                        'fuga': round(fuga[0], 4),
                        'pred': round(law(cell, 'complex' if not model.read_proj == 're' else 're',
                                          dlt, 39.0 / 128.0, 'fiel'), 4),
                        **guarded_em(em, fuga, ratios)}
            out['cells'] = cells
    return out


def em_probe(model, x, y, seqs, device):
    """EM exacto del forward del modelo (delta_nlms o wave) con la
    interfaz estandar: decode_step secuencial, logits en ANSWER."""
    B, T = x.shape
    pos_ans = torch.tensor([s.index(ANSWER_ID) for s in seqs], dtype=torch.long,
                           device=device)
    state = model.init_state(B, device)
    pred = torch.zeros(B, dtype=torch.long, device=device)
    with torch.no_grad():
        for t in range(T):
            logits, state = model.decode_step(x[:, t], state)
            if (pos_ans == t).any():
                p = pred.clone()
                p[pos_ans == t] = logits[pos_ans == t].argmax(dim=-1)
                pred = p
    return float((pred == y.to(device)).float().mean().item())


def guarded_em(em, fuga, ratios, eps=EPS_NULL):
    """Guarda de residual nulo (adicion 1 del auditor): EM_erased se
    reporta solo sobre eventos con ||residual|| > eps*||v_erased||. Si no
    hay eventos no-nulos -> 'residual nulo' (resultado mas fuerte que
    cualquier EM bajo)."""
    null_frac = float((ratios <= eps).float().mean().item())
    return {'em_guarded': em if null_frac < 1.0 else 'residual nulo',
            'null_frac': round(null_frac, 4),
            'em_on_nonnull_frac': round(1.0 - null_frac, 4)}


def tost_equivalence(diffs, eps=0.02):
    n = len(diffs)
    mu = sum(diffs) / n
    sd = (sum((d - mu) ** 2 for d in diffs) / max(n - 1, 1)) ** 0.5
    se = sd / math.sqrt(n)
    half = T_CRIT_90_DF4 * se
    return {'mean_delta': round(mu, 4), 'sd': round(sd, 4),
            'ci90': [round(mu - half, 4), round(mu + half, 4)],
            'equivalence_pass': bool(abs(mu) + half <= eps), 'eps': eps}


def probe_wave_2x2(model, seed, n_events, device, n_pairs_range=(32, 48)):
    """Probe 2x2 completo: 4 celdas x 6 delta, fuga pooled + EM_erased
    con guarda de residual nulo. Autocheck EM_alive vs harness adentro."""
    out = {}
    for kind in ('alive', 'erased'):
        x, y, seqs, v_erased = make_fr_probe_dataset(
            n_events, seed=1000 + seed, n_pairs_range=n_pairs_range, kind=kind)
        x = x.to(device)
        if kind == 'alive':
            em, _ = replay_wave(model, x, y, seqs, v_erased, 'clave', 0.0, device)
            out['em_alive'] = round(em, 4)
        else:
            cells = {}
            for cell in ('clave', 'estado'):
                cells[cell] = {}
                for dlt in DELTAS:
                    em, fuga, ratios = replay_wave(
                        model, x, y, seqs, v_erased, cell, dlt, device,
                        per_event=True)
                    cells[cell][str(dlt)] = {
                        'fuga': round(fuga[0], 4),
                        'pred': round(law(cell, 'complex' if not model.read_proj == 're' else 're',
                                          dlt, 39.0 / 128.0, 'fiel'), 4),
                        **guarded_em(em, fuga, ratios)}
            out['cells'] = cells
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--seeds', type=str, default='1,2,3,4,5')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--delta_epochs', type=int, default=150)
    ap.add_argument('--delta_train', type=int, default=1200)
    ap.add_argument('--n_probe', type=int, default=N_EVENTS)
    ap.add_argument('--d64_dir', type=str, default='outputs')
    ap.add_argument('--out', type=str, default='outputs/wave_mem/o05.json')
    args = ap.parse_args()
    device = torch.device(args.device)
    seeds = [int(s) for s in args.seeds.split(',')]
    c = 39.0 / 128.0  # n medio 40 -> c = (n-1)/D

    arms = [('wave_complex', build_wave('complex')),
            ('wave_re', build_wave('re')),
            ('delta_nlms', build_delta_nlms)]
    with KeySpace(48, 48, 105, 57):
        spec_wave = fr_spec(args.epochs)
        spec_delta = fr_spec(args.delta_epochs, n_train=args.delta_train)
        print(f'FR d=128: n_pairs [32,48] n_forget [1,2] max_seq {spec_wave.max_seq_len} '
              f'c={c} vocab={CS.VOCAB_FORGET_SIZE}', flush=True)
        print(f'  wave: {spec_wave.epochs} ep x 600 / delta_nlms: '
              f'{spec_delta.epochs} ep x {args.delta_train} (receta n1, '
              f'v4 mostro EM 0.07-0.98 sub-entrenado)', flush=True)

        t0 = time.time()
        out = {'variant': 'o05', 'd_model': 128, 'n_layers': 2, 'seeds': seeds,
               'n_probe_events': args.n_probe, 'c': c,
               'spec': 'FR only, n_pairs [32,48], 48 keys/48 vals (vocab 105, '
                       'KeySpace O05), wave 60ep x 600 / delta_nlms 150ep x 1200 '
                       '(v5; v4: wave EM 1.000 a 60ep x 600, delta sub-entrenado)',
               'eps_null': EPS_NULL, 'runs': {}}

        def _ckpt(arm, seed, epochs):
            return (f'outputs/o5_{arm}/cache/forget_retrieval_seed{seed}'
                    f'_dm128_L2_ep{epochs}.pt')

        for seed in seeds:
            for arm_name, build in arms:
                spec = spec_delta if arm_name == 'delta_nlms' else spec_wave
                label = f'{arm_name}_s{seed}'
                print(f'\n===== O05 {arm_name} seed={seed} '
                      f'({spec.epochs}ep x {spec.make_train(0).__len__()}) =====',
                      flush=True)
                ck = _ckpt(arm_name, seed, spec.epochs)
                if os.path.exists(ck):
                    ckpt = torch.load(ck, map_location=device)
                    train_em = ckpt['meta']['final']['valid_exact_match']
                    print(f'  (resume) EM={train_em:.3f}', flush=True)
                else:
                    res = run_one_task(spec, build, device, seed=seed, lr=1e-3,
                                       weight_decay=0.01, clip=1.0,
                                       d_model=128, n_layers=2,
                                       variant=f'o5_{arm_name}')
                    train_em = res['final_exact_match']
                    print(f'  train EM={train_em:.3f} loss={res["final_loss"]:.3f} '
                          f't={res["seconds"]:.0f}s', flush=True)
                model = build(vocab_size=CS.VOCAB_FORGET_SIZE,
                              max_len=spec.max_seq_len, d_model=128, n_layers=2)
                ckpt = torch.load(ck, map_location=device)
                model.load_state_dict(ckpt['model'])
                model.to(device).eval()
                if any(torch.isnan(p.detach().cpu()).any().item()
                       for p in model.parameters()):
                    print('  WARNING: NaN en ckpt', flush=True)
                    out['runs'][label] = {'train_em': train_em, 'status': 'nan'}
                    continue

                # Autocheck: EM del probe (alive) vs harness oficial.
                from functools import partial
                from torch.utils.data import DataLoader
                val_ds = ForgetRetrieveDataset(64, seed=seed + 100,
                                               n_pairs_range=(32, 48),
                                               n_forget_range=(1, 2))
                col = partial(collate_fn, answer_marker_id=ANSWER_ID,
                              mark_after_marker=False, prefix_answer=False)
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                                        collate_fn=col)
                val_res = evaluate(model, val_loader, device,
                                   vocab_size=CS.VOCAB_FORGET_SIZE)
                print(f'  autocheck harness EM={val_res["exact_match"]:.3f}',
                      flush=True)

                if arm_name in ('wave_complex', 'wave_re'):
                    probe = probe_wave_2x2(model, seed, args.n_probe, device)
                    probe['autocheck_harness_em'] = round(val_res['exact_match'], 3)
                    ok = abs(probe['em_alive'] - val_res['exact_match']) < 0.15
                    probe['autocheck_ok'] = bool(ok)
                    if not ok:
                        print('  WARNING: probe alive NO matchea harness - abort',
                              flush=True)
                        sys.exit(2)
                    # EM selectividad (TOST) a d=0 sin drift: probe erased reread.
                    x, y, seqs, v_erased = make_fr_probe_dataset(
                        args.n_probe, seed=1000 + seed, n_pairs_range=(32, 48),
                        kind='erased')
                    em_e, _, _ = replay_wave(model, x.to(device), y, seqs,
                                             v_erased, 'clave', 0.0, device,
                                             per_event=True)
                    probe['em_erased_d0'] = round(em_e, 4)
                    probe['selectividad'] = round(probe['em_alive'] - em_e, 4)
                    print(f'  probe: EM_vivo={probe["em_alive"]} '
                          f'sel={probe["selectividad"]}', flush=True)
                else:
                    # delta_nlms: EM alive/erased (selectividad para TOST).
                    probe = {}
                    for kind in ('alive', 'erased'):
                        x, y, seqs, v_erased = make_fr_probe_dataset(
                            args.n_probe, seed=1000 + seed, n_pairs_range=(32, 48),
                            kind=kind)
                        em = em_probe(model, x.to(device), y, seqs, device)
                        probe['em_' + kind] = round(em, 4)
                    probe['autocheck_harness_em'] = round(val_res['exact_match'], 3)
                    ok = abs(probe['em_alive'] - val_res['exact_match']) < 0.15
                    probe['autocheck_ok'] = bool(ok)
                    probe['selectividad'] = round(probe['em_alive'] - probe['em_erased'], 4)
                    print(f'  probe: EM_vivo={probe["em_alive"]} '
                          f'EM_fuga={probe["em_erased"]} sel={probe["selectividad"]}',
                          flush=True)
                out['runs'][label] = {'train_em': round(train_em, 4), 'probe': probe}

        # ---- TOST (wave_complex vs delta_nlms, selectividad, 5 seeds) ----
        tost_diffs = []
        for s in seeds:
            wc = out['runs'][f'wave_complex_s{s}']['probe'].get('selectividad')
            dl = out['runs'][f'delta_nlms_s{s}']['probe'].get('selectividad')
            if wc is not None and dl is not None:
                tost_diffs.append(wc - dl)
        if len(tost_diffs) == len(seeds):
            out['tost'] = tost_equivalence(tost_diffs)
        else:
            out['tost'] = {'status': 'NO COMPUTABLE', 'n_common': len(tost_diffs)}

    # ---- C1 a c emparejada: probe d=64 con n_pairs (17,24) ----
    # c = 19.5/64 = 0.3047, el mismo c que d=128. Espacio de claves
    # ORIGINAL (40/40, vocab 89) fuera del KeySpace.
    c1 = {}
    d64 = {'complex': f'{args.d64_dir}/n1_wave_complex/cache',
           're': f'{args.d64_dir}/n1_wave_re/cache'}
    for arm, chan in (('wave_complex', 'complex'), ('wave_re', 're')):
        c1[arm] = {}
        for seed in seeds[:3]:
            ck = f'{d64[chan]}/forget_retrieval_seed{seed}_dm64_L2_ep80.pt'
            if not os.path.exists(ck):
                print(f'  C1: falta {ck} - skip', flush=True)
                c1[arm][f's{seed}'] = {'status': 'skip'}
                continue
            model = build_wave(chan)(vocab_size=89, max_len=64, d_model=64,
                                     n_layers=2)
            model.load_state_dict(torch.load(ck, map_location=device)['model'])
            model.to(device).eval()
            c1[arm][f's{seed}'] = {}
            for cell in ('clave', 'estado'):
                c1[arm][f's{seed}'][cell] = {}
                for dlt in DELTAS:
                    x, y, seqs, v_erased = make_fr_probe_dataset(
                        args.n_probe, seed=1000 + seed, n_pairs_range=(17, 24),
                        kind='erased')
                    em, fuga, _ = replay_wave(model, x.to(device), y, seqs,
                                              v_erased, cell, dlt, device,
                                              per_event=True)
                    c1[arm][f's{seed}'][cell][str(dlt)] = round(fuga[0], 4)
    # diff d=128 vs d=64 por celda (max sobre delta y seeds 1-3)
    c1['diff_d128_d64'] = {}
    for arm in ('wave_complex', 'wave_re'):
        c1['diff_d128_d64'][arm] = {}
        for cell in ('clave', 'estado'):
            mx = 0.0
            for seed in seeds[:3]:
                if 'status' in c1[arm].get(f's{seed}', {}):
                    continue
                for dlt in DELTAS:
                    f128 = out['runs'][f'{arm}_s{seed}']['probe']['cells'][cell][str(dlt)]['fuga']
                    f64 = c1[arm][f's{seed}'][cell][str(dlt)]
                    mx = max(mx, abs(f128 - f64))
            c1['diff_d128_d64'][arm][cell] = round(mx, 4)
            c1['diff_d128_d64'][arm][cell + '_pass'] = bool(mx < 0.02)
    out['c1_matched_c'] = c1

    out['total_seconds'] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nEscrito: {args.out}  total={out["total_seconds"]}s', flush=True)


if __name__ == '__main__':
    main()