"""Runner O02/O03: G0 + Fase 0-lite + probes (Clase A, canales, n/D, Clase B',
selectividad plumbing) para wave_mem y delta_forget. Pre-registro en
logs/O02_test.md y logs/O03_test.md. Salida unica: outputs/wave_mem/probes.json.
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

from prototypes.wave_mem.model import WaveMemLM
from prototypes.delta_forget.model import DeltaForgetLM
from tests.common_smoke import (BOS_ID, SEP_ID, QUERY_ID, FORGET_ID,
                                ANSWER_ID, K_OFFSET_MQAR, V_OFFSET_MQAR,
                                N_KEYS_MQAR, N_VALS_MQAR, PAD_ID)


def gen_probe_sample(rng, n_pairs=6, n_forget=1, q_kind='alive'):
    ids_k = rng.sample(range(N_KEYS_MQAR), n_pairs)
    ids_v = rng.sample(range(N_VALS_MQAR), n_pairs)
    ei = rng.randrange(n_pairs)
    alive = [i for i in range(n_pairs) if i != ei]
    qi = ei if q_kind == 'erased' else rng.choice(alive)
    order = list(range(n_pairs))
    rng.shuffle(order)
    store = []
    for p in order:
        store += [K_OFFSET_MQAR + ids_k[p], V_OFFSET_MQAR + ids_v[p]]
    seq = ([BOS_ID] + store + [SEP_ID, FORGET_ID, K_OFFSET_MQAR + ids_k[ei],
            SEP_ID, QUERY_ID, K_OFFSET_MQAR + ids_k[qi], ANSWER_ID])
    return seq, ids_v[ei], ids_v[qi]


def collate_probe(seqs, targets):
    T = max(len(s) for s in seqs)
    B = len(seqs)
    x = torch.full((B, T), PAD_ID, dtype=torch.long)
    y = torch.full((B, T), PAD_ID, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.bool)
    for i, (s, t) in enumerate(zip(seqs, targets)):
        x[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        pos = s.index(ANSWER_ID)
        y[i, pos] = t
        mask[i, pos] = True
    return x, y, mask


def g0_checks(model, device, label):
    rng = random.Random(7)
    seqs = [gen_probe_sample(rng, n_pairs=6)[0] for _ in range(4)]
    extra = [rng.randrange(9, 89) for _ in range(3)]
    x1 = torch.tensor([s for s in seqs], dtype=torch.long, device=device)
    x2 = torch.tensor([s + extra for s in seqs], dtype=torch.long, device=device)
    with torch.no_grad():
        logits1 = model(x1)
        logits2 = model(x2)
    suffix_ok = bool(torch.equal(logits1, logits2[:, :x1.size(1)]))
    state = model.init_state(x1.size(0), device)
    with torch.no_grad():
        loop = []
        for t in range(x1.size(1)):
            lt, state = model.decode_step(x1[:, t], state)
            loop.append(lt)
        loop_logits = torch.stack(loop, dim=1)
    parity_ok = bool(torch.equal(logits1, loop_logits))
    seqs2, tgts = [], []
    for _ in range(8):
        s, _, t = gen_probe_sample(rng, n_pairs=6, q_kind='alive')
        seqs2.append(s)
        tgts.append(t)
    xb, yb, mb = collate_probe(seqs2, tgts)
    xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
    logits = model(xb)
    loss = torch.nn.functional.cross_entropy(
        logits[mb], yb[mb], reduction='sum')
    loss.backward()
    grads_ok = True
    grad_info = {}
    for name, p in model.named_parameters():
        if p.grad is None or float(p.grad.abs().sum()) == 0.0:
            grads_ok = False
            grad_info[name] = 'ZERO'
        else:
            grad_info[name] = round(float(p.grad.abs().sum()), 6)
    return {'model': label, 'suffix_bit_exact': suffix_ok,
            'full_equals_rollout': parity_ok, 'grads_connected': grads_ok,
            'grad_l1_by_param': grad_info}


def phase0_lite(model, device, label, steps=5):
    rng = random.Random(11)
    seqs, tgts = [], []
    for _ in range(32):
        s, _, t = gen_probe_sample(rng, n_pairs=6, q_kind='alive')
        seqs.append(s)
        tgts.append(t)
    xb, yb, mb = collate_probe(seqs, tgts)
    xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(
            logits[mb], yb[mb], reduction='mean')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(round(float(loss.item()), 4))
    finite = all(math.isfinite(l) for l in losses)
    decreasing = len(losses) >= 2 and losses[-1] < losses[0]
    return {'model': label, 'losses': losses, 'finite': finite,
            'decreasing': decreasing,
            'pass': finite and decreasing}


def unit_phasor(rng):
    th = 2 * math.pi * rng.random()
    return complex(math.cos(th), math.sin(th))


def probe_clase_a():
    D = 128
    rng = random.Random(3)
    u = torch.tensor([[unit_phasor(rng) for _ in range(D)]], dtype=torch.complex64)
    M = torch.randn(1, D, D, dtype=torch.float32)
    M = torch.complex(M, torch.randn(1, D, D, dtype=torch.float32))
    wh = torch.einsum('bd,bdj->bj', torch.conj(u), M)
    M_wave = M - u.unsqueeze(-1) * (wh / D).unsqueeze(1)
    n2 = (u.conj() * u).sum(dim=1, keepdim=True)
    M_delta = M - u.unsqueeze(-1) * (wh / n2).unsqueeze(1)
    diff = float((M_wave - M_delta).abs().max())
    nrm = float(M.abs().max())
    return {'D': D, 'max_abs_diff': diff, 'rel_to_M': diff / max(nrm, 1e-12)}


def structural_pipeline_check(model, device):
    rng = random.Random(5)
    D = model.d_model
    seq, _, _ = gen_probe_sample(rng, n_pairs=6, q_kind='erased')
    x = torch.tensor([seq], dtype=torch.long, device=device)
    state = model.init_state(1, device)
    with torch.no_grad():
        for t in range(x.size(1)):
            _, state = model.decode_step(x[:, 0, t:t + 1].squeeze(0), state)
    r = state.r[0][0]
    tok_vj = seq[seq.index(ANSWER_ID) + 1] if len(seq) > seq.index(ANSWER_ID) + 1 else None
    return {'readout_power': float(r.abs().pow(2).sum().item()),
            'r_norm': float(r.abs().norm().item()), 'D': D}


def probe_canales(D=128, n=16, trials=20):
    """v REAL (decision O03): canal complejo -> fuga exacta 0; canal re ->
    (n-1)/(2D) (sin el termino 1/2, que venia del valor complejo); orig_v
    -> (n-1)/D. Lectura siempre con la clave limpia w[k]."""
    out = {}
    fugas = {'complex': [], 're': [], 'orig_v': []}
    for tr in range(trials):
        rng = random.Random(101 + tr)
        w = [[unit_phasor(rng) for _ in range(D)] for _ in range(n)]
        v = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(n)]
        M = [[sum(w[k][i] * v[k][j] for k in range(n)) for j in range(D)]
             for i in range(D)]
        k = n // 2
        wh = [sum((w[k][i].conjugate()) * M[i][j] for i in range(D)) / D
              for j in range(D)]
        for canal, vhat in (('complex', wh),
                            ('re', [complex(x.real, 0.0) for x in wh]),
                            ('orig_v', v[k])):
            Ma = [[M[i][j] - w[k][i] * vhat[j] for j in range(D)]
                  for i in range(D)]
            r = [sum((w[k][i].conjugate()) * Ma[i][j] for i in range(D)) / D
                 for j in range(D)]
            fuga = sum(abs(x) ** 2 for x in r) / sum(x * x for x in v[k])
            fugas[canal].append(fuga)
    for canal, fs in fugas.items():
        out[canal] = {'mean': round(sum(fs) / len(fs), 4),
                      'n_trials': len(fs)}
    out['pred_complex'] = 0.0
    out['pred_re'] = round((n - 1) / (2 * D), 4)
    out['pred_orig_v'] = round((n - 1) / D, 4)
    out['note'] = 'v real (O03): pred_re sin el termino 1/2'
    return out


def probe_n_over_d():
    law = []
    grid = [(64, 4), (64, 8), (64, 16), (64, 32),
            (128, 8), (128, 16), (128, 32), (128, 64),
            (256, 16), (256, 32), (256, 64), (256, 128)]
    for D, n in grid:
        fs = []
        for s in range(5):
            rng = random.Random(11 + s)
            w = [[unit_phasor(rng) for _ in range(D)] for _ in range(n)]
            v = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(n)]
            M = [[sum(w[k][i] * v[k][j] for k in range(n)) for j in range(D)]
                 for i in range(D)]
            kk = n // 2
            Ma = [[M[i][j] - w[kk][i] * v[kk][j] for j in range(D)]
                  for i in range(D)]
            r = [sum((w[kk][i].conjugate()) * Ma[i][j] for i in range(D)) / D
                 for j in range(D)]
            fs.append(sum(abs(x) ** 2 for x in r) / sum(x * x for x in v[kk]))
        law.append({'D': D, 'n': n, 'n_over_D': round(n / D, 3),
                    'fuga_mean5': round(sum(fs) / len(fs), 3),
                    'pred_(n-1)/D': round((n - 1) / D, 3)})
    fit_rows = [p for p in law if p['n_over_D'] < 0.4]
    xs = [((p['n'] - 1) / p['D']) for p in fit_rows]
    ys = [p['fuga_mean5'] for p in fit_rows]
    slope = intercept = r2 = None
    if all(x > 0 for x in xs) and all(y > 0 for y in ys):
        lx = [math.log(x) for x in xs]
        ly = [math.log(y) for y in ys]
        bx = sum(lx) / len(lx)
        by = sum(ly) / len(ly)
        num = sum((a - bx) * (b - by) for a, b in zip(lx, ly))
        den = sum((a - bx) ** 2 for a in lx)
        slope = round(num / den, 3) if den > 0 else None
        intercept = round(by - slope * bx, 3) if slope is not None else None
        if slope is not None and den > 0:
            pred = [bx + slope * (a - bx) for a in lx]
            ss_res = sum((b - p) ** 2 for b, p in zip(ly, pred))
            ss_tot = sum((b - by) ** 2 for b in ly)
            r2 = round(1 - ss_res / ss_tot, 3) if ss_tot > 0 else None
    return {'grid': law, 'fit': {'n_points': len(fit_rows),
                                 'excluded_n_over_D_ge_04': True,
                                 'loglog_slope': slope,
                                 'loglog_intercept': intercept,
                                 'r_squared': r2},
            'pred_slope': 1.0, 'flat_in_D_at_fixed_nD': _check_flat(law)}


def _check_flat(law):
    g = {}
    for p in law:
        g.setdefault(p['n_over_D'], []).append(p['fuga_mean5'])
    return {k: {'values': v, 'spread': round(max(v) - min(v), 3)}
            for k, v in sorted(g.items())}


def _n2(x):
    return sum(abs(a) ** 2 for a in x)


def _erase_reread(M, wu):
    wh = [sum((wu[i].conjugate()) * M[i][j] for i in range(len(wu)))
          / _n2(wu) for j in range(len(wu))]
    return [[M[i][j] - wu[i] * wh[j] for j in range(len(wu))]
            for i in range(len(wu))]


def _erase_represent(M, wu, vtrue):
    return [[M[i][j] - wu[i] * vtrue[j] for j in range(len(wu))]
            for i in range(len(wu))]


def _fuga(Ma, w, vtrue):
    r = [sum((w[i].conjugate()) * Ma[i][j] for i in range(len(w))) / _n2(w)
         for j in range(len(w))]
    return sum(abs(x) ** 2 for x in r) / sum(x * x for x in vtrue)


def _cruce_phase(c, lo=0.0, hi=2.0):
    def f(s):
        x = math.exp(-s * s)
        return ((1 - x) ** 2 + c * (1 - x)
                - c - (1 - math.exp(-s * s / 2)) ** 2)
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    s = (lo + hi) / 2
    x = math.exp(-s * s)
    rr = (1 - x) ** 2 + c * (1 - x)
    return {'sigma_cruce': round(s, 3),
            'fuga_reread_en_cruce': round(rr, 3),
            'leak_amplitud_pct': round(100 * math.sqrt(rr), 1)}


def _cruce_additive(c, D, lo=0.0, hi=2.0):
    def f(s):
        q = s * s / (1 + s * s)
        return q * q + c * q - c - s * s / D
    for _ in range(80):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    s = (lo + hi) / 2
    q = s * s / (1 + s * s)
    rr = q * q + c * q
    return {'sigma_cruce': round(s, 3),
            'fuga_reread_en_cruce': round(rr, 3),
            'leak_amplitud_pct': round(100 * math.sqrt(rr), 1)}


def probe_clase_b_prime(D=128, n=16, trials=10):
    """Clase B' (O03, auditoria): drift de fase comun delta sobre la clave de
    borrado, comparando los modos de borrado con la MISMA clave drifteada:
      - reread (proyeccion con clave drifteada): fuga EXACTA 0 a todo delta
        (e^{i delta} w preserva el rango 1-D: (I - e^{id} w w^H e^{-id}/D)
        = (I - w w^H / D)). Prediccion: 0.0000.
      - represent (erase con v VERDADERO y clave drifteada): fuga ~
        (n-1)/D + (2 - 2 cos delta). Prediccion incremental exacta.
      - delta_real_rotacion (analogo del delta en R: rotacion 2-D de la clave
        real): fuga ~ (2/D) sin^2(delta) (solo 2/D de la energia de la clave
        se mueve -> proteccion dimensional). Prediccion: ~0.0006 en D=128.
    Ruido i.i.d. (O03, reconciliado O01d): DOS modelos de ruido con forma
    cerrada, convencion de sigma explicitada:
      - phase: eps_d i.i.d. N(0, sigma) radianes, w~_d = w_d e^{i eps_d}
        (potencia de ruido sigma^2 por elemento). Forma cerrada:
        reread = (1-e^{-sig^2})^2 + (n-1)/D (1-e^{-sig^2})   [item + leak]
        represent = (n-1)/D + (1-e^{-sig^2/2})^2
      - additive: eps_d complejo N(0, sigma/sqrt(2)) por componente
        (E|eps|^2 = sigma^2 por elemento, convencion del auditor). Forma:
        reread = (sig^2/(1+sig^2))^2 + (n-1)/D sig^2/(1+sig^2)
        represent = (n-1)/D + sig^2/D
    Cruce analitico (reread == represent): phase ~0.66 rad (fuga ~0.15,
    leak de amplitud ~40%) y additive ~0.72 (D=128, n=16) — FUERA de todo
    regimen operativo (reloj real: sigma << 0.1 rad): la ventaja de reread
    cubre el regimen operativo completo; la inversion es limite asintotico.
    Nota convencion: sigma por componente (Re E Im cada uno N(0,sigma) =
    potencia 2 sigma^2) corre el cruce a ~0.51; por elemento (potencia
    sigma^2) a ~0.72. La forma cerrada es el ancla, no el cruce empirico."""
    out = {}
    drift = {}
    for dlt in (0.0, 0.2, 0.4, 0.6):
        rr, rp = [], []
        for tr in range(trials):
            rng = random.Random(300 + int(round(dlt * 10)) + tr)
            w = [[unit_phasor(rng) for _ in range(D)] for _ in range(n)]
            v = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(n)]
            M = [[sum(w[k][i] * v[k][j] for k in range(n)) for j in range(D)]
                 for i in range(D)]
            k = n // 2
            ph = complex(math.cos(dlt), math.sin(dlt))
            wd = [w[k][d] * ph for d in range(D)]
            rr.append(_fuga(_erase_reread(M, wd), w[k], v[k]))
            rp.append(_fuga(_erase_represent(M, wd, v[k]), w[k], v[k]))
        drift[str(dlt)] = {'reread': round(sum(rr) / len(rr), 4),
                           'represent': round(sum(rp) / len(rp), 4),
                           'pred_represent': round(
                               (n - 1) / D + (2 - 2 * math.cos(dlt)), 4),
                           'pred_reread': 0.0}
    out['drift_fase'] = drift
    rot = {}
    for dlt in (0.0, 0.2, 0.4, 0.6):
        vals = []
        for tr in range(trials):
            rng = random.Random(350 + int(round(dlt * 10)) + tr)
            w = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(n)]
            v = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(n)]
            M = [[sum(w[k][i] * v[k][j] for k in range(n)) for j in range(D)]
                 for i in range(D)]
            k = n // 2
            wd = list(w[k])
            c, s = math.cos(dlt), math.sin(dlt)
            wd[0], wd[1] = c * w[k][0] - s * w[k][1], s * w[k][0] + c * w[k][1]
            vals.append(_fuga(_erase_reread(M, wd), w[k], v[k]))
        rot[str(dlt)] = {'fuga': round(sum(vals) / len(vals), 6),
                         'pred_(2/D)sin2': round(2 * math.sin(dlt) ** 2 / D, 6)}
    out['delta_real_rotacion'] = rot
    iid = {}
    for sig in (0.2, 0.4, 0.6, 0.8):
        ph_rr, ph_rp, ad_rr, ad_rp = [], [], [], []
        for tr in range(trials):
            rng = random.Random(400 + int(sig * 10) + tr)
            w = [[unit_phasor(rng) for _ in range(D)] for _ in range(n)]
            v = [[rng.gauss(0, 1) for _ in range(D)] for _ in range(n)]
            M = [[sum(w[k][i] * v[k][j] for k in range(n)) for j in range(D)]
                 for i in range(D)]
            k = n // 2
            eps = [rng.gauss(0, sig) for _ in range(D)]
            wn = [w[k][d] * complex(math.cos(eps[d]), math.sin(eps[d]))
                  for d in range(D)]
            ph_rr.append(_fuga(_erase_reread(M, wn), w[k], v[k]))
            ph_rp.append(_fuga(_erase_represent(M, wn, v[k]), w[k], v[k]))
            ep2 = [complex(rng.gauss(0, sig / math.sqrt(2)),
                           rng.gauss(0, sig / math.sqrt(2)))
                   for _ in range(D)]
            wa = [w[k][d] + ep2[d] for d in range(D)]
            ad_rr.append(_fuga(_erase_reread(M, wa), w[k], v[k]))
            ad_rp.append(_fuga(_erase_represent(M, wa, v[k]), w[k], v[k]))
        c = (n - 1) / D
        iid[str(sig)] = {
            'phase': {'reread': round(sum(ph_rr) / len(ph_rr), 4),
                      'represent': round(sum(ph_rp) / len(ph_rp), 4),
                      'pred_reread': round((1 - math.exp(-sig * sig)) ** 2
                                           + c * (1 - math.exp(-sig * sig)), 4),
                      'pred_represent': round(
                          c + (1 - math.exp(-sig * sig / 2)) ** 2, 4)},
            'additive': {'reread': round(sum(ad_rr) / len(ad_rr), 4),
                         'represent': round(sum(ad_rp) / len(ad_rp), 4),
                         'pred_reread': round((sig * sig / (1 + sig * sig)) ** 2
                                              + c * sig * sig / (1 + sig * sig), 4),
                         'pred_represent': round(c + sig * sig / D, 4)},
        }
    out['iid_sweep'] = iid
    out['cruce_analitico'] = {
        'phase': _cruce_phase((n - 1) / D),
        'additive': _cruce_additive((n - 1) / D, D),
        'caveat': ('cruce FUERA de regimen operativo (reloj real: sigma << '
                   '0.1 rad); leak de amplitud en el cruce ~40%; la ventaja '
                   'de reread cubre el regimen completo')}
    out['note'] = ('inmunidad exacta = drift de fase (ruido que preserva el '
                   'rango 1-D de la clave); iid: formas cerradas fuga(sigma) '
                   'para phase y additive (convencion: sigma por elemento, '
                   'potencia sigma^2); rotacion 2-D del delta real = '
                   'protegida dimensionalmente O(1/D)')
    return out


def probe_selectividad_plumbing(models, device):
    rng = random.Random(42)
    res = {}
    for label, model in models:
        em_ok = fuga_ok = 0
        em_n = fuga_n = 0
        for kind in ('alive', 'erased'):
            seqs, tgts = [], []
            for _ in range(32):
                s, ev, tv = gen_probe_sample(rng, n_pairs=6, q_kind=kind)
                seqs.append(s)
                tgts.append(tv if kind == 'alive' else ev)
            xb, yb, mb = collate_probe(seqs, tgts)
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            with torch.no_grad():
                logits = model(xb)
                pred = logits.argmax(dim=-1)
                ok = int(((pred == yb) & mb).sum().item())
            if kind == 'alive':
                em_ok, em_n = ok, 32
            else:
                fuga_ok, fuga_n = ok, 32
        em_vivo = em_ok / em_n
        fuga = fuga_ok / fuga_n
        res[label] = {'em_vivo': round(em_vivo, 4), 'fuga': round(fuga, 4),
                      'selectividad': round(em_vivo - fuga, 4),
                      'note': 'plumbing: random-init, sin interpretar'}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--d_model', type=int, default=64)
    ap.add_argument('--n_layers', type=int, default=2)
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--out', type=str, default='outputs/wave_mem/probes.json')
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    t0 = time.time()
    out = {'variant': 'wave_mem', 'preset': 'probes', 'seed': args.seed,
           'd_model': args.d_model, 'n_layers': args.n_layers,
           'device': str(device)}

    models = [
        ('wave_mem_complex', WaveMemLM(vocab_size=89, max_len=32,
                                       d_model=args.d_model,
                                       n_layers=args.n_layers,
                                       read_proj='complex')),
        ('wave_mem_re', WaveMemLM(vocab_size=89, max_len=32,
                                  d_model=args.d_model, n_layers=args.n_layers,
                                  read_proj='re')),
        ('delta_forget', DeltaForgetLM(vocab_size=89, max_len=32,
                                       d_model=args.d_model,
                                       n_layers=args.n_layers)),
    ]
    for m in models:
        m[1].to(device)

    out['g0'] = [g0_checks(m, device, label) for label, m in models]
    out['phase0_lite'] = [phase0_lite(m, device, label) for label, m in models]
    out['probe_clase_a'] = probe_clase_a()
    out['probe_canales'] = probe_canales()
    out['probe_n_over_d'] = probe_n_over_d()
    out['probe_clase_b_prime'] = probe_clase_b_prime()
    out['probe_selectividad_plumbing'] = probe_selectividad_plumbing(models, device)
    out['total_seconds'] = round(time.time() - t0, 1)

    import platform
    out['env'] = {'python': platform.python_version(), 'torch': torch.__version__}
    try:
        import subprocess
        r = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO,
                           capture_output=True, text=True, shell=True, timeout=5)
        out['git_hash'] = r.stdout.strip() or None
    except Exception:
        out['git_hash'] = None

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, default=str)

    g0_all = all(g['suffix_bit_exact'] and g['full_equals_rollout']
                 and g['grads_connected'] for g in out['g0'])
    p0_all = all(p['pass'] for p in out['phase0_lite'])
    print(f"G0 all pass: {g0_all}   Fase0-lite all pass: {p0_all}")
    print("\n===JSON_START===")
    print(json.dumps({'g0_all': g0_all, 'phase0_all': p0_all,
                      'probe_clase_a': out['probe_clase_a'],
                      'probe_canales': out['probe_canales'],
                      'probe_n_over_d': {'fit': out['probe_n_over_d']['fit'],
                                         'flat_in_D': out['probe_n_over_d']['flat_in_D_at_fixed_nD']},
                      'probe_clase_b_prime': out['probe_clase_b_prime'],
                      'selectividad': out['probe_selectividad_plumbing']},
                     indent=2, default=str))
    print("===JSON_END===")
    print(f"Escrito: {args.out}")


if __name__ == '__main__':
    main()