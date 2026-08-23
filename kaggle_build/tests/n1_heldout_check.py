"""Held-out check del 1.000 (ajuste 1 de la auditoria O03-N1):
eval FR de los 6 ckpts wave con claves/valores FRESCOS (mismo generador,
seed de datos distinta, seed=7777) vs la seed de validacion original
(seed+100). Si EM ~1.000 en ambas, el head generaliza el crosstalk
(no memoriza la distribucion de eval)."""

import sys
import torch
from functools import partial
from torch.utils.data import DataLoader

sys.path.insert(0, '.')
from tests.common_smoke import (ForgetRetrieveDataset, collate_fn, evaluate,
                                ANSWER_ID, VOCAB_FORGET_SIZE)
from tests.wave_mem_n1 import build_wave

dev = torch.device('cpu')
col = partial(collate_fn, answer_marker_id=ANSWER_ID, mark_after_marker=False,
              prefix_answer=False)
out = {}
for arm in ('wave_complex', 'wave_re'):
    for seed in (1, 2, 3):
        m = build_wave('complex' if arm == 'wave_complex' else 're')(
            VOCAB_FORGET_SIZE, 64, 64, 2)
        ck = torch.load(f'outputs/n1_{arm}/cache/forget_retrieval_seed{seed}'
                        f'_dm64_L2_ep80.pt', map_location=dev)
        m.load_state_dict(ck['model'])
        m.eval()
        row = {}
        for label, dseed in (('val_seed_plus100', seed + 100), ('held_out_7777', 7777)):
            ds = ForgetRetrieveDataset(150, seed=dseed, n_pairs_range=(16, 24),
                                       n_forget_range=(1, 2))
            vl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=col)
            r = evaluate(m, vl, dev, VOCAB_FORGET_SIZE)
            row[label] = round(float(r['exact_match']), 4)
            print(f'{arm} s{seed} {label}: EM={row[label]:.3f}')
        out[f'{arm}_s{seed}'] = row
import json
with open('outputs/wave_mem/n1_heldout.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)