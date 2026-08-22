import sys, torch, math
sys.path.insert(0, '.')
from tests.common_smoke import (ForgetRetrieveDataset, collate_fn, evaluate,
                                ANSWER_ID, PAD_ID, VOCAB_FORGET_SIZE)
from tests.wave_mem_n1 import build_delta
from torch.utils.data import DataLoader
from functools import partial
import torch.nn.functional as F
import torch.nn as nn


def quick_train(npr, epochs=25, seed=1, max_seq=32):
    dev = torch.device('cpu')
    m = build_delta(VOCAB_FORGET_SIZE, max_seq, 64, 2)
    col = partial(collate_fn, answer_marker_id=ANSWER_ID, mark_after_marker=False,
                  prefix_answer=False)
    tr = ForgetRetrieveDataset(300, seed=seed, n_pairs_range=npr, n_forget_range=(1, 2))
    va = ForgetRetrieveDataset(80, seed=seed + 1, n_pairs_range=npr, n_forget_range=(1, 2))
    tl = DataLoader(tr, batch_size=32, shuffle=True, collate_fn=col)
    vl = DataLoader(va, batch_size=32, shuffle=False, collate_fn=col)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
    peak = 0.0
    s_norm = 0.0
    for ep in range(1, epochs + 1):
        m.train()
        tl_ = 0.0
        for x, y, tm in tl:
            yt = torch.where(tm, y, torch.full_like(y, PAD_ID))
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(m(x).reshape(-1, VOCAB_FORGET_SIZE),
                                   yt.reshape(-1), ignore_index=PAD_ID)
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tl_ = loss.item()
        if not math.isfinite(tl_):
            return ep, float('nan'), peak, s_norm
        v = evaluate(m, vl, dev, VOCAB_FORGET_SIZE)
        peak = max(peak, v['exact_match'])
        with torch.no_grad():
            S = m.init_state(1, dev)
            xb, yb, tm2 = next(iter(vl))
            m(xb[:4])
            s_norm = max(s_norm, float(m.blocks[0].k_proj.weight.norm()))
        if ep % 5 == 0:
            print(f'  npr={npr} ep={ep} train_loss={tl_:.4f} valEM={v["exact_match"]:.3f}')
    return epochs, tl_, peak, s_norm


print('=== delta default n_pairs (5,9) ===')
ep, l, peak, sn = quick_train((5, 9))
print(f'RESULT default: nan_at={ep} final_loss={l} peakEM={peak}')

print('=== delta power-spec n_pairs (16,24) ===')
ep, l, peak, sn = quick_train((16, 24), max_seq=64)
print(f'RESULT power: nan_at={ep} final_loss={l} peakEM={peak}')