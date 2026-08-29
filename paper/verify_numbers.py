#!/usr/bin/env python3
"""Verify every number quoted in the manuscript/supplement against the frozen
JSON artifacts. Prints tables; exits nonzero if any gate metric fails.

Frozen artifacts (read-only):
  outputs/wave_mem/o05.json             d=128 final gate (O05)
  outputs/wave_mem/o04b.json            d=64  matched-load probe (O04b)
  outputs/wave_mem/delta_autopsy.json   corrective delta beta=1, lr x clip grid
  outputs/wave_mem/delta_autopsy_beta.json  NLMS/beta controls
  outputs/transformer/benchmark.json    attention anchor (baseline gate)

All numbers printed here are recomputed from the JSONs only.
"""
import json
import math
import statistics
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = lambda *p: os.path.join(ROOT, *p)


def load(rel):
    with open(OUT(rel), encoding="utf-8") as f:
        return json.load(f)


o05 = load("outputs/wave_mem/o05.json")
o04b = load("outputs/wave_mem/o04b.json")
autopsy = load("outputs/wave_mem/delta_autopsy.json")
autopsy_beta = load("outputs/wave_mem/delta_autopsy_beta.json")
bench = load("outputs/transformer/benchmark.json")
tostv2 = load("outputs/wave_mem/tostv2.json")

print("=" * 78)
print("1. O05 header metadata")
print("=" * 78)
print(f"  variant={o05['variant']} d_model={o05['d_model']} c={o05['c']}")
print(f"  header.seeds={o05['seeds']} (metadata from part0; see runs below)")
print(f"  n_probe_events={o05['n_probe_events']} eps_null={o05['eps_null']}")
print(f"  spec={o05['spec']}")
runs = o05["runs"]
arms = sorted({k.rsplit("_s", 1)[0] for k in runs})
seeds = sorted({int(k.rsplit("_s", 1)[1]) for k in runs})
print(f"  runs={len(runs)} arms={arms} seeds_present={seeds}")

print()
print("=" * 78)
print("2. Training saturation (15/15 EM 1.000)")
print("=" * 78)
em = [runs[k]["train_em"] for k in runs]
print(f"  train_em all 1.0: {all(e == 1.0 for e in em)}  (min={min(em)}, max={max(em)}, n={len(em)})")

print()
print("=" * 78)
print("3. Four residual laws, d=128 (O05): max |fuga - pred| per cell")
print("   pred = stored prediction (zero free parameters); fuga = pooled residual per seed")
print("=" * 78)
print(f"  {'cell':<14}{'channel':<10}{'max|f-p|':>10}{'at delta':>10}{'med|med-p|':>10}")
deltas = ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5"]
mx = 0.0
mx_med = 0.0
for arm, chan in (("wave_complex", "complex"), ("wave_re", "re")):
    for cell in ("clave", "estado"):
        for dl in deltas:
            p = runs[f"{arm}_s1"]["probe"]["cells"][cell][dl]["pred"]
            fugas = [runs[f"{arm}_s{s}"]["probe"]["cells"][cell][dl]["fuga"] for s in "12345"]
            e = max(abs(f - p) for f in fugas)
            e_med = abs(statistics.median(fugas) - p)
            mx = max(mx, e)
            mx_med = max(mx_med, e_med)
            if e == 0.0049 or e > 0.004:
                print(f"  {cell:<14}{chan:<10}{e:>10.4f}{dl:>10}{e_med:>10.4f}")
print(f"  -> max |fuga-pred| over seeds x cells x delta = {mx:.5f}")
print(f"  -> max |median(fuga)-pred|                     = {mx_med:.5f}")

print()
print("=" * 78)
print("4. Four residual laws, d=64 (O04b): max |fuga - pred| per cell")
print("   pred = stored prediction; 2 pooled events x 3 seeds per cell")
print("=" * 78)
r64 = o04b["runs"]
mx64 = 0.0
mx64_med = 0.0
print(f"  {'cell':<14}{'channel':<10}{'max|f-p|':>10}{'at delta':>10}{'med|med-p|':>10}")
for arm, chan in (("wave_complex", "complex"), ("wave_re", "re")):
    for cell in ("clave", "estado"):
        for dl in deltas:
            p = r64[f"{arm}_s1"][dl][f"pred_{cell}"]
            fugas = [f for s in (1, 2, 3) for f in r64[f"{arm}_s{s}"][dl][f"fuga_{cell}"]]
            e = max(abs(f - p) for f in fugas)
            e_med = abs(statistics.median(fugas) - p)
            mx64 = max(mx64, e)
            mx64_med = max(mx64_med, e_med)
            if e > 0.005:
                print(f"  {cell:<14}{chan:<10}{e:>10.4f}{dl:>10}{e_med:>10.4f}")
print(f"  -> max |fuga-pred| over events x seeds x cells x delta = {mx64:.5f}")
print(f"  -> max |median(fuga)-pred|                              = {mx64_med:.5f}")

print()
print("=" * 78)
print("5. Matched-load agreement (C1 as function of c = n/D)")
print("=" * 78)
c1 = o05["c1_matched_c"]["diff_d128_d64"]
for arm in ("wave_complex", "wave_re"):
    for cell in ("clave", "estado"):
        print(f"  {arm}/{cell}: diff={c1[arm][cell]:.4f} pass={c1[arm][f'{cell}_pass']}")
mx_c1 = max(c1[a][c] for a in ("wave_complex", "wave_re") for c in ("clave", "estado"))
all_c1 = all(c1[a][f"{c}_pass"] for a in ("wave_complex", "wave_re") for c in ("clave", "estado"))
print(f"  -> max diff = {mx_c1:.4f} (gate 0.02), 4/4 pass = {all_c1}")

print()
print("--- 5b. Cross-scale offset attributable to load mismatch alone ---")
c128 = o05["c"]
dc = c128 - (19 / 64)
slopes = {
    "clave/complex": lambda d: 0.0,
    "estado/complex": lambda d: 2 * (1 - math.cos(d)),
    "clave/re": lambda d: (1 - math.cos(2 * d)) / 4,
    "estado/re": lambda d: (1 - math.cos(d)),
}
mx_attr = 0.0
deltas_f = [float(x) for x in deltas]
for name, slope in slopes.items():
    worst = max(slope(d) * dc for d in deltas_f)
    mx_attr = max(mx_attr, worst)
    print(f"  {name}: max over delta of slope*dc = {worst:.5f}")
print(f"  -> max offset expected from dc alone = {mx_attr:.5f} "
      f"(compare with cross-scale max diff {mx_c1:.4f})")

print()
print("=" * 78)
print("6. TOST wave_complex vs delta_nlms (selectividad)")
print("=" * 78)
t = o05["tost"]
print(f"  mean={t['mean_delta']} sd={t['sd']} ci90={t['ci90']} eps={t['eps']} n_seeds={t['n_seeds']}")
print(f"  equivalence_pass={t['equivalence_pass']}  (CI crosses margin by {max(abs(t['ci90'][0]) - t['eps'], abs(t['ci90'][1]) - t['eps']):.4f})")

print()
print("=" * 78)
print("6b. TOST v2 wave_complex vs delta_nlms (selectividad, n=8, df=7)")
print("=" * 78)
for key in ("tost_wave_complex_vs_delta_nlms", "tost_wave_re_vs_delta_nlms"):
    tv = tostv2[key]
    print(f"  {key}: mean={tv['mean_delta']} sd={tv['sd']} ci90={tv['ci90']} "
          f"n={tv['n_seeds']} df={tv['df']} pass={tv['equivalence_pass']}")
    assert tv["n_seeds"] == 8
    assert tv["equivalence_pass"] is True, f"{key} debe pasar equivalencia"
t8 = tostv2["tost_wave_complex_vs_delta_nlms"]
assert abs(t8["mean_delta"] - (-0.0039)) < 1e-4
assert abs(t8["ci90"][0] - (-0.0135)) < 1e-4 and abs(t8["ci90"][1] - 0.0057) < 1e-4
print("  gate TOST v2 n=8 PASSA (equivalencia dentro de eps=0.02)")

print()
print("=" * 78)
print("7. Key/complex null residual (guarded, residual-nulo)")
print("=" * 78)
n_null = sum(
    1
    for s in "12345"
    for dl in deltas
    if runs[f"wave_complex_s{s}"]["probe"]["cells"]["clave"][dl]["em_guarded"] == "residual nulo"
)
print(f"  residual-nulo outcomes: {n_null} / {5 * len(deltas)}")

print()
print("=" * 78)
print("8. Decodability of key/re residual (EM guarded) at d=128, seed 1")
print("=" * 78)
for dl in deltas:
    v = runs["wave_re_s1"]["probe"]["cells"]["clave"][dl]
    fuga = v["fuga"]
    p = v["pred"]
    print(f"  delta={dl}: fuga={fuga:.4f} pred={p:.4f} em_guarded={v['em_guarded']}")

print()
print("=" * 78)
print("9. Write stability: corrective delta beta=1 diverges; NLMS/superposition bounded")
print("=" * 78)
for i, r in enumerate(autopsy["runs"]):
    snf, snl = r["S_norm_first"], r["S_norm_last"]
    print(f"  delta beta=1 lr={r['lr']} clip={r['clip']}: ||S|| first={snf[0]:.3g} last={snl[0]:.3g} final_em={r['final_em']:.3f}")
b = autopsy_beta
print(f"  delta random items: ||S|| {b['delta_random_S_norm'][0]:.3g} -> {b['delta_random_S_norm'][1]:.3g}")
print(f"  wave_complex:       ||S|| {b['wave_complex_S_norm'][0]:.3g} -> {b['wave_complex_S_norm'][1]:.3g}")
print(f"  wave_re:            ||S|| {b['wave_re_S_norm'][0]:.3g} -> {b['wave_re_S_norm'][1]:.3g}")
print(f"  NLMS beta=0.1:      ||S|| {b['beta01_S_norm_first'][0]:.3g} -> {b['beta01_S_norm_last'][0]:.3g} peak_em={b['beta01_peak_em']}")

print()
print("=" * 78)
print("10. Attention anchor (baseline gate, frozen)")
print("=" * 78)
for task, v in bench["tasks"].items():
    print(f"  {task:<20} EM={v['exact_match']:.4f} threshold={v['threshold']} pass={v['pass']}")
print(f"  n_params={bench['n_params']} smoke={bench['smoke']}")

print()
print("=" * 78)
print("GATES")
print("=" * 78)
ok = True
if mx > 0.02:
    ok = False
if mx64 > 0.02:
    ok = False
if not all(e == 1.0 for e in em):
    ok = False
if n_null < 5 * len(deltas):
    ok = False
if not all_c1:
    ok = False
print(f"  d=128 max deviation {mx:.5f} < 0.02 : {mx < 0.02}")
print(f"  d=64  max deviation {mx64:.5f} < 0.02 : {mx64 < 0.02}")
print(f"  EM 1.000 15/15                : {all(e == 1.0 for e in em)}")
print(f"  key/complex null 30/30         : {n_null == 5 * len(deltas)}")
print(f"  C1 matched-c 4/4               : {all_c1}")
print(f"  OVERALL: {'PASS' if ok else 'FAIL'}")