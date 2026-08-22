# Verification Report — ONDA lab v1

**Date**: 2026-08-20 · **Closing commit**: `cec4ffe` · **Tag**: `lab-radioactivo-v1`
**Document**: closing report of the verification rounds O01-O05 and the verified state of the claims.

**Evidence rule**: every claim in this document is backed by a JSON artifact in `outputs/audit/audit_o0*_verification.json` or `outputs/wave_mem/*.json`, by a log in `lab/logs/O0*_test.md`, or by a standalone reproducible script in `tests/`. Verification was performed in adversarial review rounds; every round's checks are re-runnable from the scripts and checkpoints shipped in this package, with no dependency on the original reviewer. Where a number below is labeled "verified", it means: reproduced by an independent reproducible check (script + data in this package) and cross-checked in a separate review session.

---

## 1. Final verdict

**This laboratory has its paper.** Not the one claimed by the original claim v1 (a new phase-interference erasure mechanism), but a smaller, truer, better defended one: **forgetting as an operator-specific property, measured with three operators at matched budget, with four closed-form laws with zero free parameters verified at two scales**.

The final empirical arc:

| Operator | State in `forget_retrieval` |
|---|---|
| Attention | No erasure operator (anchor 0.067) |
| Delta-rule beta=1 (corrective write) | Has the operator; the write diverges by LMS stability (||S|| ~10^11 in one pass) |
| Delta NLMS (||k||=1) | Stable by construction; trains FR (EM ~1.0); almost-equivalent to wave_complex (honest TOST) |
| wave_mem (phasor superposition) | Open-loop operator, unconditionally stable + 4 closed-form residual laws + conditional guarantee of exact erasure |

---

## 2. Verified claims (with evidence level)

### C-A. Equivalence wave_mem = delta_forget in the ideal erase — *theorem*
Erase by complex re-read `M - w x (w^H M)/D` and delta beta=1 with an exact key are the same operator (diff 0.0 exact, verified by the lab and independently). It is reported as a **lineage theorem**, not as superiority. The "mechanism distinct from subtraction" contribution of claim v1 is **dead and withdrawn**.

### C-B. Four closed-form residual post-erase laws — *central result*
Grid channel {complex, Re} x drift {key, state}, closed form, zero free parameters:

| cell | law | d=64 (max deviation) | d=128 (max deviation) |
|---|---|---|---|
| key/complex | 0 (flat, exact) | 0.0000 | 0.0000 |
| state/complex | 2(1-cos delta)(1+c) | 0.0086 | included in 0.0049 |
| key/re | sin^4 delta + c(1-cos 2delta)/4 | 0.0016 | included in 0.0049 |
| state/re | (1-cos delta)^2 + c(1-cos delta) | 0.0014 | included in 0.0049 |

c = (n-1)/D. Verified on **trained models** at two scales; C1 confirmed as a function of c = n/D (4/4 cells at matched c, diff <= 0.0064). The represent laws (oracle channel) are agnostic to the injection variant and remained intact.

### C-C. Write stability: LMS condition — *theorem with mechanism*
Superposition write = open loop, unconditionally stable. Corrective write = closed loop with per-direction gain |1 - beta||k||^2|: stable iff beta||k||^2 < 2. Verified: beta=1 with ||k||^2=8 diverges (||S|| -> 10^11, the NaN of N1), beta=0.1 or NLMS bound it and train. **Reading for the paper**: the beta gates of DeltaNet/GDN are (among other things) stability devices for the corrective loop; wave_mem solves the same structural problem without gates. Lineage: Widrow-Hoff LMS/NLMS.

### C-D. Conditional erasure guarantee — *theorem + functional measurement*
Only one cell of the 2x2 has true erasure: **key/complex, residual exactly null** (6 delta x 5 seeds, null-residual guard applied). Under state drift, every channel leaves a structured copy of the value that the head decodes (fuga_power 0.05-0.32 but EM_erased 0.75-1.0): **power leakage underestimates functional leakage by more than an order of magnitude**. Exact forgetting requires write-to-erase clock coherence. Honesty note: in v1 (fixed codebook, non-oscillating state) neither drift occurs naturally — the 2x2 is a stress test that delimits the guarantee, not an operational failure mode.

### C-E. Control TOST wave_complex vs delta_nlms — *almost-equivalent, reported unrounded*
d=128, 5 seeds: mean -0.0031, IC90 [-0.0206, 0.0144] (crosses eps=0.02 by 0.0006). Reported as is. If review presses: 8-10 seeds close it (future appendix).

### C-F. Consolidated negative result
- Vector geometry C^D: SNR flat in D (~2/(n-1)); erase by re-read = total erasure. **Dead, archived as evidence.**
- Learned keys (v2): break phasor uniformity -> C1 does not apply. Deferred to future work.
- EM as gate metric: saturated (1.000 x 15 combos). The probes carry the evidence; EM is sanity.
---

## 3. Verification chain (epistemic value of the process)

| Round | What died / what was corrected |
|---|---|
| O01 | Geometric ambiguity (C^D vs C^{DxD}) — C1 only lives in matrix geometry. "Antiphase != subtraction" refuted (6e-17). Crosstalk floor dominates erasure. VSA/HRR risk identified. |
| O02 | wave=delta equivalence exposed -> delta_forget comparator is empty. Bugs caught: double rng in phasors; in-place setitem breaking autograd (permanent rule: state = list of deltas). |
| O03 | Three arms = two operators; wave_complex/delta reclassified as TOST control. Re-presentation channel added. Common drift delta replaces rho machinery. Statistical power fixed (n_pairs >= 16, >=300 events). Sigma convention reconciled with closed form (crossing sigma* = 0.638, both sides). |
| N1 | wave trains FR 1.0 (6/6, generalizes to fresh keys); delta beta=1 does not train. Autopsy: structural write failure (LMS), not capacity nor erase. re/complex contrast confirmed in fuga_power (exact half). |
| O04 | 2x2xdelta on trained models. Lab correction to the review amendment: reread-Re is not flat under drift (quartic law). delta_nlms resurrects the control. Permanent rule: pooled estimator (heavy tail of ||v||^2). |
| O04b | Per-cell drift injection (the O04 probe mixed two physical interventions across arms). 16/16 synthetic cells bit-exact in both conventions. Hybrid number (B-Re 0.0870) withdrawn: no clean physical cell. No-free-lunch reformulated as conditional guarantee (only exact cell: key/complex). |
| O05 | d=128 gate confirmed: 4 laws (max 0.0049), null residual with guard, C1 at matched c 4/4, honest TOST. First and only Kaggle spend of the lab (5.44h, 2xT4). |

**Structural lesson**: each round reduced the claim to something smaller and truer. The final claim is a strict subset of the initial one — and it is publishable precisely because of that.

---

## 4. Anti-reviewer checklist (mandatory before submission)

- [ ] **VSA/HRR lineage cited**: Plate (HRR), Kanerva (HDC), Frady et al. (capacity), resonator networks. Phasor superposition + matched filter + unbinding is that formalism; the novelty is the forgetting axis + residual theory under drift + matched-budget measurement. Do not claim mechanism novelty.
- [ ] **Fast-weight / linear attention lineage**: Schlag et al. (delta rule as fast-weight programmer), DeltaNet/GDN (beta as gate). Position C-C as a stability reading of those gates.
- [ ] **LMS/NLMS lineage**: Widrow-Hoff for the beta||k||^2 < 2 condition.
- [ ] **Matched-budget protocol in writing** (appendix): same FORGET_ID interface, same budget, frozen anchors (check_frozen --strict 9/9).
- [ ] **Metrics**: pooled fuga (sum|r|^2/sum|v|^2), never mean-of-ratios with heavy-tailed denominator. EM_erased only with null-residual guard. Selectivity as primary functional metric.
- [ ] **TOST reported unrounded**; seeds and CI in every table.
- [ ] **Limitations section**: fixed codebook (no natural drift in v1), real values, synthetic tasks, saturated EM, open TOST, v2 (learned keys) pending.
- [ ] **Reproducibility**: regenerable anchors, regenerable probes, JSON evidence shipped (regenerated, not committed, per lab rule).

---

## 5. Post-v1 backlog (non-blocking)

1. **O06 — realistic state drift**: delta(t) = omega x lag write-to-forget; probe with delta growing per event; does training learn to compensate slow drift?
2. **v2 — learned keys**: residual theory without phasor uniformity (contribution (2) of the claim in full form).
3. **Hybrid channel**: if a design can cover both 2x2 cells (e.g. double complex+Re erase), characterize it — but the conjecture is that none exists (structural no-free-lunch).
4. **Scale**: d>=256 only if a reviewer asks; C1 as a function of c is already verified at two scales.

---

## 6. Verification process (how to read the numbers in this package)

Five rounds, three original claims reduced to one, two theorems gained along the way (lineage equivalence, LMS stability), four closed laws, one conditional guarantee. The process worked: **nothing in this package depends on a number that is not independently reproducible** from `tests/` + `outputs/` shipped here.

All verification artifacts:
- `outputs/audit/audit_o01b_verification.json`, `audit_o02_verification.json`, `audit_o03_verification.json`, `audit_o04b_verification.json`
- `outputs/wave_mem/{probes,n1,n1_heldout,o04,o04b,o04_laws_verify,delta_autopsy,delta_autopsy_beta}.json`
- `outputs/wave_mem/o05.json` (merged 5-seed O05 record), `o05_part0.json`, `o05_part1.json`, `o05_run.log`
- `outputs/transformer/benchmark.json` (frozen attention anchor)
- Trained checkpoints in `outputs/{n1_wave_complex,n1_wave_re,o04_delta_nlms,o5_wave_complex,o5_wave_re,o5_delta_nlms}/cache/`

Tag authorized: `lab-radioactivo-v1` on `cec4ffe`.
