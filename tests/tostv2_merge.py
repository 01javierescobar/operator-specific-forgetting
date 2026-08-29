"""Merge tostv2_*.json (seeds 1..8) into a canonical tostv2.json artifact
with the n=8 powered equivalence test (df=7). Reproducible, no scipy."""
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "wave_mem")

T_CRIT_90 = {4: 2.132, 7: 1.8946}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


d15 = load(os.path.join(OUT, "tostv2_seeds15.json"))
seeds = {}
for s in range(1, 9):
    src = d15 if s <= 5 else load(os.path.join(OUT, f"tostv2_seed{s}.json"))
    seeds[s] = {arm: round(src["runs"][f"{arm}_s{s}"]["probe"]["selectividad"], 4)
                for arm in ("wave_complex", "wave_re", "delta_nlms")}


def tost(diffs, eps=0.02, name=""):
    n = len(diffs)
    mu = sum(diffs) / n
    sd = (sum((d - mu) ** 2 for d in diffs) / (n - 1)) ** 0.5
    se = sd / math.sqrt(n)
    half = T_CRIT_90[n - 1] * se
    return {
        "n_seeds": n,
        "mean_delta": round(mu, 4),
        "sd": round(sd, 4),
        "se": round(se, 4),
        "df": n - 1,
        "ci90": [round(mu - half, 4), round(mu + half, 4)],
        "equivalence_pass": bool(abs(mu) + half <= eps),
        "margin_cross": round(half - (eps - abs(mu)), 4),
        "eps": eps,
    }


wc_dl = [seeds[s]["wave_complex"] - seeds[s]["delta_nlms"] for s in range(1, 9)]
wre_dl = [seeds[s]["wave_re"] - seeds[s]["delta_nlms"] for s in range(1, 9)]

merged = {
    "note": "Cluster B TOST v2: 8 seeds x 3 arms (wave recipe 60x600; "
            "delta_nlms 150x1200). Replay externo, per-event, n=320 events/arm.",
    "selectividad_by_seed": {str(s): seeds[s] for s in range(1, 9)},
    "tost_wave_complex_vs_delta_nlms": tost(wc_dl),
    "tost_wave_re_vs_delta_nlms": tost(wre_dl),
    "per_seed_diffs": {
        "wave_complex_minus_delta_nlms": [round(d, 4) for d in wc_dl],
        "wave_re_minus_delta_nlms": [round(d, 4) for d in wre_dl],
    },
}
with open(os.path.join(OUT, "tostv2.json"), "w", encoding="utf-8") as f:
    json.dump(merged, f, indent=2)

t = merged["tost_wave_complex_vs_delta_nlms"]
print("TOST wave_complex - delta_nlms (n=8):")
print(f"  mean={t['mean_delta']} sd={t['sd']} ci90={t['ci90']} "
      f"pass={t['equivalence_pass']}")
t2 = merged["tost_wave_re_vs_delta_nlms"]
print("TOST wave_re - delta_nlms (n=8):")
print(f"  mean={t2['mean_delta']} sd={t2['sd']} ci90={t2['ci90']} "
      f"pass={t2['equivalence_pass']}")
print("escrito:", os.path.join(OUT, "tostv2.json"))