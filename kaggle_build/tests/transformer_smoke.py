import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from prototypes.transformer.model import TransformerLM
from tests.common_smoke import run_smoke, print_summary


def build(vocab_size: int, max_len: int, d_model: int, n_layers: int):
    # Vanilla baseline: d_model=64, n_heads=4, ff_mult=2, abs-pos, no RoPE.
    # n_layers passed through; max_len + 8 margin like the other runners.
    n_heads = 4 if d_model % 4 == 0 else 2
    return TransformerLM(vocab_size=vocab_size, d_model=d_model,
                          n_layers=n_layers, n_heads=n_heads, ff_mult=2,
                          max_len=max_len + 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--d_model', type=int, default=64)
    ap.add_argument('--n_layers', type=int, default=2)
    ap.add_argument('--smoke', type=str, default='cpu_quick',
                    choices=['cpu_quick', 'smoke', 'full'])
    ap.add_argument('--tasks', type=str, default='all')
    ap.add_argument('--epochs', type=int, default=-1)
    ap.add_argument('--out', type=str,
                    default='outputs/transformer/reasoning_smoke.json')
    ap.add_argument('--save-baseline', action='store_true',
                    help='Write outputs/transformer/benchmark.json snapshot. '
                         'Run this once when the benchmark is consolidated; '
                         'proto/s4d_ham runners read it for the vs-baseline check.')
    args = ap.parse_args()
    device = torch.device(args.device)
    summary = run_smoke(variant='transformer', build_fn=build, device=device,
                        seed=args.seed, d_model=args.d_model,
                        n_layers=args.n_layers, smoke=args.smoke,
                        tasks=args.tasks, epochs_override=args.epochs)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    print_summary(summary)

    if args.save_baseline:
        baseline_path = os.path.join(os.path.dirname(args.out), 'benchmark.json')
        snapshot = {
            'variant': 'transformer',
            'smoke': args.smoke,
            'seed': args.seed,
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'n_params': summary['n_params'],
            'tasks': {t: {
                'exact_match': tr['success_metric_value'],
                'threshold': tr['success_threshold'],
                'pass': tr['success_pass'],
                'final_loss': tr['final_loss'],
                'final_token_acc': tr['final_token_acc'],
                'seconds': tr['seconds'],
            } for t, tr in summary['tasks'].items()},
        }
        with open(baseline_path, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2, default=str)
        print(f"\nBaseline snapshot guardado en: {baseline_path}")
        print("(Otros prototipos compararan contra este snapshot con epsilon=0.0)")

    print("\n===JSON_START===")
    print(json.dumps({
        'variant': summary['variant'],
        'passed_count': summary['passed_count'],
        'total_tasks': summary['total_tasks'],
        'recommend_kaggle': summary['recommend_kaggle'],
        'n_params': summary['n_params'],
        'tasks': {t: {
            'value': tr['success_metric_value'],
            'threshold': tr['success_threshold'],
            'pass': tr['success_pass'],
        } for t, tr in summary['tasks'].items()},
    }, indent=2, default=str))
    print("===JSON_END===")
    print(f"\nEscrito: {args.out}")


if __name__ == '__main__':
    main()
