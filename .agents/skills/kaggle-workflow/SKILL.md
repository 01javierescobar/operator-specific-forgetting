---
name: kaggle-workflow
description: >-
  Prepares, pushes, monitors, and downloads results for remote GPU experiments
  executed on Kaggle using the Python Kaggle CLI.
---

# Kaggle Workflow Skill

Use this skill when you need to execute GPU-intensive benchmarks, multi-seed training sweeps, or Triton/CUDA kernels on Kaggle remote GPUs.

## Prerequisites & Authentication

- Verify Kaggle CLI is functional:
  ```powershell
  python -m kaggle kernels list --mine
  ```
- Credential file resides at `%USERPROFILE%\.kaggle\kaggle.json` (or `~/.kaggle/kaggle.json`).

## Workflow Steps

### 1. Structure the Experiment
Ensure the experiment is organized in `kaggle/<experiment_name>/`:
- `kernel-metadata.json`:
  ```json
  {
    "id": "<kaggle_username>/<kernel_slug>",
    "title": "<Human Readable Title>",
    "code_file": "<notebook_name>.ipynb",
    "language": "python",
    "kernel_type": "notebook",
    "is_private": true,
    "enable_gpu": true,
    "enable_internet": true,
    "dataset_sources": [],
    "kernel_sources": [],
    "competition_sources": [],
    "model_sources": [],
    "machine_shape": "NvidiaTeslaT4"
  }
  ```
- `<notebook_name>.ipynb`: Self-contained Jupyter notebook containing data generators, model definitions, metrics, and training loops.

### 2. Push & Launch the Kernel
Run from repository root:
```powershell
python -m kaggle kernels push -p kaggle/<experiment_name>
```

### 3. Non-Blocking Status Polling
Check kernel status:
```powershell
python -m kaggle kernels status <kaggle_username>/<kernel_slug>
```
Statuses:
- `KernelWorkerStatus.QUEUED` / `KernelWorkerStatus.RUNNING`: Kernel is active.
- `KernelWorkerStatus.COMPLETE`: Ready to download outputs.
- `KernelWorkerStatus.ERROR`: Check logs for traceback.

### 4. Download Outputs and Logs
When complete, retrieve logs and JSON results:
```powershell
python -m kaggle kernels output <kaggle_username>/<kernel_slug> -p kaggle/<experiment_name>/outputs
```

### 5. Parse Output Safely
Read logs in Python with UTF-8 encoding:
```python
with open('kaggle/<experiment_name>/outputs/<kernel_slug>.log', encoding='utf-8', errors='ignore') as f:
    print(f.read())
```

## Agent Guidelines

- **GPU Concurrency**: Run only 1 active GPU notebook at a time to prevent quota exhaustion or queue blockages on Kaggle free tier.
- **CPU vs GPU Partitioning**: Keep quick smoke tests and contract checks (< 15 mins) on local CPU. Offload multi-seed runs ($N \ge 3$) and $> 2000$ steps to Kaggle GPU.
