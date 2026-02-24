# Blind-Assistance

YOLO-based object detection training workflow for blind-assistance use cases.

## Project Flow

This project follows a 3-step training process:

1. **Baseline training** started with base parameters from `hyper_parameters/base_params.yaml`.
2. **Hyperparameter sweep** was performed on Weights & Biases (W&B) using sweep parameter configs in:
   - `hyper_parameters/sweeping_params.yaml`
   - `hyper_parameters/final_params.yaml`
3. **Final training with sweeped params** is run using `training_scripts/sweep.py` with the best values selected from W&B.

## Repository Structure

- `training_scripts/baseline.py`: baseline YOLO training script.
- `training_scripts/sweep.py`: training script that uses W&B sweep-selected parameters.
- `hyper_parameters/`: baseline and sweep config files.
- `model_training_results/`: exported training outputs (`results.csv`, `args.yaml`, weights).
- `wandb/`: W&B run artifacts and hyperparameter sweep results.

## Quick Start

### 1) Baseline run

```bash
python training_scripts/baseline.py
```

### 2) W&B sweep (using sweep params)

```bash
wandb sweep hyper_parameters/sweeping_params.yaml
# or
wandb sweep hyper_parameters/final_params.yaml
```

### 3) Train with sweeped params

```bash
python training_scripts/sweep.py
```

> Note: current baseline script uses `data/dataset.yaml`. Ensure your dataset YAML exists at that path (or update script paths accordingly).

## Hyperparameter and Result Artifacts

- Sweep outcomes and run logs are stored in `wandb/`.
  - Run summary: `wandb/files/wandb-summary.json`
  - Sweep/metric curves: `wandb/files/media/table/curves/`
- Final training outputs are in `model_training_results/`.
  - Metrics by epoch: `model_training_results/results.csv`
  - Final args (including tuned values): `model_training_results/args.yaml`
  - Exported/best weights: `model_training_results/weights/`

## Best Tuned Values (from stored artifacts)

From `model_training_results/args.yaml`, tuned values include:

- `cls: 0.6`
- `lr0: 0.005893699257589709`
- `mixup: 0.05`
- `hsv_s: 0.7`

These were used for the final sweep-based training stage.
