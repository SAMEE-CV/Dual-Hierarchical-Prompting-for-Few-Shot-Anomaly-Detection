# Training and Evaluation

## Smoke test

```bash
python -m dual_prompt smoke \
  --config configs/trainers/DualPrompt/smoke.yaml
```

## Train one MVTec category

```bash
python -m dual_prompt train \
  --config configs/trainers/DualPrompt/vit_b16_4shot.yaml \
  --data-root /path/to/mvtec_anomaly_detection \
  --category bottle
```

Use `--category all` for all 15 categories. Each run saves a checkpoint, metrics, predictions, and anomaly maps under `outputs/`.

## Evaluate a checkpoint

```bash
python -m dual_prompt evaluate \
  --config configs/trainers/DualPrompt/vit_b16_4shot.yaml \
  --data-root /path/to/mvtec_anomaly_detection \
  --checkpoint outputs/mvtec/bottle/checkpoint.pt
```

The PowerShell and Bash launchers in `scripts/dual_prompt/` provide the same training command.

