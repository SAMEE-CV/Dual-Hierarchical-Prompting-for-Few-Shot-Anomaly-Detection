# Verification record

Verified on 2026-10-01 with Windows, Python 3.13.7, PyTorch 2.14.0+cpu, torchvision 0.29.0, and open-clip-torch 3.3.0.

## Checks completed

| Check | Result |
|---|---|
| `ruff check .` | Passed |
| `ruff format --check .` | Passed |
| `pytest` | 9 tests passed |
| `python -m dual_prompt smoke --config configs/smoke.yaml` | Passed one complete epoch |
| Checkpoint reload and evaluation | Passed; reproduced the original smoke metrics |
| Dual OpenCLIP ViT-B/16 prompted forward/backward | Passed; text gradients present and CLIP gradients absent |
| MVTec dataset validation | Passed for all 15 categories |
| Real MVTec, all 15 categories, 4-shot, 1 epoch | Passed for every category |
| Real checkpoint reload | Passed; reproduced every reported metric exactly |

The OpenCLIP integration check returned logits with shape `[1, 2]` and patch features with shape `[1, 196, 768]` at blocks 0, 5, and 11. Gradients reached the hierarchical textual prompts and coupled visual prompt components while all frozen CLIP parameter gradients remained absent. The full model contains 10,643,718 trainable prompt, coupling, conditioner, and gate parameters.

The final real-data test used original OpenAI ViT-B/16 QuickGELU weights, seed 42, four normal support images, one SGD epoch, and the 83-image MVTec `bottle` test split:

| Metric | Result |
|---|---:|
| Image AUROC | 0.974603 |
| Image AUPR | 0.993306 |
| Pixel AUROC | 0.975091 |
| Pixel AUPR | 0.769648 |
| Pixel Dice at fixed threshold 0.5 | 0.000000 |

The fixed-threshold Dice value is included transparently; it is sensitive to score calibration and is not selected on test labels. The one-epoch numbers verify integration, checkpointing, and inference and must not be presented as the converged paper result. The synthetic smoke fixture remains useful for CI but has no research meaning.

The subsequent full MVTec integration run completed all 15 independently adapted categories. Its unweighted macro mean was:

| Metric | One-epoch macro mean |
|---|---:|
| Image AUROC | 0.896841 |
| Image AUPR | 0.942835 |
| Pixel AUROC | 0.956207 |
| Pixel AUPR | 0.516340 |
| Pixel Dice at fixed threshold 0.5 | 0.000000 |

Per-category metrics and artifacts are under `outputs/mvtec_all_1epoch/`. The spread across categories, including weak one-epoch image detection on screw, is retained to avoid presenting an integration run as a tuned benchmark result.
