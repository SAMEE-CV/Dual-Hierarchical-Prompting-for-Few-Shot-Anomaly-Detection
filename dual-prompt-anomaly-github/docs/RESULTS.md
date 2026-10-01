# Results

The implementation was tested for one epoch on all 15 MVTec AD categories with four normal support images per category.

| Metric | Macro mean |
|---|---:|
| Image AUROC | 0.8968 |
| Image AUPR | 0.9428 |
| Pixel AUROC | 0.9562 |
| Pixel AUPR | 0.5163 |

The `bottle` category achieved 0.9746 image AUROC and 0.9751 pixel AUROC. All checkpoints, prediction files, and 1,725 anomaly maps were generated successfully.

These are one-epoch integration results, not converged paper results. The fixed 0.5 Dice threshold was not calibrated on test labels.

