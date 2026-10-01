# Datasets

Download MVTec AD from its [official page](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it without changing category names.

```text
mvtec_anomaly_detection/
├── bottle/
│   ├── train/good/
│   ├── test/good/
│   ├── test/<defect>/
│   └── ground_truth/<defect>/
└── ... 15 categories
```

Validate the dataset:

```bash
python -m dual_prompt inspect-data --root /path/to/mvtec_anomaly_detection
```

The dataset is not included in this repository.

