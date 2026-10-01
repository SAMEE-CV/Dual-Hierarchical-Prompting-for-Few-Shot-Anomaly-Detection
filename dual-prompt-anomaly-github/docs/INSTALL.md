# Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch torchvision
pip install -e ".[dev]"
```

On Windows, activate with `.venv\Scripts\activate`. Install the CUDA-specific PyTorch build recommended by the [PyTorch installer](https://pytorch.org/get-started/locally/) for GPU training.

Run the checks with:

```bash
ruff check .
pytest
```

