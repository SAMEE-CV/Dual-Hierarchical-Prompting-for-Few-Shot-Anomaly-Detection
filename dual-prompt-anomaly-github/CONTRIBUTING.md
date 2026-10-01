# Contributing

Thank you for helping improve this research implementation.

1. Create a focused branch and add tests for behavior changes.
2. Run `ruff check .`, `ruff format --check .`, and `pytest`.
3. Keep paper-faithful behavior separate from experimental extensions.
4. Document new hyperparameters and state whether they come from the paper or are implementation choices.
5. Do not commit MVTec AD data, pretrained weights, or generated experiment outputs.

Bug reports should include the command, configuration, package versions, GPU, random seed, and complete traceback.

