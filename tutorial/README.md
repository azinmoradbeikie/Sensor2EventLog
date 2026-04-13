# Tutorial

This directory contains a minimal, self-contained walkthrough for the repository.

- `toy_walkthrough.py` generates a synthetic sensor dataset, extracts features, evaluates rule diagnostics, and creates a simple interval event log.
- `toy_walkthrough.ipynb` mirrors the same flow in notebook form for interactive exploration and includes a repo-root bootstrap step for more reliable imports.
- `output/` is created when the script is run and stores CSV tutorial artifacts.

Run the tutorial script from the repository root:

```bash
python3 tutorial/toy_walkthrough.py
```

If you want to run the notebook in Jupyter, make sure your virtual environment also has `ipykernel` installed:

```bash
python -m pip install ipykernel
```
