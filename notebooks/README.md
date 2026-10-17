# Notebooks

Exploratory analysis only; production logic lives in `src/optrade_ds`.

- `01_exploration.ipynb` — load prices (DB or CSV), inspect features, fit the baseline ranker,
  and look at rank IC.

Run with Jupyter installed ad hoc (not a project dependency):

```bash
uv run --with jupyter jupyter lab notebooks/
```

Keep outputs cleared before sharing.
