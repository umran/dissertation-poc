# Bayesian Actor–Critic (BAC) — Experiments

This repo contains experiment scripts and notebooks for HMC-AC (Hamiltonian Monte Carlo Actor–Critic) and Bootstrapped-AC.

## Installation

1. **Install `uv` (if you don’t have it yet)**  
   `uv` is a fast Python package/venv manager.

-   macOS / Linux:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
-   Windows (PowerShell):
    ```powershell
    powershell -ExecutionPolicy ByPass -c "iwr https://astral.sh/uv/install.ps1 -UseBasicParsing | iex"
    ```

2. **Sync the environment** (from the project root):

```bash
uv sync
```

3. **Editable install** (from the project root):

```bash
uv pip install -e .
```

---

## Running Experiments

-   All experiment entry points are **main scripts** located under:
    ```
    src/bac/experiments/
    ```
-   To run an experiment, execute (from the project root):
    ```bash
    uv run -m bac.experiments.<experiment_script> --seed <some_integer_seed>
    ```
    Replace `<experiment_script>` with the module name (filename without `.py`) and provide an **integer** for `--seed`.  
    **All experiment scripts expect an integer seed** in order to run.

### Saving Artifacts

For experiment artifacts to be saved, create this directory structure at the project root **before** running:

```
out/
└── prod/
```

### Viewing Results

From the project root, open one of the notebooks:

-   `hmc_plots.ipynb` — plots for **HMC-AC**
-   `bs_plots.ipynb` — plots for **Bootstrapped-AC**

You can open them in Jupyter or your preferred notebook environment.

### Using Precomputed Results (Optional)

If you have a ZIP file of precomputed experiment results (available from the author), **unzip its contents into**:

```
out/prod
```

as a **flat list** (i.e., files directly inside `out/prod`, not nested in extra subfolders).

---

## Quick Examples

```bash
# Example: run an experiment with a fixed seed
uv run -m bac.experiments.hmc_cartpole --seed 42

# Example: run another experiment
uv run -m bac.experiments.bs_mujoco --seed 2025
```

If you hit issues with `uv` not being on your PATH after installation, restart your shell or follow the post-install message to add the binary directory to your PATH.
