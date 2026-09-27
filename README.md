# Task 1 — Interpreting Correlations

How to compute and interpret correlations with **numeric and categorical data**, using the insurance dataset from our linear regression lab (Kaggle `noordeen/insurance-premium-prediction`, 1,338 rows).


## Project structure

```
├── README.md
├── requirements.txt                  # Python libraries to install
├── .gitignore                        # keeps .venv and cache files out of Git
├── 01_simple_linear.py               # class script: load → inspect → correlate → interpret
├── correlation_mini_example.ipynb    # Task 1 walkthrough notebook (start here)
├── correlation_pocket_tool.py        # reusable tool: correlations for ANY CSV
└── data/
    └── insurance-premium-prediction/
        └── insurance.csv             # created automatically on first run
```

The dataset downloads automatically the first time you run `01_simple_linear.py`, the notebook, or the pocket tool (via `kagglehub`, needs internet once).

---

## Quick start

### 0. Prerequisites (one time)

- **Python 3.9+** → <https://www.python.org/downloads/>
  Windows: on the first installer screen, check **"Add python.exe to PATH"**.
- **Git** → <https://git-scm.com/downloads>
- **VS Code** → <https://code.visualstudio.com/>, plus the **Python** and **Jupyter** extensions (by Microsoft).

### 1. Download the project

```bash
git clone <https://github.com/ZiyanChen1/Linear-Regression-Concepts.git>
cd <Rhttps://github.com/ZiyanChen1/Linear-Regression-Concepts.git>
```

(Or on GitHub: **Code → Download ZIP**, unzip, then `cd` into the folder.)

### 2. Create a virtual environment

| Windows (PowerShell) | Mac / Linux |
|---|---|
| `py -m venv .venv` | `python3 -m venv .venv` |

### 3. Activate it

| Windows (PowerShell) | Mac / Linux |
|---|---|
| `.venv\Scripts\Activate.ps1` | `source .venv/bin/activate` |

Your prompt should now start with **`(.venv)`**.

> **Windows error "running scripts is disabled on this system"?** Run this, then activate again (affects only the current terminal window):
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> ```

### 4. Install the requirements

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run

**a) Class script**
```bash
python 01_simple_linear.py
```

**b) Notebook**
1. Open the folder in VS Code: `code .`
2. Open `correlation_mini_example.ipynb`.
3. Top-right: **Select Kernel → Python Environments → `.venv`**.
4. Click **Run All**, or go cell by cell with `Shift+Enter`.

**c) Pocket tool**
```bash
python correlation_pocket_tool.py                                    # insurance demo
python correlation_pocket_tool.py --plot                             # demo + save charts
python correlation_pocket_tool.py path/to/your.csv                   # YOUR data
python correlation_pocket_tool.py path/to/your.csv --target price --plot
```

With your own CSV, the tool automatically:
- detects numeric, binary, and multi-category columns;
- encodes binary categories (yes/no, female/male…) as 0/1 and prints the mapping;
- one-hot encodes multi-category columns (never 1, 2, 3, 4);
- prints Pearson and Spearman matrices and explains the strongest pairs;
- with `--target`, ranks every feature against that column and reports eta for categories;
- with `--plot`, saves `correlation_heatmap.png`.

Tips for your own CSV: Excel files must be saved as **CSV** first; remove `$` and `,` from numbers; put paths with spaces in quotes.

### Next time

Just activate the environment again (step 3). Leave it with `deactivate`.

**Strength labels** (same as `01_simple_linear.py`): |r| < 0.10 very weak · < 0.30 weak · < 0.50 moderate · < 0.70 strong · ≥ 0.70 very strong

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` / `py` not recognized | Reinstall Python with **"Add to PATH"** checked, then open a **new** terminal. |
| `running scripts is disabled` (Windows) | See the note in step 3. |
| `ModuleNotFoundError` | `.venv` isn't active (no `(.venv)` in the prompt), or the notebook kernel isn't `.venv`. |
| `.venv` not in **Select Kernel** | `Ctrl/Cmd+Shift+P` → **Python: Select Interpreter** → **Enter interpreter path** → `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Mac). Reload VS Code. |
| Dataset download fails | Check your internet connection, then run `python 01_simple_linear.py` once to download it into `data/`. |
| Files disappear from `data/insurance-premium-prediction/` | `01_simple_linear.py` deletes and re-copies that folder on every run. Don't store your own files there. |

---
