"""
Correlation Pocket Tool
=======================
A reusable script that computes correlations for ANY CSV with mixed data
(numeric + categorical). Keep it in your pocket for future projects.

HOW TO RUN (VS Code terminal, with .venv activated - see README.md)
-------------------------------------------------------------------
    python correlation_pocket_tool.py                                  # demo: insurance data
    python correlation_pocket_tool.py --plot                           # demo + save charts
    python correlation_pocket_tool.py my_data.csv                      # YOUR data
    python correlation_pocket_tool.py my_data.csv --target sales       # rank features vs a target
    python correlation_pocket_tool.py my_data.csv --target sales --plot

WHAT IT DOES
------------
Step 1  Load data (insurance demo, or your CSV)
Step 2  Detect column types: numeric / binary categorical / multi-category
Step 3  Transform categories:
          binary (yes/no)       -> 0/1   (Pearson on 0/1 = point-biserial)
          multi-category        -> one-hot 0/1 columns (NEVER 1, 2, 3, 4)
Step 4  Compute Pearson and Spearman correlation matrices
Step 5  Interpret the strongest pairs in plain English
Step 6  (with a target) rank every feature against the target, incl. eta

Strength labels match 01_simple_linear.py:
  |r| < 0.10 very weak | < 0.30 weak | < 0.50 moderate | < 0.70 strong | else very strong

Demo data: Kaggle "noordeen/insurance-premium-prediction" (insurance.csv)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
INSURANCE_CSV = HERE / "data" / "insurance-premium-prediction" / "insurance.csv"
KAGGLE_DATASET = "noordeen/insurance-premium-prediction"
ZERO_WORDS = {"no", "false", "n", "0", "female", "f"}


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------
def load_data(csv_path=None):
    """Load the user's CSV, or the insurance demo dataset."""
    if csv_path:
        print(f"Loading your file: {csv_path}")
        return pd.read_csv(csv_path)

    if INSURANCE_CSV.exists():
        print(f"Loading demo data: {INSURANCE_CSV.relative_to(HERE)}")
        return pd.read_csv(INSURANCE_CSV)

    # Not downloaded yet: fetch with kagglehub (same source as 01_simple_linear.py)
    print("Demo data not found locally; downloading with kagglehub (needs internet)...")
    import kagglehub
    folder = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    return pd.read_csv(folder / "insurance.csv")


# ---------------------------------------------------------------------------
# Step 2: Detect column types
# ---------------------------------------------------------------------------
def classify_columns(df, max_categories=10):
    """Split columns into numeric, binary categorical, and multi-category."""
    numeric, binary, multi, skipped = [], [], [], []
    for col in df.columns:
        n_unique = df[col].dropna().nunique()
        if pd.api.types.is_numeric_dtype(df[col]) and n_unique > 2:
            numeric.append(col)
        elif n_unique == 2:
            binary.append(col)       # e.g. smoker yes/no, sex female/male
        elif 2 < n_unique <= max_categories:
            multi.append(col)        # e.g. region with 4 areas
        else:
            skipped.append(col)      # IDs, free text, constants, too many levels
    return numeric, binary, multi, skipped


# ---------------------------------------------------------------------------
# Step 3: Transform categorical data
# ---------------------------------------------------------------------------
def encode_binary(df, binary_cols):
    """Encode each 2-value column as 0/1 and print the mapping used."""
    encoded = pd.DataFrame(index=df.index)
    for col in binary_cols:
        values = sorted(df[col].dropna().unique(), key=str)
        if set(values) <= {0, 1}:          # already 0/1
            encoded[col] = df[col]
            continue
        # "no", "false", "female" become 0 when present; otherwise alphabetical
        values.sort(key=lambda v: 0 if str(v).strip().lower() in ZERO_WORDS else 1)
        mapping = {values[0]: 0, values[1]: 1}
        encoded[f"{col}_encoded"] = df[col].map(mapping)
        print(f"  {col:<12} {mapping}  ->  '{col}_encoded'")
    return encoded


def one_hot(df, multi_cols):
    """One 0/1 column per category. No fake order, no fake equal spacing."""
    if not multi_cols:
        return pd.DataFrame(index=df.index)
    dummies = pd.get_dummies(df[multi_cols], prefix=multi_cols, dtype=int)
    for col in multi_cols:
        levels = sorted(df[col].dropna().unique(), key=str)
        print(f"  {col:<12} {len(levels)} levels -> {len(levels)} one-hot columns: "
              f"{', '.join(f'{col}_{v}' for v in levels)}")
    return dummies


def correlation_ratio(categories, values):
    """
    Eta: how much of a numeric variable's variation is explained by the groups
    of a categorical variable. Range 0 to 1, no direction.
    eta^2 = between-group sum of squares / total sum of squares
    """
    data = pd.DataFrame({"cat": categories, "val": values}).dropna()
    grand_mean = data["val"].mean()
    groups = data.groupby("cat")["val"]
    ss_between = (groups.count() * (groups.mean() - grand_mean) ** 2).sum()
    ss_total = ((data["val"] - grand_mean) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else np.nan


# ---------------------------------------------------------------------------
# Interpretation helpers (same cutoffs as 01_simple_linear.py)
# ---------------------------------------------------------------------------
def strength_label(r):
    a = abs(r)
    if a < 0.10:
        return "very weak"
    if a < 0.30:
        return "weak"
    if a < 0.50:
        return "moderate"
    if a < 0.70:
        return "strong"
    return "very strong"


def direction_label(r):
    if r > 0:
        return "positive"
    if r < 0:
        return "negative"
    return "no linear"


def top_pairs(corr_matrix, n=5):
    """Return the n strongest unique pairs from a correlation matrix."""
    mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
    pairs = corr_matrix.where(mask).stack().reset_index()
    pairs.columns = ["var_1", "var_2", "r"]
    pairs["abs_r"] = pairs["r"].abs()
    return pairs.sort_values("abs_r", ascending=False).head(n)


def pearson_by_hand(x, y):
    """Show the Pearson formula step by step."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    dx, dy = x - x.mean(), y - y.mean()
    table = pd.DataFrame({
        "x": x, "y": y,
        "x - x_bar": dx.round(2), "y - y_bar": dy.round(2),
        "product": (dx * dy).round(2),
        "(x - x_bar)^2": (dx ** 2).round(2), "(y - y_bar)^2": (dy ** 2).round(0),
    })
    numerator = (dx * dy).sum()
    denominator = np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    return table, numerator, denominator, numerator / denominator


def line(title=""):
    print("\n" + "=" * 72)
    if title:
        print(title)
        print("=" * 72)


# ---------------------------------------------------------------------------
# Demo-only section: the 5-row mini example
# ---------------------------------------------------------------------------
def mini_example(df):
    line("MINI EXAMPLE  Pearson r by hand on 5 rows (age vs expenses)")
    mini = df.head(5)
    table, num, den, r_hand = pearson_by_hand(mini["age"], mini["expenses"])
    print(table.to_string(index=False))
    print(f"\nx_bar = {mini['age'].mean():.2f}   y_bar = {mini['expenses'].mean():,.2f}")
    print("r = sum(product) / sqrt( sum((x - x_bar)^2) * sum((y - y_bar)^2) )")
    print(f"  = {num:,.2f} / {den:,.2f} = {r_hand:.3f}")
    print(f"pandas check: {mini['age'].corr(mini['expenses']):.3f}")

    line("MINI EXAMPLE  Why 5 rows are not enough")
    smoker01 = df["smoker"].map({"no": 0, "yes": 1})
    rows = []
    for name, series in [("age", df["age"]), ("bmi", df["bmi"]), ("smoker (0/1)", smoker01)]:
        rows.append({
            "variable": name,
            "r (5 rows)": round(series.head(5).corr(df["expenses"].head(5)), 3),
            f"r ({len(df)} rows)": round(series.corr(df["expenses"]), 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print("Lesson: bmi even flips sign. Learn the formula on 5 rows, conclude on all rows.")


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def run(csv_path=None, target=None, make_plot=False):
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 30)

    df = load_data(csv_path)
    is_demo = csv_path is None
    if is_demo and target is None:
        target = "expenses"

    line("STEP 1  First 5 rows")
    print(df.head())
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    if is_demo:
        mini_example(df)

    numeric, binary, multi, skipped = classify_columns(df)
    line("STEP 2  Column types")
    print(f"Numeric             : {numeric}")
    print(f"Binary categorical  : {binary}   -> encode 0/1")
    print(f"Multi-category      : {multi}   -> one-hot (not 1, 2, 3, 4)")
    if skipped:
        print(f"Skipped             : {skipped}   (IDs, text, constant, or > 10 levels)")

    line("STEP 3  Transform categorical data")
    print("Binary -> 0/1:")
    encoded = encode_binary(df, binary)
    if encoded.empty:
        print("  (none)")
    print("Multi-category -> one-hot:")
    dummies = one_hot(df, multi)
    if dummies.empty:
        print("  (none)")

    analysis = pd.concat([df[numeric], encoded], axis=1)
    if analysis.shape[1] < 2:
        print("\nNeed at least 2 usable columns to compute correlations.")
        return

    line("STEP 4  Correlation matrices (numeric + binary 0/1)")
    pearson = analysis.corr(method="pearson")
    spearman = analysis.corr(method="spearman")
    print("Pearson (linear relationship):")
    print(pearson.round(3))
    print("\nSpearman (rank-based; less sensitive to outliers and skew):")
    print(spearman.round(3))

    line("STEP 5  Strongest pairs (Pearson) in plain English")
    for _, row in top_pairs(pearson).iterrows():
        print(f"  {row.var_1} vs {row.var_2}: r = {row.r:+.3f} -> "
              f"{strength_label(row.r)}, {direction_label(row.r)}")

    if target:
        if target not in df.columns or target not in numeric:
            print(f"\n(--target '{target}' is not a numeric column; skipping Step 6)")
        else:
            rank_against_target(df, analysis, dummies, multi, target)

    line("REMINDERS")
    print("  * Correlation is not causation.")
    print("  * Pearson measures LINEAR relationships only; always look at a scatter plot.")
    print("  * If Pearson and Spearman differ a lot, check for skew or outliers.")
    print("  * The sign of a 0/1 variable depends on which group you coded as 1.")

    if make_plot:
        save_plots(pearson, df, target)


def rank_against_target(df, analysis, dummies, multi, target):
    """Step 6: every feature vs the target, strongest first."""
    line(f"STEP 6  All features ranked against target: {target}")
    rows = []
    features = pd.concat([analysis.drop(columns=[target]), dummies], axis=1)
    for col in features.columns:
        r = features[col].corr(df[target])
        rows.append({"feature": col, "r": r, "|r|": abs(r),
                     "strength": strength_label(r), "direction": direction_label(r)})
    ranked = pd.DataFrame(rows).sort_values("|r|", ascending=False)
    print(ranked.round(3).to_string(index=False))

    if multi:
        print("\nMulti-category variables as a whole (eta, 0 to 1, no direction):")
        for cat in multi:
            eta = correlation_ratio(df[cat], df[target])
            print(f"  {cat} -> {target}: eta = {eta:.3f} ({strength_label(eta)})")
            print(df.groupby(cat)[target].mean().round(2).to_string().replace("\n", "\n    "))


def save_plots(corr_matrix, df, target=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_xticklabels([t.get_text().replace("_", "\n") for t in ax.get_xticklabels()], rotation=0)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title("Pearson correlation matrix")
    fig.tight_layout()
    fig.savefig("correlation_heatmap.png", dpi=150)
    print("\nSaved: correlation_heatmap.png")

    if target and "smoker" in df.columns and "bmi" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.scatterplot(data=df, x="bmi", y=target, hue="smoker", alpha=0.6, ax=ax)
        ax.set_title(f"bmi vs {target}, colored by smoker")
        fig.tight_layout()
        fig.savefig("scatter_bmi_expenses_smoker.png", dpi=150)
        print("Saved: scatter_bmi_expenses_smoker.png")
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Correlation pocket tool")
    parser.add_argument("csv", nargs="?", help="path to a CSV file (omit for the insurance demo)")
    parser.add_argument("--target", help="numeric column to rank all features against")
    parser.add_argument("--plot", action="store_true", help="save a heatmap PNG")
    args = parser.parse_args()
    run(csv_path=args.csv, target=args.target, make_plot=args.plot)
