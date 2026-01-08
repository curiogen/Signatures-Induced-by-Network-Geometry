import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import zscore

INPUT_FILE = Path("citation_ensemble/citation_numeric_observables.csv")
OUTDIR = Path("citation_diagnostics")
OUTDIR.mkdir(exist_ok=True)


def coefficient_of_variation(df):
    mu = df.mean()
    sigma = df.std()
    return sigma / mu.replace(0, np.nan)


def correlation_stability(df):
    corr = df.corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    return upper.stack().describe()


def realization_outliers(df, z_thresh=3.0):
    Z = np.abs(zscore(df, nan_policy="omit"))
    mask = (Z > z_thresh).any(axis=1)
    return df.index[mask].tolist()


def main():
    df = pd.read_csv(INPUT_FILE)
    df = df.drop(columns=["realization_id"], errors="ignore")
    df = df.dropna(axis=1, how="all")

    cv = coefficient_of_variation(df)
    cv.to_csv(OUTDIR / "observable_coefficient_of_variation.csv")

    corr_stats = correlation_stability(df)
    corr_stats.to_csv(OUTDIR / "correlation_stability_summary.csv")

    outliers = realization_outliers(df)
    pd.Series(outliers, name="outlier_realization_ids").to_csv(
        OUTDIR / "outlier_realizations.csv",
        index=False
    )

    print("Citation diagnostics complete")
    print("Highest CV observables:")
    print(cv.sort_values(ascending=False).head(5))
    print("Outlier realizations:", outliers)


if __name__ == "__main__":
    main()
