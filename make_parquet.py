"""Convert the raw training CSVs to Parquet with smaller dtypes.

Run once from the project root:
    python src/data/make_parquet.py
"""
from pathlib import Path

import pandas as pd

RAW = Path("src/data")  # change to Path("data/raw") if you move the CSVs
OUT = Path("src/data/processed")


def shrink(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numbers to smaller types to save memory."""
    for col in df.select_dtypes("float64").columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes("int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ["train_transaction", "train_identity"]:
        print(f"Reading {name}.csv ...")
        df = pd.read_csv(RAW / f"{name}.csv")
        before = df.memory_usage(deep=True).sum() / 1e6

        df = shrink(df)
        after = df.memory_usage(deep=True).sum() / 1e6

        df.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"  shape {df.shape} | memory {before:.0f} MB -> {after:.0f} MB")
        del df  # free memory before loading the next file


if __name__ == "__main__":
    main()
