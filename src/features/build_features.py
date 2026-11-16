from pathlib import Path
import pandas as pd

PROCESSED = Path("data/processed")

def load_and_merge(processed_dir: Path = PROCESSED) -> pd.DataFrame:
    """Load transaction + identity tables and join them on TransactionID."""
    tx = pd.read_parquet(processed_dir / "train_transaction.parquet")
    idn = pd.read_parquet(processed_dir / "train_identity.parquet")

    df = tx.merge(
        idn,
        on= "TransactionID",
        how="left",
        validate = "one_to_one",
        indicator="_merge",
    )

    df["has_identity"] = (df["_merge"] == "both").astype("int8")
    df = df.drop(columns="_merge")

    # sort by time now so the time-based split later is straightforward
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    return df


def time_split(df: pd.DataFrame, valid_frac: float = 0.2):
    """Split a time-sorted DataFrame: earliest rows -> train, latest -> valid."""
    if not df["TransactionDT"].is_monotonic_increasing:
        raise ValueError("df must be sorted by TransactionDT first")

    cut = int(len(df) * (1 - valid_frac))
    train = df.iloc[:cut]
    valid = df.iloc[cut:]
    return train, valid

NON_FEATURES = ["isFraud", "TransactionID", "TransactionDT"]
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Modifies df in place (avoids copying ~1 GB)."""
    feature_cols = df.columns.drop(
        NON_FEATURES + ["has_identity", "n_missing"], errors="ignore"
    )
    df["n_missing"] = df[feature_cols].isna().sum(axis=1).astype("int16")
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Convert text columns to pandas 'category' dtype. Modifies df in place."""
    text_cols = df.select_dtypes(include=["object", "str"]).columns
    for col in text_cols:
        df[col] = df[col].astype("category")
    return df

def get_X_y(df: pd.DataFrame):
    """Split into model inputs (X) and the target (y)."""
    X = df.drop(columns=NON_FEATURES)
    y = df["isFraud"]
    return X, y


def build_dataset(valid_frac: float = 0.2):
    """Full pipeline: load -> features -> categories -> time split -> X/y."""
    df = load_and_merge()
    df = add_features(df)
    df = encode_categoricals(df)
    train, valid = time_split(df, valid_frac)
    X_train, y_train = get_X_y(train)
    X_valid, y_valid = get_X_y(valid)
    return X_train, y_train, X_valid, y_valid