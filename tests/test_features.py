import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    NON_FEATURES, add_features, encode_categoricals, get_X_y, time_split,
)


def make_df():
    """5 tiny transactions with known blanks, sorted by time."""
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4, 5],
        "TransactionDT": [10, 20, 30, 40, 50],
        "isFraud": [0, 1, 0, 0, 1],
        "has_identity": np.array([1, 0, 0, 1, 0], dtype="int8"),
        "TransactionAmt": [10.0, np.nan, 30.0, np.nan, 50.0],
        "card4": ["visa", None, "mastercard", "visa", None],
    })


def test_n_missing_counts_only_feature_columns():
    df = add_features(make_df())
    assert df["n_missing"].tolist() == [0, 2, 0, 1, 1]


def test_add_features_works_without_target_and_id_columns():
    # the API calls add_features on data with no isFraud/TransactionID/TransactionDT
    df = add_features(make_df().drop(columns=NON_FEATURES))
    assert df["n_missing"].tolist() == [0, 2, 0, 1, 1]


def test_encode_categoricals_converts_text_and_keeps_blanks():
    df = encode_categoricals(make_df())
    assert df["card4"].dtype == "category"
    assert df["card4"].cat.categories.tolist() == ["mastercard", "visa"]
    assert df["card4"].isna().sum() == 2


def test_time_split_puts_latest_rows_in_valid():
    train, valid = time_split(make_df(), valid_frac=0.4)
    assert len(train) == 3 and len(valid) == 2
    assert train["TransactionDT"].max() < valid["TransactionDT"].min()


def test_time_split_rejects_unsorted_data():
    unsorted = make_df().iloc[::-1]
    with pytest.raises(ValueError):
        time_split(unsorted)


def test_get_X_y_removes_target_and_ids():
    X, y = get_X_y(make_df())
    for col in NON_FEATURES:
        assert col not in X.columns
    assert y.tolist() == [0, 1, 0, 0, 1]