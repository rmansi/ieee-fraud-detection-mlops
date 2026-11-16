import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def recall_at_precision(y_true, y_score, min_precision: float = 0.9) -> float:
    """Highest recall reachable while keeping precision >= min_precision."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def evaluate(y_true, y_score) -> dict:
    """Score a model's fraud probabilities with metrics suited to rare events."""
    return {
        "roc_auc": round(roc_auc_score(y_true, y_score), 4),
        "auc_pr": round(average_precision_score(y_true, y_score), 4),
        "recall_at_p90": round(recall_at_precision(y_true, y_score), 4),
    }