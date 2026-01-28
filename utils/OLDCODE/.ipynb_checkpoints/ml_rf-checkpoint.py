__all__ = [
    "run_rf_pipeline"
]

from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    roc_curve, f1_score, precision_recall_curve, balanced_accuracy_score,
    precision_score, recall_score
)
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# -----------------------
# Training (same, but params exposed)
# -----------------------
def train_random_forest(
    X_train, y_train,
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",   # try None if you want to match older behavior
    random_state=42,
    n_jobs=-1
):
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )
    rf.fit(X_train, y_train)
    return rf

# -----------------------
# Threshold selection
# -----------------------
def _sweep_thresholds(y_true, y_prob, thresholds):
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append({
            "thr": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "acc": accuracy_score(y_true, y_pred),
            "bacc": balanced_accuracy_score(y_true, y_pred)
        })
    return pd.DataFrame(rows)

def find_best_threshold(
    y_true, y_prob,
    strategy="f1",                # "f1" | "balanced_accuracy" | "youden" | "target_recall" | "target_precision" | "cost"
    target=0.7,                   # used by target_recall / target_precision
    fp_cost=1.0, fn_cost=2.0,     # used by cost strategy
    beta=1.0,                     # if you want F-beta, set strategy="fbeta" and beta>0
    grid=None,                    # custom thresholds
):
    if grid is None:
        grid = np.linspace(0, 1, 101)

    # precompute
    sweep = _sweep_thresholds(y_true, y_prob, grid)

    if strategy == "f1":
        best_idx = sweep["f1"].idxmax()
    elif strategy == "fbeta":
        # compute F-beta on the fly
        P, R = sweep["precision"].to_numpy(), sweep["recall"].to_numpy()
        beta2 = beta**2
        fbeta = (1 + beta2) * (P * R) / np.clip(beta2 * P + R, 1e-12, None)
        best_idx = np.nanargmax(fbeta)
        sweep["fbeta"] = fbeta
    elif strategy == "balanced_accuracy":
        best_idx = sweep["bacc"].idxmax()
    elif strategy == "youden":
        # maximize TPR - FPR (i.e., Youden’s J) via ROC
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y_true, y_prob)
        j = tpr - fpr
        # map ROC thresholds to our grid by nearest neighbor
        best_thr = thr[np.argmax(j)]
        # snap to nearest in the grid for consistency
        best_idx = (np.abs(sweep["thr"] - best_thr)).argmin()
    elif strategy == "target_recall":
        # choose smallest threshold with recall >= target
        cand = sweep[sweep["recall"] >= target]
        best_idx = cand["thr"].idxmin() if not cand.empty else sweep["thr"].idxmax()
    elif strategy == "target_precision":
        cand = sweep[sweep["precision"] >= target]
        best_idx = cand["thr"].idxmin() if not cand.empty else sweep["thr"].idxmax()
    elif strategy == "cost":
        # Expected cost = FP*fp_cost + FN*fn_cost; minimize it
        # Approximate FP/FN counts from rates:
        from sklearn.metrics import confusion_matrix
        costs = []
        for t in sweep["thr"]:
            y_pred = (y_prob >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            costs.append(fp*fp_cost + fn*fn_cost)
        sweep["cost"] = costs
        best_idx = sweep["cost"].idxmin()
    else:
        raise ValueError(f"Unknown threshold strategy: {strategy}")

    best_thr = float(sweep.loc[best_idx, "thr"])
    print("Threshold sweep (head):")
    print(sweep.head(8).round(3))
    print("\nBest threshold by", strategy, "=>", round(best_thr, 3))
    return best_thr, sweep

# -----------------------
# Evaluation + importance
# -----------------------
def evaluate_predictions(y_true, y_probs, threshold):
    preds = (y_probs >= threshold).astype(int)
    accuracy = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, y_probs)
    fpr, tpr, _ = roc_curve(y_true, y_probs)

    print(f"\n🔎 Test Accuracy (threshold={threshold:.2f}): {accuracy:.2f}")
    print("Classification Report:")
    report_str = classification_report(y_true, preds, target_names=["Class 0", "Class 1"])
    print(report_str)
    report_df = pd.DataFrame(classification_report(y_true, preds, output_dict=True)).T
    print(f"\nAUC-ROC (Test): {auc:.2f}")

    return preds, accuracy, auc, fpr, tpr, report_df

def get_feature_importance_rf(model, feature_names, top_n=10):
    feature_names = list(feature_names)
    imps = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": imps,
        "Importance_%": 100 * imps / imps.sum() if imps.sum() > 0 else imps
    }).sort_values("Importance", ascending=False)
    print("\n📊 Top 10 Most Important Features (Random Forest):")
    print(importance_df[["Feature", "Importance"]].head(top_n))
    return importance_df

def plot_roc_curve(fpr, tpr, auc, title="ROC Curve - Random Forest"):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------
# Orchestrator
# -----------------------
def run_rf_pipeline(
    X_train, y_train,
    X_val, y_val,
    X_test, y_test,
    feature_names,
    # RF params
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",   # ← try None to see if it matches your previous run
    random_state=42,
    n_jobs=-1,
    # Threshold strategy
    threshold_strategy="f1",   # "f1","fbeta","balanced_accuracy","youden","target_recall","target_precision","cost"
    target=0.7,                # for target_* strategies
    fp_cost=1.0, fn_cost=2.0,
    beta=1.0
):
    model_rf = train_random_forest(
        X_train, y_train,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=n_jobs
    )

    val_probs = model_rf.predict_proba(X_val)[:, 1]
    best_threshold, sweep = find_best_threshold(
        y_val, val_probs,
        strategy=threshold_strategy,
        target=target,
        fp_cost=fp_cost, fn_cost=fn_cost, beta=beta
    )

    test_probs = model_rf.predict_proba(X_test)[:, 1]
    test_preds, accuracy, auc, fpr, tpr, report_df = evaluate_predictions(
        y_test, test_probs, best_threshold
    )
    feature_importance = get_feature_importance_rf(model_rf, feature_names)
    plot_roc_curve(fpr, tpr, auc)

    results_rf = {
        "val_probs": val_probs,
        "test_probs": test_probs,
        "test_preds": test_preds,
        "accuracy": accuracy,
        "auc": auc,
        "best_threshold": best_threshold,
        "threshold_sweep": sweep,
        "fpr": fpr,
        "tpr": tpr,
        "report_df": report_df,
        "feature_importance": feature_importance
    }
    return model_rf, results_rf
