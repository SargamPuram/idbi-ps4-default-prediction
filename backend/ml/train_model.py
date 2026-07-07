"""
Training pipeline for IDBI Default Prediction Model (PS4).

Trains XGBoost, LightGBM and Logistic Regression, combines them in a
stacking ensemble, evaluates with a stratified holdout AND an out-of-time
(vintage-based) holdout, generates SHAP explanations, and batch-scores the
full portfolio for the API to serve.

Run from the backend/ directory:
    python ml/train_model.py
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    precision_recall_curve, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.feature_engineering import build_feature_matrix, engineer_snapshot_features, FEATURE_COLUMNS
from ml.risk_logic import (
    ecl_stage, estimated_months_to_default, rag_status, recommended_action,
    risk_category, top_risk_drivers,
)

DATA_DIR = "data"
MODEL_DIR = "models"
REPORT_DIR = "reports"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


def make_models(scale_pos_weight):
    xgb_clf = xgb.XGBClassifier(
        n_estimators=350, max_depth=5, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.8, eval_metric="auc", random_state=42,
        reg_lambda=1.5, min_child_weight=3, n_jobs=-1,
    )
    lgbm_clf = lgb.LGBMClassifier(
        n_estimators=350, max_depth=6, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.8, random_state=42, class_weight="balanced", verbose=-1,
    )
    logreg = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)
    return xgb_clf, lgbm_clf, logreg


def evaluate(name, model, X_test, y_test, needs_scaling=False, scaler=None):
    X_eval = scaler.transform(X_test) if needs_scaling else X_test
    proba = model.predict_proba(X_eval)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "model": name,
        "auc_roc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
    }
    return metrics, proba


def main():
    print("Loading data...")
    accounts = pd.read_csv(f"{DATA_DIR}/accounts.csv")
    snapshots = pd.read_csv(f"{DATA_DIR}/monthly_snapshots.csv")
    accounts["disbursement_date"] = pd.to_datetime(accounts["disbursement_date"])

    print("Engineering time-series features (rolling stats, trend, acceleration)...")
    snap_feat = engineer_snapshot_features(snapshots)

    # Exclude already-impaired (NPA) accounts from the *prediction* task — they are
    # already in default, not "at risk of future default". They're still scored
    # deterministically (PD=0.99) later for portfolio/ECL views.
    trainable = accounts[accounts["current_sma_status"] != "NPA"].reset_index(drop=True)
    print(f"Training population: {len(trainable)} accounts (excluded {len(accounts) - len(trainable)} already-NPA)")

    X_full = build_feature_matrix(trainable, snapshot_features=snap_feat)
    y_full = trainable["default_flag"].astype(int).values

    # ---- Split 1: stratified random holdout (primary model selection/reporting) ----
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_full, test_size=0.2, stratify=y_full, random_state=42
    )

    # ---- Split 2: out-of-time holdout by loan vintage (earliest 75% train / most recent 25% test) ----
    # trainable has a fresh 0..n-1 RangeIndex, so these positional labels index y_full directly.
    vintage_sorted = trainable.sort_values("disbursement_date").index
    cutoff = int(len(vintage_sorted) * 0.75)
    oot_train_idx, oot_test_idx = vintage_sorted[:cutoff], vintage_sorted[cutoff:]
    X_oot_train, y_oot_train = X_full.loc[oot_train_idx], y_full[oot_train_idx]
    X_oot_test, y_oot_test = X_full.loc[oot_test_idx], y_full[oot_test_idx]

    print("Applying SMOTE to balance the training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  Before: {len(y_train)} rows ({y_train.mean()*100:.1f}% default) "
          f"-> After: {len(y_train_res)} rows ({y_train_res.mean()*100:.1f}% default)")

    scaler = StandardScaler().fit(X_train_res)

    print("\nTraining base models (XGBoost, LightGBM, Logistic Regression)...")
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_clf, lgbm_clf, logreg = make_models(scale_pos_weight)

    xgb_clf.fit(X_train_res, y_train_res)
    lgbm_clf.fit(X_train_res, y_train_res)
    logreg.fit(scaler.transform(X_train_res), y_train_res)

    all_metrics = []
    m, _ = evaluate("XGBoost", xgb_clf, X_test, y_test)
    all_metrics.append(m)
    m, _ = evaluate("LightGBM", lgbm_clf, X_test, y_test)
    all_metrics.append(m)
    m, _ = evaluate("Logistic Regression", logreg, X_test, y_test, needs_scaling=True, scaler=scaler)
    all_metrics.append(m)

    print("\nTraining stacking ensemble (meta-learner: Logistic Regression)...")
    # Base logreg estimator expects scaled input, so wrap it in a pipeline for the ensemble fit.
    from sklearn.pipeline import make_pipeline
    logreg_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
    stack = StackingClassifier(
        estimators=[("xgb", xgb_clf), ("lgbm", lgbm_clf), ("logreg", logreg_pipeline)],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5, stack_method="predict_proba", n_jobs=-1,
    )
    stack.fit(X_train_res, y_train_res)

    ens_metrics, ens_test_proba = evaluate("Stacking Ensemble", stack, X_test, y_test)
    all_metrics.append(ens_metrics)
    print("\nModel comparison (stratified holdout):")
    for m in all_metrics:
        print(f"  {m['model']:22s} AUC={m['auc_roc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
              f"P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")

    # ---- Out-of-time validation using the SAME production pipeline (retrain on OOT-train) ----
    print("\nRunning out-of-time validation (train=oldest 75% vintage, test=newest 25% vintage)...")
    smote_oot = SMOTE(random_state=42)
    X_oot_train_res, y_oot_train_res = smote_oot.fit_resample(X_oot_train, y_oot_train)
    oot_scaler = StandardScaler().fit(X_oot_train_res)
    oot_logreg_pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
    xgb_oot, lgbm_oot, _ = make_models(scale_pos_weight)
    oot_stack = StackingClassifier(
        estimators=[("xgb", xgb_oot), ("lgbm", lgbm_oot), ("logreg", oot_logreg_pipeline)],
        final_estimator=LogisticRegression(max_iter=1000), cv=5, stack_method="predict_proba", n_jobs=-1,
    )
    oot_stack.fit(X_oot_train_res, y_oot_train_res)
    oot_metrics, _ = evaluate("Stacking Ensemble (Out-of-Time)", oot_stack, X_oot_test, y_oot_test)
    print(f"  OOT AUC-ROC={oot_metrics['auc_roc']:.4f}  PR-AUC={oot_metrics['pr_auc']:.4f}  F1={oot_metrics['f1']:.4f}")

    # ---- ROC / PR curves + confusion matrix for the primary ensemble ----
    fpr, tpr, _ = roc_curve(y_test, ens_test_proba)
    prec, rec, _ = precision_recall_curve(y_test, ens_test_proba)
    cm = confusion_matrix(y_test, (ens_test_proba >= 0.5).astype(int)).tolist()

    roc_points = [{"fpr": round(float(f), 4), "tpr": round(float(t), 4)} for f, t in
                  zip(fpr[::max(1, len(fpr)//200)], tpr[::max(1, len(tpr)//200)])]
    pr_points = [{"precision": round(float(p), 4), "recall": round(float(r), 4)} for p, r in
                 zip(prec[::max(1, len(prec)//200)], rec[::max(1, len(rec)//200)])]

    # ---- SHAP explanations (on XGBoost base learner — used as the interpretable surrogate) ----
    print("\nComputing SHAP feature importance...")
    explainer = shap.TreeExplainer(xgb_clf)
    sample_idx = np.random.RandomState(42).choice(len(X_test), size=min(1500, len(X_test)), replace=False)
    X_shap_sample = X_test.iloc[sample_idx]
    shap_values = explainer.shap_values(X_shap_sample)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = sorted(
        [{"feature": f, "importance": round(float(v), 5)} for f, v in zip(FEATURE_COLUMNS, mean_abs_shap)],
        key=lambda d: -d["importance"],
    )[:20]

    # ---- Save models & artifacts ----
    print("\nSaving models and artifacts...")
    joblib.dump(xgb_clf, f"{MODEL_DIR}/xgb_model.joblib")
    joblib.dump(lgbm_clf, f"{MODEL_DIR}/lgbm_model.joblib")
    joblib.dump(stack, f"{MODEL_DIR}/stacking_ensemble.joblib")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
    with open(f"{MODEL_DIR}/feature_columns.json", "w") as f:
        json.dump(FEATURE_COLUMNS, f, indent=2)

    performance_report = {
        "model_comparison": all_metrics,
        "out_of_time_validation": oot_metrics,
        "confusion_matrix": {"labels": ["No Default", "Default"], "matrix": cm},
        "roc_curve": roc_points,
        "pr_curve": pr_points,
        "feature_importance": feature_importance,
        "training_summary": {
            "total_accounts": int(len(accounts)),
            "trainable_accounts": int(len(trainable)),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "train_rows_after_smote": int(len(y_train_res)),
            "default_rate_raw": round(float(y_full.mean()), 4),
            "primary_model": "Stacking Ensemble (XGBoost + LightGBM + Logistic Regression)",
        },
    }
    with open(f"{REPORT_DIR}/model_performance.json", "w") as f:
        json.dump(performance_report, f, indent=2)

    # ---- Batch-score the ENTIRE portfolio (incl. already-NPA accounts) for the API to serve ----
    print("\nScoring full portfolio for API serving layer...")
    score_full_portfolio(accounts, snap_feat, stack, explainer)

    print("\nDone. AUC-ROC (ensemble, stratified holdout):", ens_metrics["auc_roc"],
          "| AUC-ROC (ensemble, out-of-time):", oot_metrics["auc_roc"])


def score_full_portfolio(accounts, snap_feat, stack_model, shap_explainer):
    npa_mask = accounts["current_sma_status"] == "NPA"
    scoreable = accounts[~npa_mask].reset_index(drop=True)
    npa_accounts = accounts[npa_mask].reset_index(drop=True)

    X_scoreable = build_feature_matrix(scoreable, snapshot_features=snap_feat)
    pd_scores = stack_model.predict_proba(X_scoreable)[:, 1]

    shap_values = shap_explainer.shap_values(X_scoreable)

    results = []
    for i, row in scoreable.iterrows():
        pd_score = float(pd_scores[i])
        cat = risk_category(pd_score)
        ecl = ecl_stage(row["current_sma_status"], pd_score)
        feature_values = X_scoreable.iloc[i].to_dict()
        drivers = top_risk_drivers(shap_values[i], feature_values, FEATURE_COLUMNS, top_n=5)
        results.append({
            "account_id": row["account_id"],
            "probability_of_default": round(pd_score, 4),
            "risk_category": cat,
            "rag_status": rag_status(pd_score),
            "sma_classification": row["current_sma_status"],
            "ecl_stage": ecl,
            "estimated_months_to_default": estimated_months_to_default(pd_score, cat),
            "recommended_action": recommended_action(cat, ecl, row["loan_type"]),
            "top_risk_drivers": json.dumps(drivers),
        })

    for _, row in npa_accounts.iterrows():
        results.append({
            "account_id": row["account_id"],
            "probability_of_default": 0.99,
            "risk_category": "Critical",
            "rag_status": "Red",
            "sma_classification": "NPA",
            "ecl_stage": "Stage 3",
            "estimated_months_to_default": 0,
            "recommended_action": recommended_action("Critical", "Stage 3", row["loan_type"]),
            "top_risk_drivers": json.dumps([]),
        })

    scores_df = pd.DataFrame(results)
    merged = accounts.merge(scores_df, on="account_id", how="left")
    merged.to_csv(f"{DATA_DIR}/scored_accounts.csv", index=False)
    print(f"  Saved data/scored_accounts.csv ({len(merged)} accounts)")


if __name__ == "__main__":
    main()
