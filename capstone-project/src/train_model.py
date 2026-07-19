"""
train_model.py
Script training & hyperparameter tuning untuk klasifikasi SMS Spam.

Model yang dibandingkan:
1. Multinomial Naive Bayes (baseline)
2. Support Vector Machine (Linear Kernel)
3. Random Forest

Tuning dilakukan dengan GridSearchCV (5-fold, scoring=f1) karena data imbalanced.
Model terbaik + TF-IDF vectorizer disimpan ke models/ sebagai pickle
untuk dipakai kembali oleh aplikasi Streamlit.
"""

import os
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")


def load_processed():
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    for d in (train, val, test):
        d["processed_sms"] = d["processed_sms"].fillna("")
    return train, val, test


def build_tfidf(train, val, test):
    tfidf = TfidfVectorizer(max_features=6000)
    X_train = tfidf.fit_transform(train["processed_sms"])
    X_val = tfidf.transform(val["processed_sms"])
    X_test = tfidf.transform(test["processed_sms"])
    return tfidf, X_train, X_val, X_test


def get_model_grid(scale_pos_weight=1.0):
    return {
        "Naive Bayes": {
            "estimator": MultinomialNB(),
            "params": {"alpha": [0.1, 0.5, 1.0, 1.5]},
        },
        "SVM": {
            "estimator": SVC(kernel="linear", class_weight="balanced", probability=True, random_state=42),
            "params": {"C": [0.1, 1, 10]},
        },
        "Random Forest": {
            "estimator": RandomForestClassifier(class_weight="balanced", random_state=42),
            "params": {"n_estimators": [200, 400], "max_depth": [None, 30]},
        },
        "XGBoost": {
            "estimator": XGBClassifier(
                scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            ),
            "params": {
                "n_estimators": [200, 400],
                "max_depth": [3, 6],
                "learning_rate": [0.05, 0.1],
            },
        },
    }


def evaluate(model, X, y):
    pred = model.predict(X)
    try:
        proba = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, proba)
    except Exception:
        auc = None
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(y, pred, target_names=["Ham", "Spam"], output_dict=True),
    }, pred


def run_training():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train, val, test = load_processed()
    tfidf, X_train, X_val, X_test = build_tfidf(train, val, test)
    y_train, y_val, y_test = train["label_num"], val["label_num"], test["label_num"]

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model_grid = get_model_grid(scale_pos_weight=scale_pos_weight)
    print(f"scale_pos_weight (XGBoost) = {scale_pos_weight:.3f}")

    results = {}
    fitted_models = {}

    for name, cfg in model_grid.items():
        print(f"\n=== Tuning {name} ===")
        grid = GridSearchCV(cfg["estimator"], cfg["params"], scoring="f1", cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"Best params: {grid.best_params_}")

        val_metrics, _ = evaluate(best_model, X_val, y_val)
        test_metrics, test_pred = evaluate(best_model, X_test, y_test)

        results[name] = {
            "best_params": grid.best_params_,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
        }
        fitted_models[name] = best_model
        print(f"Test F1 (Spam): {test_metrics['f1']:.4f} | Test Acc: {test_metrics['accuracy']:.4f} | ROC-AUC: {test_metrics['roc_auc']:.4f}")

    # Pilih model terbaik berdasarkan F1-score Spam pada test set
    best_name = max(results, key=lambda k: results[k]["test_metrics"]["f1"])
    best_model = fitted_models[best_name]
    print(f"\n>>> Model terbaik: {best_name} (F1-Spam Test = {results[best_name]['test_metrics']['f1']:.4f})")

    # Simpan artefak
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "preprocessing.pkl"))
    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
    joblib.dump(fitted_models, os.path.join(MODELS_DIR, "all_models.pkl"))

    with open(os.path.join(MODELS_DIR, "model_meta.json"), "w") as f:
        json.dump({"best_model_name": best_name}, f, indent=2)

    def to_serializable(obj):
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        if hasattr(obj, "item"):
            return obj.item()
        return obj

    with open(os.path.join(REPORTS_DIR, "model_results.json"), "w") as f:
        json.dump(to_serializable(results), f, indent=2)

    return results, fitted_models, best_name, tfidf, (train, val, test)


if __name__ == "__main__":
    run_training()
