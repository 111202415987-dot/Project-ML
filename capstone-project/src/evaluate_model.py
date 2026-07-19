"""
evaluate_model.py
Evaluasi komprehensif seluruh model:
- Tabel perbandingan performa
- ROC curves gabungan
- Confusion matrix tiap model
- Interpretasi model terbaik menggunakan SHAP
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")


def load_artifacts():
    tfidf = joblib.load(os.path.join(MODELS_DIR, "preprocessing.pkl"))
    all_models = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))
    
    # Fix compatibility issues with models trained on older scikit-learn versions
    for name, model in all_models.items():
        if hasattr(model, "probability") and not hasattr(model, "_effective_probability"):
            model._effective_probability = model.probability
            
    with open(os.path.join(MODELS_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    test["processed_sms"] = test["processed_sms"].fillna("")
    return tfidf, all_models, meta["best_model_name"], test


def comparison_table(results_path):
    with open(results_path) as f:
        results = json.load(f)
    rows = []
    for name, r in results.items():
        m = r["test_metrics"]
        rows.append({
            "Model": name,
            "Best Params": str(r["best_params"]),
            "Accuracy": round(m["accuracy"], 4),
            "Precision": round(m["precision"], 4),
            "Recall": round(m["recall"], 4),
            "F1-Score": round(m["f1"], 4),
            "ROC-AUC": round(m["roc_auc"], 4),
        })
    df = pd.DataFrame(rows).sort_values("F1-Score", ascending=False)
    return df


def plot_roc_curves(all_models, tfidf, test, save_path):
    X_test = tfidf.transform(test["processed_sms"])
    y_test = test["label_num"]

    plt.figure(figsize=(7, 6))
    colors = {"Naive Bayes": "#4C72B0", "SVM": "#55A868", "Random Forest": "#C44E52", "XGBoost": "#8172B2"}
    for name, model in all_models.items():
        proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=colors.get(name), linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Perbandingan ROC Curve Seluruh Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrices(all_models, tfidf, test, save_path):
    X_test = tfidf.transform(test["processed_sms"])
    y_test = test["label_num"]

    fig, axes = plt.subplots(1, len(all_models), figsize=(6 * len(all_models), 5))
    cmaps = ["Blues", "Greens", "Oranges", "Purples"]
    for ax, (name, model), cmap in zip(axes, all_models.items(), cmaps):
        pred = model.predict(X_test)
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, ax=ax,
                    xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
        ax.set_title(f"Confusion Matrix - {name}")
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_shap_summary(best_model, tfidf, test, save_path, model_name, sample_size=200):
    import shap

    X_test = tfidf.transform(test["processed_sms"])
    feature_names = np.array(tfidf.get_feature_names_out())

    # Subsample agar komputasi SHAP ringan
    rng = np.random.default_rng(42)
    idx = rng.choice(X_test.shape[0], size=min(sample_size, X_test.shape[0]), replace=False)
    X_sample = X_test[idx]

    if model_name == "Naive Bayes":
        explainer = shap.LinearExplainer(best_model, X_sample, feature_names=feature_names)
        shap_values = explainer.shap_values(X_sample)
    elif model_name == "SVM":
        # shap.LinearExplainer bermasalah dengan coef_ sparse bawaan SVC (kernel linear).
        # Hitung SHAP value linear secara manual: shap_i = coef_i * (x_i - mean_i)
        X_sample_dense = np.asarray(X_sample.toarray())
        coef = np.asarray(best_model.coef_.toarray()).ravel()
        background_mean = X_sample_dense.mean(axis=0)
        shap_values = (X_sample_dense - background_mean) * coef
        X_sample = X_sample_dense
    else:  # Random Forest / tree-based
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

    plt.figure()
    shap.summary_plot(
        shap_values, X_sample, feature_names=feature_names, show=False, max_display=15
    )
    plt.title(f"SHAP Feature Importance - {model_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Ambil top-N kata berdasarkan mean(|shap_value|)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    if hasattr(mean_abs_shap, "A1"):
        mean_abs_shap = mean_abs_shap.A1
    top_idx = np.argsort(mean_abs_shap)[::-1][:15]
    top_words = pd.DataFrame({
        "kata": feature_names[top_idx],
        "mean_abs_shap": mean_abs_shap[top_idx],
    })
    return top_words


def run_evaluation():
    os.makedirs(FIG_DIR, exist_ok=True)
    tfidf, all_models, best_name, test = load_artifacts()

    comp_df = comparison_table(os.path.join(REPORTS_DIR, "model_results.json"))
    comp_df.to_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"), index=False)
    print(comp_df.to_string(index=False))

    plot_roc_curves(all_models, tfidf, test, os.path.join(FIG_DIR, "roc_curves.png"))
    plot_confusion_matrices(all_models, tfidf, test, os.path.join(FIG_DIR, "confusion_matrices.png"))

    best_model = all_models[best_name]
    top_words = plot_shap_summary(
        best_model, tfidf, test, os.path.join(FIG_DIR, "shap_summary.png"), best_name
    )
    top_words.to_csv(os.path.join(REPORTS_DIR, "shap_top_words.csv"), index=False)
    print(f"\nModel terbaik: {best_name}")
    print(top_words)

    return comp_df, top_words, best_name


if __name__ == "__main__":
    run_evaluation()
