"""
app.py
Aplikasi Streamlit: SMS Spam Detector
Capstone Project - Pembelajaran Mesin UAS Genap 2025/2026
Vasya Citra Narindra - A11.2024.15987

Menu:
1. Dashboard EDA
2. Model Demo
3. Evaluasi Model
4. Interpretasi Hasil
5. Dokumentasi
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from utils import preprocess_text  # noqa: E402

# ----------------------------- Konfigurasi Path -----------------------------
BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

st.set_page_config(
    page_title="SMS Spam Detector | Capstone ML",
    page_icon="📩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Cached Loaders -----------------------------

@st.cache_resource
def load_models():
    tfidf = joblib.load(os.path.join(MODELS_DIR, "preprocessing.pkl"))
    all_models = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))
    with open(os.path.join(MODELS_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    return tfidf, all_models, meta["best_model_name"]


@st.cache_data
def load_full_data():
    path = os.path.join(PROCESSED_DIR, "full_clean.csv")
    df = pd.read_csv(path)
    df["processed_sms"] = df["processed_sms"].fillna("")
    return df


@st.cache_data
def load_model_results():
    with open(os.path.join(REPORTS_DIR, "model_results.json")) as f:
        results = json.load(f)
    comparison_df = pd.read_csv(os.path.join(REPORTS_DIR, "model_comparison.csv"))
    return results, comparison_df


@st.cache_data
def load_shap_words():
    return pd.read_csv(os.path.join(REPORTS_DIR, "shap_top_words.csv"))


@st.cache_data
def load_top_words():
    spam = pd.read_csv(os.path.join(REPORTS_DIR, "top_spam_words.csv"))
    ham = pd.read_csv(os.path.join(REPORTS_DIR, "top_ham_words.csv"))
    return spam, ham


# ----------------------------- Sidebar Navigation -----------------------------

st.sidebar.title("📩 SMS Spam Detector")
st.sidebar.caption("Capstone Project — Pembelajaran Mesin UAS Genap 2025/2026")
st.sidebar.markdown("---")

PAGES = [
    "🏠 Dashboard EDA",
    "🔮 Model Demo",
    "📊 Evaluasi Model",
    "🧠 Interpretasi Hasil",
    "📖 Dokumentasi",
]
page = st.sidebar.radio("Navigasi", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Disusun oleh:**\n\nVasya Citra Narindra\n\nNIM: A11.2024.15987\n\nKelompok: A11.4404"
)

# ================================================================================
# PAGE 1 - DASHBOARD EDA
# ================================================================================
if page == "🏠 Dashboard EDA":
    st.title("🏠 Dashboard EDA — SMS Spam Collection Dataset")
    st.markdown(
        "Eksplorasi interaktif dataset SMS Spam Collection setelah proses *cleaning* "
        "(penghapusan duplikat) dan *feature engineering*."
    )

    df = load_full_data()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pesan", f"{len(df):,}")
    col2.metric("Pesan Ham", f"{(df['label']=='ham').sum():,}")
    col3.metric("Pesan Spam", f"{(df['label']=='spam').sum():,}")
    col4.metric("Rasio Spam", f"{(df['label']=='spam').mean()*100:.1f}%")

    st.markdown("### Distribusi Kelas")
    class_counts = df["label"].value_counts().reset_index()
    class_counts.columns = ["label", "jumlah"]
    fig1 = px.bar(
        class_counts, x="label", y="jumlah", color="label", text="jumlah",
        color_discrete_map={"ham": "#4C72B0", "spam": "#DD8452"},
        title="Distribusi Kelas Ham vs Spam",
    )
    st.plotly_chart(fig1, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Distribusi Panjang Karakter")
        fig2 = px.histogram(
            df, x="char_count", color="label", nbins=40, barmode="overlay",
            color_discrete_map={"ham": "#4C72B0", "spam": "#DD8452"},
            title="Jumlah Karakter per Kelas",
        )
        st.plotly_chart(fig2, width='stretch')
    with c2:
        st.markdown("### Distribusi Jumlah Kata")
        fig3 = px.box(
            df, x="label", y="word_count", color="label",
            color_discrete_map={"ham": "#4C72B0", "spam": "#DD8452"},
            title="Jumlah Kata per Kelas",
        )
        st.plotly_chart(fig3, width='stretch')

    st.markdown("### Korelasi Fitur Numerik terhadap Label")
    feat_cols = ["char_count", "word_count", "has_number", "has_currency", "exclamation_count", "label_num"]
    corr = df[feat_cols].corr()
    fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Heatmap Korelasi Fitur")
    st.plotly_chart(fig4, width='stretch')

    st.markdown("### Kata Paling Sering Muncul")
    top_spam, top_ham = load_top_words()
    c3, c4 = st.columns(2)
    with c3:
        fig5 = px.bar(top_spam.sort_values("frekuensi"), x="frekuensi", y="kata", orientation="h",
                       title="Top 15 Kata pada Pesan SPAM", color_discrete_sequence=["#DD8452"])
        st.plotly_chart(fig5, width='stretch')
    with c4:
        fig6 = px.bar(top_ham.sort_values("frekuensi"), x="frekuensi", y="kata", orientation="h",
                       title="Top 15 Kata pada Pesan HAM", color_discrete_sequence=["#4C72B0"])
        st.plotly_chart(fig6, width='stretch')

    with st.expander("🔍 Lihat Sampel Data"):
        st.dataframe(df[["label", "sms", "processed_sms", "char_count", "word_count"]].sample(10, random_state=1))

# ================================================================================
# PAGE 2 - MODEL DEMO
# ================================================================================
elif page == "🔮 Model Demo":
    st.title("🔮 Model Demo — Prediksi SMS Spam Real-Time")
    st.markdown("Masukkan teks SMS di bawah ini, lalu pilih model untuk melihat hasil prediksinya.")

    tfidf, all_models, best_name = load_models()

    model_choice = st.selectbox(
        "Pilih Model", list(all_models.keys()),
        index=list(all_models.keys()).index(best_name),
        help="Model terbaik hasil evaluasi ditandai secara default.",
    )
    if model_choice == best_name:
        st.success(f"✅ {model_choice} adalah model dengan performa terbaik (F1-Score kelas Spam tertinggi).")

    example_msgs = {
        "-- Pilih contoh --": "",
        "Contoh Spam": "URGENT! You have won a 1 week FREE membership in our £100,000 prize Jackpot! Txt the word CLAIM to No: 81010",
        "Contoh Ham": "Hey, are we still on for dinner tonight? Let me know what time works for you.",
    }
    example_key = st.selectbox("Atau gunakan contoh pesan:", list(example_msgs.keys()))
    default_text = example_msgs[example_key]

    user_text = st.text_area("Teks SMS:", value=default_text, height=120, placeholder="Ketik atau tempel isi SMS di sini...")

    if st.button("🔍 Prediksi", type="primary"):
        if not user_text.strip():
            st.warning("Silakan masukkan teks SMS terlebih dahulu.")
        else:
            processed = preprocess_text(user_text)
            X_input = tfidf.transform([processed])
            model = all_models[model_choice]
            pred = model.predict(X_input)[0]
            proba = model.predict_proba(X_input)[0]

            label = "SPAM 🚨" if pred == 1 else "HAM ✅"
            conf = proba[pred] * 100

            colA, colB = st.columns([1, 1])
            with colA:
                if pred == 1:
                    st.error(f"### Hasil Prediksi: {label}")
                else:
                    st.success(f"### Hasil Prediksi: {label}")
                st.metric("Tingkat Keyakinan Model", f"{conf:.2f}%")
            with colB:
                fig = go.Figure(go.Bar(
                    x=[proba[0] * 100, proba[1] * 100], y=["Ham", "Spam"], orientation="h",
                    marker_color=["#4C72B0", "#DD8452"],
                ))
                fig.update_layout(title="Probabilitas Kelas", xaxis_title="Probabilitas (%)", height=250)
                st.plotly_chart(fig, width='stretch')

            with st.expander("🛠️ Detail Preprocessing"):
                st.write("**Teks asli:**", user_text)
                st.write("**Setelah preprocessing:**", processed if processed else "_(kosong setelah cleaning)_")
                st.write(f"**Jumlah fitur TF-IDF aktif:** {X_input.nnz} dari {X_input.shape[1]} total fitur")

# ================================================================================
# PAGE 3 - EVALUASI MODEL
# ================================================================================
elif page == "📊 Evaluasi Model":
    st.title("📊 Evaluasi Model")
    st.markdown("Perbandingan performa seluruh model pada data uji (test set) setelah hyperparameter tuning.")

    results, comparison_df = load_model_results()

    st.markdown("### Tabel Perbandingan Performa")
    st.dataframe(
        comparison_df.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"], color="#c6efce"),
        width='stretch',
    )

    st.markdown("### Grafik Perbandingan Metrik")
    metrics_long = comparison_df.melt(
        id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
        var_name="Metrik", value_name="Nilai",
    )
    fig = px.bar(metrics_long, x="Metrik", y="Nilai", color="Model", barmode="group",
                 range_y=[0, 1.05], title="Perbandingan Metrik Evaluasi Antar Model")
    st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ROC Curve")
        roc_path = os.path.join(FIG_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, width='stretch')
    with c2:
        st.markdown("### Confusion Matrix")
        cm_path = os.path.join(FIG_DIR, "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, width='stretch')

    st.markdown("### Detail per Model")
    model_pick = st.selectbox("Pilih model untuk melihat classification report:", list(results.keys()))
    report = results[model_pick]["test_metrics"]["classification_report"]
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, width='stretch')
    st.caption(f"Best hyperparameters: `{results[model_pick]['best_params']}`")

# ================================================================================
# PAGE 4 - INTERPRETASI HASIL
# ================================================================================
elif page == "🧠 Interpretasi Hasil":
    st.title("🧠 Interpretasi Hasil & Insight Bisnis")

    tfidf, all_models, best_name = load_models()
    st.info(f"Model terbaik yang dipakai pada aplikasi ini: **{best_name}**")

    st.markdown("### SHAP Feature Importance")
    st.markdown(
        "Grafik berikut menunjukkan kata-kata (fitur TF-IDF) yang paling berpengaruh terhadap "
        f"keputusan model **{best_name}** dalam mengklasifikasikan pesan sebagai spam."
    )
    shap_path = os.path.join(FIG_DIR, "shap_summary.png")
    if os.path.exists(shap_path):
        st.image(shap_path, width='stretch')

    top_words = load_shap_words()
    fig = px.bar(
        top_words.sort_values("mean_abs_shap"), x="mean_abs_shap", y="kata", orientation="h",
        title="Top 15 Kata Berdasarkan Mean |SHAP Value|", color_discrete_sequence=["#55A868"],
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("### 💡 Insight Bisnis")
    st.markdown(
        """
- Kata-kata seperti **'free', 'txt', 'call', 'claim', 'stop'** menjadi sinyal kuat pesan spam —
  konsisten dengan pola pesan spam yang menawarkan hadiah/promosi dan mengarahkan korban untuk
  segera menghubungi/membalas.
- Pesan spam cenderung **lebih panjang** dan memiliki **jumlah kata lebih seragam** dibanding ham,
  mengindikasikan penggunaan template pesan massal oleh pengirim spam.
- Model dengan **recall tinggi pada kelas spam** lebih disarankan untuk implementasi nyata, karena
  biaya kesalahan meloloskan spam (*false negative*) — yang berpotensi phishing — lebih besar
  dibandingkan biaya salah menandai pesan penting sebagai spam (*false positive*).
- Rekomendasi: gunakan model **SVM (linear, balanced)** sebagai model produksi karena keseimbangan
  terbaik antara precision dan recall pada kelas spam, serta ROC-AUC tertinggi.
        """
    )

# ================================================================================
# PAGE 5 - DOKUMENTASI
# ================================================================================
elif page == "📖 Dokumentasi":
    st.title("📖 Dokumentasi Proyek")

    st.markdown(
        """
## Deskripsi Proyek
Aplikasi ini merupakan hasil akhir dari Capstone Project mata kuliah **Pembelajaran Mesin**
(UAS Genap 2025/2026) yang bertujuan membangun sistem klasifikasi SMS Spam end-to-end,
mulai dari akuisisi data, EDA, pemodelan, hingga deployment.

## Dataset
- **Sumber:** SMS Spam Collection Dataset (publik, berbahasa Inggris).
- **Jumlah data:** 5.572 pesan mentah → 5.169 pesan setelah pembersihan duplikat.
- **Kelas:** `ham` (bukan spam) dan `spam` — bersifat *imbalanced* (±87% vs ±13%).

## Metodologi (Alur Kerja)
1. **Data Acquisition** — memuat data mentah SMS Spam Collection.
2. **EDA & Preprocessing** — analisis kualitas data, feature engineering, text cleaning
   (case folding, cleaning, tokenizing, stopword removal, stemming), data splitting (train/val/test).
3. **Feature Extraction** — TF-IDF Vectorizer (6.000 fitur maksimum).
4. **Modeling & Tuning** — 3 algoritma (Naive Bayes, SVM, Random Forest) dituning dengan `GridSearchCV`
   (scoring F1, 5-fold CV).
5. **Evaluasi** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix.
6. **Interpretasi** — SHAP untuk mengetahui fitur/kata paling berpengaruh.
7. **Deployment** — aplikasi Streamlit ini (Dashboard EDA, Model Demo, Evaluasi, Interpretasi, Dokumentasi).

## Cara Menggunakan Aplikasi
- **Dashboard EDA**: eksplorasi statistik dan visualisasi dataset secara interaktif.
- **Model Demo**: masukkan teks SMS bebas, pilih model, lalu klik "Prediksi" untuk melihat hasil klasifikasi
  beserta tingkat keyakinannya.
- **Evaluasi Model**: bandingkan performa ketiga model pada data uji.
- **Interpretasi Hasil**: pahami kata-kata apa saja yang mendorong model memprediksi spam.
- **Dokumentasi**: halaman ini — ringkasan metodologi dan cara pakai.

## Struktur Repository
```
capstone-project/
├── data/               # Data mentah & hasil preprocessing
├── notebooks/          # 01_eda.ipynb, 02_modeling.ipynb
├── src/                # Script preprocessing, training, evaluasi
├── models/             # Model & vectorizer hasil training (.pkl)
├── app/                # Aplikasi Streamlit (file ini)
├── reports/            # Hasil evaluasi, figures, laporan teknis
├── requirements.txt
└── README.md
```

## Tim Penyusun
**Vasya Citra Narindra** — NIM A11.2024.15987 — Kelompok A11.4501-45XX
Mata Kuliah Pembelajaran Mesin, Fakultas Ilmu Komputer, Universitas Dian Nuswantoro.
        """
    )
