# 📩 SMS Spam Detector — Capstone Project Pembelajaran Mesin

Proyek klasifikasi SMS Spam end-to-end (data acquisition → EDA → modeling → deployment) sebagai
pemenuhan **Ujian Akhir Semester (UAS) Genap 2025/2026** mata kuliah **Pembelajaran Mesin**,
Fakultas Ilmu Komputer, Universitas Dian Nuswantoro.

**Disusun oleh:** Vasya Citra Narindra — NIM A11.2024.15987 — Kelompok A11.4501-45XX

🔗 **Live App:** _(isi link Streamlit Community Cloud setelah proses deploy)_
🎥 **Video Presentasi:** _(isi link YouTube)_

---

## 📌 Ringkasan Proyek

Sistem ini mengklasifikasikan pesan SMS ke dalam dua kelas — **Ham** (bukan spam) dan **Spam** —
menggunakan pipeline *machine learning* berbasis TF-IDF dan tiga algoritma klasifikasi
(Naive Bayes, SVM, Random Forest) yang telah melalui proses *hyperparameter tuning* dan
interpretasi model menggunakan SHAP.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **SVM (terpilih)** | 0.9768 | 0.9280 | 0.8855 | **0.9062** | 0.9904 |
| Random Forest | 0.9768 | 0.9820 | 0.8321 | 0.9008 | 0.9888 |
| Naive Bayes | 0.9739 | 0.9333 | 0.8550 | 0.8924 | 0.9859 |

> Model terbaik dipilih berdasarkan **F1-Score kelas Spam** pada data uji, karena dataset bersifat
> *imbalanced* (±87% ham vs ±13% spam).

## 🗂️ Struktur Repository

```
capstone-project/
├── data/
│   ├── raw/                  # Data mentah (spam.csv)
│   └── processed/            # Data hasil cleaning & split (train/val/test)
├── notebooks/
│   ├── 01_eda.ipynb          # EDA, preprocessing, feature engineering
│   └── 02_modeling.ipynb     # Modeling, tuning, evaluasi, SHAP
├── src/
│   ├── utils.py               # Fungsi preprocessing teks bersama
│   ├── data_preprocessing.py  # Pipeline cleaning & split data
│   ├── train_model.py         # Training & hyperparameter tuning
│   └── evaluate_model.py      # Evaluasi, ROC/CM, SHAP
├── models/
│   ├── best_model.pkl         # Model terbaik (SVM)
│   ├── all_models.pkl         # Seluruh model terlatih
│   └── preprocessing.pkl      # TF-IDF Vectorizer
├── app/
│   └── app.py                 # Aplikasi Streamlit (5 halaman)
├── reports/
│   ├── figures/                # Seluruh visualisasi (EDA, ROC, CM, SHAP)
│   ├── model_comparison.csv
│   ├── model_results.json
│   └── Laporan_Teknis_*.docx
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Cara Menjalankan

### 1. Clone & Install Dependencies
```bash
git clone <repo-url>
cd capstone-project
pip install -r requirements.txt
```

### 2. Jalankan Pipeline (opsional, artefak sudah tersedia di `models/`)
```bash
python src/data_preprocessing.py
python src/train_model.py
python src/evaluate_model.py
python src/eda_plots.py
```

### 3. Jalankan Aplikasi Streamlit
```bash
cd app
streamlit run app.py
```
Aplikasi akan terbuka di `http://localhost:8501` dengan 5 menu:
- 🏠 **Dashboard EDA** — eksplorasi interaktif dataset
- 🔮 **Model Demo** — prediksi real-time dari input teks bebas
- 📊 **Evaluasi Model** — perbandingan performa 3 model
- 🧠 **Interpretasi Hasil** — SHAP feature importance & insight bisnis
- 📖 **Dokumentasi** — metodologi & cara pakai

## 🧰 Tech Stack

- **Bahasa:** Python 3.12
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Machine Learning:** Scikit-learn
- **Model Interpretation:** SHAP
- **Deployment:** Streamlit, Joblib

## 📊 Dataset

**SMS Spam Collection Dataset** — dataset publik berbahasa Inggris berisi 5.572 pesan SMS berlabel
`ham`/`spam`, umum digunakan pada riset klasifikasi teks (mirror publik Kaggle/UCI Machine Learning
Repository).

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik (Tugas UAS) dan bersifat open untuk tujuan pembelajaran.
