"""
data_preprocessing.py
Script preprocessing data untuk proyek klasifikasi SMS Spam.

Alur:
1. Load data mentah (data/raw/spam.csv)
2. Analisis & penanganan kualitas data (missing value, duplikat)
3. Feature engineering (statistik pesan: panjang karakter, jumlah kata, dll.)
4. Text preprocessing (case folding, cleaning, stopword removal, stemming)
5. Split data train/validation/test (stratified)
6. Simpan hasil ke data/processed/
"""

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))
from utils import preprocess_text, count_message_stats

RAW_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "spam.csv")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def load_raw_data(path=RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    df = df.dropna(how="any", axis=1)
    df.columns = ["label", "sms"]
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Analisis kualitas data: missing value, duplikat, inconsistency."""
    report = {
        "jumlah_awal": len(df),
        "missing_values": int(df.isnull().sum().sum()),
        "duplikat": int(df.duplicated().sum()),
    }
    df = df.dropna().drop_duplicates().reset_index(drop=True)
    report["jumlah_setelah_cleaning"] = len(df)
    return df, report


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambahkan fitur statistik pesan sebelum teks dibersihkan (agar informasi panjang asli tidak hilang)."""
    stats = df["sms"].apply(count_message_stats).apply(pd.Series)
    df = pd.concat([df, stats], axis=1)
    return df


def apply_text_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df["processed_sms"] = df["sms"].apply(preprocess_text)
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df


def split_data(df: pd.DataFrame, test_size=0.2, val_size=0.1, random_state=42):
    """Split menjadi train/validation/test dengan stratifikasi label."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label_num"], random_state=random_state
    )
    val_relative = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_relative, stratify=train_val["label_num"], random_state=random_state
    )
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def run_pipeline():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = load_raw_data()
    df, quality_report = clean_data(df)
    df = engineer_features(df)
    df = apply_text_preprocessing(df)

    train, val, test = split_data(df)

    train.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(PROCESSED_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)
    df.to_csv(os.path.join(PROCESSED_DIR, "full_clean.csv"), index=False)

    print("=== Laporan Kualitas Data ===")
    for k, v in quality_report.items():
        print(f"{k}: {v}")
    print(f"\nTrain: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"Distribusi label (keseluruhan):\n{df['label'].value_counts()}")

    return df, train, val, test, quality_report


if __name__ == "__main__":
    run_pipeline()
