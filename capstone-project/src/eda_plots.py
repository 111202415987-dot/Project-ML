"""
eda_plots.py
Menghasilkan visualisasi EDA utama (5 insight kunci) dari data yang sudah dibersihkan.
Dipakai baik oleh notebook maupun dashboard Streamlit.
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")

sns.set_style("whitegrid")


def load_full():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "full_clean.csv"))
    df["processed_sms"] = df["processed_sms"].fillna("")
    return df


def insight_1_class_distribution(df, save_path):
    plt.figure(figsize=(6, 5))
    counts = df["label"].value_counts()
    colors = ["#4C72B0", "#DD8452"]
    bars = plt.bar(counts.index, counts.values, color=colors)
    for b, c in zip(bars, counts.values):
        plt.text(b.get_x() + b.get_width() / 2, c + 30, str(c), ha="center", fontweight="bold")
    plt.title("Insight 1: Distribusi Kelas (Data Setelah Cleaning)")
    plt.ylabel("Jumlah Pesan")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def insight_2_char_length(df, save_path):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x="char_count", hue="label", bins=40, kde=True,
                 palette={"ham": "#4C72B0", "spam": "#DD8452"}, alpha=0.6)
    plt.title("Insight 2: Distribusi Panjang Karakter Pesan per Kelas")
    plt.xlabel("Jumlah Karakter")
    plt.ylabel("Frekuensi")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def insight_3_word_count(df, save_path):
    plt.figure(figsize=(7, 5))
    sns.boxplot(data=df, x="label", y="word_count", hue="label",
                palette={"ham": "#4C72B0", "spam": "#DD8452"}, legend=False)
    plt.title("Insight 3: Perbandingan Jumlah Kata per Kelas")
    plt.xlabel("Kelas")
    plt.ylabel("Jumlah Kata")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def insight_4_feature_correlation(df, save_path):
    feat_cols = ["char_count", "word_count", "has_number", "has_currency", "exclamation_count", "label_num"]
    corr = df[feat_cols].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0)
    plt.title("Insight 4: Korelasi Fitur Numerik terhadap Label Spam")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def insight_5_top_words(df, save_path, label="spam", top_n=15):
    text = " ".join(df.loc[df["label"] == label, "processed_sms"])
    words = [w for w in text.split() if len(w) > 2]
    common = Counter(words).most_common(top_n)
    words_, counts_ = zip(*common)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=list(counts_), y=list(words_), hue=list(words_), palette="Reds_r" if label == "spam" else "Blues_r", legend=False)
    plt.title(f"Insight 5: Top {top_n} Kata Paling Sering Muncul pada Pesan {label.upper()}")
    plt.xlabel("Frekuensi")
    plt.ylabel("Kata (setelah stemming)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return pd.DataFrame(common, columns=["kata", "frekuensi"])


def run_eda():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = load_full()

    insight_1_class_distribution(df, os.path.join(FIG_DIR, "insight1_class_distribution.png"))
    insight_2_char_length(df, os.path.join(FIG_DIR, "insight2_char_length.png"))
    insight_3_word_count(df, os.path.join(FIG_DIR, "insight3_word_count.png"))
    insight_4_feature_correlation(df, os.path.join(FIG_DIR, "insight4_correlation.png"))
    top_spam = insight_5_top_words(df, os.path.join(FIG_DIR, "insight5_top_spam_words.png"), "spam")
    top_ham = insight_5_top_words(df, os.path.join(FIG_DIR, "insight5_top_ham_words.png"), "ham")

    top_spam.to_csv(os.path.join(BASE_DIR, "reports", "top_spam_words.csv"), index=False)
    top_ham.to_csv(os.path.join(BASE_DIR, "reports", "top_ham_words.csv"), index=False)

    print("EDA insights generated.")
    print("\nTop kata SPAM:\n", top_spam)
    print("\nTop kata HAM:\n", top_ham)

    return df


if __name__ == "__main__":
    run_eda()
