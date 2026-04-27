import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from scipy.stats import wilcoxon

def clean_text(text):
    text = str(text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^A-Za-z0-9 ]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

projects = [
    "pytorch",
    "tensorflow",
    "keras",
    "incubator-mxnet",
    "caffe"
]

REPEAT = 30

for project in projects:

    print(f"\n===== DATASET: {project.upper()} =====")

    df = pd.read_csv(f"../data/{project}.csv")

    df["text"] = df.apply(
        lambda row: row["Title"] + " " + str(row["Body"]) if pd.notna(row["Body"]) else row["Title"],
        axis=1
    )

    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["class"]

    baseline_f1 = []
    svm_f1 = []

    for i in range(REPEAT):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=i
        )

        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        # ----- BASELINE -----
        nb = MultinomialNB()
        nb.fit(X_train_tfidf, y_train)
        y_pred_nb = nb.predict(X_test_tfidf)

        baseline_f1.append(f1_score(y_test, y_pred_nb, zero_division=0))

        # ----- SVM -----
        svm = SVC(class_weight='balanced', kernel='linear')
        svm.fit(X_train_tfidf, y_train)
        y_pred_svm = svm.predict(X_test_tfidf)

        svm_f1.append(f1_score(y_test, y_pred_svm, zero_division=0))

    # ---------- RESULTS ----------
    print(f"Baseline F1: {np.mean(baseline_f1):.4f}")
    print(f"SVM F1:      {np.mean(svm_f1):.4f}")

    # ---------- STATISTICAL TEST ----------
    stat, p = wilcoxon(baseline_f1, svm_f1)

    print(f"Wilcoxon statistic: {stat}")
    print(f"p-value: {p}")

  