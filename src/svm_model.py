########## 1. Imports ##########
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from sklearn.svm import SVC

########## 2. Text Cleaning ##########

def clean_text(text):
    text = str(text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^A-Za-z0-9 ]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

########## 3. Datasets ##########

projects = [
    "pytorch",
    "tensorflow",
    "keras",
    "incubator-mxnet",
    "caffe"
]

REPEAT = 30

########## 4. Loop through datasets ##########

for project in projects:

    print(f"\n===== DATASET: {project.upper()} =====")

    df = pd.read_csv(f"../data/{project}.csv")

    # Combine Title + Body
    df["text"] = df.apply(
        lambda row: row["Title"] + " " + str(row["Body"]) if pd.notna(row["Body"]) else row["Title"],
        axis=1
    )

    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["class"]

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    for i in range(REPEAT):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=i
        )

        # TF-IDF
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        # SVM model
        model = SVC(class_weight='balanced', kernel='linear')

        model.fit(X_train_tfidf, y_train)

        y_pred = model.predict(X_test_tfidf)

        # Metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))

    # Print results per dataset
    print(f"Runs: {REPEAT}")
    print(f"Accuracy:  {np.mean(accuracies):.4f}")
    print(f"Precision: {np.mean(precisions):.4f}")
    print(f"Recall:    {np.mean(recalls):.4f}")
    print(f"F1 Score:  {np.mean(f1_scores):.4f}")