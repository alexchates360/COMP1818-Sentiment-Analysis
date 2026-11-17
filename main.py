from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
data_path = Path("data/txt_sentoken")

reviews = []
labels = []
for file in sorted((data_path / "pos").glob("*.txt")):
    reviews.append(open(file).read())
    labels.append(1)
for file in sorted((data_path / "neg").glob("*.txt")):
    reviews.append(open(file).read())
    labels.append(0)
print(len(reviews))
print(len(labels))

df = pd.DataFrame({"review": reviews, "sentiment": labels})
print(df.shape)
print(df["sentiment"].value_counts())
X = df["review"]
y = df["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)
print("Baseline accuracy:", accuracy_score(y_test, baseline_predictions))

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)
print("Training matrix:", X_train_counts.shape)
print("Testing matrix:", X_test_counts.shape)

nb_model = MultinomialNB()
nb_model.fit(X_train_counts, y_train)
nb_predictions = nb_model.predict(X_test_counts)
print("Naive Bayes accuracy:", accuracy_score(y_test, nb_predictions))

svm_model = LinearSVC(random_state=42, max_iter=10000)
svm_model.fit(X_train_counts, y_train)
svm_predictions = svm_model.predict(X_test_counts)
print("SVM accuracy:", accuracy_score(y_test, svm_predictions))

tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print("TF-IDF training matrix:", X_train_tfidf.shape)
print("TF-IDF testing matrix:", X_test_tfidf.shape)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("svm", LinearSVC(random_state=42, max_iter=10000))
])

param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "svm__C": [0.1, 1, 10]
}

search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")
search.fit(X_train, y_train)
print("Best parameters:", search.best_params_)
print("Best CV accuracy:", search.best_score_)

optimized_predictions = search.predict(X_test)
print("Optimized SVM accuracy:", accuracy_score(y_test, optimized_predictions))

print(classification_report(y_test, nb_predictions))
print(classification_report(y_test, svm_predictions))
print(classification_report(y_test, optimized_predictions))
print(confusion_matrix(y_test, svm_predictions))
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

Path("results").mkdir(exist_ok=True)

# Model accuracy comparison
models = [
    "Dummy",
    "Naive Bayes\nBoW",
    "Linear SVM\nBoW",
    "Tuned SVM\nTF-IDF"
]

accuracies = [
    accuracy_score(y_test, baseline_predictions) * 100,
    accuracy_score(y_test, nb_predictions) * 100,
    accuracy_score(y_test, svm_predictions) * 100,
    accuracy_score(y_test, optimized_predictions) * 100
]

fig, ax = plt.subplots(figsize=(8, 4))

bars = ax.bar(
    models, accuracies,
    color=["#9E9EA7", "#6689A3", "#23476A", "#47808E"]
)

ax.set_ylabel("Test accuracy (%)")
ax.set_ylim(0, 100)
ax.bar_label(bars, fmt="%.2f%%")

fig.tight_layout()
fig.savefig("results/accuracy_comparison.png", dpi=300)
plt.close(fig)

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))

cm = confusion_matrix(
    y_test, svm_predictions, labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Negative", "Positive"]
)

display.plot(ax=ax, cmap="Blues", values_format="d")

fig.tight_layout()
fig.savefig("results/confusion_matrix.png", dpi=300)
plt.close(fig)

import joblib
joblib.dump(baseline, "models/baseline.joblib")
joblib.dump(vectorizer, "models/vectorizer.joblib")
joblib.dump(nb_model, "models/naive_bayes.joblib")
joblib.dump(svm_model, "models/svm.joblib")
joblib.dump(search.best_estimator_, "models/optimized_svm.joblib")