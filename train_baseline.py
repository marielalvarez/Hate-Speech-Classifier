from utils import load_data, save_report
import joblib, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import json

# trains baseline and generales reports in json format for Streamlit.

train_df, test_df = load_data()

tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=30_000,
                        lowercase=True, stop_words="english")
X_train = tfidf.fit_transform(train_df["tweet"])
X_test  = tfidf.transform(test_df["tweet"])

clf = LogisticRegression(max_iter=200, n_jobs=-1)
clf.fit(X_train, train_df["label"])

preds = clf.predict(X_test)
with open("artifacts/baseline_preds.json", "w") as f:
    json.dump(preds.tolist(), f)

print("✅ Baseline preds saved to artifacts/baseline_preds.json")

save_report(test_df["label"], preds, "baseline")

os.makedirs("artifacts", exist_ok=True)
joblib.dump(tfidf, "artifacts/tfidf.pkl")
joblib.dump(clf,   "artifacts/baseline_lr.pkl")
print("✅ Baseline entrenado y guardado en artifacts/")
