import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

HF_PATH = "hf://datasets/thefrankhsu/hate_speech_twitter/"

def load_data():
    splits = {"train": "training set.csv", "test": "testing set.csv"}
    train = df = pd.read_csv(HF_PATH + splits["train"], usecols=["tweet", "label"]).dropna()
    test  = pd.read_csv(HF_PATH + splits["test"],  usecols=["tweet", "label"]).dropna()
    return train, test

def save_report(y_true, y_pred, name: str, out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    report = classification_report(y_true, y_pred, output_dict=True, digits=3)
    with open(f"{out_dir}/{name}_report.json", "w") as f:
        json.dump(report, f, indent=2)

    cm = confusion_matrix(y_true, y_pred)
    np.save(f"{out_dir}/{name}_cm.npy", cm)

def plot_confusion(cm: np.ndarray, labels=("non-hate", "hate"), title="Confusion Matrix"):
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted"); plt.ylabel("True")

def make_wordcloud(text, title="Word Cloud"):
    wc = WordCloud(width=800, height=400, background_color="white").generate(text)
    plt.figure(figsize=(8,4))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off"); plt.title(title)
