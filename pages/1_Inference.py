import streamlit as st, joblib, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_data
import json, os
st.title("📝 Inference Interface")

import pickle, torch.nn.functional as F

@st.cache_resource
def load_models():
    # --- baseline ---
    tfidf  = joblib.load("artifacts/tfidf.pkl")
    base   = joblib.load("artifacts/baseline_lr.pkl")

    # --- BERT ---
    tok    = AutoTokenizer.from_pretrained("artifacts/bert_model")
    bert   = AutoModelForSequenceClassification.from_pretrained("artifacts/bert_model")

    # --- LSTM ---
    with open("artifacts/lstm_vocab.pkl", "rb") as f:
        lstm_vocab = pickle.load(f)

    from train_lstm import BiLSTMClassifier, emb_dim     
    lstm_model = BiLSTMClassifier(len(lstm_vocab), emb_dim,
                                hidden_dim=128, dropout=0.3,
                                pad_idx=lstm_vocab["<pad>"])
    lstm_model.load_state_dict(torch.load("artifacts/lstm_best.pt", map_location="cpu"))
    lstm_model.eval()

    return tfidf, base, tok, bert, lstm_vocab, lstm_model

tfidf, base_lr, tok, bert_model, lstm_vocab, lstm_model = load_models()

choices = {"Baseline LR": "baseline", "LSTM": "lstm", "BERT": "bert"}
model_name = st.selectbox("Modelo", list(choices.keys()))
txt = st.text_area("Introduce texto", height=150)

def predict(text, model_sel):
    if model_sel == "baseline":
        X = tfidf.transform([text])
        prob = base_lr.predict_proba(X)[0]
        return prob
    elif model_sel == "bert":
        inputs = tok(text, return_tensors="pt")
        with torch.no_grad():
            logits = bert_model(**inputs).logits
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return prob
    else:  # lstm
        tokens = re.findall(r"\b\w+\b", text.lower())
        idxs = [lstm_vocab[t] for t in tokens]
        idxs = idxs[:60] + [lstm_vocab["<pad>"]] * (60 - len(idxs))
        logits = lstm_model(torch.tensor([idxs]))
        return F.softmax(logits, dim=1).numpy()[0]

if st.button("Predecir") and txt.strip():
    probs = predict(txt, choices[model_name])
    pred  = int(np.argmax(probs))
    st.write(f"### Predicción: **{ 'Hate' if pred else 'No-Hate' }**")
    st.write(f"Confianza: {probs[pred]:.3f}")
