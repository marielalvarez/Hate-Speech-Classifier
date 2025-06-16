import streamlit as st, joblib, torch, numpy as np, gdown, os, re, pickle, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils import load_data

st.title("Inference Interface")
st.write("The models are downloading from drive.")
choices = {"Baseline LR": "baseline", "LSTM": "lstm", "BERT": "bert"}
model_name = st.selectbox("Select a model for your predictions.", list(choices.keys()))

GDRIVE = {
    "tfidf.pkl":              "1i__vZFTIspqTZqdGDmQrkG67_5t6hnfr",
    "baseline_lr.pkl":        "1tYQ62qBSgHqAphP_FyaX1l53G1oG2ezz",
    "lstm_vocab.pkl":         "1hjWl9sfUAnn9uOz6VQhm1urH4hU6hB0H",
    "bilstm_best_params.pkl": "1FV7QBBTzLF4Vondqj9reh4G9qQB9z58W",
    "lstm_best.pt":           "1gMuF4ELouhnqPRWb2racAqOF63J5B2vT",
   
    "bert_model.zip":         "1wsnOJL7NWZ7L83dI7QryH7kEFPS4O51Z"
}

ART_DIR = "artifacts"        

def fetch_if_needed(fname):
    """Descarga de Google Drive solo si no existe localmente."""
    path = os.path.join(ART_DIR, fname)
    if not os.path.exists(path):
        os.makedirs(ART_DIR, exist_ok=True)
        url = f"https://drive.google.com/uc?id={GDRIVE[fname]}"
        st.info(f"Downloading {fname} from Drive…")
        gdown.download(url, path, quiet=False)
        if fname.endswith(".zip"):
            import zipfile, shutil
            with zipfile.ZipFile(path, 'r') as zf:
                zf.extractall(os.path.join(ART_DIR, "bert_model"))
            os.remove(path)
    return path

@st.cache_resource
def load_models():
    # ▼ BASELINE
    tfidf_path = fetch_if_needed("tfidf.pkl")
    lr_path    = fetch_if_needed("baseline_lr.pkl")
    tfidf  = joblib.load(tfidf_path)
    base   = joblib.load(lr_path)

    # ▼ BERT  (
    fetch_if_needed("bert_model.zip")
    bert_dir = os.path.join(ART_DIR, "bert_model")
    tok   = AutoTokenizer.from_pretrained(bert_dir)
    bert  = AutoModelForSequenceClassification.from_pretrained(bert_dir)

    # ▼ Bi-LSTM
    vocab   = pickle.load(open(fetch_if_needed("lstm_vocab.pkl"), "rb"))
    best_hp = pickle.load(open(fetch_if_needed("bilstm_best_params.pkl"), "rb"))
    from train_lstm import BiLSTMClassifier, emb_dim
    lstm = BiLSTMClassifier(len(vocab), emb_dim,
                            best_hp["hidden_dim"], best_hp["dropout"],
                            vocab["<pad>"])
    lstm.load_state_dict(torch.load(fetch_if_needed("lstm_best.pt"), map_location="cpu"))
    lstm.eval()
    return tfidf, base, tok, bert, vocab, lstm

tfidf, base_lr, tok, bert_model, lstm_vocab, lstm_model = load_models()

txt = st.text_area("Write a message...", height=150)

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
        return F.softmax(logits, dim=1).detach().numpy()[0]


if st.button("Predict") and txt.strip():
    probs = predict(txt, choices[model_name])
    pred  = int(np.argmax(probs))
    st.write(f"### Prediction: **{ 'Hate' if pred else 'No-Hate' }**")
    st.write(f"Confidence: {probs[pred]:.3f}")