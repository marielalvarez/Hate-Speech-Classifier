import streamlit as st, json, numpy as np, seaborn as sns, matplotlib.pyplot as plt, os, pandas as pd
from utils import plot_confusion

st.title("Model Analysis & Justification")

def load_artifact(name):
    with open(f"artifacts/{name}_report.json") as f:
        rep = json.load(f)
    cm = np.load(f"artifacts/{name}_cm.npy")
    return rep, cm


st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Baseline  ·  TF-IDF + LogReg")
    st.markdown(
        """
        **Purpose**  
        Provide a **reference point – a fast, transparent model that can be
        always beat (or at least explain).

        **Key traits**  
        - Bag-of-words TF-IDF vectors *(1-2 grams, 30 k features)*  
        - ℓ2-regularised **Logistic Regression** (`max_iter=200`)  
        - Accuracy ≈ **56 %** on the held-out test set  

        **Why keep it?**  
        - Instant inference for large batches  
        - Easy to interpret feature weights  
        """,
        unsafe_allow_html=False,
    )

with col2:
    st.subheader(" Bi-LSTM  ·  Word Embeddings")
    st.markdown(
        """
        **Purpose**  
        Capture **word order** and contextual patterns that the baseline
        ignores, while still running on commodity hardware.

        **Key traits**  
        - 128-dim **trainable embeddings** initialised with Xavier  
        - **Bidirectional LSTM** with global-max pooling  
        - Hyper-parameters tuned via **Optuna** *(hidden_dim, dropout, lr, batch_size)*  

        **Why it matters**  
        - Learns phrase-level cues such as *“go back to …”*  
        - Demonstrates the value of **representation learning** over sparse vectors  
        """,
        unsafe_allow_html=False,
    )

with col3:
    st.subheader("BERT  ·  Transformer Heavyweight")
    st.markdown(
        """
        **Purpose**  
        Serve as the **state-of-the-art** contender and production candidate,
        leveraging deep contextual representations.

        **Key traits**  
        - Fine-tuned **`bert-base-uncased`** (110 M parameters)  
        - Self-attention captures **long-range dependencies** & nuanced semantics  
        - Trained for **3 epochs** with early stopping on validation accuracy  
        - Accuracy ≈ **85 %**, outperforming Bi-LSTM by **~6 pp**  

        **Why it wins**  
        - Handles sarcasm, slurs, and spelling variations better  
        - Transfer-learning = strong performance even with limited labelled data  
        """,
        unsafe_allow_html=False,
    )

st.divider()

model_opt = st.selectbox("Choose a modle for getting metrics.", ["baseline", "lstm", "bert"])
if not os.path.exists(f"artifacts/{model_opt}_report.json"):
    st.error("The artifact was not found. Train the model first.")
    st.stop()

rep, cm = load_artifact(model_opt)
st.subheader("Classification report")


rep_df = pd.DataFrame(rep).T  
st.dataframe(rep_df.style.format(precision=3))

st.subheader("Confusion matrix")
plot_confusion(cm, title=f"{model_opt.upper()} CM")
st.pyplot(plt.gcf())

st.subheader("Error analysis")
fp_idx = np.where( (rep['1']['precision'] < 1.0) )[0]  
st.warning("Analiza falsos positivos/negativos y patrones: sarcasmo, ofensivas sutiles, etc.")
