import streamlit as st, json, numpy as np, seaborn as sns, matplotlib.pyplot as plt, os, pandas as pd
from utils import plot_confusion

st.title("📑 Model Analysis & Justification")

def load_artifact(name):
    with open(f"artifacts/{name}_report.json") as f:
        rep = json.load(f)
    cm = np.load(f"artifacts/{name}_cm.npy")
    return rep, cm

model_opt = st.selectbox("Modelo", ["baseline", "lstm", "bert"])
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
st.write("Analiza falsos positivos/negativos y patrones: sarcasmo, ofensivas sutiles, etc.")
