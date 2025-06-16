# – Model evaluation, reports & error analysis

import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from utils import load_data

sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Model Evaluation", page_icon="📈")
st.title("Evaluation & Model Justification")

model_name = st.selectbox(
    "Choose a model for metrics",
    ["baseline", "lstm", "bert"]
)

train_df, test_df = load_data()
y_true = test_df["label"].to_list()

with open(f"artifacts/{model_name}_report.json") as f:
    report = json.load(f)
cm = np.load(f"artifacts/{model_name}_cm.npy")

st.header("①  Model justification")
if model_name == "baseline":
    st.markdown(
        """
        **TF-IDF + Logistic Regression**  
        - **Why?** Fast, interpretable benchmark.  
        - 1–2-gram TF-IDF (30 k features) captures common slurs.  
        - Sets a transparent performance floor.
        """
    )
elif model_name == "lstm":
    st.markdown(
        """
        **Bi-LSTM w/ trainable embeddings**  
        - **Why?** Captures word order & compositional context.  
        - Optimized hidden size, dropout & learning rate via Optuna.  
        - Addresses sequential patterns and long-range cues.
        """
    )
else:
    st.markdown(
        """
        **BERT (bert-base-uncased)**  
        - **Why?** Deep self-attention handles polysemy, sarcasm, and
          long-distance dependencies.  
        - Fine-tuned for 3 epochs with early-stopping on validation accuracy.  
        - Achieves state-of-the-art performance on social media text.
        """
    )

st.markdown("---")

st.header("②  Classification report")
df_report = pd.DataFrame(report).T.round(3)
st.dataframe(df_report, use_container_width=True)

st.markdown("---")

st.header("③  Confusion matrix")
fig, ax = plt.subplots(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greys",
            xticklabels=["Non-Hate","Hate"],
            yticklabels=["Non-Hate","Hate"],
            ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"{model_name.upper()} confusion matrix")
st.pyplot(fig)

st.markdown("---")

st.header("④  Error analysis & examples")

# gather predictions
preds = []
if model_name == "baseline":
    # baseline predictions need a reload, so derive from report keys
    # assume save_report saved preds in JSON; otherwise reload model
    preds = pd.read_json(f"artifacts/baseline_preds.json", typ='series').tolist()
else:
    preds = pd.read_json(f"artifacts/{model_name}_preds.json", typ='series').tolist()

df_test = test_df.copy()
df_test["pred"] = preds
fp = df_test[(df_test.label == 0) & (df_test.pred == 1)].sample(3, random_state=42)
fn = df_test[(df_test.label == 1) & (df_test.pred == 0)].sample(3, random_state=42)

st.subheader("False Positives (predicted HATE but actually NON-HATE)")
for _, row in fp.iterrows():
    st.markdown(f"> {row.tweet}")

st.subheader("False Negatives (predicted NON-HATE but actually HATE)")
for _, row in fn.iterrows():
    st.markdown(f"> {row.tweet}")

st.divider()

st.header("Global Insights")

st.markdown(
     """
     Key observations
     
    - Slur ≠ hate as mdels still over-penalise reclaimed or quoted slurs.
 
    - Models are over-sensitive to profanity yet still rely on explicit slurs for hate, missing nuanced or coded language.

    - Obfuscated hate (numeronyms, rare neologisms) requires character- or byte-level coverage beyond wordpieces.

    - Universal negative statements (“all X…”, “X stunt kids”) are an enduring blind spot—addressable with linguistic rules + targeted data.

    Next actions

    - Curate micro-aggression and implicit-hate examples for fine-tuning.

    - Plug in a character-CNN or byte-BERT front-end to decode numeronyms and unseen slurs.

    - Periodic error-driven re-sampling: every week, feed 100 misclassified tweets back into fine-tuning to close the most painful gaps.

    - Curated data augmentation to catch quantifier-based generalisations and pronoun misgendering.
    """
)
