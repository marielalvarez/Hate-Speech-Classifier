import streamlit as st, matplotlib.pyplot as plt, seaborn as sns
from utils import load_data, make_wordcloud
train_df, test_df = load_data()
st.title("📊 Dataset Visualizations")

tab1, tab2 = st.tabs(["Distribución de Clases", "Token Length & Wordcloud"])

with tab1:
    counts = train_df["label"].value_counts().sort_index()
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="Pastel1")
    ax.set_xticklabels(["No-Hate", "Hate"])
    ax.set_ylabel("Tweets")
    st.pyplot(fig)

with tab2:
    train_df["n_tokens"] = train_df["tweet"].str.split().str.len()
    fig, ax = plt.subplots()
    sns.histplot(train_df["n_tokens"], bins=30, ax=ax)
    ax.set_title("Token length distribution")
    st.pyplot(fig)

    if st.checkbox("Mostrar WordCloud"):
        text = " ".join(train_df["tweet"].tolist())
        make_wordcloud(text)
        st.pyplot(plt.gcf())
