import streamlit as st, pandas as pd, optuna
import plotly.express as px, os
st.title("🔧 Hyperparameter Tuning (Optuna)")

if os.path.exists("artifacts/study.db"):
    study = optuna.load_study(study_name="lstm", storage="sqlite:///artifacts/study.db")
    df = study.trials_dataframe()
    st.subheader("Resultados de los 10 trials")
    st.dataframe(df[["number","value","params.hidden","params.dropout","params.lr","params.bs"]])
    fig = px.line(df, x="number", y="value", title="Error vs Trial")
    st.plotly_chart(fig)
    st.write("Mejores parámetros:", study.best_params)
else:
    st.info("Ejecuta `train_lstm.py` para generar el Optuna study.")
