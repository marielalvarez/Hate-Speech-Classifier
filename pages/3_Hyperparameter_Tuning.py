import streamlit as st
import optuna
import pandas as pd
from pathlib import Path
from optuna.visualization import (
    plot_optimization_history, plot_param_importances
)

DB_PATH = Path("artifacts/bilstm_study.db")
STUDY_NAME = "bilstm_search"

st.title("💻 Hyperparameter Tuning (Optuna)")

if not DB_PATH.exists():
    st.error(
        "Optuna.database does not exist yet.\n"
        "Run `python train_lstm.py` to generate the trials."
    )
    st.stop()

study = optuna.load_study(study_name=STUDY_NAME,
                          storage=f"sqlite:///{DB_PATH}")

if len(study.trials) == 0:
    st.error("The study exists but does not yet contain trials.")
    st.stop()

st.subheader("Best Hiperparameters")
best_df = pd.DataFrame(study.best_params.items(), columns=["Parameter", "Value"])
st.dataframe(best_df, use_container_width=True)

st.subheader("Convergence curve")
st.plotly_chart(plot_optimization_history(study), use_container_width=True)

st.subheader("Importance of each parameter")
st.plotly_chart(plot_param_importances(study), use_container_width=True)

df_trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
st.dataframe(df_trials, use_container_width=True)
