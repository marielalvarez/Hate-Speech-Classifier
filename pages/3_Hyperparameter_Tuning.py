import streamlit as st
import optuna
import pandas as pd
from pathlib import Path
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)

DB_PATH   = Path("artifacts/bilstm_study.db")
STUDY_NM  = "bilstm_search"
OBJ_NAME  = "Validation Error (1 - Acc.)"

st.set_page_config(page_title="Hyperparameter Tuning", page_icon="🛠️")
st.title("Hyperparameter Tuning (Optuna)")

st.write('The hyperparameter tuning was focused on the Bi-LSTM model to ' \
'explore its capacity to capture sequential and contextual patterns in tweets. ' \
'Unlike the baseline, which has no tunable architecture, and BERT, which benefits ' \
'from pre-trained configurations, the Bi-LSTM required tuning to balance model complexity, regularization, '
'and learning dynamics for this task.')

if not DB_PATH.exists():
    st.error("Optuna DB not found. Run `python train_lstm.py` first.")
    st.stop()

study = optuna.load_study(study_name=STUDY_NM, storage=f"sqlite:///{DB_PATH}")

if len(study.trials) == 0:
    st.error("The study exists but has no trials yet.")
    st.stop()

colA, colB, colC = st.columns(3)
colA.metric("Trials completed", f"{len(study.trials):,}")
colB.metric("Best value", f"{study.best_value:.4f}", help=OBJ_NAME)
colC.metric("Best trial #", study.best_trial.number)

st.markdown("---")

with st.expander("Search space & strategy (click to expand)"):
    st.markdown(
        """
        **Objective**: minimise *Validation Error = 1 − Accuracy*  
        **Sampler**: `TPESampler` (Bayesian) &nbsp;|&nbsp; **Trials**: 10–100  
        **Parameters searched**

        | Parameter | Range / Choices | Rationale |
        |-----------|-----------------|-----------|
        | `hidden_dim` | {32, 64, 128} | Controls model capacity |
        | `dropout` | 0.30 – 0.60 | Combat over-fitting |
        | `lr` | 1e-4 – 5e-3 (log-scale) | Balance convergence speed |
        | `batch_size` | {32, 64} | Memory vs. gradient stability |
        """
    )

st.subheader("Best hyperparameters")
best_df = pd.DataFrame(study.best_params.items(), columns=["Parameter", "Value"])
st.dataframe(best_df, use_container_width=True)

st.subheader("Convergence curve")
st.plotly_chart(
    plot_optimization_history(study, target_name=OBJ_NAME),
    use_container_width=True
)

st.subheader("Importance of each parameter")
st.plotly_chart(
    plot_param_importances(study, target_name=OBJ_NAME),
    use_container_width=True
)

with st.expander("Parameter interactions (parallel coords)"):
    st.plotly_chart(
        plot_parallel_coordinate(
            study, params=["hidden_dim", "dropout", "lr", "batch_size"],
            target_name=OBJ_NAME
        ),
        use_container_width=True
    )

st.subheader("Full trials")
df_trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
st.dataframe(df_trials, use_container_width=True, height=300)

st.markdown(
    """
    ### Take-aways
    * **Learning rate (`lr`) dominates** the search – accounting for ≈ 87 % of the performance variance.
    * **Optimal recipe**: 64 hidden units, ~0.49 dropout, small learning rate 3 e-4, mini-batch 32.  
      This combo yields the lowest validation error (**≈ 0.058** ⇒ ≈ 94 % accuracy).
    * **Batch size** shows minimal influence, meaning our model is not bandwidth-limited.
    """
)

