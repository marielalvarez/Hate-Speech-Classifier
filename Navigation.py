# This page is an introduction to the problem & solution.

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Hate-Speech Detection",
    page_icon="📢",
    layout="wide",
)

hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer     {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.title("Hate-Speech Detection")
st.markdown(
    """
    The purpose of this app is to experiment with models seen in the 
    Modeling Learning with Artificial Intelligence class (Gpo 302).
    """
)

st.divider()

st.header("Problem Definition & Dataset Rationale")

st.markdown(
    """
    **Task.** Detect hateful content in short, user-generated text
    (tweets). The model must assign one of two labels:

    * `0` – *Non-hate / Neutral*  
    * `1` – *Hate Speech*  

    **Motivation**  
    Hate speech on social media undermines constructive discourse, fuels harassment, and can even incite real-world harm. 
    For this, automated detection helps **content moderators** respond at scale, while **researchers** can quantify trends and design interventions.

    **Expected outcome**  
    An inference that:  
    1. Achieves ≥ **80 %** accuracy on held-out data.  
    2. Provides real-time confidence scores for transparent moderation.  
    3. Outperforms a shallow baseline and a context-aware Bi-LSTM,
       validating the benefit of transfer learning (BERT).
    """
)
st.divider()

st.header("Dataset Selection")

st.markdown("#### Dataset: `thefrankhsu/hate_speech_twitter` (Hugging Face)")

st.markdown(
    """
    | Criterion | Justification |
    |-----------|--------------|
    | **Domain fit** | Real Twitter posts in **English**, matching the target application. |
    | **Size & balance** | 6 679 rows (5 679 train / 1 000 test). 30 % hate speech overall; test split is perfectly balanced (500 vs 500) to avoid skewed evaluation. |
    | **Fine-grained tags** | 9 hate sub-categories (race, gender, disability, etc.) enable future **multi-label** extension. |
    | **Public & documented** | Hosted on HF with a card, license, and class description; original source: Kaggle “Hate Speech & Offensive Language”. |
    | **Annotation quality** | Labels verified by the authors; sub-category tags derived via GPT-3.5 then manually spot-checked. |
    """
)






st.divider()
st.header("Model Choice")

st.markdown(
    """
    **Limitations of past approaches**

    * *Shallow ML* (TF-IDF + LogReg/SVM) offers speed and interpretability
      but ignores **word order** and **context**, leading to false positives
      on generic insults and false negatives on subtle slurs.
    * *CNNs* capture local n-gram features yet struggle with long-range
      dependencies and sarcasm — highlighted by **SemEval-2019 Task 5**
      leaderboard discussions [Zhou et al. (2020)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9139953).
    * Ensemble fusion [Zhou et al. (2020)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9139953) can boost metrics but adds
      serving complexity.

    **Why our three-model line-up?**

    | Model | Role | Rationale |
    |-------|------|-----------|
    | **TF-IDF + LogReg** | *Baseline* | Fast benchmark; sets a transparent performance floor (cf. [Toktarova et al. (2023)](https://www.researchgate.net/profile/Batyrkhan-Omarov/publication/374763141_CREATING_HATE_SPEECH_DETECTION_MODEL_BY_USING_MACHINE_LEARNING_METHODS/links/66c482654b25ef677f71337a/CREATING-HATE-SPEECH-DETECTION-MODEL-BY-USING-MACHINE-LEARNING-METHODS.pdf) which reports 70–80 % accuracy). |
    | **Bi-LSTM** | *Context-aware architecture* | Proven to outperform CNNs in tweets due to sequential modelling; aligns with findings of [Toktarova et al. (2023)](https://www.researchgate.net/profile/Batyrkhan-Omarov/publication/374763141_CREATING_HATE_SPEECH_DETECTION_MODEL_BY_USING_MACHINE_LEARNING_METHODS/links/66c482654b25ef677f71337a/CREATING-HATE-SPEECH-DETECTION-MODEL-BY-USING-MACHINE-LEARNING-METHODS.pdf) where Bi-LSTM was top performer. |
    | **BERT (fine-tuned)** | *SOTA contender* | Self-attention excels at capturing polysemy and subtle hate cues; [Zhou et al. (2020)](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9139953) showed transformers plus fusion achieved the best F1. |

    This progression lets us **quantify the value added by
    increasingly sophisticated representations** while keeping the
    pipeline reproducible and modular.
    """
)

st.success("! Use the sidebar to continue to **Inference Interface** or **Dataset Visualization**.")
st.divider()

st.markdown(
    """
    ### Explore the data in an interactive dashboard
    1. **Inference Interface**  
      Input custom text and get predictions from three models:
      a TF-IDF baseline, a Bi-LSTM, and a fine-tuned BERT model.
      Confidence scores are provided to help interpret predictions.

    2. **Dataset Visualization**  
      Explore the dataset characteristics via interactive charts.
      This includes class distribution, token length histograms,
      word clouds, and examples of ambiguous or noisy samples.

    3. **Hyperparameter Tuning**  
      Review the hyperparameter search process using Optuna.
      Visual summaries of parameter performance and the best configuration
      help illustrate model optimization decisions.

    4. **Model Analysis and Justification**  
      Understand why this problem is challenging (imbalanced classes,
      noisy language, etc.), see evaluation metrics (classification reports,
      confusion matrix), and read the error analysis.


        """
)
st.divider()

st.markdown("### Other References")

st.markdown(
    """
    ① **Fusion of ELMo, BERT & CNN for SemEval-2019 Task 5**  
       *Journal of Information Science*, 2020.

    ② **Comprehensive Comparison of BiLSTM vs. shallow models for Twitter Hate-Speech**  
       *IEEE Access*, 2023.
    """
)

st.divider()

st.markdown('Mariel Álvarez Salas · A01198828 · Repo: https://github.com/marielalvarez/Hate-Speech-Classifier')