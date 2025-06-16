
# Hate‑Speech Detection

End‑to‑end NLP pipeline that detects hateful tweets and serves real‑time
predictions through a **Streamlit** web interface ([Link here](https://hate-speech-classifier.streamlit.app/)). 

<img width="1000" alt="Screenshot 2025-06-16 at 1 06 48" src="https://github.com/user-attachments/assets/1f3b5182-c85a-4323-a069-501d60f493b9" />

| Tech | Role |
|------|------|
| **Python 3.11** | Core language |
| **Streamlit** | Front‑end / dashboards |
| **scikit‑learn** | Baseline model |
| **PyTorch** | Bi‑LSTM |
| **Hugging Face Transformers** | Fine‑tuned BERT |
| **Optuna** | Hyper‑parameter search |
| **wordcloud / seaborn / matplotlib** | Dataset EDA |

---

## Project description

* **Dataset**: [`thefrankhsu/hate_speech_twitter`](https://huggingface.co/datasets/thefrankhsu/hate_speech_twitter) (6 679 tweets, 30 % hate, 9 sub‑categories).
* **Models**  
  1. **TF‑IDF + Logistic Regression** – fast, transparent benchmark.  
  2. **Bi‑LSTM** – contextual sequence model; hyper‑parameters tuned with Optuna.  
  3. **BERT (bert‑base‑uncased)** – fine‑tuned 3 epochs; weights pushed to the HuggingFaceHub <https://huggingface.co/marielalvs/bert‑hate‑speech>.
* **Artefacts**  
  * BERT checkpoint is retrieved directly from the **Hugging Face Hub**.
* **App pages**  
  1. Problem & Dataset  
  2. Live Inference  
  3. Dataset Visualisation  
  4. Hyper‑parameter Tuning  
  5. Evaluation & Error Analysis  
<img width="1000" alt="Screenshot 2025-06-16 at 1 09 05" src="https://github.com/user-attachments/assets/417fc57a-444e-49c0-9d0c-a257332a3eac" />
<img width="1000" alt="Screenshot 2025-06-16 at 1 09 53" src="https://github.com/user-attachments/assets/e3b6c2ad-db36-4e18-afec-a93dd0fdaa2e" />

---

## Repository layout

```
.
├── Navigation.py                    
├── pages/
│   ├── 0_Problem_and_Dataset.py
│   ├── 1_Inference_Interface.py
│   ├── 2_Dataset_Visualization.py
│   ├── 3_Hyperparameter_Tuning.py
│   └── 4_Evaluation_and_Justification.py
├── train_baseline.py           # TF‑IDF + LogReg
├── train_lstm.py               # Bi‑LSTM + Optuna
├── train_bert.py               # BERT fine‑tune & push to HF
├── utils.py                    # Shared helpers
├── requirements.txt
└── artifacts/                  # Generated after training / first run
```
