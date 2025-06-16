from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from datasets import Dataset
from utils import load_data, save_report
import numpy as np, os
import json

# tunes BERT and generales reports in json format for Streamlit.


train_df, test_df = load_data()
tok = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")

def tokenize(batch): return tok(batch["tweet"], truncation=True)
train_ds = Dataset.from_pandas(train_df).map(tokenize, batched=True,
                                             remove_columns=["tweet"])
test_ds  = Dataset.from_pandas(test_df).map(tokenize, batched=True,
                                            remove_columns=["tweet"])

collator = DataCollatorWithPadding(tok)
model = AutoModelForSequenceClassification.from_pretrained(
            "google-bert/bert-base-uncased", num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred          
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()     
    return {"accuracy": acc}
args = TrainingArguments(
    output_dir="artifacts/bert_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,     
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

trainer = Trainer(model, args,
                  train_dataset=train_ds,
                  eval_dataset=test_ds,
                  tokenizer=tok,
                  data_collator=collator,
                  compute_metrics=compute_metrics)

trainer.train()
print("✅ BERT fine-tuned")

pred_logits = trainer.predict(test_ds).predictions
y_pred = np.argmax(pred_logits, axis=1)
save_report(test_df["label"], y_pred, "bert")

with open("artifacts/bert_preds.json", "w") as f:
    json.dump(y_pred.tolist(), f)

print("✅ BERT preds saved to artifacts/bert_preds.json")
print("Reportes en artifacts/")

trainer.save_model("artifacts/bert_model")      
tok.save_pretrained("artifacts/bert_model")
print("✅ Modelo + tokenizer exportados a artifacts/bert_model/")
