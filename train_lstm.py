import os, re, torch, optuna, pickle
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab as build_vocab
from collections import Counter
from sklearn.model_selection import train_test_split
from utils import load_data, save_report
import json

# trains bi-lstm and generales reports in json format for Streamlit.


torch.manual_seed(42)
device   = torch.device("cpu")
emb_dim  = 128         

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, dropout, pad_idx):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        nn.init.xavier_uniform_(self.emb.weight)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1  = nn.Linear(hidden_dim * 2, 64)
        self.drop = nn.Dropout(dropout)
        self.out  = nn.Linear(64, 2)

    def forward(self, x):
        x = self.emb(x)
        o, _ = self.lstm(x)
        x, _ = torch.max(o, dim=1)         
        x = torch.relu(self.fc1(self.drop(x)))
        return self.out(x)

if __name__ == "__main__":
    train_df, test_df = load_data()
    train_txt, val_txt, train_lbl, val_lbl = train_test_split(
        train_df["tweet"], train_df["label"], test_size=0.1, random_state=42)

    PAD, UNK = "<pad>", "<unk>"
    def yield_tokens(text_series):
        for txt in text_series:
            yield re.findall(r"\b\w+\b", txt.lower())

    counter = Counter()
    for tokens in yield_tokens(train_txt):
        counter.update(tokens)

    vocab = build_vocab(counter, specials=[PAD, UNK], min_freq=2)
    vocab.set_default_index(vocab[UNK])
    print(f"Vocab size: {len(vocab)}  |  Emb dim: {emb_dim}")

    class TweetsDS(Dataset):
        def __init__(self, texts, labels, vocab, seq_len=50):
            self.labels = labels.tolist(); self.vocab = vocab; self.seq_len = seq_len
            self.texts  = [self.encode(t) for t in texts]

        def encode(self, txt):
            ids = [self.vocab[t] for t in re.findall(r"\b\w+\b", txt.lower())][:self.seq_len]
            ids += [self.vocab[PAD]] * (self.seq_len - len(ids))
            return torch.tensor(ids, dtype=torch.long)

        def __len__(self): return len(self.labels)
        def __getitem__(self, i): return self.texts[i], torch.tensor(self.labels[i])

    tr_ds  = TweetsDS(train_txt, train_lbl, vocab)
    val_ds = TweetsDS(val_txt,   val_lbl,   vocab)
    test_ds= TweetsDS(test_df["tweet"], test_df["label"], vocab)

    def objective(trial):
        h  = trial.suggest_categorical("hidden_dim", [32, 64, 128])
        dr = trial.suggest_float("dropout", 0.3, 0.6)
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        bs = trial.suggest_categorical("batch_size", [32, 64])

        model = BiLSTMClassifier(len(vocab), emb_dim, h, dr, vocab[PAD]).to(device)
        loss_fn, opt = nn.CrossEntropyLoss(), torch.optim.Adam(model.parameters(), lr=lr)
        tr_loader = DataLoader(tr_ds, bs, shuffle=True)
        val_loader= DataLoader(val_ds, bs)

        for _ in range(3):
            model.train()
            for x, y in tr_loader:
                opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()

        model.eval(); correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item(); total += len(y)
        return 1 - correct/total

    os.makedirs("artifacts", exist_ok=True)
    study = optuna.create_study(direction="minimize",
                                study_name="bilstm_search",
                                storage="sqlite:///artifacts/bilstm_study.db",
                                load_if_exists=True)
    study.optimize(objective, n_trials=10, show_progress_bar=True)
    best = study.best_params; print("🥇 Best params:", best)

    #  Train final 
    best_model = BiLSTMClassifier(len(vocab), emb_dim,
                                  best["hidden_dim"], best["dropout"],
                                  vocab[PAD]).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt     = torch.optim.Adam(best_model.parameters(), lr=best["lr"])
    tr_loader= DataLoader(tr_ds, best["batch_size"], shuffle=True)

    for epoch in range(6):
        best_model.train()
        for x, y in tr_loader:
            opt.zero_grad(); loss_fn(best_model(x), y).backward(); opt.step()
        print(f"Epoch {epoch+1}/6 ✅")

    torch.save(best_model.state_dict(), "artifacts/lstm_best.pt")
    with open("artifacts/lstm_vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    with open("artifacts/bilstm_best_params.pkl", "wb") as f:
        pickle.dump(best, f)
    print("✅ Pesos, vocab y mejores hiperparámetros guardados")

    best_model.eval(); preds=[]
    with torch.no_grad():
        for x,_ in DataLoader(test_ds, 64):
            preds.extend(best_model(x).argmax(1).tolist())
    save_report(test_df["label"], preds, "lstm")
    with open("artifacts/lstm_preds.json", "w") as f:
        json.dump(preds, f)

    print("ℹ️ Reportes listos")
    print("✅ Bi-LSTM preds saved to artifacts/lstm_preds.json")
    print("ℹ️ Reportes listos")

