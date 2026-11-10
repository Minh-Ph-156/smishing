import os
import sys
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from metrics import compute_comprehensive_metrics
from utils import resolve_columns, get_target_names


class SimpleVocab:
    def __init__(self, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.w2i = {"<PAD>": 0, "<UNK>": 1}

    def fit(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(str(t).split())
        for w, _ in counter.most_common(self.max_vocab_size - 2):
            self.w2i[w] = len(self.w2i)

    def transform(self, text, max_len=128):
        tokens = str(text).split()
        ids = [self.w2i.get(w, 1) for w in tokens][:max_len]
        if len(ids) < max_len:
            ids += [0] * (max_len - len(ids))
        return ids


class SMSDeepLearningDataset(Dataset):
    def __init__(self, texts, labels, vocab, num_feats=None, max_len=128):
        self.inputs = [vocab.transform(t, max_len) for t in texts]
        self.labels = [int(lbl) for lbl in labels]
        self.num_feats = num_feats

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_text = torch.tensor(self.inputs[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.num_feats is not None:
            x_num = torch.tensor(self.num_feats[idx], dtype=torch.float32)
            return x_text, x_num, y
        return x_text, y


class RecurrentSMSClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_classes=3, cell_type='LSTM', num_numerical_features=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.cell_type = cell_type.upper()
        self.num_numerical_features = num_numerical_features

        rnn_cls = {'RNN': nn.RNN, 'LSTM': nn.LSTM, 'GRU': nn.GRU}.get(self.cell_type)
        if not rnn_cls:
            raise ValueError(f"Không hỗ trợ cell_type: {cell_type}. Chỉ chấp nhận 'RNN', 'LSTM', 'GRU'.")

        self.rnn = rnn_cls(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        in_fc_dim = hidden_dim * 2 + num_numerical_features
        self.fc = nn.Sequential(
            nn.Linear(in_fc_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, num_feats=None):
        embedded = self.embedding(x)
        if self.cell_type == 'LSTM':
            _, (hidden, _) = self.rnn(embedded)
        else:
            _, hidden = self.rnn(embedded)

        feat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        if num_feats is not None and self.num_numerical_features > 0:
            feat = torch.cat((feat, num_feats), dim=1)

        return self.fc(feat)


def train_eval_dl_models(train_df, test_df, train_num_feats=None, test_num_feats=None, text_col=None, label_col=None, epochs=5, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
    text_col, label_col = resolve_columns(train_df, text_col, label_col)
    y_train = train_df[label_col].values.astype(int)
    y_test = test_df[label_col].values.astype(int)

    all_labels = np.concatenate([y_train, y_test])
    num_classes = len(set(all_labels))
    target_names = get_target_names(all_labels)

    if train_num_feats is not None and test_num_feats is not None:
        scaler = StandardScaler()
        train_num_scaled = scaler.fit_transform(train_num_feats)
        test_num_scaled = scaler.transform(test_num_feats)
        num_numerical_features = train_num_scaled.shape[1]
    else:
        train_num_scaled = test_num_scaled = None
        num_numerical_features = 0

    vocab = SimpleVocab()
    vocab.fit(train_df[text_col])

    train_ds = SMSDeepLearningDataset(train_df[text_col], y_train, vocab, num_feats=train_num_scaled)
    test_ds = SMSDeepLearningDataset(test_df[text_col], y_test, vocab, num_feats=test_num_scaled)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    weights_arr = np.ones(num_classes, dtype=np.float32)
    for c, w in zip(classes, cw):
        weights_arr[c] = w
    class_weights_tensor = torch.tensor(weights_arr, dtype=torch.float32).to(device)

    dl_results = {}
    for cell in ['RNN', 'LSTM', 'GRU']:
        model_name = f"Bi-{cell}"
        model = RecurrentSMSClassifier(
            vocab_size=len(vocab.w2i),
            num_classes=num_classes,
            cell_type=cell,
            num_numerical_features=num_numerical_features
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                if num_numerical_features > 0:
                    bx, bnum, by = batch
                    logits = model(bx.to(device), bnum.to(device))
                    loss = criterion(logits, by.to(device))
                else:
                    bx, by = batch
                    logits = model(bx.to(device))
                    loss = criterion(logits, by.to(device))

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        model.eval()
        preds, targets, all_probs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                if num_numerical_features > 0:
                    bx, bnum, by = batch
                    logits = model(bx.to(device), bnum.to(device))
                else:
                    bx, by = batch
                    logits = model(bx.to(device))

                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds.extend(np.argmax(probs, axis=1))
                targets.extend(by.numpy())
                all_probs.append(probs)

        all_probs = np.vstack(all_probs)
        metrics = compute_comprehensive_metrics(targets, preds, y_prob=all_probs, num_classes=num_classes)
        metrics["Model"] = model
        dl_results[model_name] = metrics

        print(f"[{model_name}] Acc: {metrics['Acc']} | P: {metrics['P']} | R: {metrics['R']} | F1: {metrics['F1']} | FPR: {metrics['FPR']} | FNR: {metrics['FNR']} | AUC: {metrics['AUC']}")

    return dl_results
