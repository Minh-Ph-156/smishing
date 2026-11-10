import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

try:
    from pyvi import ViTokenizer
    PYVI_AVAILABLE = True
except ImportError:
    PYVI_AVAILABLE = False

from metrics import compute_comprehensive_metrics
from utils import resolve_columns, get_target_names

ADDITIONAL_SPECIAL_TOKENS = ['[URL]', '[MONEY]', '[TIME]', '[DATE]', '[NUMBER]', '[NAME]', '[POINT]', '[TB]', '[QC]']


def segment_for_phobert(text: str) -> str:
    if not PYVI_AVAILABLE or not isinstance(text, str):
        return text

    token_map = {}
    for idx, tok in enumerate(ADDITIONAL_SPECIAL_TOKENS):
        if tok in text:
            placeholder = f"SPECIALTOKEN{idx}VAL"
            text = text.replace(tok, placeholder)
            token_map[placeholder] = tok

    segmented = ViTokenizer.tokenize(text)
    for placeholder, tok in token_map.items():
        segmented = segmented.replace(placeholder, tok)

    return segmented


class TransformerDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        text_list = [str(t) for t in texts]
        self.encodings = tokenizer(text_list, truncation=True, padding=True, max_length=max_len, return_tensors='pt')
        self.labels = torch.tensor([int(lbl) for lbl in labels], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


def train_eval_transformers(train_df, test_df, text_col=None, label_col=None, epochs=3, batch_size=16, lr=2e-5, device='cuda' if torch.cuda.is_available() else 'cpu'):
    text_col, label_col = resolve_columns(train_df, text_col, label_col)
    y_train = train_df[label_col].values.astype(int)
    y_test = test_df[label_col].values.astype(int)

    all_labels = np.concatenate([y_train, y_test])
    num_labels = len(set(all_labels))
    target_names = get_target_names(all_labels)

    classes = np.unique(y_train)
    cw = compute_class_weight('balanced', classes=classes, y=y_train)
    weights_arr = np.ones(num_labels, dtype=np.float32)
    for c, w in zip(classes, cw):
        weights_arr[c] = w
    class_weights_tensor = torch.tensor(weights_arr, dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    transformer_candidates = {
        "PhoBERT-v2": {
            "checkpoint": "vinai/phobert-base-v2",
            "epochs": 3,
            "lr": 2e-5,
            "batch_size": 16,
            "is_phobert": True
        },
        "BERT": {
            "checkpoint": "bert-base-multilingual-cased",
            "epochs": 3,
            "lr": 2e-5,
            "batch_size": 16,
            "is_phobert": False
        },
        "DistilBERT": {
            "checkpoint": "distilbert-base-multilingual-cased",
            "epochs": 5,
            "lr": 1.5e-5,
            "batch_size": 16,
            "is_phobert": False
        },
        "DeBERTa-v3": {
            "checkpoint": "microsoft/mdeberta-v3-base",
            "epochs": 3,
            "lr": 1.5e-5,
            "batch_size": 16,
            "use_fast": False,
            "is_phobert": False
        }
    }

    tf_results = {}
    for name, config in transformer_candidates.items():
        model_checkpoint = config["checkpoint"]
        model_epochs = config.get("epochs", epochs)
        model_lr = config.get("lr", lr)
        model_batch_size = config.get("batch_size", batch_size)
        is_phobert = config.get("is_phobert", False)
        use_fast = config.get("use_fast", True)

        print(f"\n[*] Đang khởi tạo mô hình {name} ({model_checkpoint}) | Epochs: {model_epochs} | Batch: {model_batch_size}", flush=True)
        try:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=use_fast)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
            model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels).to(device)

            special_tokens_dict = {'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS}
            num_added = tokenizer.add_special_tokens(special_tokens_dict)
            if num_added > 0:
                try:
                    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
                except TypeError:
                    model.resize_token_embeddings(len(tokenizer))
        except Exception as e:
            print(f"[!] Không thể tải model {model_checkpoint}: {e}", flush=True)
            continue

        if is_phobert and PYVI_AVAILABLE:
            train_texts = [segment_for_phobert(t) for t in train_df[text_col]]
            test_texts = [segment_for_phobert(t) for t in test_df[text_col]]
        else:
            train_texts, test_texts = train_df[text_col], test_df[text_col]

        train_ds = TransformerDataset(train_texts, y_train, tokenizer)
        test_ds = TransformerDataset(test_texts, y_test, tokenizer)

        train_loader = DataLoader(train_ds, batch_size=model_batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=model_batch_size * 2, shuffle=False)

        optimizer = torch.optim.AdamW(model.parameters(), lr=model_lr, weight_decay=0.01)
        total_steps = len(train_loader) * model_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

        model.train()
        for epoch in range(model_epochs):
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
                loss = criterion(outputs.logits, batch['labels'].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"  [{name}] Epoch {epoch + 1}/{model_epochs} | Training Loss: {avg_loss:.4f}", flush=True)

        model.eval()
        preds, targets, all_probs = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device))
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                preds.extend(np.argmax(probs, axis=1))
                targets.extend(batch['labels'].numpy())
                all_probs.append(probs)

        all_probs = np.vstack(all_probs)
        metrics = compute_comprehensive_metrics(targets, preds, y_prob=all_probs, num_classes=num_labels)
        metrics["Model"] = model
        metrics["Tokenizer"] = tokenizer
        tf_results[name] = metrics

        print(f"[{name}] Acc: {metrics['Acc']} | P: {metrics['P']} | R: {metrics['R']} | F1: {metrics['F1']} | FPR: {metrics['FPR']} | FNR: {metrics['FNR']} | AUC: {metrics['AUC']}", flush=True)

    return tf_results
