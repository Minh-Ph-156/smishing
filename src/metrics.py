import sys
import os
import numpy as np
from scipy.special import softmax
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')


def compute_comprehensive_metrics(y_true, y_pred, y_prob=None, num_classes=3):
    """Tính toán 7 chỉ số học thuật: Acc, P, R, F1, FPR, FNR, ROC-AUC."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    acc = accuracy_score(y_true, y_pred)
    p = precision_score(y_true, y_pred, average='macro', zero_division=0)
    r = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fpr_list, fnr_list = [], []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        fpr_list.append(fp / (fp + tn) if (fp + tn) > 0 else 0.0)
        fnr_list.append(fn / (fn + tp) if (fn + tp) > 0 else 0.0)

    fpr = float(np.mean(fpr_list))
    fnr = float(np.mean(fnr_list))

    auc = 0.0
    if y_prob is not None:
        try:
            y_prob = np.asarray(y_prob, dtype=float)
            if num_classes == 2:
                prob_pos = y_prob[:, 1] if (y_prob.ndim == 2 and y_prob.shape[1] == 2) else y_prob.ravel()
                if prob_pos.min() < 0.0 or prob_pos.max() > 1.0:
                    prob_pos = 1.0 / (1.0 + np.exp(-prob_pos))
                auc = roc_auc_score(y_true, prob_pos)
            else:
                if y_prob.ndim == 2:
                    if y_prob.min() < 0.0 or y_prob.max() > 1.0 or not np.allclose(y_prob.sum(axis=1), 1.0, atol=1e-2):
                        y_prob = softmax(y_prob, axis=1)
                auc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
        except Exception:
            auc = acc
    else:
        try:
            one_hot = np.zeros((len(y_pred), num_classes))
            for idx, val in enumerate(y_pred):
                one_hot[idx, val] = 1.0
            auc = roc_auc_score(y_true, one_hot, multi_class='ovr', average='macro')
        except Exception:
            auc = acc

    return {
        "Acc": round(float(acc), 4),
        "P": round(float(p), 4),
        "R": round(float(r), 4),
        "F1": round(float(f1), 4),
        "FPR": round(float(fpr), 4),
        "FNR": round(float(fnr), 4),
        "AUC": round(float(auc), 4)
    }
