import os
import sys
import numpy as np
import scipy.sparse as sp
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

from metrics import compute_comprehensive_metrics
from utils import resolve_columns, get_target_names


def train_eval_ml_models(train_df, test_df, train_num_feats=None, test_num_feats=None, text_col=None, label_col=None):
    text_col, label_col = resolve_columns(train_df, text_col, label_col)
    y_train = train_df[label_col].values.astype(int)
    y_test = test_df[label_col].values.astype(int)

    all_labels = np.concatenate([y_train, y_test])
    num_classes = len(set(all_labels))
    target_names = get_target_names(all_labels)

    # 1. TF-IDF Representation cho các mô hình truyền thống
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_text = tfidf.fit_transform(train_df[text_col].astype(str))
    X_test_text = tfidf.transform(test_df[text_col].astype(str))

    if train_num_feats is not None and test_num_feats is not None:
        scaler = StandardScaler()
        X_train_num = sp.csr_matrix(scaler.fit_transform(train_num_feats))
        X_test_num = sp.csr_matrix(scaler.transform(test_num_feats))
        X_train = sp.hstack([X_train_text, X_train_num])
        X_test = sp.hstack([X_test_text, X_test_num])
    else:
        X_train, X_test = X_train_text, X_test_text

    ml_models = {
        "SVM": LinearSVC(C=1.0, class_weight='balanced', random_state=42, max_iter=3000),
        "Decision Tree": DecisionTreeClassifier(max_depth=15, min_samples_split=5, class_weight='balanced', random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, weights='distance')
    }

    results = {}
    for name, model in ml_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_prob = None
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X_test)
            if decision.ndim == 1:
                prob_1 = 1.0 / (1.0 + np.exp(-decision))
                y_prob = np.vstack([1.0 - prob_1, prob_1]).T
            else:
                y_prob = softmax(decision, axis=1)

        metrics = compute_comprehensive_metrics(y_test, y_pred, y_prob=y_prob, num_classes=num_classes)
        metrics["Model"] = model
        results[name] = metrics

        print(f"[{name}] Acc: {metrics['Acc']} | P: {metrics['P']} | R: {metrics['R']} | F1: {metrics['F1']} | FPR: {metrics['FPR']} | FNR: {metrics['FNR']} | AUC: {metrics['AUC']}")

    # 2. FastText + XGBoost
    try:
        from gensim.models import FastText
        import xgboost as xgb
        from sklearn.utils.class_weight import compute_sample_weight
        try:
            from pyvi import ViTokenizer
            pyvi_avail = True
        except ImportError:
            pyvi_avail = False

        print("[*] Đang huấn luyện FastText embeddings & đặc trưng...")
        def _get_tokens(text):
            if pyvi_avail:
                try:
                    return ViTokenizer.tokenize(str(text)).split()
                except Exception:
                    pass
            return str(text).split()

        train_tokenized = [_get_tokens(t) for t in train_df[text_col]]
        test_tokenized = [_get_tokens(t) for t in test_df[text_col]]

        vector_size = 100
        ft_model = FastText(
            sentences=train_tokenized,
            vector_size=vector_size,
            window=5,
            min_count=1,
            workers=4,
            epochs=15,
            seed=42
        )

        def _doc_vector(tokens):
            vecs = [ft_model.wv[w] for w in tokens if w in ft_model.wv]
            if vecs:
                return np.mean(vecs, axis=0)
            return np.zeros(vector_size, dtype=np.float32)

        X_train_ft = np.array([_doc_vector(toks) for toks in train_tokenized])
        X_test_ft = np.array([_doc_vector(toks) for toks in test_tokenized])

        if train_num_feats is not None and test_num_feats is not None:
            scaler = StandardScaler()
            X_train_num_dense = scaler.fit_transform(train_num_feats)
            X_test_num_dense = scaler.transform(test_num_feats)
            X_train_xgb = np.hstack([X_train_ft, X_train_num_dense])
            X_test_xgb = np.hstack([X_test_ft, X_test_num_dense])
        else:
            X_train_xgb = X_train_ft
            X_test_xgb = X_test_ft

        print("[*] Đang huấn luyện mô hình FastText + XGBoost...")
        sample_weights = compute_sample_weight('balanced', y_train)
        xgb_model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss',
            tree_method='hist'
        )
        xgb_model.fit(X_train_xgb, y_train, sample_weight=sample_weights)

        y_pred_xgb = xgb_model.predict(X_test_xgb)
        y_prob_xgb = xgb_model.predict_proba(X_test_xgb)

        xgb_metrics = compute_comprehensive_metrics(y_test, y_pred_xgb, y_prob=y_prob_xgb, num_classes=num_classes)
        xgb_metrics["Model"] = xgb_model
        xgb_metrics["FastTextModel"] = ft_model
        results["FastText + XGBoost"] = xgb_metrics

        print(f"[FastText + XGBoost] Acc: {xgb_metrics['Acc']} | P: {xgb_metrics['P']} | R: {xgb_metrics['R']} | F1: {xgb_metrics['F1']} | FPR: {xgb_metrics['FPR']} | FNR: {xgb_metrics['FNR']} | AUC: {xgb_metrics['AUC']}")
    except Exception as e:
        print(f"[!] Lỗi khi chạy FastText + XGBoost: {e}")

    return results
