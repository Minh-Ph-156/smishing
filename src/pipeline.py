import os
import sys
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from Data_preprocessing import process_dataset
from ML_model import train_eval_ml_models
from DL_model import train_eval_dl_models
from Transformer_model import train_eval_transformers
from visualization import plot_grouped_bar_chart

NUMERICAL_FEATURE_COLS = [
    'char_length', 'word_count', 'uppercase_ratio', 'digit_ratio',
    'special_char_ratio', 'has_url', 'url_count', 'has_exclamation'
]


def _locate_data_file(filename):
    """Tìm file dữ liệu trong các vị trí ứng viên."""
    candidates = [
        os.path.join(current_dir, "..", "data", filename),
        os.path.join("data", filename),
        filename
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


def save_checkpoints(results_by_group, checkpoint_dir="checkpoints"):
    """Lưu checkpoint cho toàn bộ các mô hình đã huấn luyện (ML, DL, Transformers)."""
    base_dir = os.path.dirname(current_dir)
    save_dir = os.path.join(base_dir, checkpoint_dir) if not os.path.isabs(checkpoint_dir) else checkpoint_dir
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print(f"LƯU CHECKPOINT MÔ HÌNH (THƯ MỤC: {save_dir})")
    print("=" * 70)

    for group_name, methods in results_by_group.items():
        clean_grp = group_name.lower().replace(" ", "_")
        group_path = os.path.join(save_dir, clean_grp)
        os.makedirs(group_path, exist_ok=True)

        for model_name, metrics in methods.items():
            model = metrics.get("Model")
            if model is None:
                continue

            clean_name = model_name.lower().replace(" ", "_").replace("-", "_").replace("+", "_")

            try:
                # 1. Transformers (HuggingFace AutoModel & Tokenizer)
                if hasattr(model, "save_pretrained"):
                    model_out_dir = os.path.join(group_path, clean_name)
                    os.makedirs(model_out_dir, exist_ok=True)
                    model.save_pretrained(model_out_dir)
                    tokenizer = metrics.get("Tokenizer")
                    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
                        tokenizer.save_pretrained(model_out_dir)
                    print(f"  [+] Đã lưu checkpoint Transformer: {model_out_dir}")

                # 2. PyTorch Deep Learning (Recurrent models)
                elif isinstance(model, torch.nn.Module):
                    pt_path = os.path.join(group_path, f"{clean_name}.pt")
                    torch.save(model.state_dict(), pt_path)
                    print(f"  [+] Đã lưu checkpoint PyTorch DL: {pt_path}")

                # 3. Scikit-learn & XGBoost Machine Learning models
                else:
                    import joblib
                    joblib_path = os.path.join(group_path, f"{clean_name}.joblib")
                    joblib.dump(model, joblib_path)
                    print(f"  [+] Đã lưu checkpoint ML / XGBoost: {joblib_path}")

                    ft_model = metrics.get("FastTextModel")
                    if ft_model is not None and hasattr(ft_model, "save"):
                        ft_path = os.path.join(group_path, f"{clean_name}_ft.model")
                        ft_model.save(ft_path)
                        print(f"  [+] Đã lưu FastText model: {ft_path}")

            except Exception as e:
                print(f"  [!] Lỗi khi lưu checkpoint cho '{model_name}': {e}")


def export_latex_table(results_by_group, output_path="results/performance_table.tex"):
    """Xuất bảng tổng hợp kết quả theo chuẩn LaTeX booktabs phục vụ bài báo khoa học."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    lines = [
        r"\begin{table}[htbp]",
        r"\caption{Performance evaluation of ML, DL, and Transformer methods on Vietnamese SMS dataset}",
        r"\label{tab:performance_evaluation}",
        r"\centering",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"Method & Acc & P & R & F1 & FPR & FNR & AUC \\",
        r"\midrule"
    ]

    for group_name, methods in results_by_group.items():
        if group_name == "Machine Learning":
            group_header = "TF-IDF / FastText + Numerical Features (ML)"
        elif group_name == "Deep Learning":
            group_header = "Deep Learning (Recurrent Models) + Numerical Features"
        elif group_name == "Transformers":
            group_header = "Transformers (Pre-trained Language Models)"
        else:
            group_header = group_name

        lines.append(rf"\multicolumn{{8}}{{l}}{{\textbf{{{group_header}}}}} \\")
        for m_name, met in methods.items():
            acc = met.get('Acc', 0.0)
            p = met.get('P', 0.0)
            r = met.get('R', 0.0)
            f1 = met.get('F1', 0.0)
            fpr = met.get('FPR', 0.0)
            fnr = met.get('FNR', 0.0)
            auc = met.get('AUC', 0.0)
            lines.append(f"{m_name:<28} & {acc:.4f} & {p:.4f} & {r:.4f} & {f1:.4f} & {fpr:.4f} & {fnr:.4f} & {auc:.4f} \\\\")
        lines.append(r"\midrule")

    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")

    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    tex_content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(tex_content)
    print(f"[+] Đã xuất bảng LaTeX booktabs: {output_path}")
    return output_path


def run_full_pipeline(
    data_path=None,
    test_size=0.2,
    run_ml=True,
    run_dl=True,
    run_transformer=True,
    random_state=42,
    save_checkpoint=True,
    checkpoint_dir="checkpoints",
    chart_output="results/model_comparison_bar_chart.png",
    latex_output="results/performance_table.tex"
):
    print("\n" + "=" * 70)
    print("KHỞI CHẠY PIPELINE PHÁT HIỆN TIN NHẮN RÁC / LỪA ĐẢO (3 LỚP)")
    print("=" * 70)

    train_balanced_path = _locate_data_file("train_balanced.csv")
    test_frozen_path = _locate_data_file("test_frozen.csv")

    if train_balanced_path and test_frozen_path:
        print(f"[*] Đọc dữ liệu đóng băng:")
        print(f"    - Train: {train_balanced_path}")
        print(f"    - Test:  {test_frozen_path}")
        raw_train_df = pd.read_csv(train_balanced_path).reset_index(drop=True)
        raw_test_df = pd.read_csv(test_frozen_path).reset_index(drop=True)
        label_col = 'label_3class' if 'label_3class' in raw_train_df.columns else 'label'
    else:
        if data_path is None:
            data_path = _locate_data_file("vietnamese_sms_dataset.csv")
            if data_path is None:
                raise FileNotFoundError("Không tìm thấy file dữ liệu trong thư mục data/!")

        print(f"[*] Phân tầng Train/Test từ dữ liệu gốc: {data_path}")
        df = pd.read_csv(data_path)
        label_col = 'label_3class' if 'label_3class' in df.columns else 'label'

        raw_train_df, raw_test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df[label_col]
        )
        raw_train_df = raw_train_df.reset_index(drop=True)
        raw_test_df = raw_test_df.reset_index(drop=True)

    print("[*] Tiền xử lý dữ liệu và trích xuất đặc trưng số...")
    train_df = process_dataset(raw_train_df, text_col='message')
    test_df = process_dataset(raw_test_df, text_col='message')
    print(f"[*] Train = {len(train_df)} mẫu | Test (đóng băng) = {len(test_df)} mẫu")

    train_num_feats = train_df[NUMERICAL_FEATURE_COLS].values
    test_num_feats = test_df[NUMERICAL_FEATURE_COLS].values

    results_by_group = {}

    # 1. Machine Learning (SVM, Decision Tree, KNN, FastText + XGBoost)
    if run_ml:
        try:
            results_by_group["Machine Learning"] = train_eval_ml_models(
                train_df, test_df,
                train_num_feats=train_num_feats,
                test_num_feats=test_num_feats,
                text_col='cleaned_message',
                label_col=label_col
            )
        except Exception as e:
            print(f"[!] Lỗi khi chạy nhóm ML: {e}")

    # 2. Deep Learning (Bi-RNN, Bi-LSTM, Bi-GRU)
    if run_dl:
        try:
            results_by_group["Deep Learning"] = train_eval_dl_models(
                train_df, test_df,
                train_num_feats=train_num_feats,
                test_num_feats=test_num_feats,
                text_col='cleaned_message',
                label_col=label_col,
                epochs=5
            )
        except Exception as e:
            print(f"[!] Lỗi khi chạy nhóm DL: {e}")

    # 3. Transformers (PhoBERT-v2, mBERT, DistilBERT, DeBERTa-v3)
    if run_transformer:
        try:
            results_by_group["Transformers"] = train_eval_transformers(
                train_df, test_df,
                text_col='cleaned_message',
                label_col=label_col
            )
        except Exception as e:
            print(f"[!] Lỗi khi chạy nhóm Transformers: {e}")

    # 4. Hiển thị bảng tổng hợp kết quả Benchmark
    flat_records = []
    for grp_name, methods in results_by_group.items():
        for m_name, met in methods.items():
            flat_records.append({
                "Group": grp_name,
                "Method": m_name,
                "Acc": met.get('Acc', 0.0),
                "P": met.get('P', 0.0),
                "R": met.get('R', 0.0),
                "F1": met.get('F1', 0.0),
                "FPR": met.get('FPR', 0.0),
                "FNR": met.get('FNR', 0.0),
                "AUC": met.get('AUC', 0.0)
            })

    if flat_records:
        summary_df = pd.DataFrame(flat_records)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.colheader_justify', 'left')
        print("\n" + "=" * 90)
        print("BẢNG TỔNG HỢP HIỆU SUẤT ĐÁNH GIÁ CÁC MÔ HÌNH (UNIFIED BENCHMARK)")
        print("=" * 90)
        print(summary_df.to_string(index=False))

        # 5. Vẽ biểu đồ cột nhóm tập trung vào Macro-F1, Accuracy, ROC-AUC
        try:
            plot_grouped_bar_chart(summary_df, output_path=chart_output)
        except Exception as e:
            print(f"[!] Lỗi khi vẽ biểu đồ cột nhóm: {e}")

        # 6. Xuất bảng mã LaTeX booktabs
        try:
            export_latex_table(results_by_group, output_path=latex_output)
        except Exception as e:
            print(f"[!] Lỗi khi xuất bảng LaTeX: {e}")

    # 7. Lưu Checkpoint mô hình
    if save_checkpoint and results_by_group:
        save_checkpoints(results_by_group, checkpoint_dir=checkpoint_dir)

    return results_by_group


if __name__ == "__main__":
    run_full_pipeline()
