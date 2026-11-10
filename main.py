import sys
import os
import argparse

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pipeline import run_full_pipeline, _locate_data_file
from src.Back_translation import process_and_balance_sms_data


def parse_arguments():
    parser = argparse.ArgumentParser(description="Hệ thống phát hiện tin nhắn lừa đảo và rác SMS tiếng Việt (Smishing & Spam)")
    parser.add_argument("--data-path", type=str, default="data/vietnamese_sms_dataset.csv", help="Đường dẫn file dữ liệu gốc")
    parser.add_argument("--test-size", type=float, default=0.2, help="Tỷ lệ tập Test đóng băng (mặc định 0.2)")
    parser.add_argument("--skip-ml", action="store_true", help="Bỏ qua nhóm Machine Learning (SVM, Decision Tree, KNN, FastText+XGBoost)")
    parser.add_argument("--skip-dl", action="store_true", help="Bỏ qua nhóm Deep Learning (Bi-RNN, Bi-LSTM, Bi-GRU)")
    parser.add_argument("--skip-transformer", action="store_true", help="Bỏ qua nhóm Transformer (PhoBERT-v2, mBERT, DistilBERT, DeBERTa-v3)")
    parser.add_argument("--balance", action="store_true", help="Tự động cân bằng dữ liệu nếu chưa có train_balanced.csv")
    parser.add_argument("--max-spam-factor", type=int, default=10, help="Hệ số oversampling tối đa cho lớp Spam (mặc định 10x)")
    parser.add_argument("--chart-output", type=str, default="results/model_comparison_bar_chart.png", help="Đường dẫn xuất file biểu đồ cột nhóm")
    parser.add_argument("--latex-output", type=str, default="results/performance_table.tex", help="Đường dẫn xuất bảng mã LaTeX booktabs")
    parser.add_argument("--no-checkpoint", action="store_true", help="Không lưu checkpoint các mô hình")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    train_balanced_path = _locate_data_file("train_balanced.csv")
    test_frozen_path = _locate_data_file("test_frozen.csv")

    if (not train_balanced_path or not test_frozen_path) and args.balance:
        data_src = _locate_data_file(os.path.basename(args.data_path)) or args.data_path
        process_and_balance_sms_data(data_src, test_size=args.test_size, max_oversample_factor=args.max_spam_factor)

    run_full_pipeline(
        data_path=args.data_path,
        test_size=args.test_size,
        run_ml=not args.skip_ml,
        run_dl=not args.skip_dl,
        run_transformer=not args.skip_transformer,
        save_checkpoint=not args.no_checkpoint,
        chart_output=args.chart_output,
        latex_output=args.latex_output
    )
