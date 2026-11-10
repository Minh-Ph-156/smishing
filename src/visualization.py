import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_grouped_bar_chart(
    summary_df: pd.DataFrame,
    output_path: str = "results/model_comparison_bar_chart.png",
    title: str = "So sánh hiệu suất các mô hình (Macro-F1, Accuracy, ROC-AUC)"
) -> str:
    """
    Vẽ biểu đồ cột nhóm cho các mô hình, chỉ tập trung vào 3 chỉ số cốt lõi:
    - Accuracy (Độ chính xác tổng thể)
    - Macro-F1 (F1-Score trung bình các lớp)
    - ROC-AUC (Khả năng phân tách xác suất One-vs-Rest)
    """
    if summary_df is None or summary_df.empty:
        print("[!] Không có dữ liệu để vẽ biểu đồ.")
        return None

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    models = summary_df["Method"].tolist()
    n_models = len(models)

    acc_scores = summary_df["Acc"].tolist()
    f1_scores = summary_df["F1"].tolist()
    auc_scores = summary_df["AUC"].tolist()

    x = np.arange(n_models)
    bar_width = 0.26

    fig_width = max(13, n_models * 1.35)
    fig, ax = plt.subplots(figsize=(fig_width, 7.5), dpi=300)

    color_acc = "#20639B"   # Xanh navy
    color_f1  = "#ED553B"   # Cam đỏ
    color_auc = "#3CAEA3"   # Xanh ngọc

    rects1 = ax.bar(x - bar_width, acc_scores, bar_width, label="Accuracy", color=color_acc, edgecolor="black", linewidth=0.6, alpha=0.92)
    rects2 = ax.bar(x, f1_scores, bar_width, label="Macro-F1", color=color_f1, edgecolor="black", linewidth=0.6, alpha=0.95)
    rects3 = ax.bar(x + bar_width, auc_scores, bar_width, label="ROC-AUC", color=color_auc, edgecolor="black", linewidth=0.6, alpha=0.92)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8.5, fontweight="bold",
                rotation=0
            )

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    ax.set_ylabel("Điểm số (Score)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=25)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10.5, fontweight="semibold", rotation=25, ha="right")

    min_val = min(min(acc_scores), min(f1_scores), min(auc_scores))
    y_min = max(0.0, np.floor((min_val - 0.05) * 10) / 10.0)
    ax.set_ylim(y_min, 1.05)

    ax.yaxis.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        fontsize=11,
        frameon=True,
        facecolor="#f9f9f9",
        edgecolor="#cccccc"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path
