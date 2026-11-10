def resolve_columns(df, text_col=None, label_col=None):
    """Tự động phát hiện tên cột văn bản và nhãn để tương thích pipeline."""
    if text_col is None:
        for candidate in ['cleaned_message', 'clean_text', 'message', 'text']:
            if candidate in df.columns:
                text_col = candidate
                break
        if text_col is None:
            raise KeyError(f"Không tìm thấy cột văn bản trong DataFrame. Các cột hiện có: {list(df.columns)}")

    if label_col is None:
        for candidate in ['label_3class', 'label', 'target']:
            if candidate in df.columns:
                label_col = candidate
                break
        if label_col is None:
            raise KeyError(f"Không tìm thấy cột nhãn trong DataFrame. Các cột hiện có: {list(df.columns)}")

    return text_col, label_col


def get_target_names(y_values):
    unique_labels = sorted(list(set(y_values)))
    if len(unique_labels) == 2:
        return ['Ham (0)', 'Spam/Smishing (1)']
    elif len(unique_labels) == 3:
        return ['Ham (0)', 'Spam (1)', 'Smishing (2)']
    return [f"Class {lbl}" for lbl in unique_labels]
