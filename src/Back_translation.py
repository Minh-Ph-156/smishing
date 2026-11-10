import os
import sys
import time
import random
import re
import unicodedata
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from deep_translator import GoogleTranslator

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

TLDS = (
    r'com\.vn|edu\.vn|gov\.vn|vn|com|top|vip|xyz|me|life|cc|info|club|'
    r'online|site|net|co|biz|bid|de|tech|link|io|app|to|ws|cloud'
)

PROTECTED_PATTERN = re.compile(
    rf'(?:https?:\/\/|www\.)\S+|'
    rf'(?<![a-zA-Z0-9])[a-zA-Z0-9.-]+\.(?:{TLDS})(?=[^a-zA-Z0-9]|$)(?:\/\S*)?|'
    rf'\[[A-Z0-9_]+\]',
    flags=re.IGNORECASE
)


class TextMasker:
    """Khóa và mở khóa URL, Domain và Token nhạy cảm trước khi dịch ngược."""
    @staticmethod
    def mask(text: str):
        placeholders = {}
        def _replacer(match):
            token = match.group(0).strip()
            trailing = ""
            while token and token[-1] in ".,!?:;\"'()[]{}~^*<>$":
                trailing = token[-1] + trailing
                token = token[:-1]
            key = f"XYZ{len(placeholders)}XYZ"
            placeholders[key] = token
            return f" {key} {trailing}"

        masked_text = PROTECTED_PATTERN.sub(_replacer, text)
        return masked_text, placeholders

    @staticmethod
    def unmask(text: str, placeholders: dict):
        for key, original_val in placeholders.items():
            pattern = re.compile(rf"{key[:3]}\s*{key[3:-3]}\s*{key[-3:]}", re.IGNORECASE)
            text = pattern.sub(original_val, text)
        return text


class AccentHandler:
    """Phát hiện và bảo toàn phong cách tin nhắn tiếng Việt không dấu."""
    ACCENTED_CHARS = re.compile(r'[àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', re.IGNORECASE)

    @classmethod
    def is_unsigned(cls, text: str, threshold: float = 0.05) -> bool:
        letters = re.findall(r'[a-zA-Zàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]', text)
        if not letters:
            return False
        return (len(cls.ACCENTED_CHARS.findall(text)) / len(letters)) < threshold

    @staticmethod
    def strip_accents(text: str) -> str:
        nfkd = unicodedata.normalize('NFKD', text)
        no_accent = ''.join([c for c in nfkd if not unicodedata.combining(c)])
        return no_accent.replace('đ', 'd').replace('Đ', 'D')


def robust_back_translate(text: str, intermediate_languages: list = None, max_retries: int = 3) -> str:
    """Dịch ngược qua ngôn ngữ trung gian với Exponential Backoff."""
    if not isinstance(text, str) or not text.strip():
        return None

    if intermediate_languages is None:
        intermediate_languages = ['en', 'fr']

    was_unsigned = AccentHandler.is_unsigned(text)
    masked_text, placeholders = TextMasker.mask(text)
    inter_lang = random.choice(intermediate_languages)

    for attempt in range(max_retries):
        try:
            forward = GoogleTranslator(source='auto', target=inter_lang).translate(masked_text)
            if not forward:
                continue
            backward = GoogleTranslator(source=inter_lang, target='vi').translate(forward)
            if not backward:
                continue

            restored = TextMasker.unmask(backward, placeholders)
            if was_unsigned:
                restored = AccentHandler.strip_accents(restored)

            restored = re.sub(r'\s+', ' ', restored).strip()
            if restored and restored.lower() != text.lower():
                return restored
            return None
        except Exception:
            time.sleep((2 ** attempt) + random.uniform(0.5, 1.5))

    return None


def process_and_balance_sms_data(csv_path: str, test_size: float = 0.2, random_state: int = 42, max_oversample_factor: int = 10):
    print("\n" + "=" * 60)
    print(f"[*] Đang tải dữ liệu từ: {csv_path}")
    df = pd.read_csv(csv_path)
    label_col = 'label_3class' if 'label_3class' in df.columns else 'label'

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df[label_col])
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"[*] Train: {len(train_df)} mẫu | Test (đóng băng): {len(test_df)} mẫu")

    augmented_records = []
    if label_col == 'label_3class':
        n0 = (train_df[label_col] == 0).sum()
        n1 = (train_df[label_col] == 1).sum()
        n2 = (train_df[label_col] == 2).sum()

        needed_spam = min(max(0, n0 - n1), n1 * max_oversample_factor if max_oversample_factor else n0 - n1)
        needed_smishing = max(0, n0 - n2)

        print(f"[*] Mục tiêu sinh mẫu: Spam(1): {needed_spam} | Smishing(2): {needed_smishing}")

        # Sinh mẫu Spam
        if needed_spam > 0:
            pool = train_df[train_df[label_col] == 1]
            pbar = tqdm(total=needed_spam, desc="Sinh mẫu Spam (1)")
            count, fails = 0, 0
            while count < needed_spam and fails < 25:
                sample = pool.sample(n=1).iloc[0]
                aug_text = robust_back_translate(sample['message'])
                if aug_text:
                    count += 1
                    fails = 0
                    augmented_records.append({
                        'message_id': f"AUG_SPAM_{count:04d}",
                        'date': sample.get('date', None),
                        'message': aug_text,
                        'label': 1,
                        'label_3class': 1
                    })
                    pbar.update(1)
                else:
                    fails += 1
            pbar.close()

        # Sinh mẫu Smishing
        if needed_smishing > 0:
            pool = train_df[train_df[label_col] == 2]
            pbar = tqdm(total=needed_smishing, desc="Sinh mẫu Smishing (2)")
            count, fails = 0, 0
            while count < needed_smishing and fails < 25:
                sample = pool.sample(n=1).iloc[0]
                aug_text = robust_back_translate(sample['message'])
                if aug_text:
                    count += 1
                    fails = 0
                    augmented_records.append({
                        'message_id': f"AUG_SMISHING_{count:04d}",
                        'date': sample.get('date', None),
                        'message': aug_text,
                        'label': 2,
                        'label_3class': 2
                    })
                    pbar.update(1)
                else:
                    fails += 1
            pbar.close()

    if augmented_records:
        balanced_train_df = pd.concat([train_df, pd.DataFrame(augmented_records)], ignore_index=True)
    else:
        balanced_train_df = train_df.copy()

    balanced_train_df = balanced_train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "..", "data") if os.path.exists(os.path.join(base_dir, "..", "data")) else "."
    train_path = os.path.join(out_dir, "train_balanced.csv")
    test_path = os.path.join(out_dir, "test_frozen.csv")

    balanced_train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
    print(f"[*] Đã lưu: '{train_path}' ({len(balanced_train_df)} mẫu) và '{test_path}' ({len(test_df)} mẫu)")

    return balanced_train_df, test_df
