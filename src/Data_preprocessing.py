import re
import sys
import unicodedata
import pandas as pd

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

TLDS = (
    r'com\.vn|edu\.vn|gov\.vn|vn|com|top|vip|xyz|me|life|cc|info|club|'
    r'online|site|net|co|biz|bid|de|tech|link|io|app|to|ws|cloud'
)

URL_PATTERN = re.compile(
    rf'(?:https?:\/\/|www\.)[^\s]+|'
    rf'(?<![a-zA-Z0-9])[a-zA-Z0-9.-]+\.(?:{TLDS})(?=[^a-zA-Z0-9]|$)(?:\/[^\s]*)?',
    flags=re.IGNORECASE
)

PROTECTED_TOKENS = ['[URL]', '[MONEY]', '[TIME]', '[DATE]', '[NUMBER]', '[NAME]', '[POINT]', '[TB]', '[QC]']


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFC', text)


def mask_urls(text: str) -> str:
    def _replace(match):
        url = match.group(0)
        trailing_punct = ""
        while url and url[-1] in ".,!?:;\"'()[]{}~^*<>$":
            trailing_punct = url[-1] + trailing_punct
            url = url[:-1]
        return f" [URL] {trailing_punct}"

    return URL_PATTERN.sub(_replace, text)


def clean_special_characters(text: str) -> str:
    token_map = {}
    for idx, token in enumerate(PROTECTED_TOKENS):
        placeholder = f"XXPROTECTEDTOKEN{idx}XX"
        if token.lower() in text.lower():
            text = re.sub(re.escape(token), placeholder, text, flags=re.IGNORECASE)
            token_map[placeholder] = token

    text = re.sub(r'[^\w\s]|_', ' ', text)

    for placeholder, original_token in token_map.items():
        text = text.replace(placeholder, f" {original_token} ")
    return text


def remove_extra_spaces(text: str) -> str:
    """Thu gọn các khoảng trắng thừa thành một dấu cách duy nhất."""
    return re.sub(r'\s+', ' ', text).strip()


def extract_numerical_features(raw_text: str) -> dict:
    """Trích xuất 8 đặc trưng số từ văn bản gốc trước khi làm sạch."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {
            'char_length': 0, 'word_count': 0, 'uppercase_ratio': 0.0,
            'digit_ratio': 0.0, 'special_char_ratio': 0.0,
            'has_url': 0, 'url_count': 0, 'has_exclamation': 0
        }

    raw_text = normalize_unicode(raw_text)
    char_len = len(raw_text)
    words = raw_text.split()

    alpha_chars = [c for c in raw_text if c.isalpha()]
    uppercase_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars) if alpha_chars else 0.0

    digit_count = sum(1 for c in raw_text if c.isdigit())
    digit_ratio = digit_count / char_len if char_len > 0 else 0.0

    special_count = sum(1 for c in raw_text if not c.isalnum() and not c.isspace())
    special_char_ratio = special_count / char_len if char_len > 0 else 0.0

    urls_found = URL_PATTERN.findall(raw_text)
    url_count = len(urls_found)

    return {
        'char_length': char_len,
        'word_count': len(words),
        'uppercase_ratio': round(uppercase_ratio, 4),
        'digit_ratio': round(digit_ratio, 4),
        'special_char_ratio': round(special_char_ratio, 4),
        'has_url': 1 if url_count > 0 else 0,
        'url_count': url_count,
        'has_exclamation': 1 if '!' in raw_text else 0
    }


def preprocess_sms(text: str, lowercase: bool = True) -> str:
    """Tiền xử lý chuỗi văn bản cho tokenizer: NFC -> mask URL -> clean chars -> lowercase."""
    if not isinstance(text, str):
        return ""

    text = normalize_unicode(text)
    text = mask_urls(text)
    text = clean_special_characters(text)
    text = remove_extra_spaces(text)

    if lowercase:
        text = text.lower()
        for token in PROTECTED_TOKENS:
            text = text.replace(token.lower(), token)

    return text


def process_dataset(df: pd.DataFrame, text_col: str = 'message') -> pd.DataFrame:
    """Trích xuất song song cột văn bản làm sạch và 8 đặc trưng số (đồng bộ index)."""
    df = df.copy().reset_index(drop=True)

    if text_col not in df.columns:
        for cand in ['message', 'cleaned_message', 'clean_text', 'text']:
            if cand in df.columns:
                text_col = cand
                break

    feature_dicts = df[text_col].apply(extract_numerical_features).tolist()
    feat_df = pd.DataFrame(feature_dicts).reset_index(drop=True)
    df['cleaned_message'] = df[text_col].apply(preprocess_sms)

    return pd.concat([df, feat_df], axis=1)
