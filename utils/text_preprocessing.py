# tiền xử lý text (tokenize, normalize)
# Chuẩn hóa text tiếng Việt, tokenization, compound nouns
#version multiprocessing
import re
import unicodedata
from vncorenlp import VnCoreNLP

# ==========================
# 1. Chuẩn hóa tiếng Việt
# ==========================
def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(
        r"[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễ"
        r"ìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữ"
        r"ỳýỵỷỹđ]",
        "",
        text,
    )
    return text

# ==========================
# 2. Khởi tạo segmenter (lazy load)
# ==========================
_rdrsegmenter = None

def get_segmenter():
    global _rdrsegmenter
    if _rdrsegmenter is None:
        _rdrsegmenter = VnCoreNLP(
            r"D:\VQA\utils\VnCoreNLP-master\VnCoreNLP-1.1.1.jar",
            annotators="wseg",   # chỉ tách từ
            max_heap_size='-Xmx2g'
        )
    return _rdrsegmenter

def close_segmenter():
    global _rdrsegmenter
    if _rdrsegmenter is not None:
        _rdrsegmenter.close()
        _rdrsegmenter = None

# ==========================
# 3. Tokenization
# ==========================
def tokenize_text(text: str) -> str:
    rdrsegmenter = get_segmenter()
    sentences = rdrsegmenter.tokenize(text)
    tokens = " ".join([" ".join(sent) for sent in sentences])
    return tokens

# ==========================
# 4. Full preprocessing
# ==========================
def preprocess_text(text: str) -> str:
    text = normalize_text(text)
    text = tokenize_text(text)
    return text

# ==========================
# 5. Test nhanh
# ==========================
if __name__ == "__main__":
    s = "Ủy ban Nhân dân Quận Ngũ Hành Sơn tổ chức kỳ thi học sinh giỏi."
    print(preprocess_text(s))
    close_segmenter()
