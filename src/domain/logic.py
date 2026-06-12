import re

def get_fan_count(texts: list[str], member: str, fans: list[int]) -> int | None:
    """
    OCRテキストのリストから、特定のメンバーのファン数を抽出する。
    既存の挙動を完全に維持するために、既知のファン数(fans)によるフィルタリングを含む。
    """
    token_boundary = r"[0-9A-Za-z\u3040-\u30FF\u3400-\u9FFF]"

    pattern = re.compile(
        rf"(?<!{token_boundary})(?<![\d,])"
        r"\d{1,3}(?:,\d{3})+"
        rf"(?!{token_boundary})(?![\d,])"
    )

    fancounts = {}
    for text in texts:
        if member in text:
            match = pattern.search(text)
            if match:
                value = match.group()
                numeric_value = int(value.replace(',', ''))
                if numeric_value in fans:
                    continue
                if numeric_value in fancounts.keys():
                    fancounts[numeric_value] = fancounts[numeric_value] + 1
                else:
                    fancounts[numeric_value] = 1

    if len(fancounts) != 0:
        sorted_dict = sorted(fancounts.items(), key=lambda x: x[1], reverse=True)
        return sorted_dict[0][0]
    return None

def post_process_ocr_text(text: str) -> str:
    """OCR 後処理"""
    text = text.replace(" ", "")
    text = text.replace("①", "")
    text = text.replace("↓", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("（", "")
    text = text.replace("）", "")
    text = text.replace("@", "")
    text = text.replace("、", ",")
    text = text.replace("，", ",")
    text = text.replace("30/30", "")
    text = text.replace("人", " 人")
    text = text.replace("ファン数", "ファン数 ")
    return text
