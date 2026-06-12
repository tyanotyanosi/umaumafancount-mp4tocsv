import json
from pathlib import Path
from typing import List, Dict
from src.domain.models import MemberEntry
from src.domain.logic import get_fan_count

class FanCountExtractor:
    """メンバーごとのファン数をOCRテキストから抽出"""

    def __init__(self, member_list_path: Path, replace_json_path: Path):
        self.member_list: List[str] = []
        self.member_replace: Dict[str, List[str]] = {}
        self._load_data(member_list_path, replace_json_path)

    def _load_data(self, list_path: Path, replace_path: Path):
        if list_path.exists():
            with open(list_path, 'r', encoding='utf-8') as f:
                self.member_list = [line.strip() for line in f if line.strip()]

        if replace_path.exists():
            try:
                with open(replace_path, 'r', encoding='utf-8') as f:
                    self.member_replace = json.load(f)
            except Exception:
                self.member_replace = {}
        else:
            self.member_replace = {}

    def extract(self, texts: List[str]) -> Dict[str, int]:
        """OCRテキストからメンバーごとのファン数を抽出"""
        # テキスト後処理 + メンバー置換
        processed_texts = []
        for text in texts:
            current_text = text
            for member in self.member_list:
                if member in self.member_replace:
                    for repname in self.member_replace[member]:
                        current_text = current_text.replace(repname, member)
                if member in current_text:
                    current_text = current_text.replace(member, f"\n{member} ")
            processed_texts.append(current_text)

        # テキストの結合（正規表現に渡すため）
        full_text = "\n".join(processed_texts)
        # 行ごとに分割（元のロジックに忠実にするため、一度結合して再度分割）
        # ただし、元の get_fan_count は texts(list) を受け取るので、
        # splitした結果を渡す必要がある。
        # 元のロジック: texts = "\n".join(texts).split("\n")
        split_texts = full_text.split("\n")

        # ファン数集計
        fan_counts = {}
        fans = []
        for member in self.member_list:
            fan_count = get_fan_count(split_texts, member, fans)
            if fan_count is not None:
                fan_counts[member] = fan_count
                fans.append(fan_count)
            else:
                fan_counts[member] = 0

        return fan_counts
