import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter
from src.domain.models import MemberEntry
from src.domain.logic import get_fan_count


class FanCountExtractor:
    """メンバーごとのファン数をOCRテキストまたはVLM JSONテキストから抽出"""

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

    def _is_vlm_json_input(self, texts: List[str]) -> bool:
        """入力がVLM JSON形式かどうかを判定"""
        for text in texts:
            stripped = text.strip()
            if not stripped:
                continue
            if stripped.startswith('{'):
                try:
                    json.loads(stripped)
                    return True
                except json.JSONDecodeError:
                    continue
        return False

    def _resolve_member_name(self, key: str) -> Optional[str]:
        """VLM JSONのキーを正しいメンバー名に解決

        memberReplace.json: {"正しい名前": ["誤認識1", "誤認識2"]} の形式
        keyが誤認識なら正しい名前を返す。既に正しい名前ならそのまま返す。
        """
        # 既にmember_listに含まれる名前ならそのまま使用
        if key in self.member_list:
            return key

        # 誤認識パターンから正しい名前を探索
        for correct_name, error_patterns in self.member_replace.items():
            for pattern in error_patterns:
                if pattern in key:
                    return correct_name

        return None

    def _extract_from_vlm_json(self, texts: List[str]) -> Dict[str, int]:
        """VLM JSON形式の入力からファン数を抽出（最頻値選択付き）"""
        # 各メンバーごとの値集計
        member_values: Dict[str, List[int]] = {m: [] for m in self.member_list}

        for text in texts:
            stripped = text.strip()
            if not stripped or not stripped.startswith('{'):
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            if not isinstance(data, dict):
                continue

            for key, value in data.items():
                resolved = self._resolve_member_name(str(key))
                if resolved is None:
                    continue

                fan_value = self._parse_fan_value(value)
                if fan_value is not None:
                    member_values[resolved].append(fan_value)

        # 最頻値を選択
        fan_counts = {}
        for member in self.member_list:
            values = member_values[member]
            if values:
                counter = Counter(values)
                fan_counts[member] = counter.most_common(1)[0][0]
            else:
                fan_counts[member] = 0

        return fan_counts

    @staticmethod
    def _parse_fan_value(value) -> Optional[int]:
        """VLM JSONの値から整数のファン数を抽出"""
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            cleaned = value.replace(',', '').replace(' ', '')
            try:
                return int(cleaned)
            except ValueError:
                return None
        if isinstance(value, list):
            if len(value) == 1:
                item = value[0]
                if isinstance(item, (int, float)):
                    return int(item)
                if isinstance(item, str):
                    cleaned = item.replace(',', '').replace(' ', '')
                    try:
                        return int(cleaned)
                    except ValueError:
                        return None
            elif len(value) > 1:
                total = 0
                for item in value:
                    if isinstance(item, (int, float)):
                        total += item
                    elif isinstance(item, str):
                        cleaned = item.replace(',', '').replace(' ', '')
                        try:
                            total += int(cleaned)
                        except ValueError:
                            pass
                return int(total)
        return None

    def extract(self, texts: List[str]) -> Dict[str, int]:
        """OCRテキストまたはVLM JSONテキストからメンバーごとのファン数を抽出"""
        if self._is_vlm_json_input(texts):
            return self._extract_from_vlm_json(texts)
        return self._extract_from_ocr_text(texts)

    def _extract_from_ocr_text(self, texts: List[str]) -> Dict[str, int]:
        """OCRテキストからメンバーごとのファン数を抽出（既存ロジック）"""
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

        full_text = "\n".join(processed_texts)
        split_texts = full_text.split("\n")

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
