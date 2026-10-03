import re
from collections import Counter
from typing import Optional

from src.parser.name_mapper import NameMapper


class ResultParser:
    """カード単位のOCR結果からユーザ名とファン数をパース"""

    # 数字文脈での OCR 混同文字 → 数字 置換表
    DIGIT_CONFUSION = {
        "O": "0", "Q": "0", "D": "0",
        "l": "1", "I": "1", "|": "1",
        "Z": "2", "z": "2",
        "S": "5", "G": "5",
        "B": "8",
        "A": "4",
        "E": "3",
    }

    def _clean_number(self, num_str: str) -> str:
        num_str = num_str.replace('名', '').replace('人', '')
        cleaned = re.sub(r'[^0-9.,]', '', num_str)
        cleaned = cleaned.replace('.', '').replace(',', '')
        return cleaned

    def _clean_user_name(self, name: str) -> str:
        name = re.sub(r'[（\(].*$', '', name)
        lines = [ln.strip() for ln in name.split('\n') if ln.strip()]
        # 同一行が繰り返し検出された場合（OCR アーティファクト）は1行だけ採用
        if lines and len(set(lines)) == 1:
            lines = [lines[0]]
        name = ''.join(lines)
        name = re.sub(r'\s+', '', name)
        name = name.strip()
        return name

    def _parse_number(self, num_str: str):
        if not num_str:
            return None

        cleaned = self._clean_number(num_str)

        if not cleaned or not cleaned.isdigit():
            return None

        if len(cleaned) < 3:
            return None

        try:
            result = int(cleaned)
            if 0 <= result <= 10000000000:
                return result
        except ValueError:
            pass

        return None

    def _correct_digit_confusion(self, s: str) -> Optional[str]:
        """数字文脈（カンマ+数字の混合文字列）の混同文字を数字へ置換。
        置換後も数字以外（[非 0-9, . 空白]）が残る文字列は棄却（None 返却）"""
        corrected = ''.join(
            self.DIGIT_CONFUSION.get(ch, ch) for ch in s
        )
        if re.search(r'[^0-9,.\s]', corrected):
            return None
        return corrected

    def _parse_fan_text(self, s: str):
        """ファン数テキストを int 化（接尾辞除去 → 混同補正 → クリーン → 数値検証）"""
        if not s:
            return None
        # 人 のOCR誤読 (J) も含め接尾辞を除去
        s = s.replace('人', '').replace('名', '').replace('J', '')
        corrected = self._correct_digit_confusion(s)
        if corrected is None:
            return None
        return self._parse_number(corrected)

    def parse_batch(self, frame_results: list, mapper: Optional[NameMapper] = None) -> dict:
        """
        フレーム単位のカードOCR結果を多数決で統合

        マッピング（``NameMapper``）が渡された場合は、検知名を実ユーザ名へ
        マッピング後に集計する。``mapper`` が ``None`` の場合は現状維持
        （検知のまま集計）で、既存の挙動と完全一致する。

        Args:
            frame_results: [{"cards": [{"name"/"name_raw", "fans"/"fans_raw", ...}]}]
            mapper: マッピング適用用の ``NameMapper``（省略可）。

        Returns:
            {"ユーザ名": ファン数, ...}

        Notes:
            ``mapper`` 指定時は ``mapper.unmapped_names`` に未マッピングの検知
            （``unmapped_action="suggest"``）を、``mapper.warnings`` に近似一致の
            警告（``warn_on_approx=True``）を収集する。
        """
        candidates = {}  # ユーザ名 → [(順番, ファン数)]
        unmapped_action = "suggest"
        if mapper is not None:
            # 未マッピング一覧・警告一覧は呼び出しごとにリセットする。
            mapper.unmapped_names = []
            mapper.warnings = []
            unmapped_action = getattr(mapper, "unmapped_action", "suggest")

        def _add(name: str, count: int):
            candidates.setdefault(name, []).append(count)

        for frame in frame_results:
            cards = frame.get("cards") if isinstance(frame, dict) else None
            if not cards:
                continue
            for card in cards:
                if not isinstance(card, dict):
                    continue
                name_raw = card.get("name", card.get("name_raw"))
                name = self._clean_user_name(name_raw or "")
                if not name:
                    continue
                fans_raw = card.get("fans", card.get("fans_raw"))
                count = self._parse_fan_text(fans_raw or "")
                if count is None:
                    continue
                if mapper is None:
                    _add(name, count)
                    continue
                mapped = mapper.map(name)
                if mapped.warning is not None:
                    # 近似一致の警告を蓄積（同一メッセージは重複を省く）
                    if mapped.warning not in mapper.warnings:
                        mapper.warnings.append(mapped.warning)
                if mapped.matched:
                    _add(mapped.user_name, count)
                elif unmapped_action == "drop":
                    # 集計・一覧双方から消去
                    continue
                elif unmapped_action == "suggest":
                    # 集計には含めず、未マッピング一覧に記録のみ（同名の重複を除く）
                    if mapped.raw_name not in mapper.unmapped_names:
                        mapper.unmapped_names.append(mapped.raw_name)
                else:  # keep
                    # 検知をそのまま実ユーザ名として集計に含める
                    _add(name, count)

        # 多数決（モード）。同票時は桁数の多い方を採用（桁脱落はOCRの典型的エラー）
        result = {}
        for name, counts in candidates.items():
            counter = Counter(counts)
            top = max(counter.values())
            tied = [value for value, freq in counter.most_common() if freq == top]
            result[name] = max(tied, key=lambda v: len(str(v)))

        return result
