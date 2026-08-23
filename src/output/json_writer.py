import json
from pathlib import Path
from datetime import datetime
from typing import Optional


class JSONWriter:
    """結果をJSON形式で出力"""

    def __init__(self, output_dir: str = "output/json"):
        self.output_dir = output_dir

    def write(self, data: dict, filename: Optional[str] = None) -> str:
        """
        JSONファイルに出力

        Args:
            data: {"ユーザ名": ファン数, ...}
            filename: ファイル名（未指定の場合は日時ベース）

        Returns:
            出力ファイルパス
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result_{timestamp}.json"

        filepath = Path(self.output_dir) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return str(filepath)
