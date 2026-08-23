import csv
from pathlib import Path
from datetime import datetime
from typing import Optional


class CSVWriter:
    """結果をCSV形式で出力"""

    def __init__(self, output_dir: str = "output/csv"):
        self.output_dir = output_dir

    def write(self, data: dict, filename: Optional[str] = None) -> str:
        """
        CSVファイルに出力

        Args:
            data: {"ユーザ名": ファン数, ...}
            filename: ファイル名（未指定の場合は日時ベース）

        Returns:
            出力ファイルパス
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"result_{timestamp}.csv"

        filepath = Path(self.output_dir) / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['ユーザ名', 'ファン数'])
            for user_name, fan_count in data.items():
                writer.writerow([user_name, fan_count])

        return str(filepath)
