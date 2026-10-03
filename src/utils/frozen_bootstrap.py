"""EXE（frozen）実行時のブートストラップ。

- meiki ONNX モデルを exe 内に同梱している場合、Hugging Face Hub からの
  ダウンロードをスキップし、同梱パスを返すよう ``meikiocr.ocr._get_model_path``
 を差し替える（**オフラインで初回起動してすぐ使える**ことのための要）。
- ソース実行（開発時）では何もしない（従来どおり HF Hub 利用）。

呼び出しタイミング: GUI/CLI のエントリが OCR エンジンを構築する**前**
（``build/entry_gui.py`` / ``build/entry_cli.py`` の冒頭）。

設計: ``docs/04-Design/exe-build-pipeline-design.md``
"""

from __future__ import annotations

import sys
from pathlib import Path


def run_frozen_bootstrap() -> None:
    if not getattr(sys, "frozen", False):
        return

    model_dir = Path(getattr(sys, "_MEIPASS")) / "models"
    if not model_dir.is_dir():
        # 同梱モデルが無い（旧ビルド等）場合は従来どおり HF Hub に委ねる
        return

    import meikiocr.ocr as _meiki_ocr

    def _bundled_model_path(repo_id: str, filename: str) -> str:
        p = model_dir / filename
        if not p.is_file():
            raise FileNotFoundError(
                f"exe に meiki モデルが同梱されていません: {p}"
                "（ビルド手順の models/ 取得ステップを確認してください）"
            )
        return str(p)

    _meiki_ocr._get_model_path = _bundled_model_path
