"""meiki OCR モデルを指定ディレクトリへダウンロードする（exe 同梱 / オフライン利用用）。

使い方:
    uv run python scripts/download_models.py [出力ディレクトリ]   # 既定: models/

ダウンロード対象は ``meikiocr.ocr`` の定数（リポジトリ / ファイル名）を
**そのまま参照**するため、meikiocr のアップグレードでファイル名が変わっても
自動的に追従する（同梱パスを解決する ``src/utils/frozen_bootstrap.py`` は
ファイル名ベースなので追従 unnecessary）。

設計: ``docs/04-Design/exe-build-pipeline-design.md``
"""

import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from meikiocr import ocr as _meiki_ocr

# meikiocr.ocr の定数から直接取得（手書きリストの同期漏れをなくす）
MODELS = [
    (_meiki_ocr.DET_MODEL_REPO, _meiki_ocr.DET_MODEL_NAME),
    (_meiki_ocr.REC_MODEL_REPO, _meiki_ocr.REC_MODEL_NAME),
    (_meiki_ocr.REC_MODEL_REPO, _meiki_ocr.VREC_MODEL_NAME),
]


def main(out_dir: str = "models") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for repo_id, filename in MODELS:
        p = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(out),
            # local_dir 指定時は local_dir_use_symlinks のデフォルト(false)で平らな配置
        )
        size_mb = Path(p).stat().st_size / 1024 / 1024
        print(f"downloaded: {p} ({size_mb:.1f} MB)")
    print(f"done: {len(MODELS)} files -> {out}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "models")
