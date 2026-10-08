"""PyInstaller エントリ（GUI 側）。

spec の TOC 末尾にこのスクリプトを置くことで、C ブートローダが
``gui.__main__:main`` を ``__main__`` として実行する。

frozen 実行時のブートストラップ:
1. windowed exe（GUI、console=False）では sys.stdout/sys.stderr が None になり、
   tqdm / print / logging がクラッシュするため devnull にリダイレクトする。
2. exe に同梱した meiki モデルを使うよう ``run_frozen_bootstrap()`` を
   OCR エンジン構築より前に呼び出す（HF Hub ダウンロードをスキップ）。
"""
import os
import sys

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

from src.utils.frozen_bootstrap import run_frozen_bootstrap

run_frozen_bootstrap()

from gui.__main__ import main

if __name__ == "__main__":
    main()
