"""PyInstaller エントリ（CLI 側）。

spec の TOC 末尾にこのスクリプトを置くことで、C ブートローダが
``cli.main:main`` を ``__main__`` として実行する。
"""
from cli.main import main

if __name__ == "__main__":
    main()
