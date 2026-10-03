"""PyInstaller エントリ（GUI 側）。

spec の TOC 末尾にこのスクリプトを置くことで、C ブートローダが
``gui.__main__:main`` を ``__main__`` として実行する。
"""
from gui.__main__ import main

if __name__ == "__main__":
    main()
