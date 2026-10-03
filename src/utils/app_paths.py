"""リソースパス解決（ソース実行 / exe 実行の両方で動作）。

- ソース実行（開発時）: プロジェクトルートをベース。
- EXE 実行（frozen）: exe 同置ディレクトリをベース。
  ユーザが exe の隣に ``config/`` / ``template/`` を置いた場合はそちらが優先され、
  なければ exe 内部（``sys._MEIPASS``）に同梱されたコピーにフォールバックする。

設計: ``docs/04-Design/exe-build-pipeline-design.md``
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """PyInstaller 等で frozen exe として実行中かどうか。"""
    return bool(getattr(sys, "frozen", False))


def project_root() -> Path:
    """ユーザが利用するリソース（設定・出力）のベースディレクトリ。

    - 開発時: プロジェクトルート
    - frozen: exe のディレクトリ（ユーザが自分で config を置ける場所）
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def bundle_root() -> Path:
    """読み取り専用の同梱リソースのベースディレクトリ。

    - 開発時: プロジェクトルート（project_root と同一）
    - frozen: exe の展開ディレクトリ（``sys._MEIPASS``）
    """
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", project_root()))
    return project_root()


def data_path(relative: str | Path) -> Path:
    """相対リソースパス（config / template / name_mapping 等）を解決する。

    - 絶対パス: そのまま返す。
    - 相対パス: ``project_root()`` 以下にあればそれを返し、
      なければ ``bundle_root()``（exe 内同梱）を探す。
      両方になければ ``project_root() / relative`` を返す
      （存在しないことの判定・警告は呼び出し側で行う）。
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    for base in (project_root(), bundle_root()):
        if (base / p).exists():
            return base / p
    return project_root() / p


def work_dir() -> Path:
    """出力（output/、debug/ 等）を書くディレクトリ。frozen では exe 同置。"""
    return project_root()
