"""Tests for ``src.utils.app_paths.work_dir``.

Specification: docs/00-Architecture/src_utils_app_paths__work_dir.yaml

``work_dir`` returns the directory where outputs (output/, debug/,
etc.) are written. It delegates to ``project_root()`` as-is:

- call ``project_root()`` and return its return value directly

Mocked / stand-in dependencies (per the test-generation rules):
``src.utils.app_paths.project_root`` and
``src.utils.app_paths.is_frozen`` are patched with ``unittest.mock``;
``sys.executable`` is patched with ``mock.patch.object`` for the
frozen case; a temporary directory inside the workspace stands in for
the project root.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "ソース実行で、Path(__file__).resolve().parents が3要素未満（モジュールファイルが3階層以上のディレクトリ階層にない配置）の場合"
  behavior: "project_root() 内部の parents[2] アクセスで IndexError が発生する"
"""

import contextlib
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

from src.utils.app_paths import work_dir


@contextlib.contextmanager
def _tmp_dir():
    """Create a temporary directory inside the workspace and remove it
    on exit."""
    tmp = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent.parent))
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_01():
    """
    input: ソース実行（モジュールが <root>/src/utils/app_paths.py に配置）
    expected: プロジェクトルート <root> の絶対パスを返す。
    """
    with _tmp_dir() as pr:
        with mock.patch("src.utils.app_paths.project_root", return_value=pr) as pr_mock:
            result = work_dir()
            assert result is pr
            pr_mock.assert_called_once()


def test_edge_02():
    """
    input: frozen exe 実行中（sys.executable = 'C:/app/app.exe'）
    expected: Path('C:/app')（exe 同置ディレクトリの絶対パス）を返す。
    """
    with mock.patch("src.utils.app_paths.is_frozen", return_value=True), \
         mock.patch.object(sys, "executable", "C:/app/app.exe"):
        result = work_dir()
        assert result == Path("C:/app")
        assert result.is_absolute()
