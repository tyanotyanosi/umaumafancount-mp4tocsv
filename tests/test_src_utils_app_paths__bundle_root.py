"""Tests for ``src.utils.app_paths.bundle_root``.

Specification: docs/00-Architecture/src_utils_app_paths__bundle_root.yaml

``bundle_root`` returns the base directory of the read-only bundled
resources. During development it is the same as project_root; when
frozen, it is the exe's extraction directory (sys._MEIPASS):

- call is_frozen() to determine whether we are running as a frozen exe
- if frozen: get the ``sys._MEIPASS`` attribute and wrap it in Path to
  return; if the attribute does not exist, the return value of
  project_root() is used as the default and wrapped in Path to return
- if not frozen: call project_root() and return its return value

Mocked / stand-in dependencies (per the test-generation rules):
``src.utils.app_paths.is_frozen`` and
``src.utils.app_paths.project_root`` are patched with
``unittest.mock``; ``sys._MEIPASS`` is patched with
``mock.patch.object(..., create=True)`` for the frozen case.

``errors`` section of the spec: empty (no error conditions documented).
"""

import contextlib
import shutil
import sys
import tempfile
from pathlib import Path
from unittest import mock

from src.utils.app_paths import bundle_root


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
    input: frozen exe 実行中、sys._MEIPASS = 'C:/app/_MEI12345'（展開ディレクトリ）
    expected: Path('C:/app/_MEI12345') を返す。
    """
    with mock.patch("src.utils.app_paths.is_frozen", return_value=True), \
         mock.patch.object(sys, "_MEIPASS", "C:/app/_MEI12345", create=True):
        assert bundle_root() == Path("C:/app/_MEI12345")


def test_edge_02():
    """
    input: frozen exe 実行中、sys._MEIPASS 属性が存在しない
    expected: project_root() と同一の値（exe 同置ディレクトリ）を返す。
    """
    with _tmp_dir() as pr:
        assert not hasattr(sys, "_MEIPASS")
        with mock.patch("src.utils.app_paths.is_frozen", return_value=True), \
             mock.patch("src.utils.app_paths.project_root", return_value=pr):
            assert bundle_root() == pr


def test_edge_03():
    """
    input: ソース実行
    expected: project_root() と同一の値（プロジェクトルート）を返す。
    """
    with _tmp_dir() as pr:
        with mock.patch("src.utils.app_paths.is_frozen", return_value=False), \
             mock.patch("src.utils.app_paths.project_root", return_value=pr):
            assert bundle_root() == pr
