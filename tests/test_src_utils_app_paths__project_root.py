"""Tests for ``src.utils.app_paths.project_root``.

Specification: docs/00-Architecture/src_utils_app_paths__project_root.yaml

``project_root`` returns the base directory of the resources the user
works with (settings / output). During development it is the project
root; when frozen, it is the exe's directory:

- call is_frozen() to determine whether we are running as a frozen exe
- if frozen: resolve ``sys.executable`` to an absolute path and return
  its parent directory (the exe's co-located directory)
- if not frozen: resolve this module's ``__file__`` to an absolute
  path and return its ``parents[2]`` (two levels above the directory
  containing this file)

Mocked / stand-in dependencies (per the test-generation rules):
``src.utils.app_paths.is_frozen`` is patched with ``unittest.mock``;
``sys.executable`` is patched with ``mock.patch.object`` for the
frozen case. The source-execution case uses the real module location
(``<root>/src/utils/app_paths.py``).

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "ソース実行で、このファイルが3階層以上の親ディレクトリを持つ位置にない場合（例: ファイルシステムルート直下配置）で、Path(__file__).resolve().parents が3要素未満になる"
  behavior: "parents[2] のインデックスアクセスで IndexError が発生する"
"""

import sys
from pathlib import Path
from unittest import mock

import src.utils.app_paths as app_paths
from src.utils.app_paths import project_root


def test_edge_01():
    """
    input: frozen exe 実行中（sys.executable = 'C:/app/app.exe'）
    expected: Path('C:/app') を返す（exe 同置ディレクトリの絶対パス）。
    """
    with mock.patch("src.utils.app_paths.is_frozen", return_value=True), \
         mock.patch.object(sys, "executable", "C:/app/app.exe"):
        result = project_root()
        assert result == Path("C:/app")
        assert result.is_absolute()


def test_edge_02():
    """
    input: ソース実行、モジュールが <root>/src/utils/app_paths.py に配置されている場合
    expected: Path('<root>')（プロジェクトルート、parents[2]）を返す。
    """
    with mock.patch("src.utils.app_paths.is_frozen", return_value=False):
        result = project_root()
        assert isinstance(result, Path)
        assert result.is_absolute()
        # The returned root must be the project root that actually
        # contains this module at src/utils/app_paths.py.
        assert (result / "src" / "utils" / "app_paths.py").is_file()
        assert result == Path(app_paths.__file__).resolve().parents[2]
