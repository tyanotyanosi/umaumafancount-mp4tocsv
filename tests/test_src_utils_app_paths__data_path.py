"""Tests for ``src.utils.app_paths.data_path``.

Specification: docs/00-Architecture/src_utils_app_paths__data_path.yaml

``data_path`` resolves a relative resource path (config / template /
name_mapping, etc.). An absolute path is returned as-is; a relative
path is first searched under project_root, then under bundle_root:

- convert the argument to a Path (p = Path(relative))
- if p.is_absolute() is true, return p as-is
- otherwise, iterate over bases in the order (project_root(),
  bundle_root()) and return base / p when (base / p).exists() is true
  (the first one found)
- if the scan finds nothing, return project_root() / p (this path may
  not exist; the code emits no warning or exception, and the existence
  check is the caller's responsibility)

Mocked / stand-in dependencies (per the test-generation rules):
``src.utils.app_paths.project_root`` and
``src.utils.app_paths.bundle_root`` are patched with ``unittest.mock``
to point at temporary directories (created inside the workspace via
the ``_tmp_dir`` helper and removed in ``finally``, because the DSH
temp area is not reliably writable); the resource files themselves
are real files created in those directories.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "relative が Path() による変換ができない値（例: int）の場合"
  behavior: "Path(relative) 変換で TypeError が発生する"
"""

import contextlib
import shutil
import tempfile
from pathlib import Path
from unittest import mock

from src.utils.app_paths import data_path


@contextlib.contextmanager
def _tmp_dir():
    """Create a temporary directory inside the workspace and remove it
    on exit."""
    tmp = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent.parent))
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@contextlib.contextmanager
def _bases():
    """Create two temporary base directories (project root and bundle
    root) inside one workspace temp dir and remove it on exit."""
    tmp = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent.parent))
    try:
        pr = tmp / "pr"
        bundle = tmp / "bundle"
        pr.mkdir()
        bundle.mkdir()
        yield pr, bundle
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_01(tmp_path):
    """
    input: 絶対パス（tmp_path 由来のプラットフォーム中立な絶対パス）
    expected: 渡された絶対パスを存在探索せずにそのまま返す。

    注意: ドライブレター付きパス（'C:/abs/...'）は Windows では絶対パスだが
    POSIX では単なる相対パスとして扱われるため、リテラルで書くと Linux CI で
    失敗する。tmp_path から絶対パスを組み立ててプラットフォーム中立にする。
    """
    absolute = tmp_path / "abs" / "config.yaml"
    assert absolute.is_absolute()  # 前提: 入力は絶対パスであること
    assert data_path(str(absolute)) == absolute


def test_edge_02():
    """
    input: 相対パス 'config/settings.yaml'、かつ <project_root>/config/settings.yaml が存在する
    expected: project_root() / 'config/settings.yaml' を返す。
    """
    with _bases() as (pr, bundle):
        (pr / "config").mkdir()
        (pr / "config" / "settings.yaml").write_text("x")
        with mock.patch("src.utils.app_paths.project_root", return_value=pr), \
             mock.patch("src.utils.app_paths.bundle_root", return_value=bundle):
            assert data_path("config/settings.yaml") == pr / "config" / "settings.yaml"


def test_edge_03():
    """
    input: frozen 実行、相対パス 'template/x.csv' が project_root()（exe 同置ディレクトリ）以下に存在せず、bundle_root()（exe 内同梱コピー）以下に存在する
    expected: bundle_root() / 'template/x.csv' を返す。
    """
    with _bases() as (pr, bundle):
        (bundle / "template").mkdir()
        (bundle / "template" / "x.csv").write_text("x")
        with mock.patch("src.utils.app_paths.project_root", return_value=pr), \
             mock.patch("src.utils.app_paths.bundle_root", return_value=bundle):
            assert data_path("template/x.csv") == bundle / "template" / "x.csv"


def test_edge_04():
    """
    input: 相対パス 'no/such/file.yaml' が両方の base 以下に存在しない
    expected: project_root() / 'no/such/file.yaml' を返す（存在しない可能性があるが、例外は発生しない）。
    """
    with _bases() as (pr, bundle):
        with mock.patch("src.utils.app_paths.project_root", return_value=pr), \
             mock.patch("src.utils.app_paths.bundle_root", return_value=bundle):
            result = data_path("no/such/file.yaml")
            assert result == pr / "no" / "such" / "file.yaml"
            assert not result.exists()


def test_edge_05():
    """
    input: ソース実行（project_root() と bundle_root() が同一ディレクトリ）
    expected: 同一ディレクトリが同じ順序で2回探索される。存在すればそのパスを、存在しなければ project_root() / relative を返す。
    """
    with _tmp_dir() as pr:
        (pr / "res").mkdir()
        (pr / "res" / "a.yaml").write_text("x")
        with mock.patch("src.utils.app_paths.project_root", return_value=pr), \
             mock.patch("src.utils.app_paths.bundle_root", return_value=pr):
            assert data_path("res/a.yaml") == pr / "res" / "a.yaml"
            assert data_path("res/missing.yaml") == pr / "res" / "missing.yaml"
