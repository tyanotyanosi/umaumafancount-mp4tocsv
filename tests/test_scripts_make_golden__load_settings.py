"""Tests for ``scripts.make_golden.load_settings``.

Specification: docs/00-Architecture/scripts_make_golden__load_settings.yaml

``load_settings`` reads the project settings file
``config/settings.yaml`` and returns the object parsed by
``yaml.safe_load``:

- open ``ROOT / "config" / "settings.yaml"`` in mode "r" with
  encoding="utf-8"
- parse the whole file with ``yaml.safe_load(f)`` and return its
  result

Read-only: no writing, no global-state change.

Mocked / stand-in dependencies (per the test-generation rules):
``ROOT`` is patched with ``mock.patch`` to a fresh temporary directory
(created per test, removed in finally) so the real
``config/settings.yaml`` is never read; the settings file is created
inside the temporary ``config`` directory per edge case.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: config/settings.yaml が存在しない
  behavior: FileNotFoundError が発生する。
- condition: ファイルに読み取り権限がない
  behavior: PermissionError が発生する。
- condition: 内容が UTF-8 でデコードできない
  behavior: UnicodeDecodeError が発生する。
- condition: 内容が有効な YAML ではない
  behavior: yaml.YAMLError（またはそのサブクラス）が発生する。
"""

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import pytest
import yaml

from scripts.make_golden import load_settings


@contextmanager
def _patched_root(tmp):
    """Patch the module-level ROOT to a temporary directory."""
    with mock.patch("scripts.make_golden.ROOT", tmp):
        yield


def _tmp_dir():
    """Create a temporary directory inside the workspace root."""
    return Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent.parent))


def _settings_file(tmp):
    """Create the temporary config directory and return the settings
    file path."""
    (tmp / "config").mkdir()
    return tmp / "config" / "settings.yaml"


def test_edge_01():
    """
    input: 'ROOT/config/settings.yaml が存在しない'
    expected: 'FileNotFoundError が発生し、関数内で捕まらず伝播する。'
    """
    tmp = _tmp_dir()
    try:
        _settings_file(tmp)
        with _patched_root(tmp):
            with pytest.raises(FileNotFoundError):
                load_settings()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_02():
    """
    input: 'ファイルの中身が空（0バイト）'
    expected: 'None が返る（yaml.safe_load は空ドキュメントで None を返す）。'
    """
    tmp = _tmp_dir()
    try:
        _settings_file(tmp).write_bytes(b"")
        with _patched_root(tmp):
            assert load_settings() is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_03():
    """
    input: 'ファイルが有効な YAML マッピング（key: value の行）'
    expected: 'その構造を持つ dict が返る。'
    """
    tmp = _tmp_dir()
    try:
        _settings_file(tmp).write_text("a: 1\nb: 2\n", encoding="utf-8")
        with _patched_root(tmp):
            assert load_settings() == {"a": 1, "b": 2}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_04():
    """
    input: 'ファイルが有効な YAML だがルートがリスト'
    expected: 'list が返る（リターン型注釈が dict であってもコードは検証しない）。'
    """
    tmp = _tmp_dir()
    try:
        _settings_file(tmp).write_text("- 1\n- 2\n", encoding="utf-8")
        with _patched_root(tmp):
            assert load_settings() == [1, 2]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_05():
    """
    input: 'ファイルに無効な UTF-8 バイト列を含む'
    expected: '読み取り中に UnicodeDecodeError が発生し伝播する。'
    """
    tmp = _tmp_dir()
    try:
        _settings_file(tmp).write_bytes(b"\xff\xfe\x00\x00")
        with _patched_root(tmp):
            with pytest.raises(UnicodeDecodeError):
                load_settings()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_06():
    """
    input: 'ファイルに YAML 構文エラー（例: インデントにタブ使用）を含む'
    expected: 'yaml.YAMLError（またはそのサブクラス）が発生し伝播する。'
    """
    tmp = _tmp_dir()
    try:
        _settings_file(tmp).write_text("a:\n\tb: 1\n", encoding="utf-8")
        with _patched_root(tmp):
            with pytest.raises(yaml.YAMLError):
                load_settings()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
