"""Tests for ``gui.settings_dialog.SettingsDialog._load_settings``.

Specification: docs/00-Architecture/gui_settings_dialog__SettingsDialog__load_settings.yaml

``_load_settings`` reads the settings file pointed to by
``self.settings_file``, parses it as YAML, and returns the parse result
(``{}`` when the file is missing or the result is falsy):

- Step 1: if ``self.settings_file.exists()`` is False, return ``{}``
  immediately
- Step 2: if it exists, open the file with mode 'r', encoding utf-8
- Step 3: parse with ``yaml.safe_load(f)``; if the result is falsy return
  ``{}``, otherwise return the result as-is

The return value is not guaranteed to be a dict (a truthy non-mapping YAML
value is returned as-is).

Mocked / stand-in dependencies (per the test-generation rules): the
``SettingsDialog`` instance is created without running ``__init__``
(``object.__new__``); ``self.settings_file`` is a real ``pathlib.Path`` in
a temporary directory whose contents are written per test. ``yaml.safe_load``
runs real.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "設定ファイルが存在するが不正な YAML（構文エラー）"
  behavior: "yaml.YAMLError のサブクラス（例 yaml.parser.ParserError）が呼び出し側に伝播する（本関数では捕捉されない）"
- condition: "設定ファイルが存在するが開けない（権限不足・デバイスエラー等）"
  behavior: "OSError のサブクラス（例 PermissionError）が呼び出し側に伝播する"
- condition: "ファイルのバイト列が有効な UTF-8 ではない"
  behavior: "UnicodeDecodeError が呼び出し側に伝播する"
"""

import tempfile
from pathlib import Path

from gui.settings_dialog import SettingsDialog


def _make_self(settings_file):
    """Build a ``SettingsDialog`` instance (without running ``__init__``)
    with ``settings_file`` set per test."""
    self = object.__new__(SettingsDialog)
    self.settings_file = settings_file
    return self


def test_edge_01():
    """
    input: self.settings_file が存在しない
    expected: {} を返す；ファイルはオープンされない
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    self = _make_self(settings_file)
    assert self._load_settings() == {}
    assert not settings_file.exists()


def test_edge_02():
    """
    input: ファイルが存在し 0 バイト（空ファイル）
    expected: yaml.safe_load が None を返すため {} を返す
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    settings_file.write_bytes(b"")
    self = _make_self(settings_file)
    assert self._load_settings() == {}


def test_edge_03():
    """
    input: ファイルの内容が 'null'
    expected: 解析結果 None（偽値）のため {} を返す
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    settings_file.write_text("null", encoding="utf-8")
    self = _make_self(settings_file)
    assert self._load_settings() == {}


def test_edge_04():
    """
    input: ファイルが存在し、内容が "video:" と "  frame_interval: 1.5" の 2 行（YAML mapping）
    expected: {"video": {"frame_interval": 1.5}} に等しい dict を返す
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    settings_file.write_text("video:\n  frame_interval: 1.5\n",
                            encoding="utf-8")
    self = _make_self(settings_file)
    assert self._load_settings() == {"video": {"frame_interval": 1.5}}


def test_edge_05():
    """
    input: ファイルの内容が '{}'
    expected: 解析結果が空 dict（偽値）のため {} を返す
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    settings_file.write_text("{}", encoding="utf-8")
    self = _make_self(settings_file)
    assert self._load_settings() == {}


def test_edge_06():
    """
    input: ファイルの内容が '[]'
    expected: 解析結果が空 list（偽値）のため {} を返す
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    settings_file.write_text("[]", encoding="utf-8")
    self = _make_self(settings_file)
    assert self._load_settings() == {}


def test_edge_07():
    """
    input: ファイルの内容が 'hello'（truthy な非 mapping スカラー）
    expected: 'hello' をそのまま返す（dict ではない）
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    settings_file.write_text("hello", encoding="utf-8")
    self = _make_self(settings_file)
    ret = self._load_settings()
    assert ret == "hello"
    assert not isinstance(ret, dict)
