"""Tests for ``gui.main_window.MainWindow._load_settings``.

Specification: docs/00-Architecture/gui_main_window__MainWindow__load_settings.yaml

``_load_settings`` reads the settings file at ``data_path("config/settings.yaml")``:
if the file does not exist it returns ``{}``; if it exists it imports the ``yaml``
module (dynamic import inside the function) and returns ``yaml.safe_load(f) or {}``
(falsy parse results such as ``None``, an empty dict, or ``0`` become ``{}``).
``self`` is not used in the function body.

Mocked / stand-in dependencies (per the test-generation rules): the
``MainWindow`` instance is created without running ``__init__``
(``object.__new__``); ``data_path`` is patched (wherever it is bound:
``src.utils.app_paths`` and ``gui.main_window``) to return a path inside a
temporary directory, and the settings file content is written per test.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: ファイル内容が不正な YAML である
  behavior: yaml.safe_load により yaml.YAMLError（またはその派生種別）が発生し外へ伝播する（コード側は捕捉しない）
- condition: ファイルが有効な UTF-8 でない
  behavior: 読み取り時の UnicodeDecodeError が外へ伝播する
- condition: ファイルが存在するが権限などで開けない
  behavior: open 時の OSError（PermissionError 等）が外へ伝播する
"""

import contextlib
import tempfile
from pathlib import Path
from unittest import mock

import gui.main_window as gui_main_window
from gui.main_window import MainWindow


def _make_self():
    """Build a ``MainWindow`` instance without running ``__init__``."""
    return object.__new__(MainWindow)


@contextlib.contextmanager
def _patched_data_path(settings_file):
    """Patch ``data_path`` (wherever it is bound) to return ``settings_file``."""
    import src.utils.app_paths as app_paths

    patches = [mock.patch.object(app_paths, "data_path", return_value=settings_file)]
    if hasattr(gui_main_window, "data_path"):
        patches.append(
            mock.patch.object(gui_main_window, "data_path", return_value=settings_file)
        )
    for patch in patches:
        patch.start()
    try:
        yield
    finally:
        for patch in reversed(patches):
            patch.stop()


def _write_settings(tmp_path, content):
    """Create ``tmp_path/config/settings.yaml`` with the given text content."""
    settings_file = tmp_path / "config" / "settings.yaml"
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    settings_file.write_text(content, encoding="utf-8")
    return settings_file


def test_edge_01():
    """
    input: 設定ファイルが存在しない
    expected: 例外なしで {} を返す
    """
    self = _make_self()
    tmp_path = Path(tempfile.mkdtemp())
    settings_file = tmp_path / "config" / "settings.yaml"
    with _patched_data_path(settings_file):
        result = self._load_settings()
    assert result == {}


def test_edge_02():
    """
    input: 設定ファイルが存在し、内容が空ファイルまたは null スカラー
    expected: yaml.safe_load が None を返すため {} を返す
    """
    self = _make_self()
    tmp_path = Path(tempfile.mkdtemp())
    settings_file = _write_settings(tmp_path, "")
    with _patched_data_path(settings_file):
        result = self._load_settings()
    assert result == {}


def test_edge_03():
    """
    input: 設定ファイルが存在し、内容がスカラー 0
    expected: 0 が偽値のため {} を返す
    """
    self = _make_self()
    tmp_path = Path(tempfile.mkdtemp())
    settings_file = _write_settings(tmp_path, "0")
    with _patched_data_path(settings_file):
        result = self._load_settings()
    assert result == {}


def test_edge_04():
    """
    input: 設定ファイルが存在し、内容が YAML マッピング
    expected: 内容を解析した dict を返す
    """
    self = _make_self()
    tmp_path = Path(tempfile.mkdtemp())
    settings_file = _write_settings(tmp_path, "a: 1\nb: 2\n")
    with _patched_data_path(settings_file):
        result = self._load_settings()
    assert result == {"a": 1, "b": 2}


def test_edge_05():
    """
    input: 設定ファイルが存在し、内容が YAML シーケンス
    expected: 解析された list をそのまま返す（dict ではない）
    """
    self = _make_self()
    tmp_path = Path(tempfile.mkdtemp())
    settings_file = _write_settings(tmp_path, "- 1\n- 2\n")
    with _patched_data_path(settings_file):
        result = self._load_settings()
    assert result == [1, 2]
    assert not isinstance(result, dict)
