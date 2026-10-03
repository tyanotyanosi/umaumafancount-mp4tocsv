"""Tests for ``gui.settings_dialog.SettingsDialog._save_settings``.

Specification: docs/00-Architecture/gui_settings_dialog__SettingsDialog__save_settings.yaml

``_save_settings`` validates and converts the UI input values, updates the
``video`` / ``ocr`` / ``text_region`` / ``name_mapping`` sections of
``self.config`` in place, and persists the whole settings to the settings
file:

- Step 1: parse interval (float), diff_threshold (float),
  mapping_threshold (int) from the UI variables
- Step 2: on ValueError, show the "設定エラー" dialog and return (no config
  write at all)
- Step 3: get region_enabled (text_region_var) and use_percent
  (region_unit_var == "percent")
- Step 4: if region_enabled, parse left/top/right/bottom as float; on
  ValueError show the "設定エラー" dialog and return
- Step 5: if not region_enabled, left/top/right/bottom are 0.0 (not written
  to the file)
- Step 6: ensure video/ocr sections via setdefault; write frame_interval,
  enable_diff_check, diff_threshold, diff_only, engine
- Step 7: ensure text_region section via setdefault; write enabled and
  use_percent
- Step 8: if enabled and percent, write left/top/right/bottom; if enabled
  and pixel, write x, y, width=int(right-left), height=int(bottom-top)
- Step 9: if invalid, write enabled=False again (duplicate write of the
  same value)
- Step 10: ensure name_mapping section via setdefault; write file
  (.strip() applied), enable, edit_distance_threshold, warn_on_approx,
  unmapped_action
- Step 11: create the parent directory (parents=True, exist_ok=True) and
  dump the whole ``self.config`` with yaml.dump (utf-8,
  allow_unicode=True, default_flow_style=False)

The function returns ``None`` on every path.

Mocked / stand-in dependencies (per the test-generation rules): the
``SettingsDialog`` instance is created without running ``__init__``
(``object.__new__``); the 16 UI variables are minimal fakes whose
``.get()`` returns the per-test value; ``self.config`` is a real dict;
``self.settings_file`` is a real ``pathlib.Path`` in a temporary directory
(the file is really written and re-read for assertions);
``gui.settings_dialog.messagebox`` is patched with a mock so the
"設定エラー" dialog call is recorded instead of shown.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "self.config が dict でない（例、_load_settings が非 mapping を返した場合）"
  behavior: "setdefault 呼び出しで AttributeError が発生し呼び出し側に伝播する（本関数では捕捉されない）"
- condition: "親ディレクトリ作成またはファイル書き込みが失敗（権限・ディスク等）"
  behavior: "OSError のサブクラス（例 PermissionError）が呼び出し側に伝播する（本関数では捕捉されない）"
"""

import copy
import math
import tempfile
from pathlib import Path
from unittest import mock

import yaml

from gui.settings_dialog import SettingsDialog


class _Var:
    """Minimal stand-in for a Tk variable: ``.get()`` returns the fixed
    value set for the test."""

    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def _make_self(config, settings_file, overrides=None):
    """Build a ``SettingsDialog`` instance (without running ``__init__``)
    with valid default UI values, then apply per-test overrides."""
    self = object.__new__(SettingsDialog)
    self.config = config
    self.settings_file = settings_file
    values = {
        "interval_var": "0.5",
        "diff_threshold_var": "0.3",
        "name_mapping_threshold_var": "5",
        "text_region_var": True,
        "region_unit_var": "percent",
        "region_left_var": "0.10",
        "region_top_var": "0.20",
        "region_right_var": "0.90",
        "region_bottom_var": "0.80",
        "diff_var": True,
        "diff_only_var": False,
        "ocr_engine_var": "rapidocr",
        "name_mapping_file_var": "map.csv",
        "name_mapping_enable_var": True,
        "name_mapping_warn_var": True,
        "name_mapping_action_var": "skip",
    }
    values.update(overrides or {})
    for name, value in values.items():
        setattr(self, name, _Var(value))
    return self


def _load_yaml(settings_file):
    with open(settings_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_edge_01():
    """
    input: interval_var.get() が 'abc'（他は有効な値）
    expected: 「設定エラー」（フレーム間隔・差分閾値・編集距離閾値には有効な数値を入力してください。）ダイアログを表示；None を返す；self.config は変更されずファイルも書かれない
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    self = _make_self({}, settings_file, {"interval_var": "abc"})
    with mock.patch("gui.settings_dialog.messagebox") as mb:
        ret = self._save_settings()
    assert ret is None
    mb.showerror.assert_called_once()
    assert mb.showerror.call_args[0][0] == "設定エラー"
    assert self.config == {}
    assert not settings_file.exists()


def test_edge_02():
    """
    input: name_mapping_threshold_var.get() が '3.5'
    expected: int('3.5') が ValueError を起こすため同じ「設定エラー」ダイアログを表示；None を返す；ファイルは書かれない
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    self = _make_self({}, settings_file,
                      {"name_mapping_threshold_var": "3.5"})
    with mock.patch("gui.settings_dialog.messagebox") as mb:
        ret = self._save_settings()
    assert ret is None
    mb.showerror.assert_called_once()
    assert mb.showerror.call_args[0][0] == "設定エラー"
    assert self.config == {}
    assert not settings_file.exists()


def test_edge_03():
    """
    input: interval_var.get() が 'nan'（他は有効な値）
    expected: float('nan') は成功するためエラーなし；video.frame_interval が NaN として保存され（YAML ファイルには .nan として書き込まれ）、残りの値も通常どおり保存される
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    self = _make_self({}, settings_file, {"interval_var": "nan"})
    with mock.patch("gui.settings_dialog.messagebox") as mb:
        ret = self._save_settings()
    assert ret is None
    mb.showerror.assert_not_called()
    assert math.isnan(self.config["video"]["frame_interval"])
    data = _load_yaml(settings_file)
    assert math.isnan(data["video"]["frame_interval"])
    assert ".nan" in settings_file.read_text(encoding="utf-8")
    # 残りの値も通常どおり保存される
    assert data["video"]["diff_threshold"] == 0.3
    assert data["name_mapping"]["edit_distance_threshold"] == 5
    assert data["text_region"]["left"] == 0.1


def test_edge_04():
    """
    input: text_region_var.get() が偽値（領域無効）
    expected: 座標の解析・保存はスキップされ、text_region.enabled == False となる；座標キー（left/top/right/bottom 等）はファイルに書き込まれない
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    self = _make_self({}, settings_file, {"text_region_var": False})
    with mock.patch("gui.settings_dialog.messagebox") as mb:
        ret = self._save_settings()
    assert ret is None
    mb.showerror.assert_not_called()
    assert self.config["text_region"]["enabled"] is False
    data = _load_yaml(settings_file)
    for key in ("left", "top", "right", "bottom", "x", "y", "width", "height"):
        assert key not in data["text_region"]


def test_edge_05():
    """
    input: text_region_var.get() が真値で region_left_var.get() が 'x'
    expected: 「設定エラー」（テキスト領域の座標には有効な数値を入力してください。）ダイアログを表示；None を返す；self.config は変更されずファイルも書かれない
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    self = _make_self({}, settings_file, {"region_left_var": "x"})
    with mock.patch("gui.settings_dialog.messagebox") as mb:
        ret = self._save_settings()
    assert ret is None
    mb.showerror.assert_called_once()
    assert mb.showerror.call_args[0][0] == "設定エラー"
    assert self.config == {}
    assert not settings_file.exists()


def test_edge_06():
    """
    input: 領域有効、region_unit_var.get() == 'percent'、left='0.10', top='0.20', right='0.90', bottom='0.80'
    expected: text_region.left == 0.1、top == 0.2、right == 0.9、bottom == 0.8（float として保存）
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    self = _make_self({}, settings_file)
    with mock.patch("gui.settings_dialog.messagebox") as mb:
        ret = self._save_settings()
    assert ret is None
    mb.showerror.assert_not_called()
    region = self.config["text_region"]
    assert region["left"] == 0.1
    assert region["top"] == 0.2
    assert region["right"] == 0.9
    assert region["bottom"] == 0.8
    data = _load_yaml(settings_file)
    assert data["text_region"]["left"] == 0.1
    assert data["text_region"]["top"] == 0.2
    assert data["text_region"]["right"] == 0.9
    assert data["text_region"]["bottom"] == 0.8


def test_edge_07():
    """
    input: 領域有効、region_unit_var.get() == 'pixel'、left='10.9', top='20.3', right='110.4', bottom='60.8'
    expected: text_region.x == 10、y == 20、width == 99、height == 40（int による切り捨て）
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    self = _make_self({}, settings_file, {
        "region_unit_var": "pixel",
        "region_left_var": "10.9",
        "region_top_var": "20.3",
        "region_right_var": "110.4",
        "region_bottom_var": "60.8",
    })
    with mock.patch("gui.settings_dialog.messagebox") as mb:
        ret = self._save_settings()
    assert ret is None
    mb.showerror.assert_not_called()
    region = self.config["text_region"]
    assert region["x"] == 10
    assert region["y"] == 20
    assert region["width"] == 99
    assert region["height"] == 40
    data = _load_yaml(settings_file)
    assert data["text_region"]["x"] == 10
    assert data["text_region"]["y"] == 20
    assert data["text_region"]["width"] == 99
    assert data["text_region"]["height"] == 40


def test_edge_08():
    """
    input: name_mapping_file_var.get() が '  map.csv  '
    expected: name_mapping.file == 'map.csv'（先頭・末尾の空白は .strip() で除去される）
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    self = _make_self({}, settings_file,
                      {"name_mapping_file_var": "  map.csv  "})
    with mock.patch("gui.settings_dialog.messagebox") as mb:
        ret = self._save_settings()
    assert ret is None
    mb.showerror.assert_not_called()
    assert self.config["name_mapping"]["file"] == "map.csv"
    data = _load_yaml(settings_file)
    assert data["name_mapping"]["file"] == "map.csv"


def test_edge_09():
    """
    input: 領域有効、単位 pixel、left='100', top='0', right='50', bottom='40'（right < left）
    expected: text_region.width == -50、height == 40（負の width が検証なしでそのまま保存される）
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    self = _make_self({}, settings_file, {
        "region_unit_var": "pixel",
        "region_left_var": "100",
        "region_top_var": "0",
        "region_right_var": "50",
        "region_bottom_var": "40",
    })
    with mock.patch("gui.settings_dialog.messagebox") as mb:
        ret = self._save_settings()
    assert ret is None
    mb.showerror.assert_not_called()
    region = self.config["text_region"]
    assert region["width"] == -50
    assert region["height"] == 40
    data = _load_yaml(settings_file)
    assert data["text_region"]["width"] == -50
    assert data["text_region"]["height"] == 40


def test_edge_10():
    """
    input: self.config に既に text_region（use_percent=True と left/top/right/bottom キーあり）があり、UI は領域無効
    expected: text_region.enabled == False となるが、既存の left/top/right/bottom キーは削除されずファイルに残る
    """
    tmp = Path(tempfile.mkdtemp())
    settings_file = tmp / "settings.yaml"
    config = {
        "text_region": {
            "use_percent": True,
            "left": 0.1,
            "top": 0.2,
            "right": 0.9,
            "bottom": 0.8,
        }
    }
    original = copy.deepcopy(config)
    self = _make_self(config, settings_file, {"text_region_var": False})
    with mock.patch("gui.settings_dialog.messagebox") as mb:
        ret = self._save_settings()
    assert ret is None
    mb.showerror.assert_not_called()
    assert self.config["text_region"]["enabled"] is False
    # 既存キーは削除されない
    for key in ("left", "top", "right", "bottom"):
        assert key in self.config["text_region"]
        assert self.config["text_region"][key] == original["text_region"][key]
    data = _load_yaml(settings_file)
    for key in ("left", "top", "right", "bottom"):
        assert data["text_region"][key] == original["text_region"][key]
