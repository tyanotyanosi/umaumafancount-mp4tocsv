"""Tests for ``cli.main.load_settings``.

Specification: docs/00-Architecture/cli_main__load_settings.yaml

``load_settings`` reads the YAML settings file; when the file is missing,
empty, or invalid YAML, it returns an empty dict so that processing
continues with all default values:

- Determine path p: if settings_path is truthy, ``Path(settings_path)``;
  otherwise ``data_path('config/settings.yaml')``
- If p does not exist (``p.exists()`` is False), emit a warning log
  (text 「設定ファイルが見つかりません: %s（デフォルト値を使用）」,
  argument the path) and return ``{}``
- Open the file in utf-8; if the result of ``yaml.safe_load(f)`` is
  truthy, return it; if falsy, return ``{}``
- If an ``OSError`` or ``yaml.YAMLError`` occurs, emit a warning log
  (text 「設定ファイルの読み込みに失敗しました: %s（デフォルト値を使用）:
  %s」, arguments the path and the exception) and return ``{}``

Mocked / stand-in dependencies (per the test-generation rules):
``cli.main.data_path`` is patched to return a non-existent path in a
temporary directory (its implementation is outside the read scope); the
settings files are real files created in a temporary directory under the
session workspace; ``open`` / ``yaml.safe_load`` / the warning logging
run real (log records are captured with ``caplog``).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "ファイルの読み込み・解析中にOSErrorまたはyaml.YAMLErrorが発生する"
  behavior: "例外を捕捉しwarningログを出力して{}を返す。外部に例外は送出されない。"
"""

import shutil
import tempfile
from pathlib import Path
from unittest import mock

from cli.main import load_settings


def _tmp_dir():
    """Create a temporary directory under the session workspace (the
    platform temp area is not writable for subdirectory creation under
    the file sandbox; the workspace is)."""
    workspace = Path(__file__).resolve().parent.parent
    return Path(tempfile.mkdtemp(dir=workspace))


def test_edge_01(caplog):
    """
    input: settings_path=Noneで、data_path('config/settings.yaml')の解決先が存在しない
    expected: warningログを出力し{}を返す。例外は送出されない。
    """
    tmp = _tmp_dir()
    try:
        missing = tmp / "missing_settings.yaml"
        with mock.patch("cli.main.data_path", return_value=missing):
            ret = load_settings()
        assert ret == {}
        assert "設定ファイルが見つかりません" in caplog.text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_02(caplog):
    """
    input: settings_pathが実在しないファイルを指す
    expected: warningログを出力し{}を返す。例外は送出されない。
    """
    tmp = _tmp_dir()
    try:
        missing = tmp / "nope.yaml"
        ret = load_settings(str(missing))
        assert ret == {}
        assert "設定ファイルが見つかりません" in caplog.text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_03(caplog):
    """
    input: settings_pathが空文字列
    expected: 空文字列はfalsyのためNoneと同様に扱い、data_pathでconfig/settings.yamlを探す。
    """
    tmp = _tmp_dir()
    try:
        missing = tmp / "missing_settings.yaml"
        with mock.patch("cli.main.data_path", return_value=missing) \
                as mock_dp:
            ret = load_settings("")
        assert ret == {}
        mock_dp.assert_called_once_with("config/settings.yaml")
        assert "設定ファイルが見つかりません" in caplog.text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_04(caplog):
    """
    input: ファイルが存在し内容が空（yaml.safe_loadがNoneを返す）
    expected: {}を返す。
    """
    tmp = _tmp_dir()
    try:
        p = tmp / "settings.yaml"
        p.write_text("", encoding="utf-8")
        ret = load_settings(str(p))
        assert ret == {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_05(caplog):
    """
    input: ファイルに有効なmapping YAMLがある
    expected: 解析されたmapping dictをそのまま返す。
    """
    tmp = _tmp_dir()
    try:
        p = tmp / "settings.yaml"
        p.write_text("video:\n  frame_interval: 2.5\n",
                     encoding="utf-8")
        ret = load_settings(str(p))
        assert ret == {"video": {"frame_interval": 2.5}}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_06(caplog):
    """
    input: ファイルにmapping以外（list・文字列等）のYAMLがある
    expected: yaml.safe_loadの結果がそのままtruthyなので、dict以外の値を返す（unconfirmed参照）。
    """
    tmp = _tmp_dir()
    try:
        p = tmp / "settings.yaml"
        p.write_text("- a\n- b\n", encoding="utf-8")
        ret = load_settings(str(p))
        assert ret == ["a", "b"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_07(caplog):
    """
    input: ファイルにyaml.YAMLErrorを引き起こす内容がある
    expected: warningログを出力し{}を返す。例外は送出されない。
    """
    tmp = _tmp_dir()
    try:
        p = tmp / "settings.yaml"
        p.write_text("a: b: c\n", encoding="utf-8")
        ret = load_settings(str(p))
        assert ret == {}
        assert "設定ファイルの読み込みに失敗しました" in caplog.text
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
