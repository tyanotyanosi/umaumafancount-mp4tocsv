"""Tests for ``src.parser.name_mapper.NameMapperLoader.load_mapping``.

Specification: docs/00-Architecture/src_parser_name_mapper__NameMapperLoader_load_mapping.yaml

``load_mapping`` loads a JSON-format mapping definition file and
returns it as a dict (when the file does not exist it emits a warning
and falls back to an empty dict):

- convert to ``p = Path(path)``
- if ``p.exists()`` is false, emit
  ``logger.warning("マッピング定義ファイルが存在しません: {path} (空辞書でフォールバック)")``
  and return ``{}``
- open the file with ``open(p, 'r', encoding='utf-8')`` and parse it
  with ``json.load(f)``
- if ``json.JSONDecodeError`` is raised, convert it to
  ``ValueError("マッピング定義ファイルのJSON構文エラー: {path}: {exc}")``
  and raise it (``raise ... from exc``)
- if the parsed result ``data`` is not a dict, raise
  ``ValueError("マッピング定義ファイルはオブジェクトである必要があります: {path}")``
- return ``data`` (the dict)

Mocked / stand-in dependencies (per the test-generation rules):
the module-level ``logger`` (``logging.getLogger(__name__)``) is
patched with ``mock.patch`` for the nonexistent-path case so the
warning call can be observed deterministically. The JSON files for
the parse cases are written under a temporary directory created inside
the workspace (the ``_tmp_dir`` pattern).

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "ファイルの中身が JSON として解釈できない（json.JSONDecodeError）"
  behavior: "ValueError が送出される（from exc で元の JSONDecodeError が cause として保持）。"
- condition: "JSON 解析結果が dict でない（list / str / number / bool / null）"
  behavior: "ValueError が送出される。"
- condition: "open() / json.load() 由来のその他の例外（OSError 系列、UnicodeDecodeError、RecursionError 等）"
  behavior: "捕捉されないため、そのまま呼び出し側に送出される。"
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest

import src.parser.name_mapper as nm


def _tmp_dir():
    """Create a temporary directory inside the workspace and return
    it along with a cleanup callable."""
    tmp = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent.parent))

    def _cleanup():
        shutil.rmtree(tmp, ignore_errors=True)

    return tmp, _cleanup


def test_edge_01():
    """
    input: "存在しないパス 'config/nonexistent.json'"
    expected: "{} が返り、logger.warning が1回呼ばれる（メッセージにパスを含む）。"
    """
    with mock.patch.object(nm.logger, "warning") as warn:
        result = nm.NameMapperLoader.load_mapping("config/nonexistent.json")
    assert result == {}
    assert warn.call_count == 1
    args, _ = warn.call_args
    assert "config/nonexistent.json" in str(args)


def test_edge_02():
    """
    input: "内容が '{\"user_names\": {}}' のファイルパス"
    expected: "{'user_names': {}} が返る。"
    """
    tmp, cleanup = _tmp_dir()
    try:
        path = tmp / "mapping.json"
        path.write_text('{"user_names": {}}', encoding="utf-8")
        result = nm.NameMapperLoader.load_mapping(path)
    finally:
        cleanup()
    assert result == {"user_names": {}}


def test_edge_03():
    """
    input: "内容が '[1, 2]' のファイルパス"
    expected: "ValueError(\"マッピング定義ファイルはオブジェクトである必要があります: <path>\") が送出される。"
    """
    tmp, cleanup = _tmp_dir()
    try:
        path = tmp / "mapping.json"
        path.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ValueError) as excinfo:
            nm.NameMapperLoader.load_mapping(path)
        assert "マッピング定義ファイルはオブジェクトである必要があります" in str(excinfo.value)
        assert str(path) in str(excinfo.value)
    finally:
        cleanup()


def test_edge_04():
    """
    input: "内容が '{bad json' のファイルパス"
    expected: "json.JSONDecodeError を cause に持つ ValueError(\"マッピング定義ファイルのJSON構文エラー: <path>: ...\") が送出される。"
    """
    tmp, cleanup = _tmp_dir()
    try:
        path = tmp / "mapping.json"
        path.write_text("{bad json", encoding="utf-8")
        with pytest.raises(ValueError) as excinfo:
            nm.NameMapperLoader.load_mapping(path)
        assert "マッピング定義ファイルのJSON構文エラー" in str(excinfo.value)
        assert str(path) in str(excinfo.value)
        # The original JSONDecodeError is kept as the cause.
        assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)
    finally:
        cleanup()


def test_edge_05():
    """
    input: "内容が '\"hello\"'（JSON 文字列）のファイルパス"
    expected: "ValueError（オブジェクトである必要があります）が送出される。"
    """
    tmp, cleanup = _tmp_dir()
    try:
        path = tmp / "mapping.json"
        path.write_text('"hello"', encoding="utf-8")
        with pytest.raises(ValueError) as excinfo:
            nm.NameMapperLoader.load_mapping(path)
        assert "マッピング定義ファイルはオブジェクトである必要があります" in str(excinfo.value)
        assert str(path) in str(excinfo.value)
    finally:
        cleanup()
