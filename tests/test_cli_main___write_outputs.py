"""Tests for ``cli.main._write_outputs``.

Specification: docs/00-Architecture/cli_main___write_outputs.yaml

``_write_outputs`` writes the JSON / CSV outputs via ``JSONWriter`` /
``CSVWriter`` and returns the output file paths as a 2-tuple:

- 1. initialize json_path and csv_path to None
- 2. if fmt is in ("all","json"), create a JSONWriter with
  ``Path(output_dir)/"json"``, call ``write(merged)``, record the
  return value as json_path and print 「JSON出力: {json_path}」
- 3. if fmt is in ("all","csv"), create a CSVWriter with
  ``Path(output_dir)/"csv"``, call ``write(merged)``, record the
  return value as csv_path and print 「CSV出力: {csv_path}」
- 4. return ``(json_path, csv_path)``

Mocked / stand-in dependencies (per the test-generation rules):
``cli.main.JSONWriter`` and ``cli.main.CSVWriter`` are mocked (their
implementations are outside the read scope); ``merged`` is a plain
dict, ``output_dir`` is a stand-in string path.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "JSONWriter.write または CSVWriter.write 内部で例外（output_dir/json や output_dir/csv の作成失敗、ディスクフル、merged のデータ形式不正等）"
  behavior: "例外が呼び出し側にそのまま伝播する（関数内に try/except はない）"
"""

from pathlib import Path
from unittest import mock

from cli.main import _write_outputs

MERGED = {"user1": {"cards": [{"name": "Sakura", "fans": "123"}]}}
JSON_PATH_VALUE = "D:/out/json/results.json"
CSV_PATH_VALUE = "D:/out/csv/results.csv"


def _mocks():
    """Patch JSONWriter / CSVWriter and yield their constructor mocks."""
    mock_json = mock.patch("cli.main.JSONWriter").start()
    mock_csv = mock.patch("cli.main.CSVWriter").start()
    mock_json.return_value.write.return_value = JSON_PATH_VALUE
    mock_csv.return_value.write.return_value = CSV_PATH_VALUE
    return mock_json, mock_csv


def test_edge_01(capsys):
    """
    input: fmt="all"、merged=M（任意 dict）、output_dir=D
    expected: (json_path, csv_path) を返し、両方とも None でない; 「JSON出力: ...」と「CSV出力: ...」の両方が出力される
    """
    mock_json, mock_csv = _mocks()
    try:
        out_dir = "D:/out"
        ret = _write_outputs(MERGED, out_dir, "all")
        assert ret == (JSON_PATH_VALUE, CSV_PATH_VALUE)
        mock_json.assert_called_once_with(Path(out_dir) / "json")
        mock_csv.assert_called_once_with(Path(out_dir) / "csv")
        mock_json.return_value.write.assert_called_once_with(MERGED)
        mock_csv.return_value.write.assert_called_once_with(MERGED)
        out = capsys.readouterr().out
        assert f"JSON出力: {JSON_PATH_VALUE}" in out
        assert f"CSV出力: {CSV_PATH_VALUE}" in out
    finally:
        mock_json.stop()
        mock_csv.stop()


def test_edge_02(capsys):
    """
    input: fmt="json"
    expected: (json_path, None) を返す; 「CSV出力: ...」は出力されず CSVWriter は生成されない
    """
    mock_json, mock_csv = _mocks()
    try:
        out_dir = "D:/out"
        ret = _write_outputs(MERGED, out_dir, "json")
        assert ret == (JSON_PATH_VALUE, None)
        mock_json.assert_called_once_with(Path(out_dir) / "json")
        mock_csv.assert_not_called()
        out = capsys.readouterr().out
        assert f"JSON出力: {JSON_PATH_VALUE}" in out
        assert "CSV出力:" not in out
    finally:
        mock_json.stop()
        mock_csv.stop()


def test_edge_03(capsys):
    """
    input: fmt="csv"
    expected: (None, csv_path) を返す; 「JSON出力: ...」は出力されず JSONWriter は生成されない
    """
    mock_json, mock_csv = _mocks()
    try:
        out_dir = "D:/out"
        ret = _write_outputs(MERGED, out_dir, "csv")
        assert ret == (None, CSV_PATH_VALUE)
        mock_json.assert_not_called()
        mock_csv.assert_called_once_with(Path(out_dir) / "csv")
        out = capsys.readouterr().out
        assert f"CSV出力: {CSV_PATH_VALUE}" in out
        assert "JSON出力:" not in out
    finally:
        mock_json.stop()
        mock_csv.stop()


def test_edge_04(capsys):
    """
    input: fmt="" / fmt="txt" / fmt="JSON"（大文字不一致）
    expected: (None, None) を返す; writer は生成されず、stdout 出力もしない
    """
    mock_json, mock_csv = _mocks()
    try:
        out_dir = "D:/out"
        for fmt in ("", "txt", "JSON"):
            ret = _write_outputs(MERGED, out_dir, fmt)
            assert ret == (None, None)
        mock_json.assert_not_called()
        mock_csv.assert_not_called()
        assert capsys.readouterr().out == ""
    finally:
        mock_json.stop()
        mock_csv.stop()
