# -*- coding: utf-8 -*-
"""Tests for ``conftest.pytest_configure``.

Generated from: docs/00-Architecture/conftest__pytest_configure.yaml

function:    conftest.pytest_configure
signature:   def pytest_configure(config):
purpose:     pytest のフックとして config.option.basetemp が未設定（falsy）の場合、
             tmp_path 系フィクスチャのベースディレクトリをプロジェクトローカルの
             <プロジェクトルート>/.pytest_tmp へ向ける
precondition: pytest が設定段階で pytest_configure フックとしてこの関数を呼び出す
             （呼び出し側のコードは担当ファイル外）

（テスト対象外・文書化のみ）errors:
  - condition: 'config が option または basetemp 属性を有さない（pytest の Config
    以外が渡された）'
    behavior: 'AttributeError が関数内に try/except なく呼び出し元に送出される'

unconfirmed（未決情報）:
  - この関数は basetemp の値を設定するだけで .pytest_tmp ディレクトリを作成しない。
    下流の pytest tmpdir 機構が実際ディレクトリを作成するまでか、設定値の解釈は
    担当ファイル外のため未確認
  - config.option.basetemp の型・取り得る値（None またはパス文字列、と推定）は
    担当ファイル内に定義がなく未確認

missing（欠落情報）:
  - pytest のフック機構ソース（pytest_configure フックの呼び出し方法・config 引数の型）
    — 呼び出し契約の仕様確定に必要
  - pytest の tmpdir / pathlib 機構ソース（basetemp 値をどう下流で使うか）—
    この関数の設定が最終的にどう効くかの確定に必要
"""

import os
import sys

# 本テストは tests/ 配下からプロジェクトルートの conftest.py を直接 import する。
# pytest の import モードに依存しないよう、プロジェクトルートを sys.path に確保する。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import conftest  # noqa: E402
from conftest import pytest_configure  # noqa: E402


class _Option:
    """config.option に相当する最小オブジェクト（basetemp 属性のみ持つ）。"""

    def __init__(self, basetemp):
        self.basetemp = basetemp


class _Config:
    """pytest.Config に相当する最小オブジェクト（config.option.basetemp を持つ）。"""

    def __init__(self, basetemp):
        self.option = _Option(basetemp)


def _expected_basetemp():
    """仕様書の postcondition 通り:
    os.path.join(os.path.dirname(os.path.abspath(<conftest.py の絶対パス>)), ".pytest_tmp")
    """
    return os.path.join(os.path.dirname(os.path.abspath(conftest.__file__)), ".pytest_tmp")


def test_edge_01():
    """edge_case #1
    input: 'config.option.basetemp = None'
    expected: '呼び出し後、config.option.basetemp ==
              os.path.join(os.path.dirname(os.path.abspath(<conftest.py の絶対パス>)),
              ".pytest_tmp") となり、関数は None を返す'
    """
    config = _Config(None)
    result = pytest_configure(config)
    assert config.option.basetemp == _expected_basetemp()
    assert result is None


def test_edge_02():
    """edge_case #2
    input: 'config.option.basetemp = ""（空文字列。if not config.option.basetemp
           の判定上、同様に falsy）'
    expected: 'None の場合と同様に、basetemp がプロジェクトローカルの .pytest_tmp
               パスへ上書きされる'
    """
    config = _Config("")
    result = pytest_configure(config)
    assert config.option.basetemp == _expected_basetemp()
    assert result is None


def test_edge_03():
    """edge_case #3
    input: 'config.option.basetemp = "C:/custom/tmp"（任意の truthy 値）'
    expected: '呼び出し後、config.option.basetemp == "C:/custom/tmp" のままで
              変更されない'
    """
    config = _Config("C:/custom/tmp")
    result = pytest_configure(config)
    assert config.option.basetemp == "C:/custom/tmp"
    assert result is None
