# -*- coding: utf-8 -*-
"""Tests for ``gui.main_window.MainWindow.__init__``.

Generated from: docs/00-Architecture/gui_main_window__MainWindow___init__.yaml

function:    gui.main_window.MainWindow.__init__
signature:   def __init__(self):
purpose:     メインウィンドウを初期化する。CTkウィンドウを作成し、タイトルとサイズを
             設定し、インスタンス状態を初期化し、設定を読み込み、UIを構築する。
precondition: tkinter / customtkinter 環境が利用可能で、super().__init__() による
             CTk トップレベルウィンドウの作成に成功すること、self 上で _load_settings()
             および _setup_ui() が正常に完了すること

外部依存の mock 方針（仕様書の指示に従い）:
  - CTk トップレベルウィンドウ本体の作成（super().__init__() → ctk.CTk.__init__）
    と、表示依存の tkinter メソッド（title / geometry）は mock する（GUI表示禁止）。
  - 設定ファイル読み取りを行うプライベートメソッド _load_settings と、
    UI ウィジェットを構築するプライベートメソッド _setup_ui は mock する
    （ファイル・GUI 依存のため）。
  - それ以外の __init__ 本体の動作（video_path / result_data の初期化、
    self.settings への代入、_setup_ui の呼び出し）は real で実行する。

（テスト対象外・文書化のみ）errors:
  - condition: 'CTK ウィンドウが作成できない（表示環境がない等）'
    behavior: 'super().__init__() 内で発生した例外が外へ伝播し、インスタンスは返らない'

unconfirmed（未決情報）:
  - super().__init__() が失敗した場合の具体的例外種別はこのファイルからは視認できない
  - self.settings の型は検証されない。_load_settings() が戻す値（非 dict の可能性あり）
    をそのまま持つ

missing（欠落情報）:
  - _create_main_content の完全な実装（74〜86行は可読、87行以降は未読）
  - _create_status_bar の実装（このファイルの可読範囲外）
  - data_path の実装（src/utils/app_paths.py。設定ファイルのパス解決ルール）
"""

import os
import sys

# 本テストは tests/ 配下からプロジェクトルートの gui パッケージを直接 import する。
# pytest の import モードに依存しないよう、プロジェクトルートを sys.path に確保する。
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import customtkinter as ctk  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402

from unittest.mock import MagicMock, patch  # noqa: E402


def _build_window(load_settings_return):
    """表示・ファイル依存のみを mock して MainWindow() を real に構築する。

    戻り値: (win, title_mock, geometry_mock, setup_ui_mock, load_settings_mock)
    """
    title_mock = MagicMock()
    geometry_mock = MagicMock()
    with patch.object(MainWindow, "_load_settings",
                      return_value=load_settings_return) as load_mock, \
         patch.object(MainWindow, "_setup_ui") as setup_mock, \
         patch.object(ctk.CTk, "__init__", create=True), \
         patch.object(ctk.CTk, "title", title_mock, create=True), \
         patch.object(ctk.CTk, "geometry", geometry_mock, create=True):
        win = MainWindow()
    return win, title_mock, geometry_mock, setup_mock, load_mock


def test_edge_01():
    """edge_case #1
    input: '引数なし（MainWindow()）'
    expected: '例外なし。戻り値は video_path=None、result_data=None、タイトル
              mov-to-fan-count、geometry 1200x800 のインスタンス'
    """
    win, title_mock, geometry_mock, setup_mock, _ = _build_window({})
    # インスタンス状態の初期化（real な __init__ の代入）
    assert win.video_path is None
    assert win.result_data is None
    # ウィンドウタイトルが mov-to-fan-count に設定される
    title_mock.assert_called_with("mov-to-fan-count")
    # ウィンドウサイズが 1200x800 に設定される
    geometry_mock.assert_called_with("1200x800")
    # postcondition: _setup_ui() が1回呼び出された状態である
    setup_mock.assert_called_once()


def test_edge_02():
    """edge_case #2
    input: '_load_settings() 呼び出し時点で設定ファイルが存在しない'
    expected: '例外なし。self.settings が {} になる'
    """
    # _load_settings の仕様（gui_main_window__MainWindow__load_settings.yaml）:
    # 設定ファイルが存在しない場合は 例外なしで {} を返す。
    # ここではファイル読み取りという外部依存として _load_settings を mock し、
    # 「設定ファイルが存在しない」状況（戻り値 {}）を表現する。
    win, _, _, _, load_mock = _build_window({})
    # self.settings が {} になる
    assert win.settings == {}
    # _load_settings() が1回呼び出され、その戻り値が self.settings に入った
    load_mock.assert_called_once()
