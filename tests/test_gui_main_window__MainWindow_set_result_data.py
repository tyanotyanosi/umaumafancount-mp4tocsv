"""Tests for ``gui.main_window.MainWindow.set_result_data``.

Specification: docs/00-Architecture/gui_main_window__MainWindow_set_result_data.yaml

``MainWindow.set_result_data`` stores the processing result data on the
instance, updates the result table, and sets the status text including the
extracted count (per the spec's ``purpose`` field):

- Step 1: assign ``data`` to ``self.result_data`` (reference assignment,
  no copy)
- Step 2: call ``self.result_table.update_data(data)``
- Step 3: set the ``status_label`` text to
  "処理完了: {len(data)}件のユーザ情報を抽出" (count computed via ``len(data)``)

Per the spec's ``inputs`` constraints, the instance must already have
``self.result_table`` (callable via ``update_data(data)``) and
``self.status_label`` (callable via ``configure(text=...)``). The real
widgets are unrelated dependencies for this method, so the tests below
build a ``MainWindow`` instance without running ``__init__``
(``object.__new__``; no Tk window is created) and substitute
``unittest.mock.MagicMock`` for ``self.result_table`` and
``self.status_label`` (mock targets), then call ``set_result_data``
directly and assert input→expected.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'data が len() を支持しないオブジェクト（例: None）で、かつ self.result_table.update_data が先に例外を投げない場合'
  behavior: '217行目の len(data) 呼び出しで TypeError が発生し呼び出し元へ伝播する。その時点では self.result_data はすでに data へ設定済み（215行目）である'
- condition: 'self.result_table.update_data(data) が例外を送出する（実装未読のため具体的条件は列挙できない）'
  behavior: '例外はそのまま呼び出し元へ伝播する。status_label のテキストは更新されない（217行目に到達しない）'
"""

from unittest.mock import MagicMock

import pytest

from gui.main_window import MainWindow


def _make_window():
    """Build a MainWindow instance without running ``__init__`` (no Tk
    window is created) and attach MagicMock stand-ins for the two widgets
    the method interacts with (mock targets)."""
    win = object.__new__(MainWindow)
    win.result_table = MagicMock()
    win.status_label = MagicMock()
    return win


def test_edge_01():
    """
    input: data = {}（空dict）
    expected: self.result_data が {} になり、update_data({}) が呼び出され、status_labelのテキストが「処理完了: 0件のユーザ情報を抽出」になる
    """
    win = _make_window()
    data = {}
    result = win.set_result_data(data)
    assert result is None
    assert win.result_data is data
    win.result_table.update_data.assert_called_once_with(data)
    win.status_label.configure.assert_called_once_with(
        text="処理完了: 0件のユーザ情報を抽出"
    )


def test_edge_02():
    """
    input: data = {'user_id': 1}（要素1つのdict）
    expected: self.result_data が渡されたdictそのものを参照し、update_dataが呼び出され、status_labelのテキストが「処理完了: 1件のユーザ情報を抽出」になる
    """
    win = _make_window()
    data = {"user_id": 1}
    result = win.set_result_data(data)
    assert result is None
    assert win.result_data is data
    win.result_table.update_data.assert_called_once_with(data)
    win.status_label.configure.assert_called_once_with(
        text="処理完了: 1件のユーザ情報を抽出"
    )


def test_edge_03():
    """
    input: data = None（アノテーション外の入力）
    expected: 215行目が先に実行されるため self.result_data は None になる。その後 update_data(None) の挙動は未読のため不確定だが、update_data が例外を投げなければ 217行目の len(None) によりTypeErrorが発生する
    注: 仕様書 expected 原文「215行目が先に実行されるため self.result_data は None になる。その後 update_data(None) の挙動は未読のため不確定だが、update_data が例外を投げなければ 217行目の len(None) によりTypeErrorが発生する」。実測挙動を assert（update_data を例外を投げない MagicMock で置き換えた場合、len(None) により TypeError が発生し、status_label.configure は呼び出されない）。
    """
    win = _make_window()
    with pytest.raises(TypeError):
        win.set_result_data(None)
    assert win.result_data is None
    win.result_table.update_data.assert_called_once_with(None)
    win.status_label.configure.assert_not_called()
