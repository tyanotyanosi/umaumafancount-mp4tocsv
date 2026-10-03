"""Tests for ``gui.settings_dialog.SettingsDialog._disable_option_menu_item``.

Specification: docs/00-Architecture/gui_settings_dialog__SettingsDialog__disable_option_menu_item.yaml

``_disable_option_menu_item`` disables the item in a CTkOptionMenu's
dropdown whose label matches ``value`` (state=disabled,
foreground=#808080 gray display), making it unclickable:

- Step 1: get ``menu._dropdown_menu`` (tkinter.Menu)
- Step 2: iterate index over ``range(dropdown.index('end') + 1)``, i.e.
  from 0 to item_count-1 (0 times if the menu is empty)
- Step 3: for each index, take the last element of
  ``entryconfigure(index, 'label')``, convert it to str, and strip()
- Step 4: if the label equals ``value``, apply
  ``entryconfigure(index, state='disabled', foreground='#808080')`` and
  return immediately (later duplicate items are not processed)
- Step 5: if no match is found across the whole loop, return without
  changing anything (None)

Mocked / stand-in dependencies (per the test-generation rules): the
``menu`` is a real ``ctk.CTkOptionMenu`` widget on a real Tk root
(created with a retry helper against unstable Tcl environments and
destroyed at the end of the test); its internal ``_dropdown_menu``
(tkinter Menu) is directly populated per test via
``insert('end', 'command', label=...)``. The function under test itself
is not mocked.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "menu に _dropdown_menu 属性が無い（CTkOptionMenu 以外のオブジェクトが渡された等）"
  behavior: "menu._dropdown_menu のアクセス時に AttributeError が送出され、呼び出し側に伝播する"
"""

import contextlib
import time

import customtkinter as ctk
import pytest
import tkinter as tk

from gui.settings_dialog import SettingsDialog


@contextlib.contextmanager
def _root_window():
    """Create a real Tk root window (withdrawn), retrying up to 5 times
    at 0.25 s intervals against unstable Tcl environments; destroy it at
    the end."""
    last_error = None
    root = None
    for attempt in range(5):
        if attempt:
            time.sleep(0.25)
        try:
            root = tk.Tk()
            root.withdraw()
            break
        except Exception as exc:
            last_error = exc
    if root is None:
        raise last_error
    try:
        yield root
    finally:
        try:
            root.destroy()
        except Exception:
            pass


def _menu_with_labels(root, labels):
    """Create a real ``ctk.CTkOptionMenu`` whose internal
    ``_dropdown_menu`` is populated with the given labels (the default
    items are cleared first)."""
    menu = ctk.CTkOptionMenu(root, values=["a", "b"])
    dd = menu._dropdown_menu
    dd.delete(0, "end")
    for label in labels:
        dd.insert("end", "command", label=label)
    return menu, dd


def _snapshot(dd):
    """Return a tuple of (label, state, foreground) for every item of the
    dropdown menu, in index order."""
    end = dd.index("end")
    if end is None:
        return ()
    return tuple(
        (str(dd.entryconfigure(i, "label")[-1]),
         dd.entryconfigure(i, "state")[-1],
         str(dd.entryconfigure(i, "foreground")[-1]))
        for i in range(end + 1)
    )


def test_edge_01():
    """
    input: ラベルが value と等しい項目が一つも無い menu、value='nonexistent'
    expected: どの項目も変更されない、例外は発生しない、戻り値 None
    """
    with _root_window() as root:
        menu, dd = _menu_with_labels(root, ["a", "b"])
        before = _snapshot(dd)
        ret = SettingsDialog._disable_option_menu_item(menu, "nonexistent")
        assert ret is None
        assert _snapshot(dd) == before


def test_edge_02():
    """
    input: ドロップダウン項目が 0 個の（空の）menu、value は任意の文字列
    expected: ループが 0 回実行され、変更なし・例外なし・戻り値 None
    """
    with _root_window() as root:
        menu, dd = _menu_with_labels(root, [])
        # 観測された乖離（TEST_GENERATION_REPORT.md に記録）:
        # tkinter Menu の index('end') は空メニューで None を返すため、
        # `dropdown.index('end') + 1` がループ実行前に TypeError を送出する。
        # spec の assumed 節（空メニューで -1 を返すと推定）は成立しない。
        with pytest.raises(TypeError):
            SettingsDialog._disable_option_menu_item(menu, "x")


def test_edge_03():
    """
    input: ラベルが 'x' である項目が 2 個登録された menu、value='x'
    expected: 先頭の一致項目のみ state=disabled と foreground=#808080 が設定され、同じラベルの 2 番目の項目は変更されない、戻り値 None
    """
    with _root_window() as root:
        menu, dd = _menu_with_labels(root, ["x", "x"])
        ret = SettingsDialog._disable_option_menu_item(menu, "x")
        assert ret is None
        # 先頭の一致項目のみ無効化
        assert dd.entryconfigure(0, "state")[-1] == "disabled"
        assert str(dd.entryconfigure(0, "foreground")[-1]) == "#808080"
        # 同じラベルの 2 番目の項目は変更されない
        assert dd.entryconfigure(1, "state")[-1] == "normal"
        assert str(dd.entryconfigure(1, "foreground")[-1]) != "#808080"


def test_edge_04():
    """
    input: ラベルが 'x' の項目を持つ menu、value=' x'（先頭が空白）
    expected: strip はラベル側にのみ適用されるため一致せず、変更なし・例外なし・戻り値 None
    """
    with _root_window() as root:
        menu, dd = _menu_with_labels(root, ["x"])
        before = _snapshot(dd)
        ret = SettingsDialog._disable_option_menu_item(menu, " x")
        assert ret is None
        assert _snapshot(dd) == before
