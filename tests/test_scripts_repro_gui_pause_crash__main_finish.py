"""Tests for ``scripts.repro_gui_pause_crash.main.finish``.

Specification: docs/00-Architecture/scripts_repro_gui_pause_crash__main_finish.yaml

``finish`` is the callback for the end of the 12-second observation
window: it prints a log line, releases the player with
``frame.close()``, and ends the mainloop via ``root.destroy()``:

1. print ``'[repro] observation window elapsed -> exit'``
   (flush=True)
2. call ``frame.close()``
3. call ``root.destroy()``

Because ``finish`` is a local function of ``main`` it cannot be
imported by name; the module-level function object below is recovered
from the nested code object named ``finish`` inside
``main.__code__.co_consts`` and rebound as a plain function via
``types.FunctionType`` with the module's globals, supplying the
closure variables (``frame``, ``root`` per the spec's preconditions)
via ``types.CellType``.

Mocked / stand-in dependencies (per the test-generation rules):
``frame`` is a ``mock.Mock`` standing in for the VideoPreviewFrame
instance created in ``main`` (the ``gui/video_player.py``
implementation is out of read scope per the spec's ``missing``
section); ``root`` is a ``_FakeRoot`` stand-in recording
``destroy()`` calls (a real ``destroy()`` on a live window would end
the mainloop, which is the spec's "root.mainloop() が return する").

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "frame.close() が例外を送出する（実装が本ファイルにないため未確認）"
  behavior: "例外はコールバックの外へ送出され、その行以降（root.destroy()）は実行されない"
"""

import types
from unittest import mock

import pytest

import scripts.repro_gui_pause_crash as repro
from scripts.repro_gui_pause_crash import main


class _FakeRoot:
    """Stand-in for the ``tk.Tk()`` root window: records
    ``destroy()`` calls."""

    def __init__(self):
        self.destroy_count = 0

    def destroy(self):
        self.destroy_count += 1


def _make_finish(frame, root):
    """Recover the nested ``finish`` function from
    ``main.__code__.co_consts`` and rebind it with the closure
    variables ``frame`` and ``root``."""
    code = None
    for const in main.__code__.co_consts:
        if isinstance(const, types.CodeType) and const.co_name == "finish":
            code = const
            break
    assert code is not None
    assert code.co_freevars == ("frame", "root")
    frame_cell = types.CellType()
    frame_cell.cell_contents = frame
    root_cell = types.CellType()
    root_cell.cell_contents = root
    fn = types.FunctionType(code, repro.__dict__, "finish",
                            None, (frame_cell, root_cell))
    return fn


def test_edge_01(capsys):
    """
    input: mainloop 中に1回呼び出されたとき
    expected: stdout に '[repro] observation window elapsed -> exit' が現れ、frame.close() が1回、root.destroy() が1回呼び出され、root.mainloop() が return する
    """
    root = _FakeRoot()
    frame = mock.Mock(name="VideoPreviewFrame")
    finish = _make_finish(frame, root)
    finish()
    out = capsys.readouterr().out
    assert out == "[repro] observation window elapsed -> exit\n"
    assert frame.close.call_count == 1
    assert root.destroy_count == 1
    # root.destroy() is what makes root.mainloop() return.


def test_edge_02(capsys):
    """
    input: frame.close() が例外を送出したとき
    expected: root.destroy() は呼び出されない（コールバック本体が例外で中断される）。mainloop 以降の処理は tkinter の例外処理に依存し未確認
    """
    root = _FakeRoot()
    frame = mock.Mock(name="VideoPreviewFrame")
    frame.close.side_effect = RuntimeError("close failed")
    finish = _make_finish(frame, root)
    with pytest.raises(RuntimeError):
        finish()
    assert root.destroy_count == 0
