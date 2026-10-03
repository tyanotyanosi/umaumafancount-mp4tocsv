"""Tests for ``scripts.repro_gui_pause_crash.main.do_pause``.

Specification: docs/00-Architecture/scripts_repro_gui_pause_crash__main_do_pause.yaml

``do_pause`` is a callback simulating a PAUSE button press: it prints
a log line and pauses playback by calling ``frame._toggle_play()``:

1. print ``'[repro] PAUSE pressed (mainloop)'`` (flush=True)
2. call ``frame._toggle_play()``

Because ``do_pause`` is a local function of ``main`` it cannot be
imported by name; the module-level function object below is recovered
from the nested code object named ``do_pause`` inside
``main.__code__.co_consts`` and rebound as a plain function via
``types.FunctionType`` with the module's globals, supplying the
closure variable (``frame`` per the spec's preconditions) via
``types.CellType``.

Mocked / stand-in dependencies (per the test-generation rules):
``frame`` is a ``mock.Mock`` standing in for the VideoPreviewFrame
instance created in ``main`` with the video loaded (the
``gui/video_player.py`` implementation is out of read scope per the
spec's ``missing`` section); ``frame._toggle_play`` is a mock spy
verifying the single call.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "frame._toggle_play() が例外を送出する（実装が本ファイルにないため未確認）"
  behavior: "例外はコールバックの外へ送出される。その後の処理は tkinter の after コールバック例外処理（report_callback_exception）に委ねられる（未確認）"
"""

import types
from unittest import mock

import scripts.repro_gui_pause_crash as repro
from scripts.repro_gui_pause_crash import main


def _make_do_pause(frame):
    """Recover the nested ``do_pause`` function from
    ``main.__code__.co_consts`` and rebind it with the closure
    variable ``frame``."""
    code = None
    for const in main.__code__.co_consts:
        if isinstance(const, types.CodeType) and const.co_name == "do_pause":
            code = const
            break
    assert code is not None
    assert code.co_freevars == ("frame",)
    frame_cell = types.CellType()
    frame_cell.cell_contents = frame
    fn = types.FunctionType(code, repro.__dict__, "do_pause",
                            None, (frame_cell,))
    return fn


def test_edge_01(capsys):
    """
    input: 映像がロード済みの frame で mainloop 中に1回呼び出されたとき
    expected: stdout に '[repro] PAUSE pressed (mainloop)' が1行現れ、frame._toggle_play() が1回呼び出される（モック・スパイによる検証が可能）
    """
    frame = mock.Mock(name="VideoPreviewFrame")
    do_pause = _make_do_pause(frame)
    do_pause()
    out = capsys.readouterr().out
    assert out == "[repro] PAUSE pressed (mainloop)\n"
    assert frame._toggle_play.call_count == 1
