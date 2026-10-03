"""Tests for ``scripts.repro_gui_pause_crash.main``.

Specification: docs/00-Architecture/scripts_repro_gui_pause_crash__main.yaml

``main`` reproduces the GUI "play -> pause" crash with the same
composition as the real GUI (gui.__main__:main): it loads a fixed
video into VideoPreviewFrame, schedules play/pause/play/pause via
tkinter ``after`` at 300/2000/4000/6000 ms, observes for 12 seconds,
then closes and exits:

1. create the root window with ``tk.Tk()`` and hide it with
   ``root.withdraw()``
2. create ``frame = VideoPreviewFrame(root)`` and call
   ``frame.update()``
3. call ``frame.load_video(VIDEO)`` and print
   ``"[repro] video loaded: frames={frame.frame_count},
   fps={frame.fps}"`` (flush=True)
4. define the nested functions do_play, do_pause, finish (specified
   in their own files)
5. schedule via ``root.after``: 300ms do_play, 2000ms do_pause,
   4000ms do_play, 6000ms do_pause, 12000ms finish
6. record ``t0 = time.time()``
7. enter ``root.mainloop()`` and block until the scheduled callbacks
   fire
8. when mainloop returns, print
   ``"[repro] mainloop exited cleanly after {time.time() - t0:.1f}s"``
   (flush=True) and the function ends

Mocked / stand-in dependencies (per the test-generation rules):
``tk.Tk`` is patched (``mock.patch`` on the module's ``tk`` reference)
to return a ``_FakeRoot`` stand-in that records the scheduled
``after`` callbacks and executes them in delay order when
``mainloop`` is called (mimicking the tkinter event loop for this
script); ``VideoPreviewFrame`` is patched with a ``mock.Mock`` whose
``frame_count`` / ``fps`` attributes stand in for the loaded video
(the ``gui/video_player.py`` implementation is out of read scope per
the spec's ``missing`` section). A clean return of ``main()``
corresponds to the spec's "process exit code 0".

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "frame.load_video(VIDEO) が例外を送出する（main 本体で直接呼び出し、実装は未確認）"
  behavior: "例外は main からそのまま送出される（本ファイルでは未捕捉）"
- condition: "mainloop 中に after コールバック（do_play/do_pause/finish）が例外を送出する（_toggle_play や close が送出した場合）"
  behavior: "未確認。本ファイルに try/except はなく、その後の処理は tkinter の after コールバック例外処理（report_callback_exception）に委ねられる"
"""

from unittest import mock

import scripts.repro_gui_pause_crash as repro
from scripts.repro_gui_pause_crash import main


class _FakeRoot:
    """Stand-in for ``tk.Tk()``: records the scheduled ``after``
    callbacks and executes them in delay order when ``mainloop`` is
    called, mimicking the tkinter event loop for this script."""

    def __init__(self):
        self._scheduled = []
        self.destroy_count = 0

    def withdraw(self):
        pass

    def after(self, ms, callback):
        self._scheduled.append((ms, callback))
        return len(self._scheduled)

    def mainloop(self):
        for ms, callback in sorted(self._scheduled):
            callback()

    def destroy(self):
        self.destroy_count += 1


def _run_main():
    """Run ``main()`` with a fake root and a mocked
    VideoPreviewFrame; return (root, frame)."""
    root = _FakeRoot()
    frame = mock.Mock(name="VideoPreviewFrame")
    frame.frame_count = 731
    frame.fps = 27.5
    with mock.patch("scripts.repro_gui_pause_crash.tk.Tk", return_value=root), \
         mock.patch("scripts.repro_gui_pause_crash.VideoPreviewFrame",
                    return_value=frame):
        result = main()
    assert result is None
    return root, frame


def test_edge_01(capsys):
    """
    input: 正常な再現実行（映像がロードでき、クラッシュしない）
    expected: stdout は '[repro] video loaded: frames=<int>, fps=<数値>'、'[repro] PLAY pressed (mainloop)'、'[repro] PAUSE pressed (mainloop)'、'[repro] PLAY pressed (mainloop)'、'[repro] PAUSE pressed (mainloop)'、'[repro] observation window elapsed -> exit'、'[repro] mainloop exited cleanly after <経過秒数>' の順を出力し、プロセスの終了コードは 0
    """
    root, frame = _run_main()
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert lines[:6] == [
        "[repro] video loaded: frames=731, fps=27.5",
        "[repro] PLAY pressed (mainloop)",
        "[repro] PAUSE pressed (mainloop)",
        "[repro] PLAY pressed (mainloop)",
        "[repro] PAUSE pressed (mainloop)",
        "[repro] observation window elapsed -> exit",
    ]
    assert len(lines) == 7
    assert lines[6].startswith("[repro] mainloop exited cleanly after ")
    assert lines[6].endswith("s")
    # The scheduled delays are 300/2000/4000/6000/12000 ms in order.
    assert [ms for ms, _ in root._scheduled] == [300, 2000, 4000, 6000, 12000]
    frame.update.assert_called_once()
    frame.load_video.assert_called_once_with(repro.VIDEO)


def test_edge_02(capsys):
    """
    input: 12000ms に finish コールバックが発火したとき
    expected: 出力 '[repro] observation window elapsed -> exit' 後に frame.close() が1回、root.destroy() が1回呼ばれ、root.mainloop() が return する
    """
    root, frame = _run_main()
    out = capsys.readouterr().out
    assert "[repro] observation window elapsed -> exit" in out.splitlines()
    assert frame.close.call_count == 1
    assert root.destroy_count == 1
    # mainloop returned: main() completed without exception (clean
    # exit).
