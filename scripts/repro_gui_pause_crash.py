# -*- coding: utf-8 -*-
"""GUI「再生→一時停止」クラッシュ再現スクリプト（2026-08-25 バグ検証用）

実GUI（gui.__main__:main）と同じ構成（メインスレッドで mainloop）とし、
  300ms  再生
  2000ms 一時停止
  4000ms 再生（再開）
  6000ms 一時停止
その後 12 秒間クラッシュ監視する。

正常終了: 「mainloop exited cleanly」が出力され、exit code 0。
修正前の再現結果:
  - Assertion fctx->async_lock failed at libavcodec/pthread_frame.c:178
  - プロセス強制終了（exit code 3、Python レースバックなし）
"""
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import tkinter as tk  # noqa: E402
from gui.video_player import VideoPreviewFrame  # noqa: E402

VIDEO = os.path.join(
    REPO_ROOT,
    "inputmov",
    "UmamusumePrettyDerby_Jpn 2026-02-15 15-15-34.mp4",
)


def main():
    root = tk.Tk()
    root.withdraw()

    frame = VideoPreviewFrame(root)
    frame.update()
    frame.load_video(VIDEO)
    print(f"[repro] video loaded: frames={frame.frame_count}, fps={frame.fps}", flush=True)

    def do_play():
        print(f"[repro] PLAY pressed (mainloop)", flush=True)
        frame._toggle_play()

    def do_pause():
        print(f"[repro] PAUSE pressed (mainloop)", flush=True)
        frame._toggle_play()

    def finish():
        print("[repro] observation window elapsed -> exit", flush=True)
        frame.close()
        root.destroy()

    root.after(300, do_play)
    root.after(2000, do_pause)
    root.after(4000, do_play)
    root.after(6000, do_pause)
    root.after(12000, finish)

    t0 = time.time()
    root.mainloop()
    print(f"[repro] mainloop exited cleanly after {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
