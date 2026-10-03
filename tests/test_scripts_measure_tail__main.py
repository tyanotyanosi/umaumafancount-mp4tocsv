"""scripts.measure_tail.main の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/scripts_measure_tail__main.yaml
function: scripts.measure_tail.main

検証する行動:
  動画を 2 パスで処理する。パス 1 で cv2.VideoCapture(VIDEO) を開き cap.set(CAP_PROP_POS_FRAMES, 540)
  して seek、1 フレーム読み取り（ret は確認せず）→ gray で参照フレーム ref_gray を取得 → release。
  パス 2 で再度開き全フレームをストリーミングし、prev_gray が None でない（2 フレーム目以降）ごとに
  r = dr(g, prev_gray) を計算して last_prev_diff を更新。idx>=500 かつ idx%10==0 で consecutive 行、
  idx>=540 かつ idx%10==0 で rc = dr(g, ref_gray) の cum-from-540 行を出力（>=0.1 で "  <<<" 付加）。
  ループ終了後 release し、last frame 行と参照比較行の 2 行を出力する。fps = 27.517928。

モックした依存:
  - cv2.VideoCapture を FakeCap（read/set/get/release をスクリプト化）で置換。
    mock.patch("scripts.measure_tail.cv2.VideoCapture", side_effect=[cap1, cap2]) で
    パス 1 とパス 2 に別インスタンスを返す（パス 1 は set(POS_FRAMES,540) で 540 に定位）。
  - gray / dr は実 cv2（cv2.cvtColor / cv2.absdiff + np.count_nonzero）を使用。
  - フレームは等灰度 BGR（R=G=B=v）の 10x10 ndarray（grayscale で v になる）。

errors セクション（記録のみ・テストしない）:
  - VIDEO が存在しない/開けない/541 フレーム未満で seek 540 が範囲外 → パス 1 の read が (False, None)、
    ret 未確認のため None が gray()→cv2.cvtColor に渡され cv2 由来の例外が伝播する。
  - パス 2 のフレームが prev_gray/ref_gray と形状が異なる → dr() の cv2.absdiff が例外を送出。
  - パス 2 読み取り中の例外 → cap.release() とサマリ出力がスキップされる（try/finally なし）。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from unittest import mock

import scripts.measure_tail as m

FPS = 27.517928


class FakeCap:
    """cv2.VideoCapture のスタンドイン。read() がスクリプト化された (ret, frame) 列を返し、
    set(CAP_PROP_POS_FRAMES, N) で N 番目に定位する。"""

    def __init__(self, frames, fps=30.0):
        self._frames = frames
        self._i = 0
        self._fps = fps
        self.read_count = 0
        self.released = 0
        self.set_calls = []

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        return 0

    def set(self, prop, val):
        self.set_calls.append((prop, val))
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._i = int(val)
        return True

    def read(self):
        self.read_count += 1
        if self._i >= len(self._frames):
            return False, None
        f = self._frames[self._i]
        self._i += 1
        return True, f

    def release(self):
        self.released += 1


def _mono(v, h=10, w=10):
    """等灰度 BGR フレーム（R=G=B=v）。grayscale 変換でちょうど v になる。"""
    return np.full((h, w, 3), v, dtype=np.uint8)


def _with_255(v, count, start=0, h=10, w=10):
    """等灰度 v のフレームで、row-major の start 番目から count 個のピクセルを 255 にする。"""
    f = np.full((h, w, 3), v, dtype=np.uint8)
    for i in range(start, start + count):
        r, c = divmod(i, w)
        f[r, c] = 255
    return f


def _run_main(frames, capsys):
    cap1 = FakeCap(frames)  # パス 1（set で 540 に定位）
    cap2 = FakeCap(frames)  # パス 2（先頭から順読）
    with mock.patch("scripts.measure_tail.cv2.VideoCapture", side_effect=[cap1, cap2]):
        m.main()
    return cap1, cap2


def test_edge_01(capsys):
    """input: 動画が541フレーム以上ある通常ケース
    expected: idx が {500, 510, 520, ...} の各フレームで consecutive 行を1行ずつ、{540, 550, 560, ...} で cum-from-540 行を1行ずつ出力し、最後に last frame 行と参照比較行の2行を出力して終了する
    """
    # 561 フレーム（idx 0..560）すべて同一（gray 0）。
    frames = [_mono(0) for _ in range(561)]
    _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "  consecutive #500 (18.2s): 0.0000\n" in out
    assert "  consecutive #530 (19.3s): 0.0000\n" in out
    assert "  cum-from-540 #540 (19.6s): 0.0000\n" in out
    assert "  cum-from-540 #550 (20.0s): 0.0000\n" in out
    assert "\nlast frame #560: consecutive diff=0.0000\n" in out
    assert "#540 vs last(#560): 0.0000\n" in out


def test_edge_02(capsys):
    """input: フレームインデックス idx = 540 のフレーム到達時
    expected: 同一フレームに対して consecutive 行（540 >= 500 かつ 540 % 10 == 0）と cum-from-540 行（540 >= 540）の2行が出力される
    """
    # 541 フレーム（idx 0..540）すべて同一。idx=540 で両行が出力される。
    frames = [_mono(0) for _ in range(541)]
    _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "  consecutive #540 (19.6s): 0.0000\n" in out
    assert "  cum-from-540 #540 (19.6s): 0.0000\n" in out


def test_edge_03(capsys):
    """input: idx が 500 以上 539 以下で idx % 10 == 0 のフレーム（500, 510, 520, 530）
    expected: consecutive 行のみ出力され、cum-from-540 行は出力されない
    """
    # 541 フレームすべて同一。500/510/520/530 は consecutive のみ（idx < 540 で cum-from-540 条件を満たさない）。
    frames = [_mono(0) for _ in range(541)]
    _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "  consecutive #500 (18.2s): 0.0000\n" in out
    assert "  consecutive #530 (19.3s): 0.0000\n" in out
    assert "  cum-from-540 #500" not in out
    assert "  cum-from-540 #510" not in out
    assert "  cum-from-540 #520" not in out
    assert "  cum-from-540 #530" not in out


def test_edge_04(capsys):
    """input: idx < 500 のフレーム（例: idx = 10, 20, ...）
    expected: consecutive・cum-from-540 いずれの行も出力されないが、2フレーム目以降は毎回 dr が計算され last_prev_diff が更新される
    """
    # 541 フレーム。idx=10 のフレームのみ直前（idx=9）と 10 ピクセル差（r=0.1）。以降は同一。
    # idx=10 で r=0.1 と計算・last_prev_diff が更新されるが、idx<500 なので行は出力されない。
    # 最終行の last_prev_diff は最終フレーム（idx=540、同一）の 0.0 を反映する。
    frames = [_mono(0) for _ in range(10)]
    frames.append(_with_255(0, 10))  # idx 10
    frames += [_with_255(0, 10) for _ in range(541 - 11)]
    _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "  consecutive #500 (18.2s): 0.0000\n" in out
    assert "  consecutive #4" not in out  # idx < 500 の consecutive 行がない
    assert "  cum-from-540 #4" not in out
    assert "\nlast frame #540: consecutive diff=0.0000\n" in out


def test_edge_05(capsys):
    """input: r >= 0.1 となる consecutive 対象フレーム
    expected: 出力行の末尾にマーカー 2空白+<<< が付加される。r < 0.1 の場合マーカーは付加されない
    """
    # 541 フレーム。idx=500 のフレームが直前（idx=499）と 10 ピクセル差（r=0.1）。以降は同一。
    frames = [_mono(0) for _ in range(500)]
    frames.append(_with_255(0, 10))  # idx 500
    frames += [_with_255(0, 10) for _ in range(541 - 501)]
    _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "  consecutive #500 (18.2s): 0.1000  <<<\n" in out
    assert "  consecutive #510 (18.5s): 0.0000\n" in out
    assert "  consecutive #510 (18.5s): 0.0000  <<<" not in out


def test_edge_06(capsys):
    """input: 動画が1フレームのみ（パス2で最初の読み取り成功、2回目で失敗。パス1の seek 540 は範囲外）
    expected: パス1で cap.read() が (False, None) を返すと推定され（範囲外 seek の後戻りはバックエンド依存で未確認）、ret が確認されないため None の ref_frame が gray() → cv2.cvtColor に渡され cv2 由来の例外が発生して関数が異常終了する
    """
    frames = [_mono(0)]
    with pytest.raises(cv2.error):
        _run_main(frames, capsys)
    assert capsys.readouterr().out == ""


def test_edge_07(capsys):
    """input: 動画が0フレーム（パス2で最初の cap.read() が即 False）
    expected: パス1でも ref_frame が None となり gray(None) で cv2 由来の例外が発生する（cv2 の None 入力挙動は未確認）。仮にパス1が成功しても、最終行の dr(prev_gray, ref_gray) は prev_gray が None のまま cv2.absdiff に渡され例外となる
    """
    with pytest.raises(cv2.error):
        _run_main([], capsys)
    assert capsys.readouterr().out == ""
