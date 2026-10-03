"""scripts.measure_shift.main の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/scripts_measure_shift__main.yaml
function: scripts.measure_shift.main

検証する行動:
  VIDEO 動画を読み、リスト領域（行 Y0:Y1=470:960、列 X0:X1=580:1500）のグレースケールクロップ
  crop_gray(f) について、フレーム 240〜540 で直前クロップからの垂直シフト shift(prev_crop, c) を
  測定し、measured shifts 件数、区間 (early 240-300 / mid 300-450 / burst 450-540) ごとの
  total・avg/frame・per 28f(1s)、およびバースト区間 (440-540) のフレーム毎シフト（s>5 で "  <<<" 付加）
  を標準出力に表示する。fps はローカル定数 27.517928。

モックした依存:
  - cv2.VideoCapture を FakeCap（read/get/release をスクリプト化）で置換
    （mock.patch("scripts.measure_shift.cv2.VideoCapture", return_value=cap)）。
  - crop_gray は実 cv2（cv2.cvtColor）を使用。フレームは等灰度 BGR（R=G=B=v）の 960x1500 ndarray。
  - shift は main の集計・出力ロジックを単独で検証するため mock.patch.object(m, "shift", side_effect=...)
    で置換し、idx ごとに制御されたシフト値を返す（edge_04/05/06）。
    備考（spec 乖離、下方報告に集計）: 実 cv2 5.0.0 では cv2.phaseCorrelate が「最大 3 引数」しか
    受け付けず、shift() の 4 引数呼び出し（a, b, hanning(W), hanning(H)）は常に cv2.error を送出し
    例外捕捉で None を返す。すなわち実 shift() は常に None を返すため、非 None のシフト値を扱う
    edge_04/05/06 はモックによる検証となる（edge_01/02/03 は実 shift の全 None 挙動でそのまま検証可能）。

errors セクション（記録のみ・テストしない）:
  - フレーム読み込みループ中の例外 → 呼び出し側に伝播、cap.release() 未実行。
  - VIDEO が存在しない/開けない → 例外なし、ゼロ件結果とバーストヘッダーを出力して正常終了。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from unittest import mock

import scripts.measure_shift as m

FPS = 27.517928


class FakeCap:
    """cv2.VideoCapture のスタンドイン。read() がスクリプト化された (ret, frame) 列を返す。"""

    def __init__(self, frames, fps=30.0):
        self._frames = frames
        self._i = 0
        self._fps = fps
        self.read_count = 0
        self.released = 0

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        return 0

    def read(self):
        self.read_count += 1
        if self._i >= len(self._frames):
            return False, None
        f = self._frames[self._i]
        self._i += 1
        return True, f

    def release(self):
        self.released += 1


def _full_frame(v=128, h=960, w=1500):
    """等灰度 BGR フレーム（R=G=B=v）。960x1500 でクロップ領域 470:960, 580:1500 が完全に取れる。"""
    return np.full((h, w, 3), v, dtype=np.uint8)


def _run_main(frames, capsys):
    cap = FakeCap(frames)
    with mock.patch("scripts.measure_shift.cv2.VideoCapture", return_value=cap):
        m.main()
    return cap


def _mock_shift(values):
    """m.shift を idx -> s で制御する side_effect を返す。

    main のループでは shift が idx=240,241,...,540 の順に呼ばれるため、呼び出し順から idx を復元し、
    values（dict: idx -> s）から値を返す。values にない idx は None。
    """
    state = {"calls": 0}

    def _fake(a, b):
        state["calls"] += 1
        idx = 240 + (state["calls"] - 1)
        return values.get(idx)

    return _fake


def _perframe_none(i):
    return f"  #{i} ({i / FPS:.1f}s): None"


def test_edge_01(capsys):
    """input: VIDEO パスのファイルが存在しない、または開けない
    expected: 例外は発生しない（推定：cap.read() が即座に ret False を返す）。最初の読み取りでループを離脱し、stdout は measured shifts: 0 の行と（先行改行付きの）ヘッダー行 burst region (440-540) per-frame shift: のみで、それ以外は出力されない。None を返す
    """
    cap = _run_main([], capsys)
    out = capsys.readouterr().out
    assert out == "measured shifts: 0\n\nburst region (440-540) per-frame shift:\n"
    assert cap.released == 1


def test_edge_02(capsys):
    """input: フレーム数が 240 以下（例：100 フレーム）の動画
    expected: 測定条件 240 <= idx <= 540 が 1 度も満たされず shifts == {} となるため、measured shifts: 0 とバーストヘッダー行のみを出力し、区間統計行もフレーム毎行も出力されない。None を返す
    """
    frames = [_full_frame() for _ in range(100)]
    _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert out == "measured shifts: 0\n\nburst region (440-540) per-frame shift:\n"


def test_edge_03(capsys):
    """input: 541 フレーム以上あり、全 shift() 呼び出しが None を返す動画
    expected: shifts はキー 240〜540 の 301 件、valid は空となり measured shifts: 0 とバーストヘッダー行を出力し、#440〜#540 の 101 行がそれぞれ None（"  <<<" なし）で出力される。None を返す
    """
    # 541 フレームすべて同一（等灰度）。実 shift() は cv2 5.0.0 で常に None を返す（4 引数呼び出しが
    # cv2.error を送出し例外捕捉される）。したがってモック不要で全 None を再現できる。
    frames = [_full_frame() for _ in range(541)]
    _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "measured shifts: 0\n" in out
    assert "\nburst region (440-540) per-frame shift:\n" in out
    assert "[early]" not in out
    assert "[mid]" not in out
    assert "[burst] frames" not in out
    assert _perframe_none(440) + "\n" in out
    assert _perframe_none(540) + "\n" in out
    assert "  <<<" not in out


def test_edge_04(capsys):
    """input: 541 フレーム以上で idx = 500 の shift が s = 6.0 になる動画
    expected: #500 の行は末尾に "  <<<" を付加される（s が None でなく s > 5）。s = 6.0 は measured shifts 件数にも burst 区間（450 <= 500 < 540）の total にも含まれる
    """
    frames = [_full_frame() for _ in range(541)]
    with mock.patch.object(m, "shift", side_effect=_mock_shift({500: 6.0})):
        _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "measured shifts: 1\n" in out
    assert "[burst] frames 450-540: total shift=6.0px, avg/frame=0.067px/f, per 28f(1s)=1.9px\n" in out
    assert "  #500 (18.2s): 6.0  <<<\n" in out


def test_edge_05(capsys):
    """input: 541 フレーム以上で idx = 500 の shift が s = 5.0 になる動画
    expected: #500 の行は末尾に "  <<<" を付加されない（条件は厳密な不等式 s > 5 であり、s = 5.0 は該当しない）
    """
    frames = [_full_frame() for _ in range(541)]
    with mock.patch.object(m, "shift", side_effect=_mock_shift({500: 5.0})):
        _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "measured shifts: 1\n" in out
    assert "[burst] frames 450-540: total shift=5.0px, avg/frame=0.056px/f, per 28f(1s)=1.6px\n" in out
    assert "  #500 (18.2s): 5.0\n" in out
    assert "  #500 (18.2s): 5.0  <<<" not in out


def test_edge_06(capsys):
    """input: 541 フレーム以上で idx = 540 の shift が None 以外になる動画
    expected: idx = 540 は測定される（240 <= 540 <= 540）かつバーストのフレーム毎セクションに出力されるが、どの区間統計の total にも含まれない（全区間は lo <= i < hi を用い hi が最大 540 のため i = 540 は範囲外）
    """
    frames = [_full_frame() for _ in range(541)]
    with mock.patch.object(m, "shift", side_effect=_mock_shift({540: 6.0})):
        _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "measured shifts: 1\n" in out
    assert "[early]" not in out
    assert "[mid]" not in out
    assert "[burst] frames" not in out
    assert "  #540 (19.6s): 6.0  <<<\n" in out
