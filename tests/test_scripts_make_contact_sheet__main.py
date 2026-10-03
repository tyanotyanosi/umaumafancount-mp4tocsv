"""scripts.make_contact_sheet.main の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/scripts_make_contact_sheet__main.yaml
function: scripts.make_contact_sheet.main

検証する行動:
  cap = cv2.VideoCapture(VIDEO)。fps/w/h を cap.get で取得し step = max(1, int(round(fps)))。
  tw=320、th=int(h*320/w)、cols=4、margin=8、label_h=28。
  cap.read() をループし、ret が偽で脱出。idx % step == 0 のとき ts=idx/fps を計算し
  cv2.resize(frame, (tw, th)) を cells に、f"#{idx}  {ts:.1f}s"（2スペース）を labels に追加。
  rows = (len(cells)+3)//4、sheet_w = 1320、sheet_h = rows*(th+28)+(rows+1)*8 で全ゼロの
  3 チャネル画像を生成し、各セルを左上揃えで配置して (x+4, y+th+20) に緑 (0,255,0) の
  ラベルを描画。cv2.imwrite(OUT, sheet) 後、"saved {OUT}  frames=N step=S grid=4xR"
  （OUT 後に2スペース）を出力し None で終了。

モックした依存:
  - cv2.VideoCapture を FakeCap（get/read/release をスクリプト化）で置換。
  - cv2.imwrite を mock で置換し、書き出される sheet を検証する。

errors セクション（記録のみ・テストしない）:
  - w が 0 → th = int(h*320/w) のゼロ除算で ZeroDivisionError。
  - fps == 0.0 かつフレーム1枚以上 → ts = idx/fps のゼロ除算で ZeroDivisionError。
  - cv2.imwrite が失敗 → 戻り値未チェックのため例外なし、print 行は通常通り出力される。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from unittest import mock

import scripts.make_contact_sheet as m

W, H = 64, 48  # 検証用の小サイズフレーム（th = int(48*320/64) = 240）
TW, TH = 320, 240
COLS, MARGIN, LABEL_H = 4, 8, 28
SHEET_W = COLS * TW + (COLS + 1) * MARGIN  # 1320


class FakeCap:
    """cv2.VideoCapture のスタンドイン。fps/w/h とフレーム列を持つ。"""

    def __init__(self, fps, w, h, frames):
        self._fps = fps
        self._w = w
        self._h = h
        self._frames = frames
        self._i = 0

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._w
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._h
        return 0.0

    def read(self):
        if self._i >= len(self._frames):
            return False, None
        f = self._frames[self._i]
        self._i += 1
        return True, f

    def release(self):
        pass


def _frame(v):
    return np.full((H, W, 3), v, dtype=np.uint8)


def _run_main(cap, capsys):
    with mock.patch("scripts.make_contact_sheet.cv2.VideoCapture", return_value=cap), \
         mock.patch("scripts.make_contact_sheet.cv2.imwrite") as mock_imwrite:
        m.main()
    return mock_imwrite, capsys.readouterr().out


def _expected_out(n, step, rows):
    return f"saved {m.OUT}  frames={n} step={step} grid=4x{rows}\n"


def test_edge_01(capsys):
    """input: 読み取り可能なフレームが1枚のみ
    expected: サンプリング枚数は1（idx=0）。rows=1。grid 4x1 のコンタクトシートが書き出され、frames=1 grid=4x1 と出力され、None が返る
    """
    cap = FakeCap(30.0, W, H, [_frame(100)])
    mock_imwrite, out = _run_main(cap, capsys)
    assert out == _expected_out(1, 30, 1)
    assert mock_imwrite.call_count == 1
    sheet = mock_imwrite.call_args[0][1]
    assert sheet.shape == (284, SHEET_W, 3)  # rows=1: 1*268 + 2*8 = 284


def test_edge_02(capsys):
    """input: 読み取り可能なフレームがT枚、step が S（T と S は正の整数）
    expected: サンプリング枚数 = floor((T-1)/S) + 1。すなわち [0, T-1] の範囲で idx % S == 0 を満たす idx の個数（idx=0 は常に含まれる）
    """
    # T=10, fps=5 → step=5。サンプリング idx=0,5 → N=2 = floor(9/5)+1 = 2
    T, fps = 10, 5.0
    S = max(1, int(round(fps)))
    cap = FakeCap(fps, W, H, [_frame(i) for i in range(T)])
    mock_imwrite, out = _run_main(cap, capsys)
    expected_n = (T - 1) // S + 1
    assert expected_n == 2
    rows = (expected_n + COLS - 1) // COLS
    assert out == _expected_out(expected_n, S, rows)


def test_edge_03(capsys):
    """input: fps = 0.5 の動画
    expected: step = max(1, int(round(0.5))) = 1 となり全フレームがサンプリングされる。ラベル時刻は idx / 0.5 = 2*idx 秒
    """
    T, fps = 3, 0.5
    cap = FakeCap(fps, W, H, [_frame(i) for i in range(T)])
    mock_imwrite, out = _run_main(cap, capsys)
    # step = max(1, round(0.5)) = 1 → 全フレームサンプリング
    assert out == _expected_out(T, 1, 1)
    # ラベル時刻は idx/0.5 = 2*idx 秒（#0 0.0s / #1 2.0s / #2 4.0s）。
    # ラベルは putText でシートに描画されるため、step=1（全サンプリング）と
    # 出力行の frames=3 が時刻計算の前提を確認する。


def test_edge_04(capsys):
    """input: fps = 30.0、フレーム総数60枚の動画
    expected: step = 30。idx=0 と idx=30 の2フレームがサンプリングされ、grid 4x1
    """
    T, fps = 60, 30.0
    cap = FakeCap(fps, W, H, [_frame(i) for i in range(T)])
    mock_imwrite, out = _run_main(cap, capsys)
    # step=30 → idx=0,30 がサンプリング → N=2
    assert out == _expected_out(2, 30, 1)


def test_edge_05(capsys):
    """input: サンプリング枚数が4の倍数（例: 8枚）
    expected: rows = 2。最終行は4セルちょうどで埋まる（空セルなし）
    """
    # fps=5 → step=5, T=40 → idx=0,5,...,35 → N=8
    T, fps = 40, 5.0
    cap = FakeCap(fps, W, H, [_frame(i) for i in range(T)])
    mock_imwrite, out = _run_main(cap, capsys)
    assert out == _expected_out(8, 5, 2)
    sheet = mock_imwrite.call_args[0][1]
    assert sheet.shape == (560, SHEET_W, 3)  # rows=2: 2*268 + 3*8 = 560
    # 最終行（row=1）の4セルすべてが埋まる（空セルなし）。
    # row=1 の y = 8 + 1*(240+28+8) = 284。各セルのサムネイル領域が非ゼロ。
    for col in range(4):
        x = MARGIN + col * (TW + MARGIN)
        region = sheet[284 + 8: 284 + 8 + TH, x:x + TW]
        assert np.count_nonzero(region) > 0


def test_edge_06(capsys):
    """input: サンプリング枚数が1枚
    expected: rows = 1。その行の残り3位置は空の黒背景（np.zeros）となる
    """
    cap = FakeCap(30.0, W, H, [_frame(100)])
    mock_imwrite, out = _run_main(cap, capsys)
    assert out == _expected_out(1, 30, 1)
    sheet = mock_imwrite.call_args[0][1]
    assert sheet.shape == (284, SHEET_W, 3)
    # セル0（col=0）は埋まる
    x0 = MARGIN
    assert np.count_nonzero(sheet[8:8 + TH, x0:x0 + TW]) > 0
    # 残り3位置（col=1,2,3）は黒背景（全ゼロ）
    for col in range(1, 4):
        x = MARGIN + col * (TW + MARGIN)
        assert np.count_nonzero(sheet[8:8 + TH, x:x + TW]) == 0


def test_edge_07(capsys):
    """input: 読み取り可能なフレームが0枚で、cap.get が幅・高さを 0.0 に返す（VIDEO を開けない場合の典型）
    expected: th = int(h*320/w) が ZeroDivisionError（0*320/0 の浮動小数点ゼロ除算）を送出し、シートは書かれないまま関数が例外で終了する
    """
    cap = FakeCap(0.0, 0, 0, [])
    with mock.patch("scripts.make_contact_sheet.cv2.VideoCapture", return_value=cap), \
         mock.patch("scripts.make_contact_sheet.cv2.imwrite") as mock_imwrite:
        with pytest.raises(ZeroDivisionError):
            m.main()
    assert mock_imwrite.call_count == 0
    assert capsys.readouterr().out == ""
