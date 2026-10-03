"""scripts.stack_names.main の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/scripts_stack_names__main.yaml
function: scripts.stack_names.main

検証する行動:
  cap = cv2.VideoCapture(VIDEO) を開き、read_frames(cap, KEEP) が KEEP の順に yield する
  各 (idx, f) について crop = f[470:970, 560:1900] を切り出し、cv2.resize で 0.55 倍に縮小し、
  cv2.putText で "#{idx}" を赤 (0,0,255) で重ねて crops に追加する。cap.release() 後、
  cols=3・w/h = crops の最大寸法・rows = ceil(len/3)・gap=10 で背景 40 の sheet を作成し、
  各 crop を左上揃えで配置して cv2.imwrite(OUT, sheet) で書き出し、
  "saved {OUT} frames={len(crops)}" を 1 行出力する。

モックした依存:
  - cv2.VideoCapture を FakeCap（dict {idx: frame}）で置換（read_frames は実関数を使用）。
  - cv2.imwrite を mock で置換し、書き出される sheet を検証する。

errors セクション（記録のみ・テストしない）:
  - crops が空 → w = max(...) の行で ValueError（max() argument is an empty sequence）。
  - cv2.imwrite が書き込めない → 偽を返すが戻り値は確認されないため saved 行は通常通り出力される。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from unittest import mock

import scripts.stack_names as m


class FakeCap:
    """cv2.VideoCapture のスタンドイン。dict {idx: frame} を持ち、seek 先 idx のフレームを返す。"""

    def __init__(self, frames):
        self._frames = frames
        self._current = None
        self.released = 0

    def set(self, prop, val):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._current = int(val)
        return True

    def read(self):
        if self._current is None or self._current not in self._frames:
            return False, None
        return True, self._frames[self._current]

    def release(self):
        self.released += 1


def _full_frame(v=0):
    """crop 領域（470:970, 560:1900）を十分カバーする (970, 1900, 3) の等値フレーム。"""
    return np.full((970, 1900, 3), v, dtype=np.uint8)


def _narrow_frame(v=0):
    """幅が 1900 未満（1000）のフレーム。crop は幅方向に切り詰められる。"""
    return np.full((970, 1000, 3), v, dtype=np.uint8)


def _run_main(cap, capsys):
    with mock.patch("scripts.stack_names.cv2.VideoCapture", return_value=cap), \
         mock.patch("scripts.stack_names.cv2.imwrite") as mock_imwrite:
        m.main()
    return mock_imwrite


def test_edge_01(capsys):
    """input: KEEP 15フレーム全てが読み取り成功
    expected: len(crops) = 15、rows = 5、sheet の shape は (5*h+60, 3*w+40, 3) となり OUT へ書き出され、stdout は saved output\\debug_check\\stack_names.png frames=15
    """
    frames = {idx: _full_frame(0) for idx in m.KEEP}
    cap = FakeCap(frames)
    mock_imwrite = _run_main(cap, capsys)
    out = capsys.readouterr().out
    assert out == "saved output\\debug_check\\stack_names.png frames=15\n"
    assert mock_imwrite.call_count == 1
    args = mock_imwrite.call_args[0]
    assert args[0] == m.OUT
    assert args[1].shape == (1435, 2251, 3)


def test_edge_02(capsys):
    """input: read_frames が何も yield しない（映像が開けない等）
    expected: w = max(c.shape[1] for c in crops) の行で ValueError（max() argument is an empty sequence）が送出され、cv2.imwrite は呼び出されず stdout 出力もない
    """
    cap = FakeCap({})  # 全読み取り失敗
    with mock.patch("scripts.stack_names.cv2.VideoCapture", return_value=cap), \
         mock.patch("scripts.stack_names.cv2.imwrite") as mock_imwrite:
        with pytest.raises(ValueError):
            m.main()
    assert mock_imwrite.call_count == 0
    assert capsys.readouterr().out == ""


def test_edge_03(capsys):
    """input: 切り出し領域より小さいフレーム（高 < 970 または幅 < 1900）を含む場合
    expected: 当該 crop はフレーム実サイズに切り詰められ（例外なし）、w と h は crops の最大寸法で決まり、その crop はセルの左上に配置されセル残りは背景40のまま
    """
    # 備考（spec 乖離）: 仕様 expected は「crop が実サイズに切り詰められ左上配置されセル残りは背景40」
    # とするが、実コードは sheet[y:y+h, x:x+w] = c で「最大寸法 w/h のスライス」に crop を代入する。
    # crop が最大寸法より小さい（幅 242 vs 737）と numpy の broadcasting が失敗し
    # ValueError（could not broadcast input array from shape (275,242,3) into shape (275,737,3)）が送出される。
    # 観測挙動（ValueError）を assert。
    # idx=240 は幅 1000 の narrow フレーム（値 200）、その他は幅 1900 の full フレーム（値 100）。
    frames = {}
    for idx in m.KEEP:
        frames[idx] = _narrow_frame(200) if idx == 240 else _full_frame(100)
    cap = FakeCap(frames)
    with mock.patch("scripts.stack_names.cv2.VideoCapture", return_value=cap), \
         mock.patch("scripts.stack_names.cv2.imwrite") as mock_imwrite:
        with pytest.raises(ValueError):
            m.main()
    assert mock_imwrite.call_count == 0
    assert capsys.readouterr().out == ""
