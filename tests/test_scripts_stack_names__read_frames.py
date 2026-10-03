"""scripts.stack_names.read_frames の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/scripts_stack_names__read_frames.yaml
function: scripts.stack_names.read_frames

検証する行動:
  want の要素を順に反復し、各 idx について cap.set(cv2.CAP_PROP_POS_FRAMES, idx) で seek し
  cap.read() で 1 フレーム読み取る。ret が真なら (idx, f) を yield、偽なら無言スキップする。
  want の全要素を消費して自然終了する。

モックした依存:
  - cap を FakeCap（set/read をスクリプト化）で置換。dict {idx: frame} を持ち、
    set(POS_FRAMES, idx) で現在位置を設定し、read() は idx が dict にあれば (True, frame)、
    なければ (False, None) を返す。

errors セクション（記録のみ・テストしない）:
  - cap.read() が例外を送出する → 例外はジェネレーターから送出されイテレーションが中断される。
"""
from __future__ import annotations

import cv2
import numpy as np

import scripts.stack_names as m


class FakeCap:
    """cv2.VideoCapture のスタンドイン。dict {idx: frame} を持ち、seek 先 idx のフレームを返す。"""

    def __init__(self, frames):
        self._frames = frames
        self._current = None

    def set(self, prop, val):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._current = int(val)
        return True

    def read(self):
        if self._current is None or self._current not in self._frames:
            return False, None
        return True, self._frames[self._current]


def _frame(v=0):
    return np.full((4, 4, 3), v, dtype=np.uint8)


def test_edge_01():
    """input: want が空リスト [] のとき
    expected: 一切 yield されない（ジェネレーターが即終了）
    """
    cap = FakeCap({})
    assert list(m.read_frames(cap, [])) == []


def test_edge_02():
    """input: cap.read() が ret=False を返す idx が混在するとき
    expected: 当該 idx のみ yield されず、他 idx は通常通り (idx, f) として yield され、例外は送出されない
    """
    f10 = _frame(1)
    f30 = _frame(3)
    cap = FakeCap({10: f10, 30: f30})  # idx 20 は読み取り失敗
    result = list(m.read_frames(cap, [10, 20, 30]))
    assert len(result) == 2
    assert result[0][0] == 10
    assert result[0][1] is f10
    assert result[1][0] == 30
    assert result[1][1] is f30


def test_edge_03():
    """input: want = [240, 249, 270, 300, 330, 339, 349, 360, 390, 420, 428, 450, 510, 518, 540] かつ全読み取り成功
    expected: 15回 yield され、順は want と同一、各 f は当該フレーム位置の画像 ndarray
    """
    frames = {idx: _frame(idx % 256) for idx in m.KEEP}
    cap = FakeCap(frames)
    result = list(m.read_frames(cap, m.KEEP))
    assert len(result) == 15
    assert [r[0] for r in result] == m.KEEP
    for idx, f in result:
        assert f is frames[idx]
