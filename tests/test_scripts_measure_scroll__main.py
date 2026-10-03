"""scripts.measure_scroll.main の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/scripts_measure_scroll__main.yaml
function: scripts.measure_scroll.main

検証する行動:
  VIDEO 定数の動画を読み、インデックス 240〜540（両端含む）の窓内フレームについて
  直前フレーム差分 dr(g, prev) の分布（min/max/avg、閾値 0.10/0.05/0.03/0.02 以上の件数）と、
  「last-kept 比較」（dr(g, last_kept) >= 0.1 で keep、先頭は常に keep）と
  「直前比較」（dr(g, prev) >= 0.1 で keep）の keep 件数・インデックスリストを標準出力に出力する。
  窓外フレームは差分計算も keep 記録も行わないが prev は更新される。

モックした依存:
  - cv2.VideoCapture を FakeCap（read/get/release をスクリプト化）で置換
    （mock.patch("scripts.measure_scroll.cv2.VideoCapture", return_value=cap)）。
  - gray / dr は実 cv2（cv2.cvtColor / cv2.absdiff + np.count_nonzero）を使用。
  - フレームは等灰度 BGR（R=G=B=v）の 10x10 ndarray（grayscale で v になる）。

errors セクション（記録のみ・テストしない）:
  - 動画が開けない、または窓内にフレームがない（241 フレーム未満）→ min(vals) で ValueError。
  - 動画途中の読み取り失敗（ret=False）→ ループは正常終了として扱われる。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from unittest import mock

import scripts.measure_scroll as m


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
    cap = FakeCap(frames)
    with mock.patch("scripts.measure_scroll.cv2.VideoCapture", return_value=cap):
        m.main()
    return cap


def test_edge_01(capsys):
    """input: 動画ファイルが存在しない、または開けない（最初の read で ret = False）
    expected: ループが即終了し、「scroll region frames: 0」の出力の直後に min(vals) で ValueError（空シーケンスの min）が発生する。以降の統計行は出力されない。
    """
    with pytest.raises(ValueError):
        _run_main([], capsys)
    out = capsys.readouterr().out
    assert "scroll region frames: 0\n" in out
    assert "consecutive diff" not in out


def test_edge_02(capsys):
    """input: 全 240 フレーム（インデックス 0〜239）の動画
    expected: 窓内フレームが0件で diffs が空になり、「scroll region frames: 0」の出力の直後に min(vals) で ValueError が発生する。
    """
    frames = [_mono(0) for _ in range(240)]
    with pytest.raises(ValueError):
        _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "scroll region frames: 0\n" in out
    assert "consecutive diff" not in out


def test_edge_03(capsys):
    """input: 241 フレーム以上の動画
    expected: 窓の最初のフレーム（idx = 240）の prev は idx = 239 のフレーム（窓外）なので、窓内全フレームで r が計算され、diffs の件数は窓内フレーム数に等しい（動画が541フレーム以上なら最大220件）。
    """
    # 541 フレーム（idx 0..540）すべて同一（gray 0）。窓 240..540 = 301 フレーム。
    # 備考: spec は「最大220件」とするが、窓 240..540（両端含む）は 301 フレームであり、
    # 実測では diffs 件数 = 301（spec 乖離、下方報告に集計）。
    frames = [_mono(0) for _ in range(541)]
    _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "scroll region frames: 301\n" in out
    assert "consecutive diff: min=0.0000 max=0.0000 avg=0.0000\n" in out
    assert "last-kept kept: 1 -> [240]\n" in out
    assert "prev-frame kept (thr 0.1): 0 -> []\n" in out


def test_edge_04(capsys):
    """input: 541 フレーム未満（例: 300 フレーム）の動画
    expected: 窓は動画長で打ち切られ、diffs は動画の最後のフレームまでしか入らない。統計はその部分集合に対して計算される。
    """
    # 300 フレーム（idx 0..299）すべて同一。窓は 240..299 = 60 フレームで打ち切り。
    frames = [_mono(0) for _ in range(300)]
    _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "scroll region frames: 60\n" in out
    assert "consecutive diff: min=0.0000 max=0.0000 avg=0.0000\n" in out
    assert "last-kept kept: 1 -> [240]\n" in out
    assert "prev-frame kept (thr 0.1): 0 -> []\n" in out


def test_edge_05(capsys):
    """input: 窓内フレームで dr(g, prev) がちょうど 0.1
    expected: 0.1 >= 0.1 が真なので idx が keep_prev に追加される。
    """
    # 541 フレーム。idx<=240 は gray 0、idx>=241 は先頭 10 ピクセル 255。
    # r241 = dr(g241, prev=g240) = 10/100 = 0.1 → keep_prev に 241。
    frames = [_mono(0) for _ in range(241)]
    frames.append(_with_255(0, 10))
    frames += [_with_255(0, 10) for _ in range(541 - 242)]
    _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "prev-frame kept (thr 0.1): 1 -> [241]\n" in out
    assert "  >=0.10: 1\n" in out


def test_edge_06(capsys):
    """input: 窓内フレームで dr(g, last_kept) がちょうど 0.1
    expected: is_diff が真となり、idx が keep_lastkept に追加され、last_kept = g.copy() で更新される。
    """
    # 同上。r241 = dr(g241, last_kept=g240) = 0.1 → is_diff 真 → 241 が keep_lastkept に追加、last_kept 更新。
    frames = [_mono(0) for _ in range(241)]
    frames.append(_with_255(0, 10))
    frames += [_with_255(0, 10) for _ in range(541 - 242)]
    _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "last-kept kept: 2 -> [240, 241]\n" in out


def test_edge_07(capsys):
    """input: 窓の最初のフレーム（idx = 240）で last_kept が None
    expected: 常に is_diff = True となり idx が keep_lastkept に追加される（窓の先頭フレームはこの方式では必ず keep）。
    """
    # 541 フレームすべて同一（gray 128）。窓先頭 240 は last_kept=None → 必ず keep。
    frames = [_mono(128) for _ in range(541)]
    _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "last-kept kept: 1 -> [240]\n" in out


def test_edge_08(capsys):
    """input: 窓内フレームで直前フレームと差分ゼロ（r = 0.0）
    expected: (idx, 0.0) は diffs に追加されるが、keep_prev には追加されない。
    """
    # 541 フレームすべて同一。窓内全フレームで r = 0.0。diffs には (idx, 0.0) が入るが keep_prev は空。
    frames = [_mono(0) for _ in range(541)]
    _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "scroll region frames: 301\n" in out
    assert "consecutive diff: min=0.0000 max=0.0000 avg=0.0000\n" in out
    assert "prev-frame kept (thr 0.1): 0 -> []\n" in out


def test_edge_09(capsys):
    """input: 窓外のフレーム（idx < 240 または idx > 540）
    expected: 差分計算も keep 記録も行われないが、prev はそのフレームの g に更新される（窓の先頭フレームの prev に影響する）。
    """
    # idx 0..238 は gray 0、idx 239 は先頭 10 ピクセル 255（窓外）、idx 240 は同 255、idx 241..540 は gray 0。
    # r240 = dr(g240, prev=g239) = 0.0（prev が窓外で更新された証拠。更新されていれば 0.1 になる）。
    # r241 = dr(g241, prev=g240) = 0.1 → keep_prev = [241]。窓外フレームは記録されない。
    frames = [_mono(0) for _ in range(239)]
    frames.append(_with_255(0, 10))  # idx 239（窓外）
    frames.append(_with_255(0, 10))  # idx 240（窓先頭）
    frames += [_mono(0) for _ in range(541 - 241)]
    _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "scroll region frames: 301\n" in out
    assert "consecutive diff: min=0.0000 max=0.1000 avg=0.0003\n" in out
    assert "  >=0.10: 1\n" in out
    assert "prev-frame kept (thr 0.1): 1 -> [241]\n" in out


def test_edge_10(capsys):
    """input: 窓内フレームで直前フレームと差分あり（r >= 0.1）かつ last_kept と完全一致
    expected: keep_prev には追加されるが、keep_lastkept には追加されず（dr(g, last_kept) = 0.0 < 0.1）、last_kept も更新されない。
    """
    # 備考: spec の入力「last_kept と完全一致（g == last_kept）かつ dr(g, prev) >= 0.1」は、
    # dr の対称性と last_kept の更新ルールから不可能（証明: prev が keep されたなら last_kept==prev となり g==prev で
    # dr(g,prev)=0 となり矛盾。prev が keep されなければ last_kept はその前の keep フレーム g_K で、
    # g_{idx-1} が keep されないのは dr(prev, g_K) < 0.1 のときのみであり、g == g_K なら dr(g, prev) < 0.1 で矛盾）。
    # したがって最接近の実現可能シナリオ（g は last_kept と 0.1 未満の差、prev と 0.1 以上の差）で検証し、
    # spec expected の結果（keep_prev 追加・keep_lastkept 非追加・last_kept 非更新）を assert する（spec 乖離、下方報告に集計）。
    # idx 240: gray 0（keep、last_kept=全0）。idx 241: 9 ピクセル 255（位置 0..8、dr=0.09 < 0.1 で keep されない）。
    # idx 242: 9 ピクセル 255（位置 10..18）。dr(g242, prev=g241) = 18/100 = 0.18 >= 0.1 → keep_prev。
    #          dr(g242, last_kept=全0) = 9/100 = 0.09 < 0.1 → keep_lastkept 非追加、last_kept 非更新。
    frames = [_mono(0) for _ in range(240)]
    frames.append(_mono(0))                      # idx 240
    frames.append(_with_255(0, 9, start=0))      # idx 241
    frames.append(_with_255(0, 9, start=10))     # idx 242
    frames += [_mono(0) for _ in range(541 - 243)]
    _run_main(frames, capsys)
    out = capsys.readouterr().out
    assert "prev-frame kept (thr 0.1): 1 -> [242]\n" in out
    assert "last-kept kept: 1 -> [240]\n" in out
    assert "  >=0.10: 1\n" in out
    assert "  >=0.05: 3\n" in out
