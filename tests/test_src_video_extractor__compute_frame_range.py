"""Tests for ``src.video.extractor.compute_frame_range``.

Specification: docs/00-Architecture/src_video_extractor__compute_frame_range.yaml

``compute_frame_range`` computes the (start_frame, stop_frame) int
tuple of frames to process from second-based parameters (start, end,
limit):

- if ``total_frames <= 0``, raise ValueError ("動画にフレームがありません")
- convert ``fps`` to float and clamp negative values to 0.0
- ``start_frame = max(0, int(round(start_sec * fps)))``
- if ``end_sec > 0``: ``end_frame = int(round(end_sec * fps))``;
  otherwise ``end_frame = total_frames - 1``
- if ``limit_sec > 0``:
  ``stop = min(end_frame, start_frame + int(round(limit_sec * fps)))``;
  otherwise ``stop = end_frame``
- ``stop = max(0, stop)`` (effectively a no-op)
- if ``stop >= total_frames``, clamp ``stop = total_frames - 1``
- if ``start_frame >= total_frames``, raise ValueError
  ("開始位置が動画長を超過します: start=<start_sec>s（動画=<total_frames>
  フレーム）")
- if ``start_frame > stop``, raise ValueError
  ("空のフレーム範囲です: start=<start_sec>s, end=<end_sec>s,
  limit=<limit_sec>s")
- return ``(start_frame, stop)``

``round`` is the Python builtin, so half values are rounded to the
even side (banker's rounding).

Mocked / stand-in dependencies (per the test-generation rules):
none — the function is pure and is called directly with the spec
edge-case inputs.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: 引数が数値型でない（例 fps に list、total_frames に str、start_sec に str）
  behavior: float() 変換・<= 比較・乗算の段階で TypeError が送出される（関数側でキャッチしない）
"""

import pytest

from src.video.extractor import compute_frame_range


def test_edge_01():
    """
    input: compute_frame_range(30.0, 0)
    expected: 'ValueError が送出される（メッセージは「動画にフレームがありません」）'
    """
    with pytest.raises(ValueError) as excinfo:
        compute_frame_range(30.0, 0)
    assert "動画にフレームがありません" in str(excinfo.value)


def test_edge_02():
    """
    input: compute_frame_range(30.0, -3)
    expected: ValueError が送出される（total_frames <= 0 判定）
    """
    with pytest.raises(ValueError) as excinfo:
        compute_frame_range(30.0, -3)
    assert "動画にフレームがありません" in str(excinfo.value)


def test_edge_03():
    """
    input: compute_frame_range(30.0, 900)
    expected: (0, 899) を返す
    """
    assert compute_frame_range(30.0, 900) == (0, 899)


def test_edge_04():
    """
    input: compute_frame_range(30.0, 900, start_sec=1.5)
    expected: (45, 899) を返す
    """
    assert compute_frame_range(30.0, 900, start_sec=1.5) == (45, 899)


def test_edge_05():
    """
    input: compute_frame_range(30.0, 900, end_sec=2.0)
    expected: (0, 60) を返す
    """
    assert compute_frame_range(30.0, 900, end_sec=2.0) == (0, 60)


def test_edge_06():
    """
    input: compute_frame_range(-10.0, 900)
    expected: fps は 0.0 にクランプされ (0, 899) を返す
    """
    assert compute_frame_range(-10.0, 900) == (0, 899)


def test_edge_07():
    """
    input: compute_frame_range(30.0, 900, start_sec=-5.0)
    expected: (0, 899) を返す
    """
    assert compute_frame_range(30.0, 900, start_sec=-5.0) == (0, 899)


def test_edge_08():
    """
    input: compute_frame_range(30.0, 900, start_sec=1.0, limit_sec=2.0)
    expected: (30, 90) を返す
    """
    assert compute_frame_range(30.0, 900, start_sec=1.0, limit_sec=2.0) == (30, 90)


def test_edge_09():
    """
    input: compute_frame_range(30.0, 900, start_sec=1.0, limit_sec=-1.0)
    expected: limit_sec <= 0 のため無制限扱いで (30, 899) を返す
    """
    assert compute_frame_range(30.0, 900, start_sec=1.0, limit_sec=-1.0) == (30, 899)


def test_edge_10():
    """
    input: compute_frame_range(30.0, 900, end_sec=-1.0)
    expected: end_sec <= 0 のため最後まで扱いで (0, 899) を返す
    """
    assert compute_frame_range(30.0, 900, end_sec=-1.0) == (0, 899)


def test_edge_11():
    """
    input: compute_frame_range(30.0, 900, start_sec=40.0)
    expected: 'ValueError が送出される（メッセージは「開始位置が動画長を超過します: start=40.0s（動画=900 フレーム）」）'
    """
    with pytest.raises(ValueError) as excinfo:
        compute_frame_range(30.0, 900, start_sec=40.0)
    assert ("開始位置が動画長を超過します: start=40.0s"
            "（動画=900 フレーム）") in str(excinfo.value)


def test_edge_12():
    """
    input: compute_frame_range(30.0, 900, start_sec=30.0)
    expected: start_frame=900 となり start_frame >= total_frames 判定で ValueError が送出される
    """
    with pytest.raises(ValueError) as excinfo:
        compute_frame_range(30.0, 900, start_sec=30.0)
    assert "開始位置が動画長を超過します" in str(excinfo.value)


def test_edge_13():
    """
    input: compute_frame_range(30.0, 900, start_sec=2.0, end_sec=1.0)
    expected: 'ValueError が送出される（メッセージは「空のフレーム範囲です: start=2.0s, end=1.0s, limit=0.0s」）'
    """
    with pytest.raises(ValueError) as excinfo:
        compute_frame_range(30.0, 900, start_sec=2.0, end_sec=1.0)
    assert ("空のフレーム範囲です: start=2.0s, end=1.0s, "
            "limit=0.0s") in str(excinfo.value)


def test_edge_14():
    """
    input: compute_frame_range(30.0, 900, end_sec=100.0)
    expected: stop は 3000 から 899 にクランプされ (0, 899) を返す
    """
    assert compute_frame_range(30.0, 900, end_sec=100.0) == (0, 899)


def test_edge_15():
    """
    input: compute_frame_range(2.0, 100, start_sec=0.25)
    expected: 0.25*2=0.5 は banker's rounding で 0 に丸められ (0, 99) を返す
    """
    assert compute_frame_range(2.0, 100, start_sec=0.25) == (0, 99)


def test_edge_16():
    """
    input: compute_frame_range(10.0, 100, end_sec=0.04)
    expected: 0.04*10=0.4 は round で 0 となり (0, 0) を返す
    """
    assert compute_frame_range(10.0, 100, end_sec=0.04) == (0, 0)


def test_edge_17():
    """
    input: compute_frame_range(0.0, 900, start_sec=5.0, end_sec=10.0)
    expected: fps=0 のため start_frame=0、end_frame=0 となり (0, 0) を返す
    """
    assert compute_frame_range(0.0, 900, start_sec=5.0, end_sec=10.0) == (0, 0)


def test_edge_18():
    """
    input: compute_frame_range(30.0, 1)
    expected: (0, 0) を返す
    """
    assert compute_frame_range(30.0, 1) == (0, 0)
