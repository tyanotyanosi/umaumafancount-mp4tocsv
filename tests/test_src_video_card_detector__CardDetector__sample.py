"""Tests for ``src.video.card_detector.CardDetector._sample``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__sample.yaml

``CardDetector._sample`` is a static method (per the spec's ``purpose``
field) that reads the point ``(y, x)`` of a template-matching result map
``res`` with bounds checking:

- Step 1: check whether ``y`` is within ``[0, res.shape[0])`` and ``x`` is
  within ``[0, res.shape[1])``
- Step 2: if both are in range, return ``float(res[y, x])``
- Step 3: otherwise return ``0.0``

The method has no external dependencies (no files, GUI, network, time,
randomness, or CV2/OpenCV calls) — the ``res`` argument is a plain 2-D
array (per the spec, a ``cv2.matchTemplate`` result map) that the tests
construct with real ``numpy`` data, so the tests below invoke
``CardDetector._sample`` directly with no mocks.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'res が shape 属性を持たない（None 等）'
  behavior: 'res.shape の参照で AttributeError（未処理で呼び出し側に伝播）'
- condition: '範囲内の res[y, x] の値が float 変換できない（文字列要素等）'
  behavior: 'float() 変換で ValueError / TypeError（未処理で呼び出し側に伝播）'
"""

import numpy as np

from src.video.card_detector import CardDetector


def test_edge_01():
    """
    input: res = np.array([[0.1, 0.5], [0.9, 0.2]]), y = 0, x = 1
    expected: 0.5 が返る
    """
    res = np.array([[0.1, 0.5], [0.9, 0.2]])
    result = CardDetector._sample(res, 0, 1)
    assert result == 0.5


def test_edge_02():
    """
    input: res = np.array([[0.1, 0.5], [0.9, 0.2]]), y = 1, x = 1（最終要素）
    expected: 0.2 が返る
    """
    res = np.array([[0.1, 0.5], [0.9, 0.2]])
    result = CardDetector._sample(res, 1, 1)
    assert result == 0.2


def test_edge_03():
    """
    input: res = np.array([[0.1, 0.5], [0.9, 0.2]]), y = 2, x = 0（行が範囲外）
    expected: 0.0 が返る
    """
    res = np.array([[0.1, 0.5], [0.9, 0.2]])
    result = CardDetector._sample(res, 2, 0)
    assert result == 0.0


def test_edge_04():
    """
    input: res = np.array([[0.1, 0.5], [0.9, 0.2]]), y = 0, x = 2（列が範囲外）
    expected: 0.0 が返る
    """
    res = np.array([[0.1, 0.5], [0.9, 0.2]])
    result = CardDetector._sample(res, 0, 2)
    assert result == 0.0


def test_edge_05():
    """
    input: res = np.array([[0.1, 0.5], [0.9, 0.2]]), y = -1, x = 0（負のインデックス）
    expected: 0.0 が返る（境界チェック 0 <= y が負のインデックスを無効としているため、numpy の負インデックス参照にはならない）
    """
    res = np.array([[0.1, 0.5], [0.9, 0.2]])
    result = CardDetector._sample(res, -1, 0)
    assert result == 0.0
    # 仕様書の expected 記載通り、負の y は numpy の負インデックス参照
    # （res[-1, 0] == 0.9）にはならず 0.0 が返ることを裏付ける。
    assert result != res[-1, 0]


def test_edge_06():
    """
    input: res = np.array([[0.5]]), y = 0, x = 0（numpy スカラー値の場合）
    expected: Python の float 型 0.5 が返る（float(res[0, 0]) の変換）
    """
    res = np.array([[0.5]])
    result = CardDetector._sample(res, 0, 0)
    assert result == 0.5
    # 仕様書の expected 記載通り、Python の float 型（numpy スカラー型で
    # なく float() 変換後の純粋な float）であることを確認する。
    assert type(result) is float
