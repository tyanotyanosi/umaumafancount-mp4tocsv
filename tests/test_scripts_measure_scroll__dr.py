"""Tests for ``scripts.measure_scroll.dr``.

Specification: docs/00-Architecture/scripts_measure_scroll__dr.yaml

``dr(a, b)`` computes the element-wise absolute difference of two grayscale
images with ``cv2.absdiff`` and returns the fraction of non-zero elements
(``np.count_nonzero(d) / d.size``), i.e. a ratio in [0, 1] where 0 means all
elements are equal and 1 means all elements differ (per the spec's
``purpose`` and ``behavior``). It performs no validation of ``a`` or ``b``.

Mocked / stand-in dependencies: none — ``dr`` only calls the installed
``cv2`` and ``numpy`` libraries, which are used as real objects (no mocking).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'a と b の shape（サイズ）が一致しない'
  behavior: 'cv2.error が発生し伝播する。'
- condition: 'a と b の dtype（depth）が一致しない'
  behavior: 'cv2.error が発生し伝播する。'
"""

import cv2
import numpy as np
import pytest

from scripts.measure_scroll import dr


def test_edge_01():
    """
    input: a と b が要素ごとに等しい配列（同 shape）
    expected: 0.0 が返る（d の全要素がゼロ）。
    """
    a = np.full((4, 4), 128, dtype=np.uint8)
    b = a.copy()
    assert dr(a, b) == 0.0


def test_edge_02():
    """
    input: a と b が同 shape で全要素が異なる（例: a の全要素が 0、b の全要素が 255）
    expected: 1.0 が返る。
    """
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.full((4, 4), 255, dtype=np.uint8)
    assert dr(a, b) == 1.0


def test_edge_03():
    """
    input: a と b が同 shape で要素の半数だけが異なる
    expected: 異なる要素数÷総要素数の割合が返る（ちょうど半数なら 0.5）。
    """
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.zeros((4, 4), dtype=np.uint8)
    b[:2, :] = 255  # 16 要素のうち上段 8 要素が異なる
    assert dr(a, b) == 0.5


def test_edge_04():
    """
    input: a と b の shape が異なる（例: (100, 100) と (200, 50)）
    expected: cv2.error が発生する（absdiff のサイズ不一致）。
    """
    a = np.zeros((100, 100), dtype=np.uint8)
    b = np.zeros((200, 50), dtype=np.uint8)
    with pytest.raises(cv2.error):
        dr(a, b)


def test_edge_05():
    """
    input: a と b が同 shape で dtype が異なる（例: uint8 と float32）
    expected: cv2.error が発生する（absdiff は同 depth を要求する）。
    """
    a = np.zeros((4, 4), dtype=np.uint8)
    b = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(cv2.error):
        dr(a, b)
