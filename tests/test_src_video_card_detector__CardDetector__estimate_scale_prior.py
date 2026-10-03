"""Tests for ``src.video.card_detector.CardDetector._estimate_scale_prior``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__estimate_scale_prior.yaml

``CardDetector._estimate_scale_prior(self, fw)`` computes, from the frame
width ``fw``, the center scale ``s0`` (the scale relative to
``self.reference_width`` at which the card is shown 1:1) and the candidate
scale list ``S`` for multi-scale template matching (Layer A):

- Step 1: if ``self.reference_width`` is truthy and ``> 0``:
  ``s0 = fw / self.reference_width``,
  ``r = np.arange(scale_window_low, scale_window_high + 1e-9, scale_step)``
  and ``S = list(np.unique(np.round(s0 * r, 6)))``.
- Step 2: otherwise (``reference_width`` is 0 or negative):
  ``s0 = 1.0`` and
  ``S = list(np.round(np.arange(0.5, 2.0 + 1e-9, scale_step), 6))``.
- Step 3: ``s0 = round(float(s0), 6)`` (rounded to 6 decimal places).
- Step 4: only if ``s0`` is not already in ``S``:
  ``S = sorted(set(S + [s0]))`` (dedup + ascending order).
- Step 5: return the tuple ``(s0, S)``.

The method is pure computation with no external dependencies (no files,
GUI, network, time, or randomness; ``np.arange`` / ``np.round`` /
``np.unique`` behave per NumPy standard semantics as stated in the spec's
``assumed`` field), so the tests call it directly with no mocks. Each test
constructs a ``CardDetector`` via ``CardDetector.__new__(CardDetector)``
(bypassing ``__init__`` so no settings or I/O is involved at all) and
assigns only the four attributes the spec says are read:
``reference_width``, ``scale_window_low``, ``scale_window_high``,
``scale_step``.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'self.reference_width > 0 かつ fw が除法をサポートしない型（例: str）'
  behavior: 's0 = fw / self.reference_width での TypeError。（else 分では fw は使われないため同型の fw でもエラーにならない。）'
- condition: 'self.scale_window_low / self.scale_window_high / self.scale_step が数値でない（__init__ では float() 変換されるため通常は到達不能）'
  behavior: 'np.arange 呼び出しで TypeError 或いは ValueError。'
"""

import pytest

from src.video.card_detector import CardDetector

# The 31 values of the else-branch default window: 0.5 through 2.0 in
# 0.05 steps, as stated in spec edge_cases #3 and #4
# ("0.5 から 2.0 まで 0.05 刻みの 31 要素リスト（0.5, 0.55, ..., 2.0）").
_ELSE_BRANCH_DEFAULT_S = [
    0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
    1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45,
    1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 2.0,
]


def _make_detector(reference_width, scale_window_low=0.8,
                   scale_window_high=1.3, scale_step=0.05):
    """Build a CardDetector holding exactly the attributes _estimate_scale_prior reads.

    Construction goes through ``CardDetector.__new__(CardDetector)`` (i.e.
    ``__init__`` is bypassed) so the test does not depend on settings or
    I/O. The spec's ``behavior`` section guarantees the method only reads
    ``reference_width``, ``scale_window_low``, ``scale_window_high`` and
    ``scale_step`` from ``self``.
    """
    detector = CardDetector.__new__(CardDetector)
    detector.reference_width = reference_width
    detector.scale_window_low = scale_window_low
    detector.scale_window_high = scale_window_high
    detector.scale_step = scale_step
    return detector


def test_edge_01():
    """
    input: fw=1280、self.reference_width=2560、scale_window_low=0.8、scale_window_high=1.3、scale_step=0.05（全て __init__ デフォルト値）
    expected: s0=0.5。r は {0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3} の 11 値で、S=[0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65]（11 要素、昇順）。s0=0.5 は r=1.0 の項として既に S に含まれるため追加入されない。返り値は (0.5, S)。
    """
    detector = _make_detector(2560, 0.8, 1.3, 0.05)
    s0, S = detector._estimate_scale_prior(1280)
    assert s0 == 0.5
    assert len(S) == 11
    assert S == [0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65]


def test_edge_02():
    """
    input: fw=0、self.reference_width=2560（デフォルトのウィンドウ・ステップ）
    expected: s0=0.0。s0*r が全て 0.0 になるため np.unique 後 S=[0.0]。返り値は (0.0, [0.0])。
    """
    detector = _make_detector(2560)
    s0, S = detector._estimate_scale_prior(0)
    assert s0 == 0.0
    assert S == [0.0]


def test_edge_03():
    """
    input: fw=2560、self.reference_width=0（未設定シミュレーション）
    expected: else 分へ進む。s0=1.0、S は 0.5 から 2.0 まで 0.05 刻みの 31 要素リスト（0.5, 0.55, ..., 2.0）。1.0 は元々含まれるため追加入されない。返り値は (1.0, S)。
    """
    detector = _make_detector(0)
    s0, S = detector._estimate_scale_prior(2560)
    assert s0 == 1.0
    assert len(S) == 31
    assert 1.0 in S
    assert S == _ELSE_BRANCH_DEFAULT_S


def test_edge_04():
    """
    input: fw=1000、self.reference_width=-1
    expected: 負の reference_width も else 分（条件は self.reference_width and self.reference_width > 0）。s0=1.0、S は [0.5, 0.55, ..., 2.0]（0.05 刻み 31 要素）。
    """
    detector = _make_detector(-1)
    s0, S = detector._estimate_scale_prior(1000)
    assert s0 == 1.0
    assert len(S) == 31
    assert S == _ELSE_BRANCH_DEFAULT_S


def test_edge_05():
    """
    input: fw=1280、self.reference_width=2560、scale_step=0（不正な設定）
    expected: np.arange(0.8, 1.3+1e-9, 0) が ZeroDivisionError（float division by zero）を送出し、_estimate_scale_prior は捕捉せず呼び出し側に伝播する。
    """
    detector = _make_detector(2560, 0.8, 1.3, 0)
    with pytest.raises(ZeroDivisionError):
        detector._estimate_scale_prior(1280)


def test_edge_06():
    """
    input: fw=2560、self.reference_width=2560、scale_window_low=1.0、scale_window_high=0.8（逆転したウィンドウ）
    expected: r が空配列のため S=[]。s0=1.0 が追加入され (1.0, [1.0]) を返す。例外は発生しない。
    """
    detector = _make_detector(2560, 1.0, 0.8)
    s0, S = detector._estimate_scale_prior(2560)
    assert s0 == 1.0
    assert S == [1.0]
