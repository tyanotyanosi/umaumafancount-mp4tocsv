"""Tests for ``src.video.card_detector.CardDetector._clamp``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__clamp.yaml

``CardDetector._clamp`` is a static method (per the spec's ``purpose`` field)
that clamps a bounding box ``(x, y, w, h)`` into the frame size ``(fw, fh)``:

- Step 1: ``x = max(0, int(x))``, ``y = max(0, int(y))``
- Step 2: ``w = int(w)``, ``h = int(h)`` (no lower-bound clamp)
- Step 3: if ``x + w > fw`` then ``w = fw - x``
- Step 4: if ``y + h > fh`` then ``h = fh - y``
- Step 5: if ``w <= 0`` or ``h <= 0`` return ``None``
- Step 6: otherwise return ``(x, y, w, h)``

The method has no external dependencies (no files, GUI, network, time,
randomness, or CV2/OpenCV use), so the tests below invoke
``CardDetector._clamp`` directly with no mocks.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'x, y, w, h のいずれかが None'
  behavior: 'int() で TypeError が発生'
- condition: 'x, y, w, h のいずれかが非数値の文字列（例: "abc"）'
  behavior: 'int() で ValueError が発生'
- condition: 'x, y, w, h のいずれかが float("nan")'
  behavior: 'int() で ValueError が発生'
- condition: 'x, y, w, h のいずれかが float("inf") または float("-inf")'
  behavior: 'int() で OverflowError が発生'
- condition: 'fw または fh が非数値（例: str, None）'
  behavior: 'x + w > fw の比較または fw - x の減算で TypeError が発生'
"""

from src.video.card_detector import CardDetector


def test_edge_01():
    """
    input: x=8, y=8, w=10, h=10, fw=100, fh=100
    expected: (8, 8, 10, 10) を返す（変化なし）
    """
    result = CardDetector._clamp(8, 8, 10, 10, 100, 100)
    assert result == (8, 8, 10, 10)


def test_edge_02():
    """
    input: x=95, y=95, w=10, h=10, fw=100, fh=100
    expected: (95, 95, 5, 5) を返す（w, h がフレーム端まで縮む）
    """
    result = CardDetector._clamp(95, 95, 10, 10, 100, 100)
    assert result == (95, 95, 5, 5)


def test_edge_03():
    """
    input: x=-5, y=-3, w=10, h=10, fw=100, fh=100
    expected: (0, 0, 10, 10) を返す（負の x, y は 0 へ切り上げられるのみで、w, h は増加しない）
    """
    result = CardDetector._clamp(-5, -3, 10, 10, 100, 100)
    assert result == (0, 0, 10, 10)


def test_edge_04():
    """
    input: x=100, y=0, w=10, h=10, fw=100, fh=100
    expected: None を返す（x + w = 110 > fw なので w = fw - x = 0）
    """
    result = CardDetector._clamp(100, 0, 10, 10, 100, 100)
    assert result is None


def test_edge_05():
    """
    input: x=150, y=0, w=10, h=10, fw=100, fh=100
    expected: None を返す（x + w = 160 > fw なので w = fw - x = -50）
    """
    result = CardDetector._clamp(150, 0, 10, 10, 100, 100)
    assert result is None


def test_edge_06():
    """
    input: x=0, y=100, w=10, h=10, fw=100, fh=100
    expected: None を返す（y + h = 110 > fh なので h = fh - y = 0）
    """
    result = CardDetector._clamp(0, 100, 10, 10, 100, 100)
    assert result is None


def test_edge_07():
    """
    input: x=0, y=120, w=10, h=10, fw=100, fh=100
    expected: None を返す（y + h = 130 > fh なので h = fh - y = -20）
    """
    result = CardDetector._clamp(0, 120, 10, 10, 100, 100)
    assert result is None


def test_edge_08():
    """
    input: x=2.7, y=-0.4, w=9.9, h=1.2, fw=10, fh=10
    expected: (2, 0, 8, 1) を返す（int() 切り捨て: int(2.7)=2, int(-0.4)=0, int(9.9)=9, int(1.2)=1。次に x + w = 11 > 10 なので w = 10 - 2 = 8）
    """
    result = CardDetector._clamp(2.7, -0.4, 9.9, 1.2, 10, 10)
    assert result == (2, 0, 8, 1)


def test_edge_09():
    """
    input: x=5, y=5, w=0, h=10, fw=100, fh=100
    expected: None を返す（w = 0 で w <= 0）
    """
    result = CardDetector._clamp(5, 5, 0, 10, 100, 100)
    assert result is None


def test_edge_10():
    """
    input: x=0, y=0, w=10, h=10, fw=0, fh=100
    expected: None を返す（x + w = 10 > 0 なので w = fw - x = 0）
    """
    result = CardDetector._clamp(0, 0, 10, 10, 0, 100)
    assert result is None


def test_edge_11():
    """
    input: x="5", y="0", w="10", h="10", fw=100, fh=100
    expected: (5, 0, 10, 10) を返す（数値文字列は int() 変換で受理される）
    """
    result = CardDetector._clamp("5", "0", "10", "10", 100, 100)
    assert result == (5, 0, 10, 10)
