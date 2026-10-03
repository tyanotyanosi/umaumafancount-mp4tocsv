"""Tests for ``src.video.card_detector.CardDetector._as_box``.

仕様書: docs/00-Architecture/src_video_card_detector__CardDetector__as_box.yaml

signature: ``def _as_box(self, c, s: float = 1.0) -> tuple``

仕様書の ``errors`` セクション（テスト対象外・文書化のみ）:

- condition: c の要素数が 2 未満（例 c=(100,)）
  behavior: c[1] のアクセスで IndexError
- condition: c がインデックスアクセス不可（例 c=100、c=None）
  behavior: c[0] のアクセスで TypeError
- condition: s が非数値かつ s != 1.0（例 s='0.5'、s=None）
  behavior: 124 * s の評価で TypeError
"""

from src.video.card_detector import CardDetector


def _make_detector() -> CardDetector:
    """__init__ を実行せずに CardDetector のインスタンスを生成する。

    _as_box は self のインスタンス属性を読み取らない（仕様書 postconditions）。
    したがって __init__ が読むテンプレート画像・cv2 環境は不要であり、
    object.__new__ 経由の状態を持たないインスタンスで実関数を呼び出す。
    """
    return CardDetector.__new__(CardDetector)


def test_edge_01():
    """edge_case:
    input: c=(100, 200)、s は省略（デフォルト 1.0）
    expected: (100, 200, 124, 37) を返す
    """
    detector = _make_detector()
    assert detector._as_box((100, 200)) == (100, 200, 124, 37)


def test_edge_02():
    """edge_case:
    input: c=(0, 0, 0.85)、s=0.5
    expected: (0, 0, 62, 18) を返す（int(124*0.5)=62、int(37*0.5)=int(18.5)=18）
    """
    detector = _make_detector()
    assert detector._as_box((0, 0, 0.85), 0.5) == (0, 0, 62, 18)


def test_edge_03():
    """edge_case:
    input: c=(10, 20)、s=1.24
    expected: (10, 20, 153, 45) を返す（int(124*1.24)=153、int(37*1.24)=45）
    """
    detector = _make_detector()
    assert detector._as_box((10, 20), 1.24) == (10, 20, 153, 45)


def test_edge_04():
    """edge_case:
    input: c=(5, 5)、s=2.0
    expected: (5, 5, 248, 74) を返す
    """
    detector = _make_detector()
    assert detector._as_box((5, 5), 2.0) == (5, 5, 248, 74)


def test_edge_05():
    """edge_case:
    input: c=(5, 5)、s=0.0
    expected: (5, 5, 0, 0) を返す
    """
    detector = _make_detector()
    assert detector._as_box((5, 5), 0.0) == (5, 5, 0, 0)


def test_edge_06():
    """edge_case:
    input: c=(5, 5)、s=-1.0
    expected: (5, 5, -124, -37) を返す（負の幅・高さが検証されない）
    """
    detector = _make_detector()
    assert detector._as_box((5, 5), -1.0) == (5, 5, -124, -37)


def test_edge_07():
    """edge_case:
    input: c=(5, 5)、s=True
    expected: True == 1.0 であるため s == 1.0 の分岐に入り (5, 5, 124, 37) を返す
    """
    detector = _make_detector()
    assert detector._as_box((5, 5), True) == (5, 5, 124, 37)
