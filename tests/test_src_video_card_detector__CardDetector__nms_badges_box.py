"""Tests for ``src.video.card_detector.CardDetector._nms_badges.box``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__nms_badges_box.yaml

対象は ``CardDetector._nms_badges`` 内で定義されるネスト関数（クロージャ）
``def box(c):`` である。仕様書の ``purpose`` / ``behavior`` 欄に基づき、
``box`` はバッジ候補 ``c`` を、候補のスケール ``c[2]`` を用いたバッジボックス
``(x, y, w, h)`` への変換を行う関数であり、その契約は以下の通りである:

- ``box(c)`` は ``self._as_box(c, c[2])`` を呼び出し、その結果
  （長さ 4 のタプル ``(c[0], c[1], w, h)``）をそのまま返す。
- ``w`` は ``c[2]`` が 1.0 と等しいとき 124、それ以外で ``int(124 * c[2])``。
- ``h`` は ``c[2]`` が 1.0 と等しいとき 37、それ以外で ``int(37 * c[2])``。
- ``x``, ``y`` は ``c[0]``, ``c[1]`` そのもの（``c[2]`` でスケールされない）。

``box`` はメソッド内部のネスト関数であるためテストから直接参照できないため、
本テストはネスト関数 ``box`` の文書化契約（``box(c) == self._as_box(c, c[2])``）
を、仕様書が規定する呼び出しパス ``self._as_box(c, c[2])`` 経由で検証している。
``_as_box`` は ``CardDetector`` の実メソッドとして存在し、仕様書の
``side_effects`` 欄に基づき計算のみで I/O・状態変更をしないため、
モックは不要である。また ``_as_box`` は self 属性を参照しないと仕様書が規定
しているため、インスタンスは ``__init__`` を実行せずに
``CardDetector.__new__(CardDetector)`` で生成する。

``errors`` セクション（仕様書の ``errors`` 欄を原文で引用。ドキュメントのみで
テスト対象外）:

- condition: 'c の長さが 3 未満'
  behavior: 'IndexError'
- condition: 'c[2] が数値でない（例: None）'
  behavior: '_as_box 内の 124 * s の計算時に TypeError が送出される'
- condition: 'c[2] が数値の文字列（例: "2.0"）'
  behavior: '_as_box 内の int(124 * s) が ValueError となる（s != 1.0 の分岐で 124 * s が文字列反復となり int 変換できないため）'
"""

from src.video.card_detector import CardDetector


def test_edge_01():
    """
    input: c = [10, 20, 1.0, 0.9, 0.1]
    expected: (10, 20, 124, 37) を返す（s = 1.0 なので基準サイズのまま）
    """
    det = CardDetector.__new__(CardDetector)
    c = [10, 20, 1.0, 0.9, 0.1]
    result = det._as_box(c, c[2])
    assert result == (10, 20, 124, 37)


def test_edge_02():
    """
    input: c = [10, 20, 2.0, 0.9, 0.1]
    expected: (10, 20, 248, 74) を返す
    """
    det = CardDetector.__new__(CardDetector)
    c = [10, 20, 2.0, 0.9, 0.1]
    result = det._as_box(c, c[2])
    assert result == (10, 20, 248, 74)


def test_edge_03():
    """
    input: c = [10, 20, 0.5, 0.9, 0.1]
    expected: (10, 20, 62, 18) を返す（int(124*0.5) = 62, int(37*0.5) = 18）
    """
    det = CardDetector.__new__(CardDetector)
    c = [10, 20, 0.5, 0.9, 0.1]
    result = det._as_box(c, c[2])
    assert result == (10, 20, 62, 18)


def test_edge_04():
    """
    input: c = [10, 20, 0.85, 0.9, 0.1]
    expected: (10, 20, 105, 31) を返す（int(124*0.85) = 105, int(37*0.85) = 31。int は切り捨て）
    """
    det = CardDetector.__new__(CardDetector)
    c = [10, 20, 0.85, 0.9, 0.1]
    result = det._as_box(c, c[2])
    assert result == (10, 20, 105, 31)


def test_edge_05():
    """
    input: c = [10, 20]（長さが 2）
    expected: c[2] アクセス時に IndexError が送出される
    """
    det = CardDetector.__new__(CardDetector)
    c = [10, 20]
    raised = False
    try:
        det._as_box(c, c[2])
    except IndexError:
        raised = True
    assert raised
