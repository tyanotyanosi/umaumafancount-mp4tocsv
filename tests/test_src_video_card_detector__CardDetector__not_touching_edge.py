"""Tests for ``CardDetector._not_touching_edge``.

仕様書: docs\\00-Architecture\\src_video_card_detector__CardDetector__not_touching_edge.yaml

function: src.video.card_detector.CardDetector._not_touching_edge
signature: def _not_touching_edge(self, badge, fw: int, fh: int) -> bool
purpose: バッジのバウンディングボックスがフレーム端から edge_margin 幅以内には接していない
（完全に内側にある）かを bool で返す（見切り除外の検証）。

behavior（ステップ）:
  - ステップ1 badge の先頭 4 要素を (x, y, w, h) としてデストラクトし、余分な要素を無視する
  - ステップ2 m を self.edge_margin から読む
  - ステップ3 x >= m, y >= m, x + w <= fw - m, y + h <= fh - m がすべて成立すれば True、
    そうでなければ False を返す

postconditions:
  - bool を返す
  - x >= m, y >= m, x + w <= fw - m, y + h <= fh - m がすべて成立する場合のみ True を返す
    （m = self.edge_margin）
  - インスタンス状態・グローバル状態は一切変更しない

仕様書の errors セクション（{condition, behavior}）はテスト化せず、ここに文書化のみ行う:
  - condition: badge が 4 要素未満のシーケンス（例: (8, 8, 10) の 3 要素タプル）
    behavior: ValueError（not enough values to unpack）
  - condition: badge が反復不能な型（None、int など）
    behavior: TypeError が発生
  - condition: fw または fh が非数値（例: str）
    behavior: >= や <= の比較演算で TypeError が発生
  - condition: badge の先頭 4 要素に非数値（例: str）が含まれる
    behavior: >= や <= の比較演算で TypeError が発生
"""

import pytest

from src.video.card_detector import CardDetector


def _make_detector(edge_margin: int = 8):
    """対象関数のみ ``self.edge_margin`` を読むため、それのみ設定した
    ``CardDetector`` インスタンスを生成する。

    対象関数はファイル・CV2・ネットワーク等の外部依存を触らない。
    テンプレート読み込みを含む ``__init__`` を実行せず、``__new__`` で
    インスタンスを生成し、関数が読む ``edge_margin`` 属性のみを設定する。
    """
    detector = CardDetector.__new__(CardDetector)
    detector.edge_margin = edge_margin
    return detector


def test_edge_01():
    """edge_case 1:
    input: 'badge=(8, 8, 10, 10), fw=100, fh=100, self.edge_margin=8'
    expected: 'True を返す'
    """
    detector = _make_detector(8)
    assert detector._not_touching_edge((8, 8, 10, 10), 100, 100) is True


def test_edge_02():
    """edge_case 2:
    input: 'badge=(0, 0, 10, 10), fw=100, fh=100, self.edge_margin=8'
    expected: 'False を返す（x = 0 < 8）'
    """
    detector = _make_detector(8)
    assert detector._not_touching_edge((0, 0, 10, 10), 100, 100) is False


def test_edge_03():
    """edge_case 3:
    input: 'badge=(92, 92, 8, 8), fw=100, fh=100, self.edge_margin=8'
    expected: 'False を返す（x + w = 100 > fw - m = 92）'
    """
    detector = _make_detector(8)
    assert detector._not_touching_edge((92, 92, 8, 8), 100, 100) is False


def test_edge_04():
    """edge_case 4:
    input: 'badge=(90, 8, 2, 10), fw=100, fh=100, self.edge_margin=8'
    expected: 'True を返す（x + w = 92 == fw - m で境界は含む）'
    """
    detector = _make_detector(8)
    assert detector._not_touching_edge((90, 8, 2, 10), 100, 100) is True


def test_edge_05():
    """edge_case 5:
    input: 'badge=(8, 8, 10, 10, 0.9), fw=100, fh=100, self.edge_margin=8'
    expected: 'True を返す（第 5 要素は無視される）'
    """
    detector = _make_detector(8)
    assert detector._not_touching_edge((8, 8, 10, 10, 0.9), 100, 100) is True


def test_edge_06():
    """edge_case 6:
    input: 'badge=(8, 8, 10, 10), fw=15, fh=100, self.edge_margin=8'
    expected: 'False を返す（fw < 2 * m なので、非負の w では x >= m と x + w <= fw - m は同時に成立し得ない）'
    """
    detector = _make_detector(8)
    assert detector._not_touching_edge((8, 8, 10, 10), 15, 100) is False


def test_edge_07():
    """edge_case 7:
    input: 'badge=(8, 8, 10), fw=100, fh=100, self.edge_margin=8'
    expected: 'ValueError が発生（アンパック要素数不足）'
    """
    detector = _make_detector(8)
    with pytest.raises(ValueError) as excinfo:
        detector._not_touching_edge((8, 8, 10), 100, 100)
    assert "not enough values" in str(excinfo.value)


def test_edge_08():
    """edge_case 8:
    input: 'badge=None, fw=100, fh=100, self.edge_margin=8'
    expected: 'TypeError が発生（None は反復不能）'
    """
    detector = _make_detector(8)
    with pytest.raises(TypeError) as excinfo:
        detector._not_touching_edge(None, 100, 100)
    assert "iterable" in str(excinfo.value)
