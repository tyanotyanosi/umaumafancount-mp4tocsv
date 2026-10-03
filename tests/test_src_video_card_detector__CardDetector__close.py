"""src.video.card_detector.CardDetector._close のテスト。

仕様書: docs/00-Architecture/src_video_card_detector__CardDetector__close.yaml

purpose: 候補 cands のうち、点 (x, y) から x 方向・y 方向ともに距離が r 以下にある
（境界を含む正方形領域内の）候補が存在するかどうかを判定する。

signature: 'def _close(cands: list, x: int, y: int, r: int) -> bool'

呼び出し方針: 仕様書の signature に self 相当の引数が含まれないため、
クラス属性として未バインドで `CardDetector._close(cands, x, y, r)` を呼び出す。
side_effects が空で外部依存（ファイル・GUI・ネットワーク・時刻・乱数・CV2 等）も
仕様書上に言及がないため、モックは不要。

errors セクション（本ファイルでは文書化のみ、テスト化しない）:
  - condition: cands の要素が 2 要素未満（例 cands=[(10,)]）
    behavior: アンパック時に ValueError
  - condition: cands が反復不可（例 cands=0）
    behavior: 走査開始時に TypeError
  - condition: cx, cy, x, y のいずれかが非数値（例 cands=[('a', 0)]）
    behavior: 引き算または abs() の評価で TypeError
  - condition: r が負
    behavior: 例外は発生せず、比較が真にならないため False を返す
"""

from src.video.card_detector import CardDetector


def test_edge_01():
    """edge_cases[0]
    input: cands=[]、x=0、y=0、r=10
    expected: False
    """
    cands = []
    x = 0
    y = 0
    r = 10
    assert CardDetector._close(cands, x, y, r) is False


def test_edge_02():
    """edge_cases[1]
    input: cands=[(10, 10)]、x=10、y=10、r=0
    expected: True（両座標差 0 が 0 <= 0 を満たす）
    """
    cands = [(10, 10)]
    x = 10
    y = 10
    r = 0
    assert CardDetector._close(cands, x, y, r) is True


def test_edge_03():
    """edge_cases[2]
    input: cands=[(10, 10)]、x=15、y=10、r=5
    expected: True（差がちょうど 5 で、境界は <= により包含される）
    """
    cands = [(10, 10)]
    x = 15
    y = 10
    r = 5
    assert CardDetector._close(cands, x, y, r) is True


def test_edge_04():
    """edge_cases[3]
    input: cands=[(10, 10)]、x=16、y=10、r=5
    expected: False（差 6 が 5 を超える）
    """
    cands = [(10, 10)]
    x = 16
    y = 10
    r = 5
    assert CardDetector._close(cands, x, y, r) is False


def test_edge_05():
    """edge_cases[4]
    input: cands=[(15, 15)]、x=10、y=10、r=5
    expected: True（両座標差 5。対角点も判定対象で、判定領域は正方形であり円ではない）
    """
    cands = [(15, 15)]
    x = 10
    y = 10
    r = 5
    assert CardDetector._close(cands, x, y, r) is True


def test_edge_06():
    """edge_cases[5]
    input: cands=[(10, 10, 0.85)]、x=10、y=10、r=0
    expected: True（3 要素目以降は * _ で無視される）
    """
    cands = [(10, 10, 0.85)]
    x = 10
    y = 10
    r = 0
    assert CardDetector._close(cands, x, y, r) is True


def test_edge_07():
    """edge_cases[6]
    input: cands=[(0, 0)]、x=1、y=1、r=-1
    expected: False（abs(...) >= 0 が -1 <= にはなれない）
    """
    cands = [(0, 0)]
    x = 1
    y = 1
    r = -1
    assert CardDetector._close(cands, x, y, r) is False


def test_edge_08():
    """edge_cases[7]
    input: cands=[(30, 0), (10, 10)]、x=10、y=10、r=5
    expected: True（1 要素目は一致しないが 2 要素目が一致するため）
    """
    cands = [(30, 0), (10, 10)]
    x = 10
    y = 10
    r = 5
    assert CardDetector._close(cands, x, y, r) is True
