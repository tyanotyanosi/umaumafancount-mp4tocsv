"""src.video.card_detector.CardDetector._iou のテスト。

対象関数: src.video.card_detector.CardDetector._iou
シグネチャ: def _iou(b1, b2) -> float
仕様書:   docs/00-Architecture/src_video_card_detector__CardDetector__iou.yaml

仕様書の errors セクション（テスト化せず、この docstring のみで文書化）:
  - b1 または b2 の要素数が 4 未満（例 b1=(0, 0, 10)）:
      インデックス 3 のアクセスで IndexError
  - b1 または b2 がインデックスアクセス不可（例 b1=0）:
      b1[0] のアクセスで TypeError
  - 要素が非数値（例 b1=(0, 0, 10, 'a')）:
      足し算または比較の評価で TypeError
  - a1 + a2 - inter が 0 以下:
      分母が 1e-6 にクランプされ、1e-6 による除算結果を返す
      （ZeroDivisionError は発生しない）

_iou は仕様書に外部依存（ファイル・CV2 等）の言及のない純粋関数であるため、
コンストラクタ（テンプレート画像 4 枚の cv2 読込）を実行せず、
CardDetector.__new__ によるインスタンスに対して実関数を呼び出す。
"""

from src.video.card_detector import CardDetector


def test_edge_01():
    """edge_cases 第1件
    input: b1=(0, 0, 10, 10)、b2=(20, 20, 10, 10)
    expected: 交差なし → 0.0
    """
    detector = CardDetector.__new__(CardDetector)
    result = detector._iou((0, 0, 10, 10), (20, 20, 10, 10))
    assert result == 0.0
    assert isinstance(result, float)


def test_edge_02():
    """edge_cases 第2件
    input: b1=(0, 0, 10, 10)、b2=(10, 0, 10, 10)
    expected: 辺で接するだけ（x2=10 が x1=10 と等しい）→ 0.0
    """
    detector = CardDetector.__new__(CardDetector)
    result = detector._iou((0, 0, 10, 10), (10, 0, 10, 10))
    assert result == 0.0
    assert isinstance(result, float)


def test_edge_03():
    """edge_cases 第3件
    input: b1=(0, 0, 10, 10)、b2=(0, 0, 10, 10)
    expected: 完全一致。inter=100、a1+a2-inter=100 → 1.0
    """
    detector = CardDetector.__new__(CardDetector)
    result = detector._iou((0, 0, 10, 10), (0, 0, 10, 10))
    assert result == 1.0
    assert isinstance(result, float)


def test_edge_04():
    """edge_cases 第4件
    input: b1=(0, 0, 20, 20)、b2=(5, 5, 5, 5)
    expected: b2 が b1 に内包。inter=25、a1+a2-inter=400 → 0.0625
    """
    detector = CardDetector.__new__(CardDetector)
    result = detector._iou((0, 0, 20, 20), (5, 5, 5, 5))
    assert result == 0.0625
    assert isinstance(result, float)


def test_edge_05():
    """edge_cases 第5件
    input: b1=(0, 0, 10, 10)、b2=(3, 3, 4, 4)
    expected: inter=16、a1+a2-inter=100 → 0.16
    """
    detector = CardDetector.__new__(CardDetector)
    result = detector._iou((0, 0, 10, 10), (3, 3, 4, 4))
    assert result == 0.16
    assert isinstance(result, float)


def test_edge_06():
    """edge_cases 第6件
    input: b1=(0, 0, 0, 10)、b2=(0, 0, 10, 10)
    expected: 幅 0 のボックス（x2=0 が x1=0 と等しい）→ 0.0
    """
    detector = CardDetector.__new__(CardDetector)
    result = detector._iou((0, 0, 0, 10), (0, 0, 10, 10))
    assert result == 0.0
    assert isinstance(result, float)


def test_edge_07():
    """edge_cases 第7件
    input: b1=(0, 0, -10, 10)、b2=(5, 0, 10, 10)
    expected: 幅負のボックス（x2 = min(-10, 15) = -10 が x1 = 5 を下回る）→ 0.0
    """
    detector = CardDetector.__new__(CardDetector)
    result = detector._iou((0, 0, -10, 10), (5, 0, 10, 10))
    assert result == 0.0
    assert isinstance(result, float)
