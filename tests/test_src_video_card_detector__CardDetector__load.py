"""Tests for ``src.video.card_detector.CardDetector._load``.

Specification: docs/00-Architecture/src_video_card_detector__CardDetector__load.yaml

``CardDetector._load`` is an instance method (signature:
``def _load(self, template_dir: str, name: str) -> dict:``) whose purpose is
「テンプレートディレクトリから 1 つのテンプレート画像を読み込み、グレースケール化して、グレースケール画像と幅・高さを dict で返す。」
Behavior (per the spec's ``behavior`` field):

- Step 1: the path is joined as ``f'{template_dir}/{name}'`` (the separator
  is always the slash ``'/'``; neither argument is validated).
- Step 2: the image is loaded in BGR color mode via
  ``cv2.imread(path, cv2.IMREAD_COLOR)``.
- Step 3: if the read result is ``None``, a ``FileNotFoundError`` is raised
  with the message ``'テンプレートが見つかりません: <path>'``.
- Step 4: otherwise the image is converted to grayscale via
  ``cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`` and
  ``{'gray': gray, 'w': int(gray.shape[1]), 'h': int(gray.shape[0])}`` is
  returned. The result dict has only the keys ``'gray'``, ``'w'`` and
  ``'h'``, and the method neither reads nor modifies any attribute of
  ``self`` (per the spec's postconditions).

External dependencies: the method reads a file through ``cv2.imread`` and
builds numpy arrays through ``cv2.cvtColor`` (per the spec's
``side_effects``/``missing`` fields), so the tests mock ``cv2.imread`` and
``cv2.cvtColor`` with ``unittest.mock`` and never touch the filesystem. The
detector instance is created with ``CardDetector.__new__`` (skipping
``__init__``) because the spec's postconditions state that ``_load``
references and modifies no attribute of ``self``.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'cv2.imread が None を返す（ファイル不存在・アクセス不能・デコード不能等）'
  behavior: "FileNotFoundError。メッセージは 'テンプレートが見つかりません: {template_dir}/{name}'。"
"""

from unittest import mock

import cv2

from src.video.card_detector import CardDetector


class _FakeImage:
    """Minimal stand-in for a cv2 array.

    ``_load`` only reads ``.shape`` off the grayscale array that
    ``cv2.cvtColor`` returns (``gray.shape[1]`` / ``gray.shape[0]``), so a
    plain object carrying a ``shape`` tuple is sufficient.
    """

    def __init__(self, shape):
        self.shape = shape


def _make_detector():
    """Return a ``CardDetector`` instance without running ``__init__``.

    The spec's postconditions state that ``_load`` neither references nor
    modifies any attribute of ``self``
    （「self の属性もグローバル状態も変更されない（本メソッドは self の属性を
    参照・変更しない）」）, so the uninitialized instance is a valid receiver
    for the tests below.
    """
    return CardDetector.__new__(CardDetector)


def test_edge_01():
    """
    input: template_dir='tpl', name='missing.png'（ファイルが存在しない）
    expected: FileNotFoundError が送出される。メッセージは 'テンプレートが見つかりません: tpl/missing.png'。
    """
    detector = _make_detector()
    with mock.patch('cv2.imread', return_value=None) as mock_imread, \
            mock.patch('cv2.cvtColor') as mock_cvtColor:
        try:
            detector._load('tpl', 'missing.png')
        except FileNotFoundError as exc:
            message = str(exc)
        else:
            message = None
    # パスは f'{template_dir}/{name}' = 'tpl/missing.png' で、cv2.IMREAD_COLOR で読まれる（behavior 1・2）
    assert mock_imread.call_args == mock.call('tpl/missing.png', cv2.IMREAD_COLOR)
    # imread が None を返したのでグレースケール化には到達しない（behavior 3・4）
    assert mock_cvtColor.call_args is None
    assert message == 'テンプレートが見つかりません: tpl/missing.png'


def test_edge_02():
    """
    input: template_dir='tpl', name='broken.png'（ファイルは存在するが画像としてデコードできない）
    expected: cv2.imread が None を返す場合、同じメッセージの FileNotFoundError が送出される（cv2.imread が None を返す条件は OpenCV の実装次第であり、このファイルでは未検証）。
    """
    detector = _make_detector()
    with mock.patch('cv2.imread', return_value=None) as mock_imread, \
            mock.patch('cv2.cvtColor') as mock_cvtColor:
        try:
            detector._load('tpl', 'broken.png')
        except FileNotFoundError as exc:
            message = str(exc)
        else:
            message = None
    assert mock_imread.call_args == mock.call('tpl/broken.png', cv2.IMREAD_COLOR)
    assert mock_cvtColor.call_args is None
    # 「同じメッセージ」= 'テンプレートが見つかりません: ' + 'tpl/broken.png'
    assert message == 'テンプレートが見つかりません: tpl/broken.png'


def test_edge_03():
    """
    input: template_dir='tpl', name='card.png'（card.png が 幅640x高さ480 の画像ファイル）
    expected: {'gray': shape (480, 640) の ndarray, 'w': 640, 'h': 480} が返る。
    """
    detector = _make_detector()
    bgr = _FakeImage((480, 640, 3))  # 幅640 x 高さ480 の BGR 元画像（ダミー）
    gray = _FakeImage((480, 640))    # グレースケール画像、shape は (h, w)
    with mock.patch('cv2.imread', return_value=bgr) as mock_imread, \
            mock.patch('cv2.cvtColor', return_value=gray) as mock_cvtColor:
        result = detector._load('tpl', 'card.png')
    assert mock_imread.call_args == mock.call('tpl/card.png', cv2.IMREAD_COLOR)
    assert mock_cvtColor.call_args == mock.call(bgr, cv2.COLOR_BGR2GRAY)
    # postconditions: キーは 'gray', 'w', 'h' のみ。'w' = int(gray.shape[1]) = 640、'h' = int(gray.shape[0]) = 480
    assert result == {'gray': gray, 'w': 640, 'h': 480}


def test_edge_04():
    """
    input: template_dir='tpl/'（末尾スラッシュ付き）, name='i_icon.png'
    expected: パスは 'tpl//i_icon.png' になる。ファイルが読込可能なら通常どおりロードされる（OS・cv2 の挙向に依存）。
    """
    detector = _make_detector()
    bgr = _FakeImage((8, 8, 3))
    gray = _FakeImage((8, 8))
    with mock.patch('cv2.imread', return_value=bgr) as mock_imread, \
            mock.patch('cv2.cvtColor', return_value=gray):
        result = detector._load('tpl/', 'i_icon.png')
    # 末尾スラッシュ + 硬結合スラッシュでパスは 'tpl//i_icon.png' になる（behavior 1）
    assert mock_imread.call_args == mock.call('tpl//i_icon.png', cv2.IMREAD_COLOR)
    # 読込可能なら通常どおり dict が返る（w = shape[1] = 8, h = shape[0] = 8）
    assert result == {'gray': gray, 'w': 8, 'h': 8}


def test_edge_05():
    """
    input: template_dir='tpl', name='sub/a.png'（サブディレクトリを含む）
    expected: パスは 'tpl/sub/a.png' になる。ファイルが読込可能なら通常どおりロードされる。
    """
    detector = _make_detector()
    bgr = _FakeImage((16, 32, 3))
    gray = _FakeImage((16, 32))
    with mock.patch('cv2.imread', return_value=bgr) as mock_imread, \
            mock.patch('cv2.cvtColor', return_value=gray):
        result = detector._load('tpl', 'sub/a.png')
    assert mock_imread.call_args == mock.call('tpl/sub/a.png', cv2.IMREAD_COLOR)
    # 読込可能なら通常どおり dict が返る（w = shape[1] = 32, h = shape[0] = 16）
    assert result == {'gray': gray, 'w': 32, 'h': 16}
