"""src.utils.image_processor.ImageProcessor.crop_region の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/src_utils_image_processor__ImageProcessor_crop_region.yaml
function: src.utils.image_processor.ImageProcessor.crop_region

検証する行動:
  image.shape[:2] から img_height（高さ）・img_width（幅）を抽出。
  use_percent=False なら region=(x, y, w, h) をアンパックし、
    x = max(0, min(int(x), img_width)), y = max(0, min(int(y), img_height)),
    w = max(0, min(int(w), img_width - x)), h = max(0, min(int(h), img_height - y))
    で image[y:y+h, x:x+w] を返す。
  use_percent=True なら region=(left%, top%, right%, bottom%) を
    left = int(region[0]/100*img_width) 等に変換し、各値を [0, 上限] にクランプして
    image[top:bottom, left:right] を返す。
  範囲外の値はクランプされクラッシュしない（ゼロ行・ゼロ列の空配列になり得る）。

モックした依存: なし（純粋な numpy スライス処理）。
検証用画像は pixel (r, c) = r * img_width + c の 2 次元 int64 配列で、
クロップ後の値・形状が一意に検証できる。

errors セクション（記録のみ・テストしない）:
  - use_percent=False で region が 4 要素でない → ValueError / TypeError。
  - use_percent=True で region が 4 要素未満 → IndexError。
  - image に shape 属性がない → AttributeError。
  - image が 1 次元配列 → ValueError。
  - region の要素が数値でない → TypeError。
"""
from __future__ import annotations

import numpy as np

from src.utils.image_processor import ImageProcessor


def _img(h, w):
    """pixel (r, c) = r * w + c の 2 次元 int64 画像（各ピクセルが一意）。"""
    return np.arange(h * w, dtype=np.int64).reshape(h, w)


def test_edge_01():
    """input: image shape (100, 200)（高100, 幅200）、use_percent=False、region=(50, 20, 40, 30)
    expected: ビュー image[20:50, 50:90]（高さ30 × 幅40 の領域）を返す。
    """
    img = _img(100, 200)
    result = ImageProcessor.crop_region(img, (50, 20, 40, 30))
    assert result.shape == (30, 40)
    assert result[0, 0] == img[20, 50] == 20 * 200 + 50
    assert result[-1, -1] == img[49, 89] == 49 * 200 + 89


def test_edge_02():
    """input: image shape (100, 200)、use_percent=False、region=(300, 0, 50, 10)（x が幅超過）
    expected: x は 200 に、w は min(50, 200-200)=0 にクランプされ、image[0:10, 200:200]（ゼロ列の空配列）を返す。例外は発生しない。
    """
    img = _img(100, 200)
    result = ImageProcessor.crop_region(img, (300, 0, 50, 10))
    assert result.shape == (10, 0)
    assert result.size == 0


def test_edge_03():
    """input: image shape (100, 200)、use_percent=False、region=(-10, -5, 40, 30)（負の値）
    expected: x=0, y=0, w=min(40, 200-0)=40, h=min(30, 100-0)=30 となり、image[0:30, 0:40] を返す。例外は発生しない。
    """
    img = _img(100, 200)
    result = ImageProcessor.crop_region(img, (-10, -5, 40, 30))
    assert result.shape == (30, 40)
    assert result[0, 0] == img[0, 0] == 0
    assert result[-1, -1] == img[29, 39] == 29 * 200 + 39


def test_edge_04():
    """input: image shape (100, 200)、use_percent=True、region=(0, 0, 100, 100)
    expected: left=0, top=0, right=200, bottom=100 となり、image[0:100, 0:200]（画像全体）を返す。
    """
    img = _img(100, 200)
    result = ImageProcessor.crop_region(img, (0, 0, 100, 100), use_percent=True)
    assert result.shape == (100, 200)
    assert result[0, 0] == img[0, 0]
    assert result[-1, -1] == img[99, 199]


def test_edge_05():
    """input: image shape (100, 200)、use_percent=True、region=(-5, 0, 105, 100)（負 / 100% 超過）
    expected: left=0, top=0, right=200, bottom=100 にクランプされ、画像全体を返す。クラッシュしない。
    """
    img = _img(100, 200)
    result = ImageProcessor.crop_region(img, (-5, 0, 105, 100), use_percent=True)
    assert result.shape == (100, 200)
    assert result[0, 0] == img[0, 0]
    assert result[-1, -1] == img[99, 199]


def test_edge_06():
    """input: image shape (100, 200)、use_percent=True、region=(50, 0, 30, 100)（変換後 right < left）
    expected: left=100, right=60 となり、image[0:100, 100:60]（ゼロ列の空配列）を返す。例外も座標の入れ替えも行われない。
    """
    img = _img(100, 200)
    result = ImageProcessor.crop_region(img, (50, 0, 30, 100), use_percent=True)
    assert result.shape == (100, 0)
    assert result.size == 0


def test_edge_07():
    """input: image shape (100, 200)、use_percent=False、region=(10.9, 0.2, 100.7, 50.9)（float 要素）
    expected: 各要素が int() で切り捨てられ（10, 0, 100, 50）、x=10, y=0, w=min(100, 190)=100, h=min(50, 100)=50 となり、image[0:50, 10:110] を返す。
    """
    img = _img(100, 200)
    result = ImageProcessor.crop_region(img, (10.9, 0.2, 100.7, 50.9))
    assert result.shape == (50, 100)
    assert result[0, 0] == img[0, 10] == 10
    assert result[-1, -1] == img[49, 109] == 49 * 200 + 109
