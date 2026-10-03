"""src.utils.image_processor.ImageProcessor.resize の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/src_utils_image_processor__ImageProcessor_resize.yaml
function: src.utils.image_processor.ImageProcessor.resize

検証する行動:
  cv2.resize(image, (width, height)) を呼び出し、その戻り値（np.ndarray）をそのまま返す。

モックした依存: なし（実 cv2 を使用）。

errors セクション（記録のみ・テストしない）:
  - image / width / height が cv2.resize に受け入れられない組み合わせ →
    cv2.resize が送出する例外がそのまま伝播する（例外種別は cv2 の実装で決まる）。
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.utils.image_processor import ImageProcessor


def test_edge_01():
    """input: image shape (100, 200, 3)、width=50, height=25
    expected: cv2.resize(image, (50, 25)) の戻り値、すなわち高さ25 × 幅50 のリサイズ済み画像を返す（画素値は cv2 の既定の補間方法で決まる）。
    """
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    result = ImageProcessor.resize(img, 50, 25)
    assert result.shape == (25, 50, 3)


def test_edge_02():
    """input: cv2.resize が受け入れない入力（画像として無効な配列、幅・高さの値が不正等）
    expected: cv2.resize が送出する例外がそのまま伝播する（例外の種類・メッセージは cv2 の実装で決まる、読込対象コードで確定できない）。
    """
    with pytest.raises(cv2.error):
        ImageProcessor.resize(None, 50, 25)
