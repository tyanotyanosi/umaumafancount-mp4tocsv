"""src.ocr.base.OCRBase.recognize の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/src_ocr_base__OCRBase_recognize.yaml
function: src.ocr.base.OCRBase.recognize

検証する行動:
  @abstractmethod により抽象メソッドとして定義され、本体は pass のみ。
  recognize を実装していないサブクラスをインスタンス化すると TypeError が送出される
  （インスタンス化時であり、メソッド呼び出し時ではない）。
  未バウンドで OCRBase.recognize(image) を直接呼び出せば pass 本体のため None が返る。

モックした依存: なし（抽象クラス本体の直接検証）。

errors セクション（記録のみ・テストしない）:
  - 抽象メソッドを実装していないサブクラスのインスタンス化 → TypeError（Python インタプリタが送出）。
"""
from __future__ import annotations

import pytest

from src.ocr.base import OCRBase


def test_edge_01():
    """input: recognize をオーバーライドしていないサブクラスのインスタンス化を試みる
    expected: TypeError（抽象メソッド recognize 未実装のため抽象クラスをインスタンス化できない）が送出される。このエラーはメソッド呼び出し時ではなくインスタンス化時に発生する
    """
    # recognize_with_confidence は実装し、recognize のみ未実装にする。
    class Sub(OCRBase):
        def recognize_with_confidence(self, image):
            return []

    with pytest.raises(TypeError):
        Sub()


def test_edge_02():
    """input: 未バインドで OCRBase.recognize(任意のimage) を直接呼び出す
    expected: None が返る（pass 本体）
    """
    assert OCRBase.recognize(None, None) is None
