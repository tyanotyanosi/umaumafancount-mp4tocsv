"""src.ocr.meiki_ocr.MeikiOCRWrapper.recognize の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/src_ocr_meiki_ocr__MeikiOCRWrapper_recognize.yaml
function: src.ocr.meiki_ocr.MeikiOCRWrapper.recognize

検証する行動:
  self.engine.run_ocr(image, det_threshold=self.det_threshold, rec_threshold=self.rec_threshold) を
  呼び、各結果行の line['text']（falsy な行は除外）を出現順に改行（"\n"）で連結した単一文字列を返す。
  空でないテキスト行が一つもない場合 "" を返す。

モックした依存:
  - self.engine を mock.Mock() で置換し、engine.run_ocr の戻り値を各 edge の結果リストに設定。
  - MeikiOCRWrapper インスタンスは object.__new__(MeikiOCRWrapper) で __init__ を回避して作成し
    （meikiocr/onnxruntime が未インストールのため _MeikiOCREngine を実生成しない）、
    engine / det_threshold / rec_threshold を手動で設定する。

errors セクション（記録のみ・テストしない）:
  - 結果要素に "text" キーがない → KeyError("text")。
  - run_ocr がイテラブル以外（例: None）を返す → TypeError（str.join が結果を反復しようとして送出）。
  - self.engine.run_ocr からの例外 → 捕捉されず元の例外として送出される。
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest import mock

import src.ocr.meiki_ocr as m


def _make_wrapper(results):
    """__init__ を回避した MeikiOCRWrapper を作成し、engine.run_ocr が results を返すように設定する。"""
    w = object.__new__(m.MeikiOCRWrapper)
    w.engine = mock.Mock()
    w.engine.run_ocr.return_value = results
    w.det_threshold = 0.8
    w.rec_threshold = 0.2
    return w


def _image():
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_edge_01():
    """input: run_ocr が [] を返す（認識行ゼロの画像）
    expected: ""（空文字列）を返す
    """
    w = _make_wrapper([])
    assert w.recognize(_image()) == ""


def test_edge_02():
    """input: run_ocr が [{"text": "hello"}, {"text": "world"}] を返す
    expected: "hello\nworld"（改行1個で連結した2行文字列）を返す
    """
    w = _make_wrapper([{"text": "hello"}, {"text": "world"}])
    assert w.recognize(_image()) == "hello\nworld"


def test_edge_03():
    """input: run_ocr が [{"text": ""}, {"text": "a"}, {"text": ""}] を返す
    expected: "a" を返す（空テキスト行は連結から除外され、区切りも挿入されない）
    """
    w = _make_wrapper([{"text": ""}, {"text": "a"}, {"text": ""}])
    assert w.recognize(_image()) == "a"


def test_edge_04():
    """input: run_ocr が [{"text": "abc"}, {"text": "def"}, {"text": "ghi"}] を返す
    expected: "abc\ndef\nghi"（改行で連結した3行文字列）を返す
    """
    w = _make_wrapper([{"text": "abc"}, {"text": "def"}, {"text": "ghi"}])
    assert w.recognize(_image()) == "abc\ndef\nghi"


def test_edge_05():
    """input: run_ocr が "text" キーのない行を含む。例: [{"conf": 0.9}]
    expected: KeyError("text") が送出される
    """
    w = _make_wrapper([{"conf": 0.9}])
    with pytest.raises(KeyError):
        w.recognize(_image())
