"""src.ocr.meiki_ocr.MeikiOCRWrapper.recognize_with_confidence の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/src_ocr_meiki_ocr__MeikiOCRWrapper_recognize_with_confidence.yaml
function: src.ocr.meiki_ocr.MeikiOCRWrapper.recognize_with_confidence

検証する行動:
  self.engine.run_ocr(image, det_threshold=self.det_threshold, rec_threshold=self.rec_threshold) を
  呼び、各行の line['text'] が falsy（空文字列等）ならスキップ。残りは line.get('chars') or [] を取り、
  非空なら conf = sum(float(c.get('conf', 0.0))) / len(chars)、空なら 0.0 を計算し、
  {"text": line['text"], "confidence": conf} を出現順にリストへ追加して返す。

モックした依存:
  - self.engine を mock.Mock() で置換し、engine.run_ocr の戻り値を各 edge の結果リストに設定。
  - MeikiOCRWrapper インスタンスは object.__new__(MeikiOCRWrapper) で __init__ を回避して作成し
    （meikiocr/transformers/torch が未インストールのため _MeikiOCREngine を実生成しない）、
    engine / det_threshold / rec_threshold を手動で設定する。

errors セクション（記録のみ・テストしない）:
  - 結果要素に "text" キーがない → KeyError("text")。
  - chars 内の 'conf' が float に変換不能（例: None）→ TypeError（float(...) 由来）。
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
    """input: run_ocr が [] を返す
    expected: [] を返す
    """
    w = _make_wrapper([])
    assert w.recognize_with_confidence(_image()) == []


def test_edge_02():
    """input: run_ocr が [{"text": "ab", "chars": [{"conf": 0.9}, {"conf": 0.7}]}] を返す
    expected: [{"text": "ab", "confidence": 0.8}] を返す
    """
    w = _make_wrapper([{"text": "ab", "chars": [{"conf": 0.9}, {"conf": 0.7}]}])
    assert w.recognize_with_confidence(_image()) == [{"text": "ab", "confidence": 0.8}]


def test_edge_03():
    """input: run_ocr が [{"text": "ab"}] を返す（"chars" キーなし）
    expected: [{"text": "ab", "confidence": 0.0}] を返す
    """
    w = _make_wrapper([{"text": "ab"}])
    assert w.recognize_with_confidence(_image()) == [{"text": "ab", "confidence": 0.0}]


def test_edge_04():
    """input: run_ocr が [{"text": "ab", "chars": []}] を返す
    expected: [{"text": "ab", "confidence": 0.0}] を返す
    """
    w = _make_wrapper([{"text": "ab", "chars": []}])
    assert w.recognize_with_confidence(_image()) == [{"text": "ab", "confidence": 0.0}]


def test_edge_05():
    """input: run_ocr が [{"text": "abc", "chars": [{"conf": 0.6}, {}]}] を返す（"conf" 欠落の文字あり）
    expected: [{"text": "abc", "confidence": 0.3}] を返す（欠落 conf は 0.0 として平均に集計）
    """
    w = _make_wrapper([{"text": "abc", "chars": [{"conf": 0.6}, {}]}])
    assert w.recognize_with_confidence(_image()) == [{"text": "abc", "confidence": 0.3}]


def test_edge_06():
    """input: run_ocr が [{"text": "ab", "chars": [{"conf": None}]}] を返す
    expected: TypeError が送出される（float(None) 由来）
    """
    w = _make_wrapper([{"text": "ab", "chars": [{"conf": None}]}])
    with pytest.raises(TypeError):
        w.recognize_with_confidence(_image())


def test_edge_07():
    """input: run_ocr が [{"text": ""}, {"text": "a", "chars": [{"conf": 0.5}]}] を返す
    expected: [{"text": "a", "confidence": 0.5}] を返す（空テキスト行はスキップ）
    """
    w = _make_wrapper([{"text": ""}, {"text": "a", "chars": [{"conf": 0.5}]}])
    assert w.recognize_with_confidence(_image()) == [{"text": "a", "confidence": 0.5}]


def test_edge_08():
    """input: run_ocr が "text" キーのない行を含む。例: [{"chars": [{"conf": 0.9}]}]
    expected: KeyError("text") が送出される
    """
    w = _make_wrapper([{"chars": [{"conf": 0.9}]}])
    with pytest.raises(KeyError):
        w.recognize_with_confidence(_image())
