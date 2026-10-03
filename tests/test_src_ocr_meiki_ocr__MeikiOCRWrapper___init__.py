"""src.ocr.meiki_ocr.MeikiOCRWrapper.__init__ の仕様書駆動 pytest テスト。

spec: docs/00-Architecture/src_ocr_meiki_ocr__MeikiOCRWrapper___init__.yaml
function: src.ocr.meiki_ocr.MeikiOCRWrapper.__init__

検証する行動:
  無引数で _MeikiOCREngine()（meikiocr.MeikiOCR）を構築し self.engine に格納し、
  det_threshold（デフォルト 0.8）を self.det_threshold、rec_threshold（デフォルト 0.2）を
  self.rec_threshold に格納する。しきい値は値域検証なしで任意の値がそのまま格納される。

モックした依存:
  - _MeikiOCREngine を mock.patch.object で MagicMock に置換する（meikiocr/onnxruntime が
    未インストールのため実エンジン構築を回避）。self.engine はその Mock の return_value になる。

errors セクション（記録のみ・テストしない）:
  - meikiocr モジュールが未インストール → モジュールレベルの ImportError（import 時点で送出）。
  - _MeikiOCREngine() の構築中例外 → 捕捉されず送出され、self.engine が未設定のまま送出。
"""
from __future__ import annotations

from unittest import mock

import src.ocr.meiki_ocr as m


def test_edge_01():
    """input: MeikiOCRWrapper()（引数なし）
    expected: self.engine が None でなく、self.det_threshold == 0.8 かつ self.rec_threshold == 0.2
    """
    with mock.patch.object(m, "_MeikiOCREngine") as mock_engine:
        w = m.MeikiOCRWrapper()
    assert w.engine is not None
    assert w.engine is mock_engine.return_value
    assert w.det_threshold == 0.8
    assert w.rec_threshold == 0.2


def test_edge_02():
    """input: MeikiOCRWrapper(det_threshold=0.5, rec_threshold=0.3)
    expected: self.det_threshold == 0.5 かつ self.rec_threshold == 0.3
    """
    with mock.patch.object(m, "_MeikiOCREngine"):
        w = m.MeikiOCRWrapper(det_threshold=0.5, rec_threshold=0.3)
    assert w.det_threshold == 0.5
    assert w.rec_threshold == 0.3


def test_edge_03():
    """input: MeikiOCRWrapper(det_threshold=2.0)
    expected: コード上に検証がないため例外は発生せず self.det_threshold == 2.0（後のエンジンでの使用時の挙動は未確認）
    """
    with mock.patch.object(m, "_MeikiOCREngine"):
        w = m.MeikiOCRWrapper(det_threshold=2.0)
    assert w.det_threshold == 2.0
    assert w.rec_threshold == 0.2
