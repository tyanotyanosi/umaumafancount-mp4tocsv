"""Tests for ``cli.main._build_ocr_engines``.

Specification: docs/00-Architecture/cli_main___build_ocr_engines.yaml

``_build_ocr_engines`` builds the OCR engines according to the
``ocr_engine`` name and returns the 3-tuple ``(name_ocr, name_ocr_low,
fan_ocr)``; when the ocr cache is enabled (default) each engine is
wrapped in a ``CachingOCR``:

- ``cache_enabled = bool((settings or {}).get('ocr', {}).get('cache', True))``
- If ``ocr_engine == 'meiki'``: if not quiet, print
  「meikiOCR初期化中...」
- (meiki) read 6 thresholds from the ``settings['ocr']['meiki']``
  section (default {}): name_det_threshold=0.3, name_rec_threshold=0.2,
  fan_det_threshold=0.3, fan_rec_threshold=0.05,
  name_det_threshold_low=0.2, name_rec_threshold_low=0.1
- (meiki) if not quiet, print the 4 thresholds (name_det / name_rec /
  fan_det / fan_rec) in 2 lines
- (meiki) create 3 ``MeikiOCRWrapper`` instances and return
  ``(_wrap(name_ocr, 'name'), _wrap(name_ocr_low, 'name_low'),
  _wrap(fan_ocr, 'fan'))``
- Otherwise: if not quiet, print 「Gemma4初期化中...」, create one
  ``Gemma4Wrapper`` and return ``(_wrap(ocr, 'name'), None,
  _wrap(ocr, 'fan'))``
- ``_wrap(eng, name)`` returns ``CachingOCR(eng, name=name)`` if
  cache_enabled is truthy, otherwise ``eng`` itself

Mocked / stand-in dependencies (per the test-generation rules):
``cli.main.MeikiOCRWrapper`` and ``cli.main.Gemma4Wrapper`` are mocks
(distinct stand-in instances are handed out per constructor call via
``side_effect``); ``CachingOCR`` runs real (lightweight wrapper holding
``engine`` / ``name`` attributes).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "settings['ocr']が存在するがNone（例: YAMLでocr:が空）"
  behavior: "cache_enabled判定またはmeiki分岐の.get('...')呼び出しでAttributeErrorが送出される"
- condition: "MeikiOCRWrapper / Gemma4Wrapper / CachingOCRのコンストラクタが例外を送出する"
  behavior: "本関数は捕捉せず、例外は呼び出し元へそのまま送出される"
"""

import contextlib
from unittest import mock

import pytest

from cli.main import _build_ocr_engines
from src.ocr.cache import CachingOCR


@contextlib.contextmanager
def _mocks():
    """Patch the OCR wrapper constructors and yield their mocks."""
    with mock.patch("cli.main.MeikiOCRWrapper") as mock_mow, \
            mock.patch("cli.main.Gemma4Wrapper") as mock_gw:
        yield mock_mow, mock_gw


def _thresholds(call):
    """Return ``(det_threshold, rec_threshold)`` from a constructor
    call, whether passed positionally or as keywords."""
    args = list(call.args)
    kwargs = dict(call.kwargs)
    det = kwargs.get("det_threshold", args[0] if args else None)
    rec = kwargs.get("rec_threshold", args[1] if len(args) > 1 else None)
    return det, rec


def test_edge_01(capsys):
    """
    input: ocr_engine='meiki', settings={}, quiet=True
    expected: (CachingOCR(MeikiOCRWrapper(det_threshold=0.3, rec_threshold=0.2), name='name'), CachingOCR(MeikiOCRWrapper(det_threshold=0.2, rec_threshold=0.1), name='name_low'), CachingOCR(MeikiOCRWrapper(det_threshold=0.3, rec_threshold=0.05), name='fan')) を返し、printは出ない。
    """
    with _mocks() as (mock_mow, mock_gw):
        inst_name = mock.Mock(name="meiki_name")
        inst_low = mock.Mock(name="meiki_low")
        inst_fan = mock.Mock(name="meiki_fan")
        mock_mow.side_effect = [inst_name, inst_low, inst_fan]
        ret = _build_ocr_engines("meiki", {}, quiet=True)
        assert isinstance(ret, tuple) and len(ret) == 3
        assert isinstance(ret[0], CachingOCR)
        assert ret[0].name == "name"
        assert ret[0].engine is inst_name
        assert isinstance(ret[1], CachingOCR)
        assert ret[1].name == "name_low"
        assert ret[1].engine is inst_low
        assert isinstance(ret[2], CachingOCR)
        assert ret[2].name == "fan"
        assert ret[2].engine is inst_fan
        assert _thresholds(mock_mow.call_args_list[0]) == (0.3, 0.2)
        assert _thresholds(mock_mow.call_args_list[1]) == (0.2, 0.1)
        assert _thresholds(mock_mow.call_args_list[2]) == (0.3, 0.05)
        mock_gw.assert_not_called()
        assert capsys.readouterr().out == ""


def test_edge_02(capsys):
    """
    input: ocr_engine='meiki', settings={'ocr': {'cache': False}}
    expected: ラップせず、生の(MeikiOCRWrapper(0.3, 0.2), MeikiOCRWrapper(0.2, 0.1), MeikiOCRWrapper(0.3, 0.05)) が返る。
    """
    with _mocks() as (mock_mow, mock_gw):
        inst_name = mock.Mock(name="meiki_name")
        inst_low = mock.Mock(name="meiki_low")
        inst_fan = mock.Mock(name="meiki_fan")
        mock_mow.side_effect = [inst_name, inst_low, inst_fan]
        ret = _build_ocr_engines("meiki", {"ocr": {"cache": False}})
        assert ret[0] is inst_name
        assert ret[1] is inst_low
        assert ret[2] is inst_fan
        assert not isinstance(ret[0], CachingOCR)
        assert not isinstance(ret[1], CachingOCR)
        assert not isinstance(ret[2], CachingOCR)
        assert _thresholds(mock_mow.call_args_list[0]) == (0.3, 0.2)
        assert _thresholds(mock_mow.call_args_list[1]) == (0.2, 0.1)
        assert _thresholds(mock_mow.call_args_list[2]) == (0.3, 0.05)
        mock_gw.assert_not_called()


def test_edge_03(capsys):
    """
    input: ocr_engine='meiki', settings={'ocr': {'meiki': {'name_det_threshold': 0.5}}}
    expected: nameエンジンはMeikiOCRWrapper(det_threshold=0.5, rec_threshold=0.2)となり、その他の閾値は既定値のまま。
    """
    with _mocks() as (mock_mow, mock_gw):
        inst_name = mock.Mock(name="meiki_name")
        inst_low = mock.Mock(name="meiki_low")
        inst_fan = mock.Mock(name="meiki_fan")
        mock_mow.side_effect = [inst_name, inst_low, inst_fan]
        ret = _build_ocr_engines(
            "meiki", {"ocr": {"meiki": {"name_det_threshold": 0.5}}})
        assert _thresholds(mock_mow.call_args_list[0]) == (0.5, 0.2)
        assert _thresholds(mock_mow.call_args_list[1]) == (0.2, 0.1)
        assert _thresholds(mock_mow.call_args_list[2]) == (0.3, 0.05)
        assert isinstance(ret[0], CachingOCR)
        assert ret[0].engine is inst_name
        mock_gw.assert_not_called()


def test_edge_04(capsys):
    """
    input: ocr_engine='gemma4', settings={}, quiet=True
    expected: (CachingOCR(ocr, name='name'), None, CachingOCR(ocr, name='fan')) を返す（ocrは同じ1つのGemma4Wrapperインスタンス）。printは出ない。
    """
    with _mocks() as (mock_mow, mock_gw):
        ocr_inst = mock.Mock(name="gemma4")
        mock_gw.side_effect = [ocr_inst]
        ret = _build_ocr_engines("gemma4", {}, quiet=True)
        assert isinstance(ret, tuple) and len(ret) == 3
        assert ret[1] is None
        assert isinstance(ret[0], CachingOCR)
        assert ret[0].name == "name"
        assert ret[0].engine is ocr_inst
        assert isinstance(ret[2], CachingOCR)
        assert ret[2].name == "fan"
        assert ret[2].engine is ocr_inst
        mock_gw.assert_called_once()
        mock_mow.assert_not_called()
        assert capsys.readouterr().out == ""


def test_edge_05(capsys):
    """
    input: ocr_engine='other'（'meiki'以外の任意の文字列）
    expected: Gemma4分岐として処理される（エラーにはならない）。
    """
    with _mocks() as (mock_mow, mock_gw):
        ocr_inst = mock.Mock(name="gemma4")
        mock_gw.side_effect = [ocr_inst]
        ret = _build_ocr_engines("other", {})
        assert isinstance(ret, tuple) and len(ret) == 3
        assert ret[1] is None
        assert isinstance(ret[0], CachingOCR)
        assert ret[0].engine is ocr_inst
        assert isinstance(ret[2], CachingOCR)
        assert ret[2].engine is ocr_inst
        mock_gw.assert_called_once()
        mock_mow.assert_not_called()


def test_edge_06(capsys):
    """
    input: ocr_engine=None
    expected: None != 'meiki' であるためGemma4分岐に入る。
    """
    with _mocks() as (mock_mow, mock_gw):
        ocr_inst = mock.Mock(name="gemma4")
        mock_gw.side_effect = [ocr_inst]
        ret = _build_ocr_engines(None, {})
        assert isinstance(ret, tuple) and len(ret) == 3
        assert ret[1] is None
        assert isinstance(ret[0], CachingOCR)
        assert ret[0].engine is ocr_inst
        assert isinstance(ret[2], CachingOCR)
        assert ret[2].engine is ocr_inst
        mock_gw.assert_called_once()
        mock_mow.assert_not_called()


def test_edge_07(capsys):
    """
    input: ocr_engine='meiki', settings=None
    expected: meiki分岐のsettings.getでAttributeErrorが送出される。
    """
    with _mocks() as (mock_mow, mock_gw):
        with pytest.raises(AttributeError):
            _build_ocr_engines("meiki", None)
