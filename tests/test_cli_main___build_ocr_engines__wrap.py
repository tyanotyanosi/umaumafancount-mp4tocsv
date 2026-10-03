"""Tests for ``cli.main._build_ocr_engines._wrap``.

Specification: docs/00-Architecture/cli_main___build_ocr_engines__wrap.yaml

``_wrap`` is a nested function inside ``_build_ocr_engines``: when the
cache is enabled it returns ``CachingOCR(eng, name=name)``, otherwise
it returns ``eng`` unchanged as a wrapper:

- reads the closure variable ``cache_enabled`` from the outer
  ``_build_ocr_engines``
  (``= bool((settings or {}).get('ocr', {}).get('cache', True))``)
- if ``cache_enabled`` is truthy, return ``CachingOCR(eng, name=name)``;
  otherwise return ``eng`` as-is

Because ``_wrap`` is a nested function and cannot be called directly
from outside the module (it depends on the closure variable
``cache_enabled``), it is exercised through the call path of the outer
``_build_ocr_engines`` (nested-function call-path policy): the meiki
branch with ``settings={}`` yields ``cache_enabled=True`` and the
``name_ocr`` slot of the returned tuple is exactly
``_wrap(name_ocr, 'name')``; the meiki branch with
``settings={'ocr': {'cache': False}}`` yields ``cache_enabled=False`` and
the ``name_ocr`` slot is ``_wrap(name_ocr, 'name')`` returning the raw
engine.

Mocked / stand-in dependencies (per the test-generation rules):
``cli.main.MeikiOCRWrapper`` and ``cli.main.Gemma4Wrapper`` are mocks
(distinct stand-in instances are handed out per constructor call via
``side_effect``); ``CachingOCR`` runs real (lightweight wrapper holding
``engine`` / ``name`` attributes).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: "CachingOCRのコンストラクタが例外を送出する（実装依存）"
  behavior: "本関数は捕捉せず、例外は呼び出し元（_build_ocr_engines）へそのまま送出される"
- condition: "_build_ocr_enginesのスコープ外で呼び出される（クロージャ変数cache_enabledが存在しない）"
  behavior: "NameError（name 'cache_enabled' is not defined）が送出される"
"""

import contextlib
from unittest import mock

from cli.main import _build_ocr_engines
from src.ocr.cache import CachingOCR


@contextlib.contextmanager
def _mocks():
    """Patch the OCR wrapper constructors and yield their mocks."""
    with mock.patch("cli.main.MeikiOCRWrapper") as mock_mow, \
            mock.patch("cli.main.Gemma4Wrapper") as mock_gw:
        yield mock_mow, mock_gw


def test_edge_01():
    """
    input: cache_enabled=True, engがMeikiOCRWrapperインスタンス, name='name'
    expected: CachingOCR(eng, name='name') の新インスタンスを返す。
    """
    with _mocks() as (mock_mow, mock_gw):
        inst_name = mock.Mock(name="meiki_name")
        mock_mow.side_effect = [inst_name, mock.Mock(), mock.Mock()]
        ret = _build_ocr_engines("meiki", {}, quiet=True)
        wrapped = ret[0]
        assert isinstance(wrapped, CachingOCR)
        assert wrapped is not inst_name
        assert wrapped.name == "name"
        assert wrapped.engine is inst_name
        mock_gw.assert_not_called()


def test_edge_02():
    """
    input: cache_enabled=False, engが任意のエンジンインスタンス
    expected: engをそのまま（同一オブジェクトとして）返す。
    """
    with _mocks() as (mock_mow, mock_gw):
        inst_name = mock.Mock(name="meiki_name")
        mock_mow.side_effect = [inst_name, mock.Mock(), mock.Mock()]
        ret = _build_ocr_engines("meiki", {"ocr": {"cache": False}})
        assert ret[0] is inst_name
        assert not isinstance(ret[0], CachingOCR)
        mock_gw.assert_not_called()
