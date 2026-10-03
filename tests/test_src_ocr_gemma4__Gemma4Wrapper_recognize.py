"""Tests for ``src.ocr.gemma4.Gemma4Wrapper.recognize``.

Specification: docs/00-Architecture/src_ocr_gemma4__Gemma4Wrapper_recognize.yaml

``recognize`` is supposed to recognize text from an image (unimplemented:
after the model-load check it always returns an empty string):

1. call ``self._ensure_model()`` (on first use: imports
   transformers/torch, reads the model files, sets
   ``self._model`` / ``self._processor``)
2. perform no recognition and return ``""`` (the body is a
   placeholder comment; the implementation is unfinished)

The ``image`` argument is not referenced at all; there is no type or
content validation.

Mocked / stand-in dependencies (per the test-generation rules):
transformers and torch are NOT installed in this environment. For the
model-loaded case, stand-in modules are injected into ``sys.modules``
(a fake ``transformers`` module whose ``AutoModelForCausalLM`` /
``AutoProcessor`` classes have ``from_pretrained`` class methods
returning stand-in objects, and a fake ``torch`` module with a
``float16`` attribute); the ImportError case relies on the genuine
absence of the packages.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "self._ensure_model() からの例外（ImportError、モデル読み込み失敗など）"
  behavior: "捕捉されず、元の例外種別・メッセージのまま送出される"
"""

import sys
import types

import pytest

from src.ocr.gemma4 import Gemma4Wrapper


def _make_fake_transformers(model_factory, processor_factory):
    """Build a stand-in ``transformers`` module."""
    mod = types.ModuleType("transformers")

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return model_factory(path, kwargs)

    class AutoProcessor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return processor_factory(path, kwargs)

    mod.AutoModelForCausalLM = AutoModelForCausalLM
    mod.AutoProcessor = AutoProcessor
    return mod


def _make_fake_torch():
    """Build a stand-in ``torch`` module with a ``float16``
    attribute."""
    mod = types.ModuleType("torch")
    mod.float16 = "fake-float16"
    return mod


def _install_fakes(model_factory, processor_factory):
    """Install the stand-in transformers/torch modules into
    ``sys.modules`` and return a cleanup callable that restores the
    previous state."""
    saved = {name: sys.modules.get(name) for name in ("transformers", "torch")}
    sys.modules["transformers"] = _make_fake_transformers(model_factory,
                                                          processor_factory)
    sys.modules["torch"] = _make_fake_torch()

    def _cleanup():
        for name, value in saved.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value

    return _cleanup


def test_edge_01():
    """
    input: 'image = None, self._model は None でない'
    expected: '"" を返す。モデル読み込み処理は行われない'
    """
    inst = Gemma4Wrapper()
    sentinel = object()
    inst._model = sentinel
    result = inst.recognize(None)
    assert result == ""
    # No model loading was performed.
    assert inst._model is sentinel
    assert "transformers" not in sys.modules


def test_edge_02():
    """
    input: 'image = None, self._model is None, transformers/torch インストール済み かつ self.model_path 有効'
    expected: '_ensure_model() によりモデルが読み込まれ、その後に "" を返す'
    """
    inst = Gemma4Wrapper("/fake/model")
    assert inst._model is None

    def _model_factory(path, kwargs):
        return "FAKE_MODEL"

    def _processor_factory(path, kwargs):
        return "FAKE_PROCESSOR"

    cleanup = _install_fakes(_model_factory, _processor_factory)
    try:
        result = inst.recognize(None)
    finally:
        cleanup()
    assert result == ""
    assert inst._model == "FAKE_MODEL"
    assert inst._processor == "FAKE_PROCESSOR"


def test_edge_03():
    """
    input: 'image = None, self._model is None, transformers 未インストール'
    expected: 'ImportError が送出される（メッセージ: Gemma4を使用するには `pip install torch transformers` が必要です）'
    """
    inst = Gemma4Wrapper()
    assert inst._model is None
    with pytest.raises(ImportError) as excinfo:
        inst.recognize(None)
    assert "Gemma4を使用するには `pip install torch transformers` が必要です" in str(excinfo.value)
    assert inst._model is None
