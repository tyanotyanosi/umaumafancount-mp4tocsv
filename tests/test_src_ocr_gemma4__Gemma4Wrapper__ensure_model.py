"""Tests for ``src.ocr.gemma4.Gemma4Wrapper._ensure_model``.

Specification: docs/00-Architecture/src_ocr_gemma4__Gemma4Wrapper__ensure_model.yaml

``_ensure_model`` lazily loads the Gemma4 model (only when not yet
loaded: imports transformers/torch and loads the model and processor
from ``self.model_path``):

- if ``self._model`` is not None, return immediately
- import ``AutoModelForCausalLM`` and ``AutoProcessor`` from
  transformers, and import torch
- call ``AutoModelForCausalLM.from_pretrained(self.model_path,
  torch_dtype=torch.float16, device_map="auto")`` and store the
  result in ``self._model``
- call ``AutoProcessor.from_pretrained(self.model_path)`` and store
  the result in ``self._processor``
- if an ImportError occurs in any of the steps above, raise a new
  ImportError (message: Gemma4 requires `pip install torch
  transformers`)

Mocked / stand-in dependencies (per the test-generation rules):
transformers and torch are NOT installed in this environment. For
the success and non-ImportError-failure cases, stand-in modules are
injected into ``sys.modules`` (a fake ``transformers`` module whose
``AutoModelForCausalLM`` / ``AutoProcessor`` classes have
``from_pretrained`` class methods that record their arguments and
return stand-in objects, and a fake ``torch`` module with a
``float16`` attribute); the ImportError case relies on the genuine
absence of the packages.

``errors`` section of the spec (documented here only, NOT tested, per
the test-generation rules):

- condition: "transformers / torch の import 中、または from_pretrained 呼び出し中に ImportError が発生"
  behavior: "捕捉し、新しい ImportError として送出（メッセージ: Gemma4を使用するには `pip install torch transformers` が必要です）"
- condition: "AutoModelForCausalLM.from_pretrained / AutoProcessor.from_pretrained が ImportError 以外の例外（例: パス不存在）を送出"
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
    input: 'self._model = <None 以外のオブジェクト>'
    expected: 'import もファイルアクセスも行わず None を返し、self._model は不変'
    """
    inst = Gemma4Wrapper()
    sentinel = object()
    inst._model = sentinel
    result = inst._ensure_model()
    assert result is None
    assert inst._model is sentinel
    # No import was performed: transformers remains absent.
    assert "transformers" not in sys.modules


def test_edge_02():
    """
    input: 'self._model is None かつ transformers（または torch）が未インストール'
    expected: 'ImportError が送出される（メッセージ: Gemma4を使用するには `pip install torch transformers` が必要です）。self._model は None のまま'
    """
    inst = Gemma4Wrapper()
    assert inst._model is None
    with pytest.raises(ImportError) as excinfo:
        inst._ensure_model()
    assert "Gemma4を使用するには `pip install torch transformers` が必要です" in str(excinfo.value)
    assert inst._model is None


def test_edge_03():
    """
    input: 'self._model is None かつ self.model_path が有効な Gemma4 モデルを指す（初回呼び出し）'
    expected: 'self._model と self._processor が設定され、None が返る'
    """
    inst = Gemma4Wrapper("/fake/model")
    calls = []

    def _model_factory(path, kwargs):
        calls.append(("model", path, kwargs))
        return "FAKE_MODEL"

    def _processor_factory(path, kwargs):
        calls.append(("processor", path, kwargs))
        return "FAKE_PROCESSOR"

    cleanup = _install_fakes(_model_factory, _processor_factory)
    try:
        result = inst._ensure_model()
    finally:
        cleanup()
    assert result is None
    assert inst._model == "FAKE_MODEL"
    assert inst._processor == "FAKE_PROCESSOR"
    # from_pretrained calls carry the spec's arguments.
    assert calls[0] == ("model", "/fake/model",
                       {"torch_dtype": "fake-float16", "device_map": "auto"})
    assert calls[1] == ("processor", "/fake/model", {})


def test_edge_04():
    """
    input: 'self._model is None かつ self.model_path が存在しないパス かつ transformers/torch はインストール済み'
    expected: 'from_pretrained 由来の例外（ImportError 以外、例: OSError）は捕捉されずそのまま送出される'
    """
    inst = Gemma4Wrapper("/no/such/model")

    def _model_factory(path, kwargs):
        raise OSError("model file not found")

    def _processor_factory(path, kwargs):
        return "FAKE_PROCESSOR"

    cleanup = _install_fakes(_model_factory, _processor_factory)
    try:
        with pytest.raises(OSError, match="model file not found"):
            inst._ensure_model()
    finally:
        cleanup()
    # The failed assignment left _model unset.
    assert inst._model is None
