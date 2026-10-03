"""Tests for ``src.ocr.gemma4.Gemma4Wrapper.__init__``.

Specification: docs/00-Architecture/src_ocr_gemma4__Gemma4Wrapper___init__.yaml

``__init__`` stores the model path and initializes the model as
not-yet-loaded:

- store ``model_path`` in ``self.model_path``
- initialize ``self._model`` to None (the model body itself is lazily
  loaded by ``_ensure_model``)

No validation or error handling exists in the code: any value (including
``None``) is stored as-is, and no file I/O or external calls occur.

Mocked / stand-in dependencies (per the test-generation rules): none —
``Gemma4Wrapper`` is instantiated directly (it has no abstract
methods and ``__init__`` performs no I/O).

``errors`` section of the spec: empty (no error conditions documented).
"""

from src.ocr.gemma4 import Gemma4Wrapper


def test_edge_01():
    """
    input: 'Gemma4Wrapper()（引数なし）'
    expected: 'self.model_path == "./models" かつ self._model is None'
    """
    inst = Gemma4Wrapper()
    assert inst.model_path == "./models"
    assert inst._model is None


def test_edge_02():
    """
    input: 'Gemma4Wrapper("/opt/models/gemma4")'
    expected: 'self.model_path == "/opt/models/gemma4" かつ self._model is None。ファイルへのアクセスは行われない'
    """
    inst = Gemma4Wrapper("/opt/models/gemma4")
    assert inst.model_path == "/opt/models/gemma4"
    assert inst._model is None
    # No file access: constructing with a non-existent path raised no
    # exception.


def test_edge_03():
    """
    input: 'Gemma4Wrapper(None)'
    expected: '例外は発生しない（コード上に検証がない）。self.model_path is None'
    """
    inst = Gemma4Wrapper(None)
    assert inst.model_path is None
    assert inst._model is None
