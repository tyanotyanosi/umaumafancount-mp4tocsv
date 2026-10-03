"""Tests for ``src.utils.frozen_bootstrap.run_frozen_bootstrap``.

Specification: docs/00-Architecture/src_utils_frozen_bootstrap__run_frozen_bootstrap.yaml

Purpose (per the spec): when running as a frozen exe, replace
``meikiocr.ocr._get_model_path`` with a function that uses the model bundled
with the exe (so the OCR engine works offline right after first launch). In
source execution it does nothing.

Behavior verified by these tests (per the spec's ``behavior`` and
``postconditions``):

- if ``getattr(sys, "frozen", False)`` is falsy, the function returns
  immediately (``None``)
- ``model_dir = Path(getattr(sys, "_MEIPASS")) / "models"`` is computed; if
  ``model_dir.is_dir()`` is falsy, the function returns (old builds without a
  bundled model fall back to HF Hub as before)
- otherwise ``meikiocr.ocr`` is imported as ``_meiki_ocr`` and the inner
  closure ``_bundled_model_path(repo_id, filename)`` is assigned to
  ``_meiki_ocr._get_model_path``
- the function always returns ``None``

Mocked / stand-in dependencies (per the test-generation rules): ``sys.frozen``
and ``sys._MEIPASS`` are patched with ``mock.patch.object`` (``create=True``)
to simulate frozen execution; ``meikiocr.ocr`` is replaced in ``sys.modules``
by a fake module carrying a fresh sentinel ``_get_model_path`` function, so
the real ``meikiocr.ocr`` module is never imported or modified and the spec's
"not imported / attribute unchanged" expectations are observed via the
sentinel identity.

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'frozen 実行で sys._MEIPASS 属性が存在しない場合（getattr(sys, "_MEIPASS") はデフォルト値なしで None を返す）'
  behavior: 'Path(None) で TypeError が発生する（Path は str または PathLike を要求する）。'
- condition: 'frozen 実行で sys._MEIPASS/models がディレクトリだが、meikiocr.ocr モジュールが不在の場合'
  behavior: 'import meikiocr.ocr 句で ImportError（またはそのサブクラス）が発生する。'
"""

import contextlib
import shutil
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

from src.utils.frozen_bootstrap import run_frozen_bootstrap


def _make_sentinel():
    def sentinel(repo_id, filename):
        raise AssertionError("sentinel _get_model_path was called")

    return sentinel


@contextlib.contextmanager
def _fake_meiki_ocr():
    """Install a fake ``meikiocr`` / ``meikiocr.ocr`` in ``sys.modules`` with a
    fresh sentinel ``_get_model_path``; restore the previous entries on exit."""
    fake_pkg = types.ModuleType("meikiocr")
    fake_ocr = types.ModuleType("meikiocr.ocr")
    sentinel = _make_sentinel()
    fake_ocr._get_model_path = sentinel
    fake_pkg.ocr = fake_ocr
    saved = {name: sys.modules.get(name) for name in ("meikiocr", "meikiocr.ocr")}
    sys.modules["meikiocr"] = fake_pkg
    sys.modules["meikiocr.ocr"] = fake_ocr
    try:
        yield fake_ocr, sentinel
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_edge_01():
    """
    input: ソース実行（sys.frozen 属性が存在しない）
    expected: None を返し、meikiocr.ocr は import されず、_get_model_path 属性は変更されない。
    """
    had_frozen = hasattr(sys, "frozen")
    orig_frozen = getattr(sys, "frozen", None)
    if had_frozen:
        del sys.frozen
    try:
        with _fake_meiki_ocr() as (fake_ocr, sentinel):
            ret = run_frozen_bootstrap()
            assert ret is None
            # The import line is never reached: the sentinel attribute is unchanged,
            # i.e. the fake meikiocr.ocr was neither imported nor patched.
            assert fake_ocr._get_model_path is sentinel
    finally:
        if had_frozen:
            sys.frozen = orig_frozen


def test_edge_02():
    """
    input: frozen 実行だが sys._MEIPASS 直下に 'models' ディレクトリが存在しない（旧ビルド等）
    expected: None を返し、meikiocr.ocr は import されず、_get_model_path 属性は変更されない。
    """
    tmp = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent.parent))
    try:
        with _fake_meiki_ocr() as (fake_ocr, sentinel), \
             mock.patch.object(sys, "frozen", True, create=True), \
             mock.patch.object(sys, "_MEIPASS", str(tmp), create=True):
            ret = run_frozen_bootstrap()
            assert ret is None
            # model_dir.is_dir() is False: the import line is never reached and
            # the sentinel attribute is unchanged.
            assert fake_ocr._get_model_path is sentinel
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_03():
    """
    input: frozen 実行、sys._MEIPASS 直下に 'models' ディレクトリが存在する
    expected: meikiocr.ocr が import され、_meiki_ocr._get_model_path がクロージャ _bundled_model_path に差し替わる（以降の呼び出しは <_MEIPASS>/models 配下のパスを str で返す、または FileNotFoundError を送出する）。
    """
    tmp = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent.parent))
    try:
        models = tmp / "models"
        models.mkdir()
        (models / "meiki.onnx").write_bytes(b"bundled-model-bytes")
        with _fake_meiki_ocr() as (fake_ocr, sentinel), \
             mock.patch.object(sys, "frozen", True, create=True), \
             mock.patch.object(sys, "_MEIPASS", str(tmp), create=True):
            ret = run_frozen_bootstrap()
            assert ret is None
            # The attribute was replaced with the closure.
            assert fake_ocr._get_model_path is not sentinel
            fn = fake_ocr._get_model_path
            # Subsequent calls return a str path under <_MEIPASS>/models ...
            path = fn("any-repo", "meiki.onnx")
            assert isinstance(path, str)
            assert path == str(models / "meiki.onnx")
            # ... or raise FileNotFoundError.
            try:
                fn("any-repo", "missing.onnx")
            except FileNotFoundError:
                raised = True
            else:
                raised = False
            assert raised
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
