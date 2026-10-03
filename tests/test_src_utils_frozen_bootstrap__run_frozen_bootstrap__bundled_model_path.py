"""Tests for ``src.utils.frozen_bootstrap.run_frozen_bootstrap._bundled_model_path``.

Specification: docs/00-Architecture/src_utils_frozen_bootstrap__run_frozen_bootstrap__bundled_model_path.yaml

Purpose (per the spec): ``_bundled_model_path`` is the inner closure
``def _bundled_model_path(repo_id: str, filename: str) -> str:`` defined inside
``run_frozen_bootstrap`` (per the spec's ``function`` field). After the patch it
functions as ``meikiocr.ocr._get_model_path`` and returns the local path of a
model file bundled with the exe as a str:

- ``p = model_dir / filename`` (``model_dir`` is the closure variable
  ``Path(sys._MEIPASS) / "models"`` bound at the time of the
  ``run_frozen_bootstrap`` call)
- if ``p.is_file()`` is falsy, a ``FileNotFoundError`` is raised with a message
  that mentions the path and the build procedure
- otherwise ``str(p)`` is returned
- ``repo_id`` is not used anywhere in the function body (the model location is
  determined solely by the closure variable ``model_dir``)

Because ``_bundled_model_path`` is a local function of ``run_frozen_bootstrap``
it cannot be imported by name; the function object is obtained by running
``run_frozen_bootstrap`` in a mocked frozen environment (``sys.frozen`` True,
``sys._MEIPASS`` a temporary directory containing a ``models`` directory,
``meikiocr.ocr`` a fake module in ``sys.modules``) and reading the patched
``_get_model_path`` attribute of the fake module.

Mocked / stand-in dependencies (per the test-generation rules):
``sys.frozen`` / ``sys._MEIPASS`` via ``mock.patch.object``; ``meikiocr.ocr``
replaced in ``sys.modules`` by a fake module (the real module is never
imported or modified).

``errors`` section of the spec (documented here only, NOT tested, per the
test-generation rules):

- condition: 'model_dir/filename が存在しない、またはファイルでない（ディレクトリ等）'
  behavior: 'FileNotFoundError を送出する。'
"""

import contextlib
import shutil
import sys
import tempfile
import types
from pathlib import Path
from unittest import mock

import pytest

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


@contextlib.contextmanager
def _frozen_env(tmp):
    """Run ``run_frozen_bootstrap`` in a mocked frozen environment where
    ``tmp`` plays the role of ``sys._MEIPASS`` and contains a ``models``
    directory; yield the obtained closure and the models directory."""
    models = tmp / "models"
    models.mkdir()
    with _fake_meiki_ocr() as (fake_ocr, sentinel):
        with mock.patch.object(sys, "frozen", True, create=True), \
             mock.patch.object(sys, "_MEIPASS", str(tmp), create=True):
            run_frozen_bootstrap()
        fn = fake_ocr._get_model_path
        assert fn is not sentinel  # the closure replaced the sentinel
        try:
            yield fn, models
        finally:
            pass


def test_edge_01():
    """
    input: (repo_id='任意の値', filename='meiki.onnx')、かつ <model_dir>/meiki.onnx がファイルとして存在する
    expected: str(<model_dir>/meiki.onnx) を返す。repo_id の値は戻り値に影響しない。
    """
    tmp = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent.parent))
    try:
        with _frozen_env(tmp) as (fn, models):
            (models / "meiki.onnx").write_bytes(b"bundled-model-bytes")
            r1 = fn("repo-a", "meiki.onnx")
            r2 = fn("a-different-repo-id", "meiki.onnx")
            assert isinstance(r1, str)
            assert r1 == str(models / "meiki.onnx")
            # repo_id does not affect the return value.
            assert r1 == r2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_02():
    """
    input: (repo_id='任意の値', filename='missing.onnx')、かつ <model_dir> 直下にそのファイルが存在しない
    expected: FileNotFoundError が送出される（メッセージ: 'exe に meiki モデルが同梱されていません: <パス>' と '（ビルド手順の models/ 取得ステップを確認してください）' が連結）。
    """
    tmp = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent.parent))
    try:
        with _frozen_env(tmp) as (fn, models):
            with pytest.raises(FileNotFoundError) as excinfo:
                fn("any-repo", "missing.onnx")
            expected_msg = (
                f"exe に meiki モデルが同梱されていません: {models / 'missing.onnx'}"
                "（ビルド手順の models/ 取得ステップを確認してください）"
            )
            assert str(excinfo.value) == expected_msg
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_edge_03():
    """
    input: filename が <model_dir> 直下のディレクトリ名を指す（model_dir/filename がディレクトリ）
    expected: FileNotFoundError が送出される（is_file() が偽になるため）。
    """
    tmp = Path(tempfile.mkdtemp(dir=Path(__file__).resolve().parent.parent))
    try:
        with _frozen_env(tmp) as (fn, models):
            (models / "a-subdir").mkdir()
            with pytest.raises(FileNotFoundError):
                fn("any-repo", "a-subdir")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
