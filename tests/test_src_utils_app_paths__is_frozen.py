"""Tests for ``src.utils.app_paths.is_frozen``.

Specification: docs/00-Architecture/src_utils_app_paths__is_frozen.yaml

``is_frozen`` determines whether the current process is running as a
frozen executable (PyInstaller, etc.):

- obtain the value of the ``sys.frozen`` attribute via
  ``getattr(sys, 'frozen', False)`` (returns False if the attribute
  does not exist)
- return ``bool()`` of the obtained value

Mocked / stand-in dependencies (per the test-generation rules):
``sys.frozen`` is patched with ``mock.patch.object(sys, "frozen", ...,
create=True)`` for the frozen cases; the source-execution case relies
on the attribute genuinely not existing in this environment.

``errors`` section of the spec: empty (no error conditions documented).
"""

import sys
from unittest import mock

from src.utils.app_paths import is_frozen


def test_edge_01():
    """
    input: frozen exe 実行中（sys.frozen 属性が存在し真値、PyInstaller 等により設定）
    expected: True を返す。
    """
    with mock.patch.object(sys, "frozen", True, create=True):
        assert is_frozen() is True


def test_edge_02():
    """
    input: ソース実行（sys.frozen 属性が存在しない）
    expected: False を返す（getattr のデフォルト値を使用）。
    """
    assert not hasattr(sys, "frozen")
    assert is_frozen() is False


def test_edge_03():
    """
    input: sys.frozen 属性が存在するが偽値（例: None）
    expected: False を返す。
    """
    with mock.patch.object(sys, "frozen", None, create=True):
        assert is_frozen() is False
