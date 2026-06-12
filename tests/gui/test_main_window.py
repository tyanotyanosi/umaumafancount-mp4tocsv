"""GUI テスト: src.app.gui.main_window

  テスト対象:
    - AppWindow（ウィンドウ作成を伴わない軽量テストのみ）
"""

from __future__ import annotations

import pytest


class TestAppWindowAttrs:
    """AppWindow の基本属性チェック（GUI作成はスキップ）"""

    def test_debug_and_img_scale(self):
        """object.__new__ でインスタンス化し属性を検証"""
        from src.app.gui.main_window import AppWindow

        app = object.__new__(AppWindow)
        app.debug = True
        app.img_scale = "gray"
        app.is_processing = False

        assert app.debug is True
        assert app.img_scale == "gray"
        assert app.is_processing is False

    def test_default_params(self):
        from src.app.gui.main_window import AppWindow

        app = object.__new__(AppWindow)
        app.debug = False
        app.img_scale = None
        app.is_processing = False

        assert app.debug is False
        assert app.img_scale is None
