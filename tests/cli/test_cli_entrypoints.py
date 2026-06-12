"""CLI テスト: エントリポイント（main.py）

  テスト対象:
    - parse_args()
    - _run_batch_mode()
    - main() の分岐ロジック
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


class TestParseArgs:
    """parse_args() — CLI引数パース"""

    def test_video_positional(self):
        """main.py video.mp4 → args.video == 'video.mp4'"""
        import main
        args = main.parse_args(["video.mp4"])
        assert args.video == "video.mp4"

    def test_video_optional_omitted(self):
        """引数なし → args.video is None"""
        import main
        args = main.parse_args([])
        assert args.video is None

    def test_gui_flag(self):
        """--gui フラグ → args.gui is True"""
        import main
        args = main.parse_args(["--gui"])
        assert args.gui is True
        assert args.video is None

    def test_debug_flag(self):
        """--debug → args.debug is True"""
        import main
        args = main.parse_args(["video.mp4", "--debug"])
        assert args.debug is True

    def test_img_scale_gray(self):
        """--img-scale gray → args.img_scale == 'gray'"""
        import main
        args = main.parse_args(["video.mp4", "--img-scale", "gray"])
        assert args.img_scale == "gray"

    def test_all_flags_combined(self):
        """--gui + --debug + --img-scale gray の同時指定"""
        import main
        args = main.parse_args(["--gui", "--debug", "--img-scale", "gray"])
        assert args.gui is True
        assert args.debug is True
        assert args.img_scale == "gray"
        assert args.video is None

    def test_video_and_gui_together(self):
        """video + --gui の同時指定"""
        import main
        args = main.parse_args(["video.mp4", "--gui"])
        assert args.video == "video.mp4"
        assert args.gui is True


class TestBatchMode:
    """_run_batch_mode() のモックテスト"""

    def test_batch_mode_success(self, tmp_path):
        import main
        from src.domain.models import PipelineConfig, OCRResult

        config = PipelineConfig(video_path=tmp_path / "video.mp4", debug=False)
        mock_result = OCRResult(
            fan_counts={"メンバーA": 100}, texts=[], error=None
        )

        with (
            patch.object(main.PipelineRunner, "run", return_value=mock_result),
            patch("builtins.print"),
        ):
            main._run_batch_mode(config)

        # 例外が発生しなかったこと自体が成功

    def test_batch_mode_error_exits(self, tmp_path):
        import main
        from src.domain.models import PipelineConfig, OCRResult

        config = PipelineConfig(video_path=tmp_path / "video.mp4", debug=False)
        mock_result = OCRResult(
            fan_counts={}, texts=[], error="テストエラー"
        )

        with (
            patch.object(main.PipelineRunner, "run", return_value=mock_result),
            patch("builtins.print"),
            patch.object(sys, "exit") as mock_exit,
        ):
            main._run_batch_mode(config)

        mock_exit.assert_called_once_with(1)


class TestMainEntrypoint:
    """main() のGUI/バッチモード分岐"""

    def test_batch_mode_when_video_specified(self):
        """video引数が指定された場合、_run_batch_mode が呼ばれる
        main() は sys.argv を読むため、事前に差し替えが必要"""
        import main

        with (
            patch.object(sys, "argv", ["main.py", "test.mp4"]),
            patch.object(main, "_run_batch_mode") as mock_batch,
            patch.object(main, "AppWindow"),
        ):
            main.main()
            mock_batch.assert_called_once()

    def test_batch_mode_called_with_video_arg(self):
        """video引数あり → _run_batch_mode が呼ばれる"""
        import main

        with (
            patch.object(sys, "argv", ["main.py", "test.mp4"]),
            patch.object(main, "_run_batch_mode") as mock_batch,
            patch.object(main, "AppWindow"),
        ):
            main.main()
            mock_batch.assert_called_once()
            # config の video_path が "test.mp4" であること
            config = mock_batch.call_args[0][0]
            assert "test.mp4" in str(config.video_path)
