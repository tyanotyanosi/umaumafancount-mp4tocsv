"""インテグレーションテスト: VLM パイプライン

    テスト対象:
    - PipelineRunner.run with VLM mode
    - VLMFrameProcessor.process_video_vlm
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import numpy as np
import pytest

from src.domain.models import PipelineConfig, VLMConfig, OCRResult
from src.core.pipeline import PipelineRunner, VLMFrameProcessor


class TestPipelineRunnerVLMMode:
    """PipelineRunner の VLM モードテスト"""

    def test_run_vlm_mode_uses_vlm_processor(self, tmp_path):
        """VLM有効時は VLMFrameProcessor が使用される"""
        config = PipelineConfig(
            video_path=tmp_path / "test.mp4",
            debug=False,
            vlm_config=VLMConfig(enabled=True),
        )

        mock_vlm_processor = MagicMock()
        mock_vlm_processor.process_video_vlm = MagicMock(return_value=["{}"])

        with (
            patch("src.core.pipeline.VLMFrameProcessor", return_value=mock_vlm_processor),
            patch("src.core.pipeline.FanCountExtractor.extract", return_value={}),
            patch("src.core.pipeline.shutil.rmtree"),
            patch("src.core.pipeline.Path.exists", return_value=True),
            patch("src.core.pipeline.Path.mkdir"),
            patch("src.core.pipeline.os.makedirs"),
        ):
            runner = PipelineRunner()
            result = runner.run(config, lambda msg, pct=None: None)

        assert isinstance(result, OCRResult)
        mock_vlm_processor.process_video_vlm.assert_called_once()

    def test_run_ocr_mode_uses_stream_processor(self, tmp_path):
        """VLM無効時は StreamFrameProcessor が使用される"""
        config = PipelineConfig(
            video_path=tmp_path / "test.mp4",
            debug=False,
            vlm_config=None,
        )

        with (
            patch("src.core.pipeline.StreamFrameProcessor.process_video", return_value=[]),
            patch("src.core.pipeline.FanCountExtractor.extract", return_value={}),
            patch("src.core.pipeline.shutil.rmtree"),
            patch("src.core.pipeline.Path.exists", return_value=True),
            patch("src.core.pipeline.Path.mkdir"),
            patch("src.core.pipeline.os.makedirs"),
        ):
            runner = PipelineRunner()
            result = runner.run(config, lambda msg, pct=None: None)

        assert isinstance(result, OCRResult)
        assert result.fan_counts == {}


class TestVLMFrameProcessor:
    """VLMFrameProcessor のテスト"""

    @pytest.mark.asyncio
    async def test_process_video_vlm_calls_vlm_service(self, tmp_path):
        """VLMフレームプロセッサが VLMService を呼び出す"""
        video_path = tmp_path / "test.mp4"
        video_path.touch()

        config = PipelineConfig(
            video_path=video_path,
            vlm_config=VLMConfig(enabled=True),
        )

        mock_vlm_service = MagicMock()
        mock_vlm_service.stop = AsyncMock()
        mock_vlm_service._prompt_template = """\
以下の画像からファン数表を読み取り、JSON 形式で出力してください。

出力形式:
{{
  "メンバー名 1": ファン数 1,
  "メンバー名 2": ファン数 2,
  ...
}}

注意:
- メンバー名が不明な場合はその行をスキップしてください
- ファン数はカンマ区切りの整数で出力してください
- JSON 以外は一切出力しないでください
"""
        mock_vlm_service.analyze_image = AsyncMock(return_value={"メンバー A": 12345})

        with (
            patch("src.core.pipeline.cv2.VideoCapture") as mock_cap,
            patch("src.core.pipeline.cv2.cvtColor", return_value=np.zeros((100, 100, 3), dtype=np.uint8)),
            patch("src.core.pipeline.Image.fromarray", return_value=MagicMock()),
            patch("src.core.pipeline.VLMService", return_value=mock_vlm_service),
            patch("src.core.pipeline.Path.exists", return_value=True),
        ):
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.return_value = 10
            mock_cap_instance.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8)), (False, None)]
            mock_cap.return_value = mock_cap_instance

            processor = VLMFrameProcessor(config)
            results = await processor.process_video_vlm()

        assert len(results) == 1
        assert "12345" in results[0]

    @pytest.mark.asyncio
    async def test_process_video_vlm_handles_read_error(self, tmp_path):
        """動画読込エラー時に RuntimeError を送出"""
        video_path = tmp_path / "nonexistent.mp4"

        config = PipelineConfig(
            video_path=video_path,
            vlm_config=VLMConfig(enabled=True),
        )

        with patch("src.core.pipeline.cv2.VideoCapture") as mock_cap:
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = False
            mock_cap.return_value = mock_cap_instance

            processor = VLMFrameProcessor(config)

            with pytest.raises(RuntimeError) as exc_info:
                await processor.process_video_vlm()

        assert "動画ファイルを読み込めませんでした" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_process_video_vlm_handles_vlm_error(self, tmp_path):
        """VLM推論エラー時にエラーメッセージを結果に含めて処理を継続する"""
        video_path = tmp_path / "test.mp4"
        video_path.touch()

        config = PipelineConfig(
            video_path=video_path,
            vlm_config=VLMConfig(enabled=True),
        )

        mock_vlm_service = MagicMock()
        mock_vlm_service.stop = AsyncMock()
        mock_vlm_service.analyze_image = AsyncMock(side_effect=Exception("推論エラー"))

        with (
            patch("src.core.pipeline.cv2.VideoCapture") as mock_cap,
            patch("src.core.pipeline.cv2.cvtColor", return_value=np.zeros((100, 100, 3), dtype=np.uint8)),
            patch("src.core.pipeline.Image.fromarray", return_value=MagicMock()),
            patch("src.core.pipeline.VLMService", return_value=mock_vlm_service),
            patch("src.core.pipeline.Path.exists", return_value=True),
        ):
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.get.return_value = 1
            mock_cap_instance.read.side_effect = [(True, np.zeros((480, 640, 3), dtype=np.uint8)), (False, None)]
            mock_cap.return_value = mock_cap_instance

            processor = VLMFrameProcessor(config)
            results = await processor.process_video_vlm()

        assert len(results) == 1
        assert "VLM_ERROR" in results[0]
