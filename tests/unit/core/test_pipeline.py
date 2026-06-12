"""ユニットテスト: src.core.pipeline

  テスト対象:
    - PipelineRunner
    - PipelineWorker
"""

from __future__ import annotations

import queue
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.domain.models import PipelineConfig, OCRResult
from src.core.pipeline import PipelineRunner, PipelineWorker


class TestPipelineRunner:
    """PipelineRunner のテスト（モック環境）"""

    def test_run_returns_ocr_result_on_error(self, tmp_path):
        """エラー発生時は error フィールドに内容が入る"""
        config = PipelineConfig(
            video_path=tmp_path / "dummy.mp4", debug=False
        )

        # StreamFrameProcessor.process_video で例外を投げる
        with patch(
            "src.core.processor.StreamFrameProcessor.process_video",
            side_effect=RuntimeError("テストエラー"),
        ):
            runner = PipelineRunner()
            result = runner.run(config, lambda msg, pct=None: None)

        assert isinstance(result, OCRResult)
        assert result.error is not None
        assert "テストエラー" in result.error

    def test_run_success_path(self, tmp_path):
        """正常系: 空の texts が返され、fan_counts が空辞書になる"""
        config = PipelineConfig(
            video_path=tmp_path / "dummy.mp4", debug=False
        )

        with (
            patch("src.core.processor.StreamFrameProcessor.process_video",
                  return_value=[]),
            patch("src.core.extractor.FanCountExtractor.extract",
                  return_value={}),
        ):
            runner = PipelineRunner()
            result = runner.run(config, lambda msg, pct=None: None)

        assert isinstance(result, OCRResult)
        assert result.error is None
        assert result.fan_counts == {}


class TestPipelineWorkerInit:
    """PipelineWorker の初期化と属性チェック"""

    def test_worker_attributes(self):
        config = PipelineConfig(video_path=Path("test.mp4"))
        q = queue.Queue()
        worker = PipelineWorker(config, q)

        assert worker.config is config
        assert worker.result_queue is q
        assert isinstance(worker.runner, PipelineRunner)
