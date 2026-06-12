"""ユニットテスト: src.core.processor

  テスト対象:
    - StreamFrameProcessor
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.domain.models import PipelineConfig
from src.core.processor import StreamFrameProcessor


class TestStreamFrameProcessor:
    """StreamFrameProcessor の軽量テスト"""

    def test_init(self):
        """初期化時に config と ocr_service が保持される"""
        config = PipelineConfig(video_path=Path("dummy.mp4"))
        ocr_mock = MagicMock()
        processor = StreamFrameProcessor(config, ocr_mock)
        assert processor.config is config
        assert processor.ocr_service is ocr_mock
