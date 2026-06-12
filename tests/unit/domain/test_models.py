"""ユニットテスト: src.domain.models

  テスト対象:
    - PipelineConfig
    - OCRResult
    - ProgressUpdate
    - AppSettings
    - MemberEntry
"""

from __future__ import annotations

from pathlib import Path

from src.domain.models import (
    PipelineConfig,
    OCRResult,
    ProgressUpdate,
    AppSettings,
    MemberEntry,
)


class TestPipelineConfig:
    """PipelineConfig dataclass"""

    def test_defaults(self):
        config = PipelineConfig(video_path=Path("test.mp4"))
        assert config.video_path == Path("test.mp4")
        assert config.debug is False
        assert config.img_scale is None
        assert config.roi_y_start == 0.45
        assert config.roi_y_end == 0.88
        assert config.roi_x_start == 0.15
        assert config.roi_x_end == 0.45

    def test_custom_values(self):
        config = PipelineConfig(
            video_path=Path("/tmp/video.mp4"),
            debug=True,
            img_scale="gray",
            roi_y_start=0.3,
            roi_y_end=0.9,
            roi_x_start=0.1,
            roi_x_end=0.6,
        )
        assert config.video_path == Path("/tmp/video.mp4")
        assert config.debug is True
        assert config.img_scale == "gray"
        assert config.roi_y_start == 0.3


class TestOCRResult:
    """OCRResult dataclass"""

    def test_success(self):
        result = OCRResult(
            fan_counts={"メンバーA": 1000}, texts=["テキスト"], error=None
        )
        assert result.fan_counts == {"メンバーA": 1000}
        assert result.texts == ["テキスト"]
        assert result.error is None

    def test_error(self):
        result = OCRResult(fan_counts={}, texts=[], error="エラーメッセージ")
        assert result.error == "エラーメッセージ"
        assert result.fan_counts == {}
        assert result.texts == []


class TestProgressUpdate:
    """ProgressUpdate dataclass"""

    def test_fields(self):
        pu = ProgressUpdate(
            message="OCR文字認識中",
            percent=0.5,
            frame_current=50,
            frame_total=100,
        )
        assert pu.message == "OCR文字認識中"
        assert pu.percent == 0.5
        assert pu.frame_current == 50
        assert pu.frame_total == 100

    def test_defaults(self):
        pu = ProgressUpdate(message="テスト", percent=0.3)
        assert pu.message == "テスト"
        assert pu.percent == 0.3
        assert pu.frame_current is None
        assert pu.frame_total is None


class TestAppSettings:
    """AppSettings dataclass"""

    def test_defaults(self):
        settings = AppSettings()
        assert settings.roi_y_start == 0.45
        assert settings.roi_y_end == 0.88
        assert settings.roi_x_start == 0.15
        assert settings.roi_x_end == 0.45
        assert settings.img_scale is None
        assert settings.debug is False

    def test_custom_values(self):
        settings = AppSettings(
            roi_y_start=0.3,
            roi_y_end=0.9,
            roi_x_start=0.2,
            roi_x_end=0.5,
            img_scale="gray",
            debug=True,
        )
        assert settings.roi_y_start == 0.3
        assert settings.roi_y_end == 0.9
        assert settings.img_scale == "gray"
        assert settings.debug is True


class TestMemberEntry:
    """MemberEntry dataclass"""

    def test_defaults(self):
        entry = MemberEntry(name="万丈目準")
        assert entry.name == "万丈目準"
        assert entry.replace_patterns == []

    def test_custom_patterns(self):
        pats = ["メンハー", "準→準"]
        entry = MemberEntry(name="テスト", replace_patterns=pats)
        assert entry.name == "テスト"
        assert len(entry.replace_patterns) == 2
