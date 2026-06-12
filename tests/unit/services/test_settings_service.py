"""ユニットテスト: src.services.settings_service

  テスト対象:
    - SettingsService
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.domain.models import AppSettings
from src.services.settings_service import SettingsService


class TestSettingsService:
    """SettingsService のテスト"""

    def test_default_constructor_uses_home_dir(self):
        """デフォルトコンストラクタはホームディレクトリ以下のパスを使用"""
        sm = SettingsService()
        path = sm.settings_path
        assert ".umamusume-fan-count" in str(path)
        assert path.name == "settings.json"

    def test_load_default_when_no_file(self, tmp_path):
        """設定ファイルが存在しない場合はデフォルト値を返す"""
        sm = SettingsService(settings_path=tmp_path / "nonexistent.json")
        result = sm.load()
        assert isinstance(result, AppSettings)
        assert result.roi_y_start == 0.45

    def test_load_existing_file(self, tmp_path):
        """設定ファイルが存在する場合は読み込む"""
        settings_file = tmp_path / "test_settings.json"
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump({
                "roi_y_start": 0.30,
                "roi_y_end": 0.90,
                "roi_x_start": 0.20,
                "roi_x_end": 0.50,
                "img_scale": "gray",
                "debug": True,
            }, f)

        sm = SettingsService(settings_path=settings_file)
        result = sm.load()
        assert result.roi_y_start == 0.30
        assert result.roi_x_end == 0.50
        assert result.img_scale == "gray"
        assert result.debug is True

    def test_save_and_roundtrip(self, tmp_path):
        """保存→読み込みのラウンドトリップが正しく動作"""
        sm = SettingsService(settings_path=tmp_path / "settings.json")

        settings = AppSettings(roi_y_start=0.25, debug=True)
        sm.save(settings)

        loaded = sm.load()
        assert loaded.roi_y_start == 0.25
        assert loaded.debug is True

    def test_save_creates_directory(self, tmp_path):
        """ディレクトリが存在しない場合は自動作成"""
        deep_dir = tmp_path / "sub" / "deep"
        settings_file = deep_dir / "settings.json"
        sm = SettingsService(settings_path=settings_file)

        settings = AppSettings(debug=False)
        sm.save(settings)  # 例外が発生してはならない
        assert settings_file.exists()
        assert deep_dir.exists()

    def test_load_malformed_json_returns_default(self, tmp_path):
        """壊れた JSON ファイルでもデフォルト値を返す"""
        settings_file = tmp_path / "bad.json"
        settings_file.write_text("{不正なJSON}", encoding="utf-8")

        sm = SettingsService(settings_path=settings_file)
        result = sm.load()
        assert isinstance(result, AppSettings)

    def test_validate_roi_valid(self):
        """有効なROI値はエラーなし"""
        errors = SettingsService.validate_roi(0.1, 0.9, 0.2, 0.8)
        assert errors == []

    def test_validate_roi_out_of_range(self):
        """範囲外（>1.0）の値はエラー"""
        errors = SettingsService.validate_roi(-0.1, 0.9, 0.2, 1.5)
        assert len(errors) == 2

    def test_validate_roi_inverted(self):
        """start >= end はエラー"""
        errors = SettingsService.validate_roi(0.8, 0.3, 0.2, 0.5)
        assert any("y_start" in e for e in errors)

    def test_validate_roi_x_inverted(self):
        """x_start >= x_end もエラー"""
        errors = SettingsService.validate_roi(0.1, 0.9, 0.8, 0.3)
        assert any("x_start" in e for e in errors)

    def test_apply_to_config(self, tmp_path):
        """apply_to_config が PipelineConfig の ROI を上書き"""
        settings_file = tmp_path / "settings.json"
        with open(settings_file, 'w', encoding='utf-8') as f:
            json.dump({
                "roi_y_start": 0.10,
                "roi_y_end": 0.95,
                "roi_x_start": 0.05,
                "roi_x_end": 0.60,
                "img_scale": None,
                "debug": False,
            }, f)

        from src.domain.models import PipelineConfig
        config = PipelineConfig(video_path=tmp_path / "test.mp4")
        sm = SettingsService(settings_path=settings_file)
        sm.apply_to_config(config)

        assert config.roi_y_start == 0.10
        assert config.roi_y_end == 0.95
        assert config.roi_x_start == 0.05
        assert config.roi_x_end == 0.60
