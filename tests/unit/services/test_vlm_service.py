"""ユニットテスト: src.services.vlm_service

    テスト対象:
    - VLMService._parse_json_response
    - VLMService._check_model_files
    - VLMService.analyze_image (mock)
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

import pytest
import asyncio

from src.domain.models import VLMConfig
from src.services.vlm_service import VLMService, DEFAULT_PROMPT_TEMPLATE
from src.utils.exceptions import AppError


@pytest.fixture
def vlm_config():
    return VLMConfig(
        enabled=True,
        model_path="models/gemma-4-e2b-it-edited-q4_0.gguf",
           mmproj_path="models/mmproj-gemma-4-e2b-it-q4_0.gguf",
        temperature=0.1,
        max_tokens=512,
        port=8080,
    )


@pytest.fixture
def vlm_service(vlm_config):
    vlm_config.prompt_template = DEFAULT_PROMPT_TEMPLATE
    return VLMService(vlm_config)


class TestVLMServiceParseJsonResponse:
    """VLMService._parse_json_response のテスト"""

    def test_parse_valid_json(self, vlm_service):
        text = '{"メンバー A": 12345, "メンバー B": 67890}'
        result = vlm_service._parse_json_response(text)
        assert result == {"メンバー A": 12345, "メンバー B": 67890}

    def test_parse_json_with_commas(self, vlm_service):
        text = '{"メンバー A": "12,345", "メンバー B": "67,890"}'
        result = vlm_service._parse_json_response(text)
        assert result == {"メンバー A": 12345, "メンバー B": 67890}

    def test_parse_json_with_braces_in_text(self, vlm_service):
        text = '以下のように出力します：{"メンバー A": 12345}'
        result = vlm_service._parse_json_response(text)
        assert result == {"メンバー A": 12345}

    def test_parse_invalid_json_raises_error(self, vlm_service):
        text = 'これは JSON ではありません'
        with pytest.raises(AppError) as exc_info:
            vlm_service._parse_json_response(text)
        assert "VLM の応答から JSON を抽出できませんでした" in str(exc_info.value)

    def test_parse_json_with_float_values(self, vlm_service):
        text = '{"メンバー A": 12345.0, "メンバー B": 67890.5}'
        result = vlm_service._parse_json_response(text)
        assert result == {"メンバー A": 12345, "メンバー B": 67890}

    def test_parse_json_skips_invalid_values(self, vlm_service):
        text = '{"メンバー A": 12345, "メンバー B": "invalid"}'
        result = vlm_service._parse_json_response(text)
        assert result == {"メンバー A": 12345}
        assert "メンバー B" not in result

    def test_parse_json_skips_empty_keys(self, vlm_service):
        text = '{"": 12345, "メンバー A": 67890}'
        result = vlm_service._parse_json_response(text)
        assert result == {"メンバー A": 67890}


class TestVLMServiceCheckModelFiles:
    """VLMService._check_model_files のテスト"""

    def test_raises_when_vlm_disabled(self):
        config = VLMConfig(enabled=False)
        service = VLMService(config)
        with pytest.raises(AppError) as exc_info:
            service._check_model_files()
        assert "VLM が無効です" in str(exc_info.value)

    def test_raises_when_model_file_not_found(self, vlm_config):
        vlm_config.model_path = "nonexistent_model.gguf"

        with patch.object(Path, "exists", side_effect=[True, True, False]):
            service = VLMService(vlm_config)
            with pytest.raises(FileNotFoundError) as exc_info:
                service._check_model_files()
        assert "モデルファイルが見つかりません" in str(exc_info.value)

    def test_raises_when_mmproj_file_not_found(self, vlm_config):
        vlm_config.mmproj_path = "nonexistent_mmproj.gguf"
        vlm_config.prompt_template = "test prompt"

        with patch.object(Path, "exists", side_effect=[True, True, False]):
            service = VLMService(vlm_config)
            with pytest.raises(FileNotFoundError) as exc_info:
                service._check_model_files()
        assert "マルチモーダルプロジェクトファイルが見つかりません" in str(exc_info.value)

    def test_succeeds_when_all_files_exist(self, vlm_config):
        with patch.object(Path, "exists", return_value=True):
            service = VLMService(vlm_config)
            service._check_model_files()  # Should not raise


class TestVLMServiceAnalyzeImage:
    """VLMService.analyze_image のテスト (mock)"""

    @pytest.mark.asyncio
    async def test_analyze_image_returns_parsed_result(self, vlm_service):
        mock_image = MagicMock()
        mock_image.size = (640, 480)
        
        mock_enhanced = MagicMock()
        mock_enhanced.size = (1280, 960)
        
        with (
            patch.object(vlm_service, "_check_model_files"),
            patch.object(vlm_service, "_enhance_image", return_value=mock_enhanced),
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch.object(vlm_service, "_run_llama_cli", return_value='{"メンバー A": 12345, "メンバー B": 67890}'),
        ):
            mock_temp.return_value.__enter__ = lambda s: s
            mock_temp.return_value.__exit__ = lambda s, *a: None
            mock_temp.return_value.name = "temp.png"
            
            result = await vlm_service.analyze_image(mock_image)

        assert result == {"メンバー A": 12345, "メンバー B": 67890}

    @pytest.mark.asyncio
    async def test_analyze_image_raises_on_invalid_response_format(self, vlm_service):
        mock_image = MagicMock()
        mock_image.size = (640, 480)
        
        mock_enhanced = MagicMock()
        mock_enhanced.size = (1280, 960)

        with (
            patch.object(vlm_service, "_check_model_files"),
            patch.object(vlm_service, "_enhance_image", return_value=mock_enhanced),
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch.object(vlm_service, "_run_llama_cli", return_value='invalid json'),
        ):
            mock_temp.return_value.__enter__ = lambda s: s
            mock_temp.return_value.__exit__ = lambda s, *a: None
            mock_temp.return_value.name = "temp.png"
            
            with pytest.raises(AppError) as exc_info:
                await vlm_service.analyze_image(mock_image)
        assert "VLM の応答から JSON を抽出できませんでした" in str(exc_info.value)


class TestVLMServicePromptTemplate:
    """VLMServiceのプロンプトテンプレート読み込みテスト"""

    def test_uses_default_prompt_when_no_file(self, vlm_config):
        with patch("pathlib.Path.exists", return_value=False):
            service = VLMService(vlm_config)
            assert service._prompt_template == DEFAULT_PROMPT_TEMPLATE

    def test_uses_custom_prompt_from_config(self, vlm_config):
        custom_prompt = "カスタムプロンプト"
        vlm_config.prompt_template = custom_prompt
        service = VLMService(vlm_config)
        assert service._prompt_template == custom_prompt

    def test_loads_prompt_from_file_if_exists(self, vlm_config, tmp_path):
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("ファイルから読み込んだプロンプト", encoding="utf-8")

        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.read_text", return_value="ファイルから読み込んだプロンプト"):
                service = VLMService(vlm_config)
                assert service._prompt_template == "ファイルから読み込んだプロンプト"
