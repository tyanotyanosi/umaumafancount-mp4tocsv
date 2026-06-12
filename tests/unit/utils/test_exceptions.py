"""ユニットテスト: src.utils.exceptions

  テスト対象:
    - AppError
    - wrap_error
"""

from __future__ import annotations

from src.utils.exceptions import AppError, wrap_error


class TestAppError:
    """AppError の基本動作"""

    def test_basic(self):
        err = AppError("エラー発生", "対策してください")
        assert err.message == "エラー発生"
        assert err.hint == "対策してください"

    def test_no_hint(self):
        err = AppError("単独エラー")
        assert err.message == "単独エラー"
        assert err.hint == ""

    def test_is_exception(self):
        err = AppError("テスト")
        assert isinstance(err, Exception)


class TestWrapError:
    """wrap_error() のエラーマッピング"""

    def test_file_not_found_member_list(self):
        e = FileNotFoundError(2, "No such file", "input/memberList.txt")
        app_err = wrap_error(e)
        assert "設定ファイル" in app_err.message
        assert "memberList.txt" in app_err.message

    def test_file_not_found_video(self):
        e = FileNotFoundError(2, "No such file", "video.mp4")
        app_err = wrap_error(e)
        assert "動画ファイル" in app_err.message

    def test_file_not_found_other(self):
        """その他の FileNotFoundError は汎用メッセージに"""
        e = FileNotFoundError(2, "No such file", "other.txt")
        app_err = wrap_error(e)
        assert app_err.message  # 何らかのメッセージが入っている

    def test_cv2_module_error(self):
        """__module__ が cv2 のエラーは動画読み込みエラーとして扱われる"""
        class FakeCv2Error(Exception):
            pass
        FakeCv2Error.__module__ = "cv2"
        e = FakeCv2Error("test opencv error")
        app_err = wrap_error(e)
        assert "動画の読み込み" in app_err.message

    def test_ocr_engine_error(self):
        """OCR関連のキーワードを含むエラーはOCRエラーとして扱われる"""
        e = RuntimeError("OcrEngine failed to initialize")
        app_err = wrap_error(e)
        assert "OCRエンジン" in app_err.message

    def test_generic_fallback(self):
        e = ValueError("generic_value_error")
        app_err = wrap_error(e)
        assert "予期しないエラー" in app_err.message
