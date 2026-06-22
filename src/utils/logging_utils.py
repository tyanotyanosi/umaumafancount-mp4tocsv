"""VLM推論の成功/エラー時の詳細ログ出力用ユーティリティ"""

import traceback
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("vlm_error_logger")
logger.setLevel(logging.DEBUG)

success_logger = logging.getLogger("vlm_success_logger")
success_logger.setLevel(logging.DEBUG)


def _get_or_create_error_logger() -> logging.Logger:
    """エラー用ロガーを取得（重複追加防止）"""
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _get_or_create_success_logger() -> logging.Logger:
    """成功用ロガーを取得（重複追加防止）"""
    if not success_logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        success_logger.addHandler(handler)
    return success_logger


def log_vlm_success(
    frame_index: Optional[int] = None,
    raw_response: Optional[str] = None,
    prompt: Optional[str] = None,
    extracted_json: Optional[str] = None,
    parsed_result: Optional[dict] = None,
    image_size: Optional[tuple] = None,
) -> None:
    """VLM推論成功の詳細ログを出力

    引数:
        frame_index: フレーム番号
        raw_response: VLMからの生レスポンス
        prompt: 送信されたプロンプト
        extracted_json: 抽出されたJSON文字列
        parsed_result: パース後の結果辞書
        image_size: 画像サイズ (width, height)
    """
    _get_or_create_success_logger()

    context_parts = []
    if frame_index is not None:
        context_parts.append(f"フレーム: {frame_index}")
    if image_size is not None:
        context_parts.append(f"画像サイズ: {image_size[0]}x{image_size[1]}")
    if raw_response is not None:
        context_parts.append(f"生レスポンス長: {len(raw_response)} chars")
    if extracted_json is not None:
        context_parts.append(f"抽出JSON: {repr(extracted_json[:300])}")
    if parsed_result is not None:
        context_parts.append(f"解析結果: {parsed_result}")

    log_msg = f"[VLM_SUCCESS] 推論成功"
    if context_parts:
        log_msg += f"\n{'=' * 60}\n" + "\n".join(context_parts)
    log_msg += f"\n{'=' * 60}"

    success_logger.info(log_msg)

    _save_success_to_file(frame_index, raw_response, prompt, extracted_json, parsed_result, image_size)


def _save_success_to_file(
    frame_index: Optional[int],
    raw_response: Optional[str],
    prompt: Optional[str],
    extracted_json: Optional[str],
    parsed_result: Optional[dict],
    image_size: Optional[tuple],
) -> None:
    """成功情報をファイルに保存（debugモード時）"""
    try:
        success_dir = Path("output/debug/success_logs")
        success_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        success_file = success_dir / f"success_{timestamp}.log"

        with open(success_file, "w", encoding="utf-8") as f:
            f.write(f"処理時刻: {datetime.now().isoformat()}\n")
            if frame_index is not None:
                f.write(f"フレーム番号: {frame_index}\n")
            if image_size is not None:
                f.write(f"画像サイズ: {image_size[0]}x{image_size[1]}\n")
            f.write("\n" + "=" * 60 + "\n")

            if raw_response is not None:
                f.write("生レスポンス:\n")
                f.write(raw_response)
                f.write("\n" + "=" * 60 + "\n")

            if extracted_json is not None:
                f.write("抽出JSON:\n")
                f.write(extracted_json)
                f.write("\n" + "=" * 60 + "\n")

            if parsed_result is not None:
                f.write("解析結果:\n")
                f.write(str(parsed_result))
                f.write("\n" + "=" * 60 + "\n")

            if prompt is not None:
                f.write("プロンプト:\n")
                f.write(prompt)
                f.write("\n" + "=" * 60 + "\n")

    except Exception:
        pass


def log_vlm_error(
    error: Exception,
    raw_response: Optional[str] = None,
    prompt: Optional[str] = None,
    extracted_json: Optional[str] = None,
    frame_index: Optional[int] = None,
    image_size: Optional[tuple] = None,
    base64_length: Optional[int] = None,
) -> str:
    """VLM推論エラーの詳細ログを出力し、エラーメッセージを返す

    引数:
        error: 発生した例外
        raw_response: VLMからの生レスポンス
        prompt: 送信されたプロンプト
        extracted_json: 抽出されたJSON文字列
        frame_index: エラー発生時のフレーム番号
        image_size: 画像サイズ (width, height)
        base64_length: base64エンコーディング後の長さ

    戻り値:
        ユーザー表示用のエラーメッセージ
    """
    _get_or_create_error_logger()

    # トレースバックの取得
    tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
    tb_text = "".join(tb_lines)

    # コンテキスト情報の構築
    context_parts = []
    if frame_index is not None:
        context_parts.append(f"フレーム: {frame_index}")
    if image_size is not None:
        context_parts.append(f"画像サイズ: {image_size[0]}x{image_size[1]}")
    if base64_length is not None:
        context_parts.append(f"base64長: {base64_length} chars")
    if raw_response is not None:
        context_parts.append(f"生レスポンス: {repr(raw_response[:500])}")
    if extracted_json is not None:
        context_parts.append(f"抽出JSON: {repr(extracted_json[:500])}")

    # エラーログの出力
    log_msg = f"[VLM_ERROR] {type(error).__name__}: {error}"
    if context_parts:
        log_msg += f"\n{'=' * 60}\n" + "\n".join(context_parts)
    log_msg += f"\n{'=' * 60}\nトレースバック:\n{tb_text}"

    logger.error(log_msg)

    # 詳細エラーメッセージの構築（ユーザー表示用）
    error_msg = f"VLM 推論中にエラーが発生しました: {type(error).__name__}: {error}"
    if context_parts:
        error_msg += f"\n[詳細] {' | '.join(context_parts[:2])}"

    # エラーログファイルへの保存
    _save_error_to_file(error, tb_text, raw_response, prompt, extracted_json, frame_index)

    return error_msg


def _save_error_to_file(
    error: Exception,
    traceback_text: str,
    raw_response: Optional[str],
    prompt: Optional[str],
    extracted_json: Optional[str],
    frame_index: Optional[int],
    image_size: Optional[tuple] = None,
    base64_length: Optional[int] = None,
) -> None:
    """エラー情報をファイルに保存"""
    try:
        error_dir = Path("output/debug/error_logs")
        error_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        error_file = error_dir / f"error_{timestamp}.log"

        with open(error_file, "w", encoding="utf-8") as f:
            f.write(f"エラー発生時刻: {datetime.now().isoformat()}\n")
            f.write(f"エラー種類: {type(error).__name__}\n")
            f.write(f"エラーメッセージ: {error}\n")
            if frame_index is not None:
                f.write(f"フレーム番号: {frame_index}\n")
            if image_size is not None:
                f.write(f"画像サイズ: {image_size[0]}x{image_size[1]}\n")
            if base64_length is not None:
                f.write(f"base64長: {base64_length} chars\n")
            f.write("\n" + "=" * 60 + "\n")
            f.write("トレースバック:\n")
            f.write(traceback_text)
            f.write("\n" + "=" * 60 + "\n")

            if raw_response is not None:
                f.write("生レスポンス:\n")
                f.write(raw_response)
                f.write("\n" + "=" * 60 + "\n")

            if extracted_json is not None:
                f.write("抽出JSON:\n")
                f.write(extracted_json)
                f.write("\n" + "=" * 60 + "\n")

            if prompt is not None:
                f.write("プロンプト:\n")
                f.write(prompt)
                f.write("\n" + "=" * 60 + "\n")

    except Exception:
        pass
