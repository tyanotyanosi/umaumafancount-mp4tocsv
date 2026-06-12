import cv2

class AppError(Exception):
    """アプリ固有のエラー。ユーザー向けメッセージを含む"""

    def __init__(self, message: str, hint: str = ""):
        self.message = message  # ユーザーに表示するエラー内容
        self.hint = hint        # 復元手順のヒント
        super().__init__(message)


def wrap_error(e: Exception) -> AppError:
    """一般的な例外をユーザー向けのメッセージにマッピング"""

    if isinstance(e, FileNotFoundError):
        path = str(getattr(e, 'filename', '不明'))
        if "memberList.txt" in path or "memberReplace.json" in path:
            return AppError(
                message=f"設定ファイルが見つかりません: {path}",
                hint="input/ ディレクトリに memberList.txt と memberReplace.json が存在するか確認してください。",
            )
        elif ".mp4" in path or "frames" in path:
            return AppError(
                message=f"動画ファイルが見つかりません: {path}",
                hint="ファイルが存在し、アクセス権があるか確認してください。",
            )

    # OpenCV のエラー判定
    is_cv2_error = False
    try:
        if hasattr(cv2, 'error') and isinstance(e, type) is False:
            is_cv2_error = isinstance(e, cv2.error)
    except TypeError:
        pass

    if not is_cv2_error and "cv2" in str(type(e).__module__):
        is_cv2_error = True

    if is_cv2_error:  # OpenCV のエラー
        return AppError(
            message=f"動画の読み込み中にエラーが発生しました: {e.args[0] if e.args else '不明'}",
            hint="ファイルが破損しているか、サポートされていない形式かもしれません。MP4(H.264)形式をお試しください。",
        )

    # WinRT OCR 関連エラーのパターンマッチ
    err_str = str(e)
    if any(kw in err_str for kw in ("OcrEngine", "BitmapDecoder", "WinRT")):
        return AppError(
            message=f"OCRエンジンの初期化に失敗しました: {e}",
            hint="Windowsの言語設定（日本語）を確認してください。",
        )

    # 汎用フォールバック
    return AppError(
        message=f"予期しないエラーが発生しました: {e}",
        hint="ログを確認するか、開発者に報告してください.",
    )
