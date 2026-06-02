# ocr_winrt_safe.py — Umamusume Fan Count Extractor
import sys
import asyncio
from io import BytesIO
from PIL import Image
import cv2
from typing import Iterable, Tuple, Optional, List, Callable, Dict
import numpy as np
import pathlib
import os
import glob
import shutil
from tqdm import tqdm
import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict, Field
import threading
import queue
import tkinter as tk
import tkinter.messagebox as mb
import tkinter.ttk as ttk

# --- Windows の一部環境で念のため ---
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

# WinRT (PyWinRT)
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.graphics.imaging import BitmapDecoder
from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream


# ==============================================================================
# データ構造体（データ層）
# ==============================================================================

@dataclass
class PipelineConfig:
    """OCR パイプライン実行の設定"""
    video_path: Path
    debug: bool = False
    img_scale: Optional[str] = None  # "gray" or None
    roi_y_start: float = 0.45       # ROI Y軸開始（画面高さの割合）
    roi_y_end: float = 0.88         # ROI Y軸終了（画面高さの割合）
    roi_x_start: float = 0.15       # ROI X軸開始（画面幅の割合）
    roi_x_end: float = 0.45         # ROI X軸終了（画面幅の割合）


from collections import namedtuple

# B2: 進捗更新データ（namedtuple ベースの dataclass 互換）
# __slots__ + default は namedtuple では使えないため、純粋な dataclass に変更
@dataclass
class ProgressUpdate:
    """進捗更新データ"""
    message: str            # 進捗メッセージ
    percent: float          # 0.0 〜 1.0 の進捗率
    frame_current: int | None = None  # OCRフェーズでの現在フレーム数
    frame_total: int | None = None   # OCRフェーズでの総フレーム数


@dataclass
class OCRResult:
    """OCR パイプラインの完了結果"""
    fan_counts: dict
    texts: list
    error: str | None = None


# A2: アプリ設定（ROI カスタマイズ用）
@dataclass
class AppSettings:
    """アプリ全体の設定値"""
    roi_y_start: float = 0.45
    roi_y_end: float = 0.88
    roi_x_start: float = 0.15
    roi_x_end: float = 0.45
    img_scale: str | None = None
    debug: bool = False


# C2: メンバーエントリ
@dataclass
class MemberEntry:
    """1人のメンバーの定義"""
    name: str                    # メンバー名（memberList.txt に出力）
    replace_patterns: List[str] = field(default_factory=list)  # OCR誤認識パターン


# ==============================================================================
# エラーハンドリング（A3: エラーメッセージ改善）
# ==============================================================================

class AppError(Exception):
    """アプリ固有のエラー。ユーザー向けメッセージを含む"""

    def __init__(self, message: str, hint: str = ""):
        self.message = message  # ユーザーに表示するエラー内容
        self.hint = hint        # 復旧手順のヒント
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

    # OpenCV のエラー（モック環境で cv2.error が偽物でも安全に判定）
    is_cv2_error = False
    try:
        if hasattr(cv2, 'error') and isinstance(e, type) is False:
            is_cv2_error = isinstance(e, cv2.error)
    except TypeError:
        # モック環境で cv2.error が型として機能しない場合にフォールバック
        pass

    # モックでは cv2.error が Exception のサブクラスではないため、
    # 例外クラス名で判定するフォールバックも用意
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
        hint="ログを確認するか、開発者に報告してください。",
    )


# ==============================================================================
# A2: 設定管理モジュール（SettingManager）
# ==============================================================================

class SettingsManager:
    """アプリ設定の保存・読み込み"""

    _SETTINGS_DIR = ".umamusume-fan-count"

    def __init__(self):
        self._settings_path = Path.home() / self._SETTINGS_DIR / "settings.json"

    @property
    def settings_path(self) -> Path:
        return self._settings_path

    @settings_path.setter
    def settings_path(self, value: Path):
        """テスト用：設定ファイルパスを上書き可能"""
        self._settings_path = value

    def load(self) -> AppSettings:
        """ファイルから設定を読み込む。存在しない場合はデフォルト値を返す"""
        if not self.settings_path.exists():
            return AppSettings()  # デフォルト値
        with open(self.settings_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # dataclass にマージ（存在しないフィールドはデフォルト）
        defaults = {k.name: k.default for k in AppSettings.__dataclass_fields__.values()}
        merged = {}
        for key, default_val in defaults.items():
            if isinstance(default_val, Field):
                # default_factory の場合は空リストなど
                merged[key] = data.get(key, [])
            else:
                merged[key] = data.get(key, default_val)
        return AppSettings(**merged)

    def save(self, settings: AppSettings):
        """設定をファイルに保存"""
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.settings_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(settings), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _validate_roi(y_start, y_end, x_start, x_end):
        """ROI値のバリデーション"""
        errors = []
        for name, val in [("y_start", y_start), ("y_end", y_end),
                          ("x_start", x_start), ("x_end", x_end)]:
            if not (0.0 <= val <= 1.0):
                errors.append(f"{name} は 0.0〜1.0 の範囲で指定してください。")
        if y_start >= y_end:
            errors.append("y_start は y_end より小さい値にしてください。")
        if x_start >= x_end:
            errors.append("x_start は x_end より小さい値にしてください。")
        return errors

    def apply_to_config(self, config: PipelineConfig) -> PipelineConfig:
        """AppSettings の値を PipelineConfig に反映"""
        settings = self.load()
        config.roi_y_start = settings.roi_y_start
        config.roi_y_end = settings.roi_y_end
        config.roi_x_start = settings.roi_x_start
        config.roi_x_end = settings.roi_x_end
        return config


# ==============================================================================
# C2: メンバー管理モジュール（MemberManager）
# ==============================================================================

class MemberManager:
    """メンバーリストと置換JSONの読み込み・書き出し"""

    def load(self, list_path: Path, replace_path: Path) -> Dict[str, MemberEntry]:
        members = {}
        # memberList.txt を読む
        if list_path.exists():
            with open(list_path, 'r', encoding='utf-8') as f:
                for line in f:
                    name = line.strip()
                    if name:
                        members[name] = MemberEntry(name=name)

        # memberReplace.json の置換パターンをマージ
        if replace_path.exists():
            with open(replace_path, 'r', encoding='utf-8') as f:
                try:
                    replace_data = json.load(f)
                except json.JSONDecodeError:
                    replace_data = {}
            for name, patterns in replace_data.items():
                if name in members and isinstance(patterns, list):
                    members[name].replace_patterns = patterns

        return members

    def save(self, members: Dict[str, MemberEntry], list_path: Path, replace_path: Path):
        """memberList.txt と memberReplace.json を両方書き出す"""
        # ディレクトリ確保
        list_path.parent.mkdir(parents=True, exist_ok=True)

        # memberList.txt を書き出し
        with open(list_path, 'w', encoding='utf-8') as f:
            for entry in members.values():
                f.write(f"{entry.name}\n")

        # memberReplace.json を書き出し（空パターンは除外）
        replace_data = {}
        for entry in members.values():
            if entry.replace_patterns:
                replace_data[entry.name] = entry.replace_patterns

        with open(replace_path, 'w', encoding='utf-8') as f:
            json.dump(replace_data, f, ensure_ascii=False, indent=2)


# ==============================================================================
# CLI引数パース（既存互換）
# ==============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Umamusume Pretty Derby の動画からファン数を抽出するツール(WinRt OCR使用)"
    )
    p.add_argument(
        "video", nargs="?", type=str, default=None,
        help="入力動画のファイル名。省略時はGUIが起動します。"
    )
    p.add_argument("--gui", action="store_true", help="GUIを起動する")
    p.add_argument("--debug", action="store_true", help="中間ファイルを削除せずに残す")
    p.add_argument(
        "--img-scale", type=str, default=None,
        help="グレースケールで文字認識する場合は'gray'を指定"
    )
    return p.parse_args()


# ==============================================================================
# ファイル選択ダイアログ（既存互換）
# ==============================================================================

def select_video_gui() -> str | None:
    """Windows標準のファイル選択ダイアログを開き、選択された動画ファイルの絶対パスを返す。"""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("エラー: tkinter が利用できません。")
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    file_path = filedialog.askopenfilename(
        title="解析する動画ファイルを選択してください",
        initialdir=os.path.abspath("input"),
        filetypes=[("MP4 動画ファイル", "*.mp4"), ("すべてのファイル", "*.*")]
    )

    root.destroy()
    return file_path if file_path else None


# ==============================================================================
# OCR エンジン・ユーティリティ（既存関数シグネチャを維持）
# ==============================================================================

def cleanup(base_path, debug_path):
    shutil.rmtree(base_path, ignore_errors=True)
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(debug_path / "crop", exist_ok=True)
    os.makedirs(debug_path / "gray", exist_ok=True)
    os.makedirs(debug_path / "frames", exist_ok=True)
    os.makedirs(debug_path, exist_ok=True)
    os.makedirs(debug_path / "text", exist_ok=True)


def save_all_frames(video_path, dir_path, basename, ext='png'):
    video_path = Path(video_path)
    if video_path.is_absolute():
        cap = cv2.VideoCapture(str(video_path))
    else:
        cap = cv2.VideoCapture(str(Path("./input/") / video_path))

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)
    digit = len(str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    n = 0
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imwrite('{}_{}.{}'.format(base_path, str(n).zfill(digit), ext), frame)
            n += 1
        else:
            break
    cap.release()


def _crop_image(img, dir_path, basename, ext='png'):
    h, w = img.shape[:2]
    cropped_img = img[int(h*0.45):int(h*0.88), int(w*0.15):int(w*0.45)]

    base_path = os.path.join(dir_path, basename)
    cv2.imwrite(f'{base_path}.{ext}', cropped_img)
    return f'{base_path}.{ext}'


def to_gray(path, dir_path, basename, ext='png'):
    im = cv2.imread(path)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    base_path = os.path.join(dir_path, basename)
    cv2.imwrite(f'{base_path}.{ext}', im_gray)
    return f'{base_path}.{ext}'


async def _ocr_image_bytes(img_bytes: bytes) -> str:
    stream = InMemoryRandomAccessStream()
    writer = DataWriter(stream)
    writer.write_bytes(img_bytes)
    await writer.store_async()
    await writer.flush_async()
    stream.seek(0)

    decoder = await BitmapDecoder.create_async(stream)
    software_bitmap = await decoder.get_software_bitmap_async()

    engine = OcrEngine.try_create_from_user_profile_languages()
    result = await engine.recognize_async(software_bitmap)
    return result.text


def ocr_with_winrt(image_path: str) -> str:
    with Image.open(image_path) as im:
        buf = BytesIO()
        im.save(buf, format="PNG")
        return asyncio.run(_ocr_image_bytes(buf.getvalue()))


# ==============================================================================
# StreamFrameProcessor（A1: ストリーミング型フレーム処理）
# ==============================================================================

class StreamFrameProcessor:
    """フレームをファイルI/Oせず、メモリ上でcrop→gray→OCRを実行"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    async def _ocr_frame_bytes(self, img_bytes: bytes) -> str:
        """既存の _ocr_image_bytes() と同等。モジュールレベル関数からメソッドへ移動"""
        stream = InMemoryRandomAccessStream()
        writer = DataWriter(stream)
        writer.write_bytes(img_bytes)
        await writer.store_async()
        await writer.flush_async()
        stream.seek(0)

        decoder = await BitmapDecoder.create_async(stream)
        software_bitmap = await decoder.get_software_bitmap_async()

        engine = OcrEngine.try_create_from_user_profile_languages()
        result = await engine.recognize_async(software_bitmap)
        return result.text

    def _crop_roi(self, frame: np.ndarray) -> np.ndarray:
        """numpy array から ROI を切り抜く"""
        h, w = frame.shape[:2]
        y1 = int(h * self.config.roi_y_start)
        y2 = int(h * self.config.roi_y_end)
        x1 = int(w * self.config.roi_x_start)
        x2 = int(w * self.config.roi_x_end)
        return frame[y1:y2, x1:x2]

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        """グレースケール変換"""
        if self.config.img_scale == "gray":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    async def process_video(
        self,
        on_progress: Callable[[str], None] = None,
    ) -> List[str]:
        """動画から逐次フレームを処理し、OCRテキストのリストを返す"""
        cap = cv2.VideoCapture(str(self.config.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"動画ファイルを読み込めませんでした: {self.config.video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        texts: List[str] = []
        debug_path = Path("output/debug")

        # debug モード用のサブディレクトリ準備
        if self.config.debug:
            os.makedirs(debug_path / "crop", exist_ok=True)
            os.makedirs(debug_path / "gray", exist_ok=True)
            os.makedirs(debug_path / "frames", exist_ok=True)
            os.makedirs(debug_path / "text", exist_ok=True)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cropped = self._crop_roi(frame)
            processed = self._to_gray(cropped)

            # debug モード時は中間ファイルを保存
            if self.config.debug:
                cv2.imwrite(
                    str(debug_path / f"frames/frame_{frame_idx:04d}.png"), frame
                )
                cv2.imwrite(
                    str(debug_path / f"crop/crop_{frame_idx:03d}.png"), cropped
                )
                if self.config.img_scale == "gray":
                    cv2.imwrite(
                        str(debug_path / f"gray/crop_gray_{frame_idx:03d}.png"), processed
                    )

            # numpy array → PNG bytes 変換
            buf = BytesIO()
            # OCRに渡す画像はRGBに変換が必要（PILがBGRを想定しないため）
            if self.config.img_scale == "gray":
                # グレースケールの場合は直接uint8として処理
                img_for_ocr = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            else:
                img_for_ocr = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            Image.fromarray(img_for_ocr).save(buf, format="PNG")

            text = await self._ocr_frame_bytes(buf.getvalue())
            processed_text = _post_process_ocr_text(text)

            # debug モード時はテキストも保存
            if self.config.debug:
                with open(debug_path / f"text/text-{frame_idx:03d}.txt", 'w', encoding='utf-8') as f:
                    f.write(processed_text)

            texts.append(processed_text)

            # 進捗通知
            if on_progress is not None:
                pct = 0.2 + (frame_idx / max(frame_count, 1)) * 0.8  # OCRは全体の80%を占める想定
                msg = f"OCR文字認識中 ({frame_idx + 1}/{frame_count})"
                if callable(on_progress) and on_progress.__code__.co_argcount >= 2:
                    # 新しいシグネチャ (msg, pct) をサポート
                    try:
                        on_progress(msg, pct)
                    except TypeError:
                        on_progress(msg)
                else:
                    # 旧シグネチャ (msg,) のみ対応
                    on_progress(msg)

            frame_idx += 1

        cap.release()
        return texts


# ==============================================================================
# ファン数抽出ロジック（既存互換）
# ==============================================================================

def get_fan_count(texts, member, fans):
    token_boundary = r"[0-9A-Za-z\u3040-\u30FF\u3400-\u9FFF]"

    pattern = re.compile(
        rf"(?<!{token_boundary})(?<![\d,])"
        r"\d{1,3}(?:,\d{3})+"
        rf"(?!{token_boundary})(?![\d,])"
    )

    fancounts = {}
    for text in texts:
        if member in text:
            match = pattern.search(text)
            if match:
                value = match.group()
                numeric_value = int(value.replace(',', ''))
                if numeric_value in fans:
                    continue
                if numeric_value in fancounts.keys():
                    fancounts[numeric_value] = fancounts[numeric_value] + 1
                else:
                    fancounts[numeric_value] = 1

    if len(fancounts) != 0:
        sorted_dict = sorted(fancounts.items(), key=lambda x: x[1], reverse=True)
        return sorted_dict[0][0]
    return None


def _post_process_ocr_text(text: str) -> str:
    """OCR 後処理（既存の12連鎖 .replace() を関数化）"""
    text = text.replace(" ", "")
    text = text.replace("①", "")
    text = text.replace("↓", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("（", "")
    text = text.replace("）", "")
    text = text.replace("@", "")
    text = text.replace("、", ",")
    text = text.replace("，", ",")
    text = text.replace("30/30", "")
    text = text.replace("人", " 人")
    text = text.replace("ファン数", "ファン数 ")
    return text


# ==============================================================================
# FanCountExtractor（既存ロジックをラップ）
# ==============================================================================

class FanCountExtractor:
    """メンバーごとのファン数をOCRテキストから抽出"""

    def __init__(self, member_list_path: Path, replace_json_path: Path):
        with open(member_list_path, 'r', encoding='utf-8') as f:
            self.member_list = [line.strip() for line in f]
        with open(replace_json_path, 'r', encoding='utf-8') as f:
            self.member_replace = json.load(f)

    def extract(self, texts: List[str]) -> dict:
        """OCRテキストからメンバーごとのファン数を抽出"""
        # テキスト後処理 + メンバー置換
        for i, text in enumerate(texts):
            for member in self.member_list:
                if member in self.member_replace.keys():
                    for repname in self.member_replace[member]:
                        texts[i] = texts[i].replace(repname, member)
                if member in text:
                    texts[i] = texts[i].replace(member, f"\n{member} ")

        texts = "\n".join(texts).split("\n")

        # ファン数集計
        fan_counts = {}
        fans = []
        for member in self.member_list:
            fan_count = get_fan_count(texts, member, fans)
            if fan_count is not None:
                fan_counts[member] = fan_count
                fans.append(fan_count)
            else:
                fan_counts[member] = 0

        return fan_counts


# ==============================================================================
# PipelineRunner（全パイプライン統合）
# ==============================================================================

class PipelineRunner:
    """OCR パイプラインの全ステップを実行"""

    def run(
        self, config: PipelineConfig, on_progress: Callable[[str], None]
    ) -> OCRResult:
        try:
            base_path = Path("output/")
            debug_path = base_path / "debug"

            # 出力ディレクトリ初期化（0%）
            try:
                on_progress("出力ディレクトリを初期化します...", pct=0.0)
            except TypeError:
                on_progress("出力ディレクトリを初期化します...")
            cleanup(base_path, debug_path)

            video_path = config.video_path
            try:
                on_progress(f"動画を読み込みます: {video_path.name}", pct=0.05)
            except TypeError:
                on_progress(f"動画を読み込みます: {video_path.name}")

            # ストリーミング方式でフレーム処理 + OCR実行（A1対応）
            import time
            ocr_start_time = time.time()
            processor = StreamFrameProcessor(config)

            # 進捗ラッパー：パーセント値も渡す
            def wrapped_on_progress(msg: str, pct: float = 0.0):
                try:
                    on_progress(msg, pct=pct)
                except TypeError:
                    on_progress(msg)

            texts = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                texts = loop.run_until_complete(
                    processor.process_video(on_progress=wrapped_on_progress)
                )
            finally:
                loop.close()

            # debug モード時は全テキストの結合ファイルも出力
            if config.debug:
                join_text = "\n".join(texts)
                with open(debug_path / "text/output.txt", 'w', encoding='utf-8') as f:
                    f.write(join_text)

            # メンバーごとのファン数を抽出（98%）
            try:
                on_progress("メンバーごとのファン数を抽出します...", pct=0.98)
            except TypeError:
                on_progress("メンバーごとのファン数を抽出します...")
            extractor = FanCountExtractor(
                Path('input/memberList.txt'),
                Path('input/memberReplace.json')
            )
            fan_counts = extractor.extract(texts)

            # 結果出力
            with open('output/output.json', 'w', encoding='UTF-8') as f:
                json.dump(fan_counts, f, ensure_ascii=False, indent=4)

            # デバッグファイルのクリーンアップ
            if not config.debug:
                shutil.rmtree(debug_path, ignore_errors=True)

            try:
                on_progress("処理が完了しました。", pct=1.0)
            except TypeError:
                on_progress("処理が完了しました。")
            return OCRResult(fan_counts=fan_counts, texts=texts)

        except Exception as e:
            return OCRResult(fan_counts={}, texts=[], error=str(e))


# ==============================================================================
# PipelineWorker（バックグラウンドスレッド）
# ==============================================================================

class PipelineWorker(threading.Thread):
    """OCRパイプラインを別スレッドで実行"""

    def __init__(self, config: PipelineConfig, result_queue: queue.Queue):
        super().__init__()
        self.config = config
        self.result_queue = result_queue
        self.runner = PipelineRunner()

    def run(self):
        # このスレッド内で新しい asyncio event loop を作成（WinRT用）
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            # 進捗コールバック：メッセージとパーセントの両方をキューに投入
            def on_progress(msg: str, pct: float = None):
                if pct is not None:
                    self.result_queue.put(("progress_with_pct", {
                        "message": msg,
                        "percent": pct,
                    }))
                else:
                    self.result_queue.put(("progress", msg))

            result = self.runner.run(self.config, on_progress)
            if result.error:
                # PipelineRunner 内部で catch されたエラーも wrap_error に通す
                app_err = wrap_error(RuntimeError(result.error))
                self.result_queue.put(("error", {
                    "message": app_err.message,
                    "hint": app_err.hint,
                }))
            else:
                self.result_queue.put(("result", result))
        except Exception as e:
            app_err = wrap_error(e)
            self.result_queue.put(("error", {
                "message": app_err.message,
                "hint": app_err.hint,
            }))
        finally:
            self.result_queue.put(("done", None))
            loop.close()


# ==============================================================================
# A2: SettingsDialog（ROI設定ダイアログ）
# ==============================================================================

class SettingsDialog:
    """ROIと各種オプションを設定するToplevelウィンドウ"""

    def __init__(self, parent, settings_manager: SettingsManager):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("設定")
        self.dialog.geometry("450x350")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.settings_manager = settings_manager
        self.current_settings = settings_manager.load()

        # --- 内容フレーム ---
        main_frame = tk.Frame(self.dialog, padx=15, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ROI Y軸設定
        self._create_roi_entry(main_frame, "Y 軸開始（上端比）", self.current_settings.roi_y_start)
        self._create_roi_entry(main_frame, "Y 軸終了（下端比）", self.current_settings.roi_y_end)
        self._create_roi_entry(main_frame, "X 軸開始（左端比）", self.current_settings.roi_x_start)
        self._create_roi_entry(main_frame, "X 軸終了（右端比）", self.current_settings.roi_x_end)

        # img_scale ラジオボタン
        scale_frame = tk.LabelFrame(main_frame, text="画像スケーリング", font=("Meiryo UI", 9), padx=5, pady=5)
        scale_frame.pack(fill=tk.X, pady=(10, 5))
        self.var_img_scale = tk.StringVar(value=self.current_settings.img_scale or "")
        rb_none = tk.Radiobutton(scale_frame, text="なし（カラー画像をそのままOCRに渡す）", variable=self.var_img_scale, value="", font=("Meiryo UI", 9))
        rb_gray = tk.Radiobutton(scale_frame, text="グレースケール変換後にOCR実行", variable=self.var_img_scale, value="gray", font=("Meiryo UI", 9))
        rb_none.pack(anchor=tk.W)
        rb_gray.pack(anchor=tk.W)

        # Debug チェックボックス
        chk_frame = tk.Frame(main_frame)
        chk_frame.pack(fill=tk.X, pady=(5, 10))
        self.var_debug = tk.BooleanVar(value=self.current_settings.debug)
        self.chk_debug = tk.Checkbutton(
            chk_frame, text="デバッグモード（中間ファイルを保存）",
            variable=self.var_debug, font=("Meiryo UI", 9)
        )
        self.chk_debug.pack(side=tk.LEFT)

        # ヘルプテキスト
        help_text = (
            "ROI値は画面の相対座標です。\n"
            "Y軸: 上端(0.0)→下端(1.0), X軸: 左端(0.0)→右端(1.0)\n"
            "デフォルト値はウマ娘のファン数表示画面に最適化されています。\n"
            "解像度やゲームバージョンが異なる場合は調整してください。"
        )
        lbl_help = tk.Label(main_frame, text=help_text, font=("Meiryo UI", 8), fg="#666", justify=tk.LEFT)
        lbl_help.pack(anchor=tk.W)

        # --- OK/Cancel ボタン ---
        btn_frame = tk.Frame(self.dialog, padx=15, pady=10)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_ok = tk.Button(btn_frame, text="保存", font=("Meiryo UI", 10), command=self._on_save)
        self.btn_ok.pack(side=tk.RIGHT, padx=(5, 0))

        self.btn_cancel = tk.Button(btn_frame, text="キャンセル", font=("Meiryo UI", 10), command=self.dialog.destroy)
        self.btn_cancel.pack(side=tk.RIGHT)

    def _create_roi_entry(self, parent, label_text, default_value):
        """ROI値のラベル+Entry行を作成"""
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        lbl = tk.Label(row, text=label_text, width=18, anchor=tk.W, font=("Meiryo UI", 9))
        lbl.pack(side=tk.LEFT)
        entry = tk.Entry(row, width=10, font=("Meiryo UI", 9))
        entry.insert(0, str(default_value))
        entry.pack(side=tk.LEFT)
        return entry

    def _get_values(self):
        """Entryウィジェットから値を取得し、AppSettingsを返す"""
        # main_frame の子要素から Entry を探索（簡易的な方法）
        entries = []
        for widget in self.dialog.winfo_children():
            if isinstance(widget, tk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, tk.LabelFrame):
                        continue
                    if isinstance(child, tk.Frame):
                        for grandchild in child.winfo_children():
                            if isinstance(grandchild, tk.Entry):
                                entries.append(grandchild)

        try:
            y_start = float(entries[0].get())
            y_end = float(entries[1].get())
            x_start = float(entries[2].get())
            x_end = float(entries[3].get())
        except (ValueError, IndexError):
            return None

        errors = SettingsManager._validate_roi(y_start, y_end, x_start, x_end)
        if errors:
            mb.showerror("入力エラー", "\n".join(errors))
            return None

        img_scale_val = self.var_img_scale.get() or None
        debug = self.var_debug.get()

        return AppSettings(
            roi_y_start=y_start,
            roi_y_end=y_end,
            roi_x_start=x_start,
            roi_x_end=x_end,
            img_scale=img_scale_val if img_scale_val else None,
            debug=debug,
        )

    def show(self):
        """ダイアログ表示 → OK ボタン押下で設定を保存して AppSettings を返す。"""
        self.dialog.wait_window()
        # 保存が成功した場合は saved_settings がセットされる
        return getattr(self, '_saved_settings', None)

    def _on_save(self):
        """OKボタン: 値を取得→バリデーション→保存→ダイアログ閉じる"""
        settings = self._get_values()
        if settings is not None:
            self.settings_manager.save(settings)
            self._saved_settings = settings
            self.dialog.destroy()


# ==============================================================================
# C2: MemberEditorDialog（メンバー管理ダイアログ）
# ==============================================================================

class MemberEditorDialog:
    """メンバーを追加・削除・編集するToplevelウィンドウ"""

    def __init__(self, parent, members: Dict[str, MemberEntry]):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("メンバー管理")
        self.dialog.geometry("650x480")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.members = dict(members)  # コピー（元の辞書は変更しない）
        self.ok_pressed = False
        self.editing_mode = False  # 編集モードのフラグ

        main_frame = tk.Frame(self.dialog, padx=10, pady=5)
        main_frame.pack(fill=tk.BOTH, expand=True)

        lbl_title = tk.Label(main_frame, text="メンバーリストとOCR誤認識パターンを編集できます。", font=("Meiryo UI", 9), fg="#666")
        lbl_title.pack(anchor=tk.W)

        # 左右2ペイン：左=リストビュー、右=編集フォーム
        paned = tk.PanedWindow(main_frame, orient=tk.HORIZONTAL, showhandle=True)
        paned.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        # --- 左ペイン: Treeview ---
        left_frame = tk.Frame(paned)
        tv_columns = ("名前", "置換パターン数")
        self.tv_members = ttk.Treeview(left_frame, columns=tv_columns, show="headings", height=12)
        self.tv_members.heading("名前", text="メンバー名")
        self.tv_members.heading("置換パターン数", text="パターン数")
        self.tv_members.column("名前", width=200)
        self.tv_members.column("置換パターン数", width=60, anchor=tk.CENTER)

        tv_scr = tk.Scrollbar(left_frame, orient="vertical", command=self.tv_members.yview)
        self.tv_members.configure(yscrollcommand=tv_scr.set)
        self.tv_members.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tv_scr.pack(side=tk.RIGHT, fill=tk.Y)

        self.tv_members.bind("<<TreeviewSelect>>", self._on_member_selected)
        paned.add(left_frame)

        # --- 右ペイン: 編集フォーム ---
        right_frame = tk.Frame(paned, padx=5, pady=5)

        form_frame = tk.LabelFrame(right_frame, text="選択したメンバーの編集", font=("Meiryo UI", 9), padx=5, pady=5)
        form_frame.pack(fill=tk.BOTH, expand=True)

        # メンバー名入力
        name_row = tk.Frame(form_frame)
        name_row.pack(fill=tk.X, pady=(0, 5))
        tk.Label(name_row, text="名前:", font=("Meiryo UI", 9)).pack(side=tk.LEFT)
        self.entry_name = tk.Entry(name_row, font=("Meiryo UI", 9))
        self.entry_name.pack(side=tk.LEFT, padx=(5, 0), fill=tk.X, expand=True)

        # パターンリスト表示
        pat_frame = tk.LabelFrame(form_frame, text="誤認識パターン（置換元）", font=("Meiryo UI", 9), padx=3, pady=3)
        pat_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        self.lst_pattern = tk.Listbox(pat_frame, height=6, font=("Meiryo UI", 9))
        lst_pat_scr = tk.Scrollbar(pat_frame, orient=tk.VERTICAL, command=self.lst_pattern.yview)
        self.lst_pattern.configure(yscrollcommand=lst_pat_scr.set)
        self.lst_pattern.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        lst_pat_scr.pack(side=tk.RIGHT, fill=tk.Y)

        # パターン追加行
        add_row = tk.Frame(form_frame)
        add_row.pack(fill=tk.X, pady=(0, 5))
        self.entry_new_pattern = tk.Entry(add_row, font=("Meiryo UI", 9), width=15)
        self.entry_new_pattern.pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(add_row, text="+ 追加", width=6, font=("Meiryo UI", 8), command=self._add_pattern).pack(side=tk.LEFT, padx=(3, 0))

        # パターン削除ボタン
        tk.Button(form_frame, text="パターンを削除", font=("Meiryo UI", 9), command=self._remove_pattern).pack(fill=tk.X)

        # 編集・変更適用ボタン（同じ行に配置）
        edit_btn_row = tk.Frame(form_frame)
        edit_btn_row.pack(fill=tk.X, pady=(5, 0))
        self.btn_edit_mode = tk.Button(
            edit_btn_row, text="✏ 編集", font=("Meiryo UI", 9), command=self._enter_edit_mode
        )
        self.btn_edit_mode.pack(side=tk.LEFT)
        self.btn_apply = tk.Button(
            edit_btn_row, text="✓ 変更を適用", font=("Meiryo UI", 9), bg="#e6ffe6", command=self._apply_changes
        )
        self.btn_apply.pack(side=tk.RIGHT)

        # メンバー追加/削除ボタン
        btn_row = tk.Frame(form_frame)
        btn_row.pack(fill=tk.X, pady=(5, 0))
        tk.Button(btn_row, text="新しいメンバーを追加", font=("Meiryo UI", 9), command=self._add_member).pack(side=tk.LEFT)
        tk.Button(btn_row, text="メンバーを削除", font=("Meiryo UI", 9), fg="red", command=self._remove_member).pack(side=tk.RIGHT)

        # --- OK/Cancel ボタン（ダイアログ最下部）---
        btn_frame = tk.Frame(self.dialog, padx=15, pady=10)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_save = tk.Button(btn_frame, text="保存して閉じる", font=("Meiryo UI", 10), command=self._on_close_ok)
        self.btn_save.pack(side=tk.RIGHT, padx=(5, 0))

        self.btn_cancel = tk.Button(btn_frame, text="キャンセル", font=("Meiryo UI", 10), command=self.dialog.destroy)
        self.btn_cancel.pack(side=tk.RIGHT)

        # 初期化: Treeview にメンバーリストを反映 + フィールドロック
        self._populate_treeview()
        self._set_editable(False)

    def _set_editable(self, editable):
        """右パネルの編集フィールドを有効/無効にする"""
        state = tk.NORMAL if editable else tk.DISABLED
        # entry_name の状態変更（Entry は state で制御）
        self.entry_name.config(state=state)
        # btn_apply と btn_edit_mode を切り替え
        if editable:
            self.btn_apply.config(state=tk.NORMAL)
            self.btn_edit_mode.config(state=tk.DISABLED)
        else:
            self.btn_apply.config(state=tk.DISABLED)
            self.btn_edit_mode.config(state=tk.NORMAL)

    def _enter_edit_mode(self):
        """編集モードに入る"""
        self.editing_mode = True
        self._set_editable(True)

    def _populate_treeview(self):
        """Treeview に現在の members を反映"""
        for item in self.tv_members.get_children():
            self.tv_members.delete(item)
        for entry in self.members.values():
            self.tv_members.insert("", tk.END, values=(entry.name, len(entry.replace_patterns)))

    def _on_member_selected(self, event):
        """Treeview でメンバーが選択されたら右ペインのフォームを更新"""
        selected = self.tv_members.selection()
        if not selected:
            return

        # 編集モード中で別のメンバーを選択した場合、編集モードを終了
        if self.editing_mode:
            self.editing_mode = False
            self._set_editable(False)

        item = self.tv_members.item(selected[0])
        name = item["values"][0]
        entry = self.members.get(name)
        if not entry:
            return

        # Entry が無効状態でも値が書き込めるよう一時的に有効化
        was_editable = self.editing_mode
        if not was_editable:
            self.entry_name.config(state=tk.NORMAL)
        self.entry_name.delete(0, tk.END)
        self.entry_name.insert(0, entry.name)

        # パターンリスト更新
        self.lst_pattern.delete(0, tk.END)
        for pat in entry.replace_patterns:
            self.lst_pattern.insert(tk.END, pat)

        if not was_editable:
            self.entry_name.config(state=tk.DISABLED)

    def _apply_changes(self):
        """フォームの編集内容（名前変更・パターンの追加/削除）を self.members に反映"""
        sel = self.tv_members.selection()
        if not sel:
            return

        old_name = self.tv_members.item(sel[0])["values"][0]
        new_name = self.entry_name.get().strip()
        entry = self.members.get(old_name)
        if not entry:
            return

        # パターンリストボックスの内容を entry.replace_patterns に同期
        # Listbox から現在表示されているパターンを取得
        current_patterns = list(self.lst_pattern.get(0, tk.END))
        entry.replace_patterns = list(current_patterns)

        old_item_id = sel[0]
        old_vals = self.tv_members.item(old_item_id)["values"]

        if new_name and new_name != old_name:
            # 名前が変更された場合：辞書のキーとTreeviewを更新
            if new_name in self.members:
                mb.showwarning("警告", f"「{new_name}」は既に存在します。")
                return
            # entry.name を更新して辞書キーを変更
            entry.name = new_name
            del self.members[old_name]
            self.members[new_name] = entry
            # Treeview の行を更新
            self.tv_members.set(old_item_id, values=(new_name, len(entry.replace_patterns)))
        else:
            # 名前変更なしの場合はTreeviewのパターン数だけ更新
            self.tv_members.set(old_item_id, values=(old_vals[0], len(entry.replace_patterns)))

        # 編集モードを終了し、フィールドをロック状態に戻す
        self.editing_mode = False
        self._set_editable(False)

    def _add_pattern(self):
        """新しいパターンを追加"""
        selected = self.tv_members.selection()
        if not selected:
            mb.showinfo("情報", "先にメンバーを選択してください。")
            return
        new_pat = self.entry_new_pattern.get().strip()
        if not new_pat:
            return

        # 現在のフォームの状態を一旦コミットしてからパターン追加
        old_name = self.tv_members.item(selected[0])["values"][0]
        entry = self.members.get(old_name)
        if not entry or new_pat in entry.replace_patterns:
            self.entry_new_pattern.delete(0, tk.END)
            return

        entry.replace_patterns.append(new_pat)
        self.lst_pattern.insert(tk.END, new_pat)
        # Treeview のパターン数更新
        item_id = selected[0]
        self.tv_members.set(item_id, "置換パターン数", len(entry.replace_patterns))

        self.entry_new_pattern.delete(0, tk.END)

    def _remove_pattern(self):
        """選択されたパターンを削除"""
        selected_idx = self.lst_pattern.curselection()
        if not selected_idx:
            mb.showinfo("情報", "削除するパターンを選択してください。")
            return

        sel_item = self.tv_members.selection()
        if not sel_item:
            return
        old_name = self.tv_members.item(sel_item[0])["values"][0]
        entry = self.members.get(old_name)
        if not entry:
            return

        idx = selected_idx[0]
        removed_pat = self.lst_pattern.delete(idx)
        if removed_pat in entry.replace_patterns:
            entry.replace_patterns.remove(removed_pat)

        # Treeview のパターン数更新
        self.tv_members.set(sel_item[0], "置換パターン数", len(entry.replace_patterns))

    def _add_member(self):
        """新しいメンバーを追加"""
        new_name = mb.askstring("新しいメンバー", "メンバー名を入力してください:")
        if not new_name or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name in self.members:
            mb.showwarning("警告", f"「{new_name}」は既に存在します。")
            return

        self.members[new_name] = MemberEntry(name=new_name)
        self._populate_treeview()
        # 自動的に選択してフォームに反映
        for item in self.tv_members.get_children():
            if self.tv_members.item(item)["values"][0] == new_name:
                self.tv_members.selection_set(item)
                self.tv_members.see(item)
                break

    def _remove_member(self):
        """選択されたメンバーを削除"""
        selected = self.tv_members.selection()
        if not selected:
            mb.showinfo("情報", "削除するメンバーを選択してください。")
            return
        name = self.tv_members.item(selected[0])["values"][0]
        if not mb.askyesno("確認", f"「{name}」を削除しますか？"):
            return

        del self.members[name]
        self._populate_treeview()

    def _on_close_ok(self):
        """保存して閉じる"""
        self.ok_pressed = True
        self.dialog.destroy()

    def show(self) -> Dict[str, MemberEntry] | None:
        """ダイアログ表示 → OK で辞書を返す。キャンセル時はNone"""
        self.dialog.wait_window()
        return dict(self.members) if self.ok_pressed else None



class AppWindow:
    """永続表示の tkinter GUI アプリ"""

    def __init__(self, debug: bool = False, img_scale: Optional[str] = None):
        self.debug = debug
        self.img_scale = img_scale
        self.is_processing = False

        # A2/C2: SettingsManager と MemberManager の初期化
        self.settings_manager = SettingsManager()
        self.current_settings = self.settings_manager.load()
        self.member_manager = MemberManager()

        # --- ウィンドウ作成 ---
        import tkinter as tk
        from tkinter import filedialog, ttk
        self.root = tk.Tk()
        self.root.title("Umamusume Fan Count Extractor")
        self.root.geometry("700x550")
        self.root.resizable(True, True)

        # キュー（スレッド間通信用）
        self.result_queue = queue.Queue()

        # --- 上部：ファイル選択エリア ---
        top_frame = tk.Frame(self.root, padx=10, pady=10)
        top_frame.pack(fill=tk.X)

        self.btn_select = tk.Button(
            top_frame, text="📂 動画ファイルを選択",
            font=("Meiryo UI", 11), width=20, height=2,
            command=self._on_select_video
        )
        self.btn_select.pack(side=tk.LEFT)

        self.lbl_path = tk.Label(
            top_frame, text="動画が選択されていません",
            font=("Meiryo UI", 9), fg="#666666"
        )
        self.lbl_path.pack(side=tk.LEFT, padx=(15, 0))

        # A2/C2: 設定・メンバー管理ボタン（右端に配置）
        btn_right = tk.Frame(top_frame)
        btn_right.pack(side=tk.RIGHT)

        self.btn_settings = tk.Button(
            btn_right, text="⚙ 設定", font=("Meiryo UI", 10), width=8,
            command=self._on_open_settings
        )
        self.btn_settings.pack(side=tk.LEFT, padx=(0, 5))

        self.btn_members = tk.Button(
            btn_right, text="メンバー管理", font=("Meiryo UI", 10), width=10,
            command=self._on_open_member_editor
        )
        self.btn_members.pack(side=tk.LEFT)

        # --- 中央：ステータスパネル（進捗表示）---
        status_frame = tk.LabelFrame(self.root, text="処理進捗", font=("Meiryo UI", 10), padx=5, pady=5)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 5))

        # Progressbar（B2: 進捗バー追加）
        self.progress_bar = ttk.Progressbar(
            status_frame, orient=tk.HORIZONTAL, length=0, mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X, pady=(0, 5))

        # 推定残り時間のラベル
        self.lbl_eta = tk.Label(status_frame, text="", font=("Meiryo UI", 8), fg="#666")
        self.lbl_eta.pack(anchor=tk.W)

        # OCR処理開始時刻（ETA計算用）
        self._ocr_start_time: float | None = None

        self.txt_status = tk.Text(
            status_frame, height=12, width=78,
            font=("Meiryo UI", 9), state=tk.DISABLED
        )
        scr = tk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.txt_status.yview)
        self.txt_status.configure(yscrollcommand=scr.set)
        self.txt_status.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scr.pack(side=tk.RIGHT, fill=tk.Y)

        # --- 下部：結果パネル（Treeview / JSON Text の切り替え表示）---
        result_frame = tk.Frame(self.root, padx=10, pady=5)
        result_frame.pack(fill=tk.X)

        # ラジオボタン行
        radio_frame = tk.Frame(result_frame)
        radio_frame.pack(side=tk.LEFT)

        self.btn_copy = tk.Button(
            radio_frame, text="📋 クリップボードにコピー",
            font=("Meiryo UI", 9), width=18, command=self._on_copy_results
        )
        self.btn_copy.grid(row=0, column=0)

        self.var_display_mode = tk.IntVar(value=0)  # 0: テーブル形式, 1: JSON形式

        rb_table = tk.Radiobutton(
            radio_frame, text="テーブル", variable=self.var_display_mode, value=0,
            font=("Meiryo UI", 9), command=self._update_display
        )
        rb_table.grid(row=1, column=0)

        rb_json = tk.Radiobutton(
            radio_frame, text="JSON", variable=self.var_display_mode, value=1,
            font=("Meiryo UI", 9), command=self._update_display
        )
        rb_json.grid(row=2, column=0)

        # --- 結果表示エリア（切り替え用 Frame）---
        self.result_container = tk.Frame(result_frame)
        self.result_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Treeview（テーブル形式用）
        tv_frame = tk.Frame(self.result_container)
        columns = ("メンバー名", "ファン数")
        self.tv_results = ttk.Treeview(
            tv_frame, columns=columns, show="headings", height=6
        )
        self.tv_results.heading("メンバー名", text="メンバー名")
        self.tv_results.heading("ファン数", text="ファン数")
        self.tv_results.column("メンバー名", width=200)
        self.tv_results.column("ファン数", width=150, anchor=tk.E)

        tv_scr = tk.Scrollbar(tv_frame, orient=tk.VERTICAL, command=self.tv_results.yview)
        self.tv_results.configure(yscrollcommand=tv_scr.set)
        self.tv_results.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tv_scr.pack(side=tk.RIGHT, fill=tk.Y)
        self.treeview_widget = tv_frame

        # Text（JSON形式用）
        json_frame = tk.Frame(self.result_container)
        self.txt_json = tk.Text(
            json_frame, height=6, width=78,
            font=("Consolas", 10), state=tk.DISABLED
        )
        json_scr = tk.Scrollbar(json_frame, orient=tk.VERTICAL, command=self.txt_json.yview)
        self.txt_json.configure(yscrollcommand=json_scr.set)
        self.txt_json.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        json_scr.pack(side=tk.RIGHT, fill=tk.Y)
        self.json_widget = json_frame

        # 初期表示：テーブル形式を表示
        self.treeview_widget.pack(fill=tk.BOTH, expand=True)

        # --- クロージャ処理 ---
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _add_status(self, msg: str):
        """ステータスペネルにメッセージを追加（スレッドセーフ）"""
        self.txt_status.configure(state=tk.NORMAL)
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.txt_status.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.txt_status.see(tk.END)
        self.txt_status.yview_moveto(1.0)
        self.txt_status.configure(state=tk.DISABLED)

    def _update_display(self):
        """ラジオボタンの選択に応じて表示を切り替え"""
        mode = self.var_display_mode.get()
        # 両方のウィジェットを非表示にし、選択された方を再表示
        self.treeview_widget.pack_forget()
        self.json_widget.pack_forget()
        if mode == 0:
            self.treeview_widget.pack(in_=self.result_container, fill=tk.BOTH, expand=True)
        else:
            self.json_widget.pack(in_=self.result_container, fill=tk.BOTH, expand=True)

    def _on_open_settings(self):
        """⚙ 設定ボタンのイベントハンドラ"""
        dlg = SettingsDialog(self.root, self.settings_manager)
        result = dlg.show()
        if result is not None:
            # 保存が成功したら current_settings を更新
            self.current_settings = result

    def _on_open_member_editor(self):
        """メンバー管理ボタンのイベントハンドラ"""
        # 現在の memberList.txt / memberReplace.json から読み込む
        members = self.member_manager.load(
            Path('input/memberList.txt'),
            Path('input/memberReplace.json')
        )
        dlg = MemberEditorDialog(self.root, members)
        result = dlg.show()
        if result is not None:
            # 保存してファイルに書き出す
            self.member_manager.save(
                result,
                Path('input/memberList.txt'),
                Path('input/memberReplace.json')
            )

    def _on_select_video(self):
        """「動画ファイルを選択」ボタンのイベントハンドラ"""
        if self.is_processing:
            return  # 処理中は無効化

        import tkinter as tk
        from tkinter import filedialog

        file_path = filedialog.askopenfilename(
            title="解析する動画ファイルを選択してください",
            initialdir=os.path.abspath("input"),
            filetypes=[("MP4 動画ファイル", "*.mp4"), ("すべてのファイル", "*.*")]
        )

        if not file_path:
            return  # キャンセル時は何もしない

        video_path = Path(file_path)
        self.lbl_path.config(text=video_path.name, fg="#000000")

        # ステータスリセット
        self.txt_status.configure(state=tk.NORMAL)
        self.txt_status.delete(1.0, tk.END)
        self.txt_status.configure(state=tk.DISABLED)
        # Treeview クリア
        for item in self.tv_results.get_children():
            self.tv_results.delete(item)
        # JSON Text クリア
        self.txt_json.configure(state=tk.NORMAL)
        self.txt_json.delete(1.0, tk.END)
        self.txt_json.configure(state=tk.DISABLED)
        # Progressbarリセット（B2対応）
        self.progress_bar["value"] = 0
        self.lbl_eta.config(text="")
        self._ocr_start_time = None

        # ワーカースレッド起動
        config = PipelineConfig(
            video_path=video_path,
            debug=self.debug,
            img_scale=self.img_scale
        )
        # A2: SettingsManager の値を ROI に反映
        self.settings_manager.apply_to_config(config)
        worker = PipelineWorker(config, self.result_queue)
        self.is_processing = True
        self.btn_select.config(state=tk.DISABLED)
        self._add_status("処理を開始します...")
        worker.start()

        # キューポーリング開始（100msごと）
        self.root.after(100, self._check_queue)

    def _check_queue(self):
        """キューからのメッセージをGUIに反映"""
        try:
            while True:
                msg_type, data = self.result_queue.get_nowait()

                if msg_type == "progress":
                    self._add_status(data)

                elif msg_type == "progress_with_pct":
                    # B2: 進捗率付きメッセージ（Progressbar + ETA更新）
                    pct_data = data if isinstance(data, dict) else {"message": str(data), "percent": 0.0}
                    msg_text = pct_data.get("message", "")
                    percent = float(pct_data.get("percent", 0.0))

                    self._add_status(msg_text)
                    # Progressbar更新（0〜100の値に変換）
                    self.progress_bar["value"] = int(percent * 100)

                    # ETA（推定残り時間）計算
                    import time as _time
                    if self._ocr_start_time is not None and percent > 0.2:
                        elapsed = _time.time() - self._ocr_start_time
                        if elapsed > 0:
                            rate = percent / elapsed  # progress per second
                            remaining = (1.0 - percent) / rate
                            if remaining > 60:
                                eta_str = f"推定残り時間: {int(remaining // 60)}分{int(remaining % 60)}秒"
                            elif remaining > 0:
                                eta_str = f"推定残り時間: {int(remaining)}秒"
                            else:
                                eta_str = "完了間近..."
                            self.lbl_eta.config(text=eta_str)

                    # OCRフェーズに入った際に開始時刻を記録
                    if percent > 0.2 and self._ocr_start_time is None:
                        self._ocr_start_time = _time.time()

                elif msg_type == "result":
                    # 結果をTreeviewに反映
                    for member, count in data.fan_counts.items():
                        formatted_count = f"{count:,}" if count > 0 else "非検出"
                        self.tv_results.insert("", tk.END, values=(member, formatted_count))
                    # JSON Text にフォーマット済みJSONを設定
                    json_text = json.dumps(data.fan_counts, ensure_ascii=False, indent=4)
                    self.txt_json.configure(state=tk.NORMAL)
                    self.txt_json.delete(1.0, tk.END)
                    self.txt_json.insert(tk.END, json_text)
                    self.txt_json.configure(state=tk.DISABLED)
                    # 現在の表示モードに合わせてウィジェットを切り替え
                    self._update_display()

                elif msg_type == "error":
                    # 新しい dict 形式 (message + hint) または旧 string 形式の両方に対応
                    if isinstance(data, dict):
                        err_msg = data.get("message", str(data))
                        err_hint = data.get("hint", "")
                    else:
                        err_msg = str(data)
                        err_hint = ""
                    self._add_status(f"エラー: {err_msg}")
                    import tkinter.messagebox as mb
                    if err_hint:
                        mb.showwarning(
                            "処理エラー",
                            f"{err_msg}\n\n{err_hint}",
                        )
                    else:
                        mb.showwarning("処理エラー", err_msg)

                elif msg_type == "done":
                    self.is_processing = False
                    self.btn_select.config(state=tk.NORMAL)
                    if not hasattr(self, '_last_result') or getattr(self, '_last_result', None):
                        # エラーでなければ完了メッセージ
                        pass

            return  # キューが空なら終了

        except queue.Empty:
            pass

        # キューが空でない場合は引き続きポーリング
        if self.is_processing:
            self.root.after(100, self._check_queue)

    def _on_copy_results(self):
        """現在の表示モードに応じてクリップボードにコピー"""
        mode = self.var_display_mode.get()

        if mode == 0:
            # テーブル形式：Treeview からタブ区切りテキストを生成
            lines = []
            for item in self.tv_results.get_children():
                values = self.tv_results.item(item)["values"]
                if len(values) == 2:
                    lines.append(f"{values[0]}\t{values[1]}")
            if not lines:
                return
            text = "\n".join(lines)
        else:
            # JSON形式：Text ウィジェットの内容をそのままコピー
            text = self.txt_json.get(1.0, tk.END).strip()
            if not text:
                return

        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _on_close(self):
        """ウィンドウクローズ処理"""
        if self.is_processing:
            import tkinter.messagebox as mb
            if not mb.askyesno("確認", "処理中です。本当に終了しますか？"):
                return
        self.root.destroy()

    def run(self):
        """メインループ開始"""
        self._add_status("アプリを起動しました。動画ファイルを選択してください。")
        self.root.mainloop()


# ==============================================================================
# エントリポイント（__main__）
# ==============================================================================

def _run_batch_mode(config: PipelineConfig):
    """GUIを開かず、直接パイプラインを実行（バッチモード用）"""
    runner = PipelineRunner()
    result = runner.run(
        config,
        lambda msg: print(msg)  # CLIでは標準出力にメッセージを表示
    )

    if result.error:
        print(f"エラーが発生しました: {result.error}")
        sys.exit(1)

    print("=== 処理完了 ===")
    print(json.dumps(result.fan_counts, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    args = parse_args()

    # バッチモード：動画パスがCLIで指定された場合、GUIを開かずに直接実行
    if args.video is not None:
        config = PipelineConfig(
            video_path=Path(args.video),
            debug=args.debug,
            img_scale=args.img_scale
        )
        _run_batch_mode(config)

    # GUI モード：動画パスなし、または --gui フラグ指定時
    else:
        app = AppWindow(debug=args.debug, img_scale=args.img_scale)
        app.run()
