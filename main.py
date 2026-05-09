# ocr_winrt_safe.py — Umamusume Fan Count Extractor
import sys
import asyncio
from io import BytesIO
from PIL import Image
import cv2
from typing import Iterable, Tuple, Optional, List, Callable
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
from dataclasses import dataclass
import threading
import queue
import tkinter as tk

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


@dataclass
class OCRResult:
    """OCR パイプラインの完了結果"""
    fan_counts: dict
    texts: list
    error: str | None = None


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

    def run(self, config: PipelineConfig, on_progress: Callable[[str], None]) -> OCRResult:
        try:
            base_path = Path("output/")
            debug_path = base_path / "debug"

            on_progress("出力ディレクトリを初期化します...")
            cleanup(base_path, debug_path)

            video_path = config.video_path
            on_progress(f"動画を読み込みます: {video_path.name}")
            save_all_frames(video_path, debug_path, "frames/frame", "png")
            images = glob.glob(f"{debug_path}/frames/*.png")

            on_progress("画像を切り抜きます...")
            crop_images = []
            for i, img in enumerate(images):
                im = cv2.imread(img)
                crop_path = _crop_image(im, debug_path / "crop", f"crop_{i:03d}")
                crop_images.append(crop_path)
            crop_images = glob.glob(f"{debug_path}/crop/*.png")

            on_progress("グレースケール変換します...")
            gray_images = []
            for i, crop in enumerate(crop_images):
                im_gray = to_gray(crop, debug_path / "gray", f"crop_gray_{i:03d}")
                gray_images.append(im_gray)
            gray_images = glob.glob(f"{debug_path}/gray/*.png")

            target_images = gray_images if config.img_scale == "gray" else crop_images

            on_progress("OCR文字認識を実行します...")
            texts = []
            for i, img in enumerate(target_images):
                text = ocr_with_winrt(img)
                text = _post_process_ocr_text(text)
                with open(debug_path / f"text/text-{i:03d}.txt", 'w', encoding='utf-8') as f:
                    f.write(text)
                texts.append(text)

            on_progress("メンバーごとのファン数を抽出します...")
            extractor = FanCountExtractor(
                Path('input/memberList.txt'),
                Path('input/memberReplace.json')
            )
            fan_counts = extractor.extract(texts)

            # 結果出力
            join_text = "\n".join(texts)
            with open(debug_path / "text/output.txt", 'w', encoding='utf-8') as f:
                f.write(join_text)

            with open('output/output.json', 'w', encoding='UTF-8') as f:
                json.dump(fan_counts, f, ensure_ascii=False, indent=4)

            # デバッグファイルのクリーンアップ
            if not config.debug:
                shutil.rmtree(debug_path, ignore_errors=True)

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
            result = self.runner.run(
                self.config,
                lambda msg: self.result_queue.put(("progress", msg))
            )
            if result.error:
                self.result_queue.put(("error", result.error))
            else:
                self.result_queue.put(("result", result))
        except Exception as e:
            self.result_queue.put(("error", str(e)))
        finally:
            self.result_queue.put(("done", None))
            loop.close()


# ==============================================================================
# AppWindow（tkinter GUI）
# ==============================================================================

class AppWindow:
    """永続表示の tkinter GUI アプリ"""

    def __init__(self, debug: bool = False, img_scale: Optional[str] = None):
        self.debug = debug
        self.img_scale = img_scale
        self.is_processing = False

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

        # --- 中央：ステータスパネル（進捗表示）---
        status_frame = tk.LabelFrame(self.root, text="処理進捗", font=("Meiryo UI", 10), padx=5, pady=5)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 5))

        self.txt_status = tk.Text(
            status_frame, height=12, width=78,
            font=("Meiryo UI", 9), state=tk.DISABLED
        )
        scr = tk.Scrollbar(status_frame, orient=tk.VERTICAL, command=self.txt_status.yview)
        self.txt_status.configure(yscrollcommand=scr.set)
        self.txt_status.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scr.pack(side=tk.RIGHT, fill=tk.Y)

        # --- 下部：結果パネル（Treeview）---
        result_frame = tk.Frame(self.root, padx=10, pady=5)
        result_frame.pack(fill=tk.X)

        self.btn_copy = tk.Button(
            result_frame, text="📋 クリップボードにコピー",
            font=("Meiryo UI", 9), width=20, command=self._on_copy_results
        )
        self.btn_copy.pack(side=tk.LEFT)

        tv_frame = tk.Frame(result_frame)
        tv_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)

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

        # ワーカースレッド起動
        config = PipelineConfig(
            video_path=video_path,
            debug=self.debug,
            img_scale=self.img_scale
        )
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

                elif msg_type == "result":
                    # 結果をTreeviewに反映
                    for member, count in data.fan_counts.items():
                        formatted_count = f"{count:,}" if count > 0 else "非検出"
                        self.tv_results.insert("", tk.END, values=(member, formatted_count))

                elif msg_type == "error":
                    self._add_status(f"エラーが発生しました: {data}")

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
        """Treeviewの内容をクリップボードにコピー"""
        lines = []
        for item in self.tv_results.get_children():
            values = self.tv_results.item(item)["values"]
            if len(values) == 2:
                lines.append(f"{values[0]}\t{values[1]}")

        if not lines:
            return

        text = "\n".join(lines)
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
