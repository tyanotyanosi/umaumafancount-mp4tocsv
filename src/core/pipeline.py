import asyncio
import threading
import queue
import time
from pathlib import Path
from typing import List, Callable, Any
import shutil

from src.domain.models import PipelineConfig, OCRResult
from src.core.processor import StreamFrameProcessor
from src.core.extractor import FanCountExtractor
from src.services.ocr_service import OCRService
from src.utils.exceptions import wrap_error, AppError

class PipelineRunner:
    """OCR パイプラインの全ステップを実行"""

    def __init__(self):
        pass

    def run(
        self, config: PipelineConfig, on_progress: Callable[[str, float], None]
    ) -> OCRResult:
        try:
            base_path = Path("output/")
            debug_path = base_path / "debug"

            # 出力ディレクトリ初期化
            on_progress("出力ディレクトリを初期化します...", 0.0)
            shutil.rmtree(base_path, ignore_errors=True)
            base_path.mkdir(parents=True, exist_ok=True)
            os.makedirs(debug_path / "crop", exist_ok=True)
            os.makedirs(debug_path / "gray", exist_ok=True)
            os.makedirs(debug_path / "frames", exist_ok=True)
            os.makedirs(debug_path / "text", exist_ok=True)

            video_path = config.video_path
            on_progress(f"動画を読み込みます: {video_path.name}", 0.05)

            # 1. フレーム処理 + OCR
            ocr_start_time = time.time()
            ocr_service = OCRService()
            processor = StreamFrameProcessor(config, ocr_service)

            texts = []
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                def wrapped_on_progress(msg: str, pct: float = 0.0):
                    on_progress(msg, pct)

                texts = loop.run_until_complete(
                    processor.process_video(on_progress=wrapped_on_progress)
                )
            finally:
                loop.close()

            if config.debug:
                join_text = "\n".join(texts)
                with open(debug_path / "text/output.txt", 'w', encoding='utf-8') as f:
                    f.write(join_text)

            # 2. メンバーごとのファン数を抽出
            on_progress("メンバーごとのファン数を抽出します...", 0.98)
            extractor = FanCountExtractor(
                Path('input/memberList.txt'),
                Path('input/memberReplace.json')
            )
            fan_counts = extractor.extract(texts)

            # 結果出力
            with open('output/output.json', 'w', encoding='UTF-8') as f:
                import json
                json.dump(fan_counts, f, ensure_ascii=False, indent=4)

            if not config.debug:
                shutil.rmtree(debug_path, ignore_errors=True)

            on_progress("処理が完了しました。", 1.0)
            return OCRResult(fan_counts=fan_counts, texts=texts)

        except Exception as e:
            return OCRResult(fan_counts={}, texts=[], error=str(e))

import os # Required for os.makedirs
import time # Required for time.time
import shutil # Required for shutil.rmtree

class PipelineWorker(threading.Thread):
    """OCRパイプラインを別スレッドで実行"""

    def __init__(self, config: PipelineConfig, result_queue: queue.Queue):
        super().__init__()
        self.config = config
        self.result_queue = result_queue
        self.runner = PipelineRunner()

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
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
