import asyncio
import concurrent.futures
import threading
import queue
import time
import json
from pathlib import Path
from typing import List, Callable, Any, Optional
import shutil
import os
import numpy as np

from src.domain.models import PipelineConfig, OCRResult, VLMConfig
from src.core.processor import StreamFrameProcessor
from src.core.extractor import FanCountExtractor
from src.services.ocr_service import OCRService
from src.services.vlm_service import VLMService
from src.utils.exceptions import wrap_error, AppError
from src.utils.logging_utils import log_vlm_success

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

            is_vlm = config.vlm_config is not None and config.vlm_config.enabled

            # 1. フレーム処理 + OCR/VLM
            ocr_start_time = time.time()
            
            if is_vlm:
                texts = []
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                executor = None
                try:
                    def wrapped_on_progress(msg: str, pct: float = 0.0):
                        on_progress(msg, pct)

                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    processor = VLMFrameProcessor(config, executor=executor)
                    texts = loop.run_until_complete(
                        processor.process_video_vlm(on_progress=wrapped_on_progress)
                    )
                finally:
                    if executor:
                        executor.shutdown(wait=True)
                    loop.close()
            else:
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

import cv2
from PIL import Image
from src.utils.image import crop_roi, check_motion

class VLMFrameProcessor:
    """VLM 用のフレーム処理クラス - フルフレームを対象に画像を抽出"""

    def __init__(self, config: PipelineConfig, executor=None):
        self.config = config
        self.vlm_config = config.vlm_config
        self._last_frame_gray: Optional[np.ndarray] = None
        self._executor = executor

    async def process_video_vlm(
        self,
        on_progress: Callable[[str, float], None] = None,
    ) -> List[str]:
        """動画からフレームを抽出し、VLM で解析して JSON テキストのリストを返す"""
        print(f"[VLMFrameProcessor] Starting VLM processing for: {self.config.video_path}")
        cap = cv2.VideoCapture(str(self.config.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"動画ファイルを読み込めませんでした: {self.config.video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[VLMFrameProcessor] Total frames: {frame_count}")
        results: List[str] = []
        debug_path = Path("output/debug")

        vlm_service = VLMService(self.vlm_config, executor=self._executor)
        try:
            # VLM 用の ROI 設定（OCR と同じ領域）
            y_start, y_end = 0.45, 0.88
            x_start, x_end = 0.15, 0.45
            print(f"[VLMFrameProcessor] ROI: y=[{y_start}-{y_end}], x=[{x_start}-{x_end}]")

            frame_idx = 0
            processed_count = 0
            error_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"[VLMFrameProcessor] End of video. Processed: {processed_count}, Errors: {error_count}")
                    break

                # ROI を crop（ファン数表示領域のみ）
                h, w = frame.shape[:2]
                y1, y2 = int(h * y_start), int(h * y_end)
                x1, x2 = int(w * x_start), int(w * x_end)
                roi = frame[y1:y2, x1:x2]

                # Motion Detection
                if self.config.motion_detection_enabled:
                    skip, self._last_frame_gray = check_motion(
                        self._last_frame_gray, roi, self.config.motion_threshold
                    )
                    if skip:
                        frame_idx += 1
                        if on_progress is not None:
                            pct = 0.2 + (frame_idx / max(frame_count, 1)) * 0.8
                            on_progress(f"モーション検知によりスキップ中... ({frame_idx + 1}/{frame_count})", pct)
                        continue

                # RGB に変換して PIL Image に
                rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_roi)

                # デバッグ用保存
                if self.config.debug:
                    os.makedirs(debug_path / "frames", exist_ok=True)
                    cv2.imwrite(str(debug_path / f"frames/frame_{frame_idx:04d}.png"), frame)
                    os.makedirs(debug_path / "crop", exist_ok=True)
                    cv2.imwrite(str(debug_path / f"crop/frame_{frame_idx:04d}.png"), roi)

                try:
                    fan_dict = await vlm_service.analyze_image(image, frame_index=frame_idx)
                    json_str = json.dumps(fan_dict, ensure_ascii=False, indent=2)
                    results.append(json_str)
                    processed_count += 1
                    if self.config.debug:
                        print(f"[VLMFrameProcessor] Frame {frame_idx} 成功: {json_str[:200]}")
                        log_vlm_success(
                            frame_index=frame_idx,
                            parsed_result=fan_dict,
                            image_size=image.size,
                        )
                    else:
                        if processed_count <= 3:
                            print(f"[VLMFrameProcessor] Frame {frame_idx} 成功: {json_str[:200]}")
                except Exception as e:
                    error_count += 1
                    results.append(f"VLM_ERROR: {str(e)}")
                    if self.config.debug:
                        print(f"[VLMFrameProcessor] Frame {frame_idx} VLM推論エラー: {e}")

                # 進捗通知
                if on_progress is not None:
                    pct = 0.2 + (frame_idx / max(frame_count, 1)) * 0.8
                    on_progress(f"VLM 推論中 ({frame_idx + 1}/{frame_count})", pct)

                frame_idx += 1
        finally:
            await vlm_service.stop()
            cap.release()

        print(f"[VLMFrameProcessor] Processing complete. Total results: {len(results)}")
        return results


class PipelineWorker(threading.Thread):
    """OCRパイプラインを別スレッドで実行"""

    def __init__(self, config: PipelineConfig, result_queue: queue.Queue):
        super().__init__()
        self.config = config
        self.result_queue = result_queue
        self.runner = PipelineRunner()

    def run(self):
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
