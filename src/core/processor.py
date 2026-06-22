import os
import asyncio
import cv2
from pathlib import Path
from typing import List, Callable, Optional
import numpy as np
from src.domain.models import PipelineConfig
from src.services.ocr_service import OCRService
from src.utils.image import crop_roi, to_gray, img_to_bytes, check_motion

class StreamFrameProcessor:
    """フレームをファイルI/Oせず、メモリ上でcrop→gray→OCRを実行"""

    def __init__(self, config: PipelineConfig, ocr_service: OCRService):
        self.config = config
        self.ocr_service = ocr_service
        self._last_frame_gray: Optional[np.ndarray] = None

    async def process_video(
        self,
        on_progress: Callable[[str, float], None] = None,
    ) -> List[str]:
        """動画から逐次フレームを処理し、OCRテキストのリストを返す"""
        import cv2
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
        self._last_frame_gray = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cropped = crop_roi(frame, self.config.roi_y_start, self.config.roi_y_end, 
                               self.config.roi_x_start, self.config.roi_x_end)
            
            # デバッグ用保存（元のロジックを維持）
            if self.config.debug:
                os.makedirs(debug_path / "frames", exist_ok=True)
                os.makedirs(debug_path / "crop", exist_ok=True)
                os.makedirs(debug_path / "gray", exist_ok=True)
                cv2.imwrite(str(debug_path / f"frames/frame_{frame_idx:04d}.png"), frame)
                cv2.imwrite(str(debug_path / f"crop/crop_{frame_idx:03d}.png"), cropped)

            # 加工
            processed = cropped
            if self.config.img_scale == "gray":
                processed = to_gray(cropped)
                if self.config.debug:
                    cv2.imwrite(str(debug_path / f"gray/crop_gray_{frame_idx:03d}.png"), processed)

            # Motion Detection
            is_skipped = False
            if self.config.motion_detection_enabled:
                is_skipped, self._last_frame_gray = check_motion(
                    self._last_frame_gray, processed, self.config.motion_threshold
                )

            if is_skipped:
                frame_idx += 1
                if on_progress is not None:
                    pct = 0.2 + (frame_idx / max(frame_count, 1)) * 0.8
                    on_progress(f"モーション検知によりスキップ中... ({frame_idx + 1}/{frame_count})", pct)
                continue

            # OCR実行
            img_bytes = img_to_bytes(processed, self.config.img_scale == "gray")
            text = await self.ocr_service.recognize_image(img_bytes)
            
            # テキスト後処理は domain.logic で行う
            from src.domain.logic import post_process_ocr_text
            processed_text = post_process_ocr_text(text)

            if self.config.debug:
                os.makedirs(debug_path / "text", exist_ok=True)
                with open(debug_path / f"text/text-{frame_idx:03d}.txt", 'w', encoding='utf-8') as f:
                    f.write(processed_text)

            texts.append(processed_text)

            # 進捗通知
            if on_progress is not None:
                pct = 0.2 + (frame_idx / max(frame_count, 1)) * 0.8
                on_progress(f"OCR文字認識中 ({frame_idx + 1}/{frame_count})", pct)

            frame_idx += 1

        cap.release()
        return texts
