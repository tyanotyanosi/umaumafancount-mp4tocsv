import asyncio
from io import BytesIO
from PIL import Image
import numpy as np
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.graphics.imaging import BitmapDecoder
from winrt.windows.storage.streams import DataWriter, InMemoryRandomAccessStream

class OCRService:
    """WinRT OCR エンジンへのアクセスを管理"""

    async def _ocr_image_bytes(self, img_bytes: bytes) -> str:
        """画像バイト列からテキストを認識する"""
        stream = InMemoryRandomAccessStream()
        writer = DataWriter(stream)
        writer.write_bytes(img_bytes)
        await writer.store_async()
        await writer.flush_async()
        stream.seek(0)

        decoder = await BitmapDecoder.create_async(stream)
        software_bitmap = await decoder.get_software_bitmap_async()

        engine = OcrEngine.try_create_from_user_profile_languages()
        if not engine:
            raise RuntimeError("OCRエンジンを初期化できませんでした。")

        result = await engine.recognize_async(software_bitmap)
        return result.text

    async def recognize_image(self, img_bytes: bytes) -> str:
        """非同期で画像を認識する"""
        return await self._ocr_image_bytes(img_bytes)
