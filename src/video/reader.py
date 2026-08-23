import cv2
from typing import Optional


class VideoReader:
    """動画ファイルの読み込み・デコード"""

    def __init__(self, video_path: str):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.cap.release()
            raise FileNotFoundError(f"動画を開けません: {video_path}")

    def get_frame_count(self) -> int:
        """フレーム総数を取得"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_fps(self) -> float:
        """FPSを取得"""
        return self.cap.get(cv2.CAP_PROP_FPS)

    def get_resolution(self) -> tuple[int, int]:
        """解像度 (幅, 高さ) を取得"""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def read_frame(self) -> Optional[object]:
        """1フレーム読み込み"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def seek(self, frame_idx: int) -> bool:
        """指定フレームにシーク"""
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    def release(self):
        """リソース解放"""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
