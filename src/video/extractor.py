from src.video.reader import VideoReader


class FrameExtractor:
    """指定間隔でフレームを抽出"""

    def __init__(self, video_reader: VideoReader, interval_sec: float = 1.0):
        self.video_reader = video_reader
        self.interval_sec = interval_sec

    def iter_frames(self):
        """フレームを1枚ずつ生成するジェネレータ（メモリ効率・ストリーミング差分判定用）。

        - interval_sec > 0: int(fps * interval_sec) 間隔のフレームのみ
        - interval_sec == 0: 全フレーム（サンプリングせず、差分判定のみで採用を決定）
        """
        fps = self.video_reader.get_fps()
        interval_frames = max(1, int(fps * self.interval_sec))
        count = 0

        while True:
            frame = self.video_reader.read_frame()
            if frame is None:
                break
            if count % interval_frames == 0:
                yield frame
            count += 1

    def extract(self) -> list:
        """間隔 interval_sec ごとにフレームを抽出（リストで返す）"""
        return list(self.iter_frames())
