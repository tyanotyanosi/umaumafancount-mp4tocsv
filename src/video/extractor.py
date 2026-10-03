from typing import Optional

from src.video.reader import VideoReader


def compute_frame_range(fps: float, total_frames: int, start_sec: float = 0.0,
                        end_sec: float = 0.0, limit_sec: float = 0.0) -> tuple:
    """処理対象の ``(start_frame, stop_frame)`` を返す（stop_frame は最終処理フレーム・両端を含む）。

    - ``start_sec``: 動画開始からの秒（デフォルト 0）
    - ``end_sec``: 動画開始からの絶対秒（デフォルト 0 = 最後のフレームまで）
    - ``limit_sec``: start からの最大処理時間（秒、デフォルト 0 = 無制限）

    フレームインデックスは 0..total_frames-1 として扱う。
    動画が空、または範囲が空（開始位置が動画超過、start > stop）の場合 ``ValueError``。
    """
    if total_frames <= 0:
        raise ValueError("動画にフレームがありません")
    fps = max(0.0, float(fps))

    # start_frame を先にクランプしてから stop を計算する
    # （負の start で limit 計算がずれるのを防ぐ）
    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = int(round(end_sec * fps)) if end_sec > 0 else total_frames - 1
    stop = end_frame
    if limit_sec > 0:
        stop = min(stop, start_frame + int(round(limit_sec * fps)))

    stop = max(0, stop)
    if stop >= total_frames:
        stop = total_frames - 1
    if start_frame >= total_frames:
        raise ValueError(f"開始位置が動画長を超過します: start={start_sec}s（動画={total_frames} フレーム）")
    if start_frame > stop:
        raise ValueError(f"空のフレーム範囲です: start={start_sec}s, end={end_sec}s, limit={limit_sec}s")
    return (start_frame, stop)


class FrameExtractor:
    """指定間隔でフレームを抽出"""

    def __init__(self, video_reader: VideoReader, interval_sec: float = 1.0):
        self.video_reader = video_reader
        self.interval_sec = interval_sec

    def iter_frames(self, max_frames: Optional[int] = None):
        """フレームを1枚ずつ生成するジェネレータ（メモリ効率・ストリーミング差分判定用）。

        - interval_sec > 0: int(fps * interval_sec) 間隔のフレームのみ
        - interval_sec == 0: 全フレーム（サンプリングせず、差分判定のみで採用を決定）
        - max_frames: 現在位置からデコードする総フレーム数の上限（範囲指定用）。
          None = 最後まで。
        """
        fps = self.video_reader.get_fps()
        interval_frames = max(1, int(fps * self.interval_sec))
        count = 0

        while True:
            if max_frames is not None and count >= max_frames:
                break
            frame = self.video_reader.read_frame()
            if frame is None:
                break
            if count % interval_frames == 0:
                yield frame
            count += 1

    def extract(self, max_frames: Optional[int] = None) -> list:
        """間隔 interval_sec ごとにフレームを抽出（リストで返す）"""
        return list(self.iter_frames(max_frames=max_frames))
