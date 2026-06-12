from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict

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

@dataclass
class ProgressUpdate:
    """進捗更新データ"""
    message: str            # 進捗メッセージ
    percent: float          # 0.0 〜 1.0 の進捗率
    frame_current: Optional[int] = None  # OCRフェーズでの現在フレーム数
    frame_total: Optional[int] = None    # OCRフェーズでの総フレーム数

@dataclass
class OCRResult:
    """OCR パイプラインの完了結果"""
    fan_counts: Dict[str, int]
    texts: List[str]
    error: Optional[str] = None

@dataclass
class AppSettings:
    """アプリ全体の設定値"""
    roi_y_start: float = 0.45
    roi_y_end: float = 0.88
    roi_x_start: float = 0.15
    roi_x_end: float = 0.45
    img_scale: Optional[str] = None
    debug: bool = False

@dataclass
class MemberEntry:
    """1人のメンバーの定義"""
    name: str                    # メンバー名（memberList.txt に出力）
    replace_patterns: List[str] = field(default_factory=list)  # OCR誤認識パターン
