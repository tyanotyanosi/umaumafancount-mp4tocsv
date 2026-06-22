from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict

@dataclass
class VLMConfig:
    """VLM (Vision-Language Model) 推論の設定"""
    enabled: bool = False
    model_path: str = "models/gemma-4-e2b-it-edited-q4_0.gguf"
    mmproj_path: str = "models/mmproj-gemma-4-e2b-it-q4_0.gguf"
    temperature: float = 0.0
    max_tokens: int = 8192
    prompt_template: Optional[str] = None
    port: int = 8080
    ngl: int = 99
    debug: bool = False

@dataclass
class PipelineConfig:
    """OCR/VLM パイプライン実行の設定"""
    video_path: Path
    debug: bool = False
    img_scale: Optional[str] = None  # "gray" or None
    roi_y_start: float = 0.45       # ROI Y軸開始（画面高さの割合）
    roi_y_end: float = 0.88         # ROI Y軸終了（画面高さの割合）
    roi_x_start: float = 0.15       # ROI X軸開始（画面幅の割合）
    roi_x_end: float = 0.45         # ROI X軸終了（画面幅の割合）
    vlm_config: Optional[VLMConfig] = None
    mode: str = "ocr"  # "ocr" or "vlm"
    motion_detection_enabled: bool = False
    motion_threshold: float = 0.01

@dataclass
class ProgressUpdate:
    """進捗更新データ"""
    message: str            # 進捗メッセージ
    percent: float          # 0.0 〜 1.0 の進捗率
    frame_current: Optional[int] = None  # OCRフェーズでの現在フレーム数
    frame_total: Optional[int] = None    # OCRフェーズでの総フレーム数

@dataclass
class OCRResult:
    """OCR/VLM パイプラインの完了結果"""
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
    use_vlm: bool = False
    mode: str = "ocr"  # "ocr" or "vlm"
    motion_detection_enabled: bool = False
    motion_threshold: float = 0.01
    vlm_port: int = 8080

@dataclass
class MemberEntry:
    """1人のメンバーの定義"""
    name: str                    # メンバー名（memberList.txt に出力）
    replace_patterns: List[str] = field(default_factory=list)  # OCR誤認識パターン
