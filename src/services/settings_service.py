from pathlib import Path
import json
from dataclasses import asdict
from src.domain.models import AppSettings, VLMConfig, PipelineConfig
from src.utils.exceptions import AppError

class SettingsService:
    """アプリ設定の保存・読み込み"""

    _SETTINGS_DIR = ".umamusume-fan-count"

    def __init__(self, settings_path: Path | None = None):
        if settings_path:
            self._settings_path = settings_path
        else:
            self._settings_path = Path.home() / self._SETTINGS_DIR / "settings.json"

    @property
    def settings_path(self) -> Path:
        return self._settings_path

    @settings_path.setter
    def settings_path(self, value: Path):
        self._settings_path = value

    def load(self) -> AppSettings:
        """ファイルから設定を読み込む。存在しない場合はデフォルト値を返す"""
        if not self._settings_path.exists():
            return AppSettings()
        
        try:
            with open(self._settings_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            # 読み込み失敗時はデフォルトを返す
            return AppSettings()

        # dataclass にマージ（存在しないフィールドはデフォルト）
        # 注意: AppSettings は dataclass なので、dict から直接展開できる
        # ただし、既存のフィールドを確実にカバーするためにマージ処理を行う
        defaults = AppSettings()
        merged = {}
        for key in defaults.__dataclass_fields__:
            if key in data:
                merged[key] = data[key]
            else:
                merged[key] = getattr(defaults, key)
        
        return AppSettings(**merged)

    def save(self, settings: AppSettings):
        """設定をファイルに保存"""
        self._settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._settings_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(settings), f, ensure_ascii=False, indent=2)

    @staticmethod
    def validate_roi(y_start: float, y_end: float, x_start: float, x_end: float) -> list[str]:
        """ROI値のバリデーション"""
        errors = []
        for name, val in [("y_start", y_start), ("y_end", y_end),
                          ("x_start", x_start), ("x_end", x_end)]:
            if not (0.0 <= val <= 1.0):
                errors.append(f"{name} は 0.0〜1.0 の範囲で指定してください。")
        if y_start >= y_end:
            errors.append("y_start は y_end より小さい値にしてください。")
        if x_start >= x_end:
            errors.append("x_start は x_end より小さい値にしてください。")
        return errors

    def apply_to_config(self, config) -> None:
        """AppSettings の値を PipelineConfig に反映"""
        settings = self.load()
        config.roi_y_start = settings.roi_y_start
        config.roi_y_end = settings.roi_y_end
        config.roi_x_start = settings.roi_x_start
        config.roi_x_end = settings.roi_x_end
        config.img_scale = settings.img_scale
        config.debug = settings.debug
        config.motion_detection_enabled = settings.motion_detection_enabled
        config.motion_threshold = settings.motion_threshold
        config.vlm_config = VLMConfig(
            enabled=settings.use_vlm or settings.mode == "vlm",
            model_path="models/gemma-4-e2b-it-edited-q4_0.gguf",
            mmproj_path="models/mmproj-gemma-4-e2b-it-q4_0.gguf",
            port=settings.vlm_port,
        )
