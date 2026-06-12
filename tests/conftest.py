"""pytest 共有フィクスチャと一元化モック設定

全てのテストファイルに先立ち、外部依存モジュール（cv2, PIL, winrt, tqdm）を
Mock で差し替える。これにより pytest 単体で完結する。
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# モジュールインポート前のモック（src/ 以下のモジュールが import 時に
# cv2, PIL, winrt.* を読み込むため、import 前に sys.modules に差し込む）
# ---------------------------------------------------------------------------
_MOCKS: dict[str, MagicMock] = {}

for _mod_name in [
    "cv2",
    "PIL",
    "PIL.Image",
    "winrt",
    "winrt.windows.media.ocr",
    "winrt.windows.graphics.imaging",
    "winrt.windows.storage.streams",
    "winrt.windows.foundation",
    "winrt.windows.storage",
    "winrt.runtime",
    "tqdm",
]:
    _m = MagicMock()
    _MOCKS[_mod_name] = _m
    sys.modules[_mod_name] = _m


# ---------------------------------------------------------------------------
# プロジェクトルートを sys.path に追加（src/ パッケージ発見のため）
# ---------------------------------------------------------------------------
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# 各テストファイルで再利用可能なフィクスチャ
# ---------------------------------------------------------------------------
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def mock_cv2() -> MagicMock:
    """cv2 モックへの参照を提供"""
    return _MOCKS["cv2"]
