"""検知名（OCR 検知の生名前）を実ユーザ名へマッピングするモジュール。

設計書 ``docs/name_mapping_design.md`` の §8.1 に従い、以下の3要素で構成される。

- ``MappedName``: マッピング適用結果を保持するデータクラス。
- ``NameMapper``: 単一の検知に対して完全一致 / 近似一致 / 未マッピング判定を行う。
- ``NameMapperLoader``: JSON 形式のマッピング定義ファイルを読み込む。

マッピング定義ファイル（``config/name_mapping.json``）は、利用者が高頻度で
編集することを前提として JSON 形式とする。
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MappedName:
    """マッピング適用結果。

    Attributes:
        user_name: マッピング適用後の実ユーザ名（一致なし時は ``None``）。
        raw_name: 元の検知名。
        matched: ``user_names`` / ``raw_to_user`` に一致したか。
        match_type: ``"exact"`` | ``"approx"`` | ``None``。
        alias_hit: 一致した alias（近似時は最も近い alias）。
        warning: 近似時の警告メッセージ等（無い場合は ``None``）。
    """

    user_name: str | None
    raw_name: str
    matched: bool
    match_type: str | None
    alias_hit: str | None
    warning: str | None


def levenshtein_distance(a: str, b: str) -> int:
    """二つの文字列のレーベンシュタイン距離（編集距離）を返す。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # 1D DP。行 a の文字ごとに前行(prev)を更新する。
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


class NameMapper:
    """検知名を実ユーザ名へマッピングする。

    Args:
        mapping: マッピング定義辞書（``NameMapperLoader.load_mapping`` の返り値）。
        edit_distance_threshold: 近似一致のレーベンシュタイン距離閾値。
        unmapped_action: 未マッピング検知名の扱い (``"suggest"`` | ``"keep"`` | ``"drop"``)。
        warn_on_approx: 近似一致時に警告を生成するかどうか。
    """

    def __init__(
        self,
        mapping: dict | None = None,
        edit_distance_threshold: int = 2,
        unmapped_action: str = "suggest",
        warn_on_approx: bool = True,
    ):
        self.mapping = mapping or {}
        self.edit_distance_threshold = edit_distance_threshold
        self.unmapped_action = unmapped_action
        self.warn_on_approx = warn_on_approx
        self.user_names = self.mapping.get("user_names", {})
        self.raw_to_user = self.mapping.get("raw_to_user", {})
        # unmapped_action="suggest" のとき、マッピングされなかった検知名を記録する。
        self.unmapped_names: list[str] = []
        # 近似一致時の警告メッセージ（``warn_on_approx=True``）を記録する。
        self.warnings: list[str] = []

    def map(self, raw_name: str) -> MappedName:
        """単一の検知に対して 5.1〜5.3 の優先順位で一致判定を行い ``MappedName`` を返す。

        優先順位:
            1. 完全一致（``user_names`` の ``aliases`` 内）→ ``exact``
            2. 完全一致（``raw_to_user``）→ ``exact``
            3. 近似一致（``user_names`` の ``aliases`` 内、距離 <= 閾値）→ ``approx``
            4. 一致なし → ``matched=False``（``unmapped_action`` に従う）
        """
        # 1. 完全一致（user_names の aliases 内）
        for user_name, entry in self.user_names.items():
            aliases = entry.get("aliases", [])
            if raw_name in aliases:
                return MappedName(user_name, raw_name, True, "exact", raw_name, None)

        # 2. 完全一致（raw_to_user）
        if raw_name in self.raw_to_user:
            user_name = self.raw_to_user[raw_name]
            return MappedName(user_name, raw_name, True, "exact", raw_name, None)

        # 3. 近似一致（user_names の aliases 内）
        #    最小距離の実ユーザ名を採用。同距離の場合は実ユーザ名をアルファベット順
        #    （Unicode コードポイント順）で小さい方を優先する（設計書 5.2）。
        best: tuple[int, str, str] | None = None
        for user_name, entry in self.user_names.items():
            for alias in entry.get("aliases", []):
                dist = levenshtein_distance(raw_name, alias)
                if dist == 0:
                    # 完全一致は1で処理済み。念のため再帰的に除外する。
                    continue
                if dist > self.edit_distance_threshold:
                    continue
                if best is None or dist < best[0] or (dist == best[0] and user_name < best[1]):
                    best = (dist, user_name, alias)

        if best is not None:
            dist, user_name, alias = best
            warning = None
            if self.warn_on_approx:
                warning = (
                    f"近似一致 (edit_distance={dist}): "
                    f"'{raw_name}' -> '{user_name}' (alias '{alias}')"
                )
            return MappedName(user_name, raw_name, True, "approx", alias, warning)

        # 4. 一致なし
        return MappedName(None, raw_name, False, None, None, None)


class NameMapperLoader:
    """マッピング定義ファイル（JSON）の読み込み。

    - ファイルが存在しない場合は空辞書を返し、警告を出力する（フォールバック）。
    - JSON 構文エラーや形式不正の場合は ``ValueError`` を返す。
    """

    @staticmethod
    def load_mapping(path: str | Path) -> dict:
        """マッピング定義ファイルを読み込み、辞書として返す。

        Args:
            path: マッピング定義ファイルのパス。

        Returns:
            マッピング辞書（ファイル未存在時は空辞書）。

        Raises:
            ValueError: ファイルの JSON 構文エラー or 形式不正の場合。
        """
        p = Path(path)
        if not p.exists():
            logger.warning(
                "マッピング定義ファイルが存在しません: %s (空辞書でフォールバック)", path
            )
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"マッピング定義ファイルのJSON構文エラー: {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(
                f"マッピング定義ファイルはオブジェクトである必要があります: {path}"
            )
        return data


def ensure_name_mapping_file(path: str | Path) -> Path:
    """マッピング定義ファイルが存在しない場合はデフォルトテンプレートを作成する。

    ユーザー別データのため zip に同梱せず、初回実行時に作成する。
    作成時は空のマッピング（``user_names`` / ``raw_to_user`` が空）を
    書き出す。ユーザーは以降に編集してマッピングを追加する。

    Args:
        path: マッピング定義ファイルのパス。

    Returns:
        ファイルパス。
    """
    p = Path(path)
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        default = {"user_names": {}, "raw_to_user": {}}
        with open(p, "w", encoding="utf-8") as f:
            json.dump(default, f, ensure_ascii=False, indent=2)
        logger.info("マッピング定義ファイルを作成しました: %s", p)
    return p
