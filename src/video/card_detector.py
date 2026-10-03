import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Card:
    """1枚のユーザカードの情報"""
    role: str                 # "member" | "leader"
    badge_box: tuple          # バッジのボックス (x, y, w, h)
    name_box: Optional[tuple]  # ユーザ名 OCR 領域 (x, y, w, h) / 範囲外等で確定できない場合は None
    fan_box: Optional[tuple]  # ファン数 OCR 領域 (x, y, w, h) / None


class CardDetector:
    """テンプレートマッチングでユーザカードを検出する。

    検出の流れ:
    1. メンバー/リーダーバッジを検出 (matchTemplate + ピーク探索 + NMS)
    2. バッジ位置でフレーム端接していないか検証 (見切り除外)
    3. バッジごとに i_icon (ヘッダ行右端) と fan_count_label (ファン行) を探索
    4. ユーザ名領域 / ファン数領域を構築
    """

    MEMBER_TPL = "header_member.png"
    LEADER_TPL = "header_leader.png"
    ICON_TPL = "i_icon.png"
    LABEL_TPL = "fan_count_label.png"

    def __init__(self, template_dir: str, settings: Optional[dict] = None):
        settings = settings or {}
        cd = settings.get("card_detection", {})
        self.badge_threshold = float(cd.get("badge_match_threshold", 0.6))
        self.label_threshold = float(cd.get("label_match_threshold", 0.7))
        self.icon_threshold = float(cd.get("icon_match_threshold", 0.7))
        self.max_cards = int(cd.get("max_cards", 3))
        self.edge_margin = int(cd.get("edge_margin", 8))
        self.name_margin = int(cd.get("name_margin", 8))
        self.name_v_margin = int(cd.get("name_v_margin", 10))
        self.max_name_width = int(cd.get("max_name_width", 400))
        self.fan_width = int(cd.get("fan_width", 280))
        # マルチスケール（多解像度）検出
        self.reference_width = int(cd.get("reference_width", 2560))
        self.multi_scale = bool(cd.get("multi_scale", True))
        self.scale_window_low = float(cd.get("scale_window_low", 0.8))
        self.scale_window_high = float(cd.get("scale_window_high", 1.3))
        self.scale_step = float(cd.get("scale_step", 0.05))
        self.coarse_to_fine = bool(cd.get("coarse_to_fine", True))
        self.coarse_scale = float(cd.get("coarse_scale", 0.5))
        self.refine_radius = float(cd.get("refine_radius", 200))
        self.scale_cache = bool(cd.get("scale_cache", True))
        self._scale = None  # 前回フレームの勝者スケール（キャッシュ）
        self._scale_fw = None  # キャッシュ時のフレーム幅
        self._scale_fh = None  # キャッシュ時のフレーム高
        self._pyramid = {}  # スケールピラミッド（テンプレートを各 s でリサイズ済み）

        self.templates = {
            "member": self._load(template_dir, self.MEMBER_TPL),
            "leader": self._load(template_dir, self.LEADER_TPL),
            "i_icon": self._load(template_dir, self.ICON_TPL),
            "label": self._load(template_dir, self.LABEL_TPL),
        }

    def _load(self, template_dir: str, name: str) -> dict:
        path = f"{template_dir}/{name}"
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"テンプレートが見つかりません: {path}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return {"gray": gray, "w": int(gray.shape[1]), "h": int(gray.shape[0])}

    def _estimate_scale_prior(self, fw: int) -> tuple:
        """Layer A: フレーム幅 fw から中心スケール s0 と候補スケール集合 S を算出。

        s0 = fw / reference_width（参照解像度に対する等倍率）
        S  = { s0 * r | r ∈ [scale_window_low, scale_window_high], step = scale_step }
        reference_width が未設定 (0 未満) の場合は [0.5, 2.0] のフルスウィープにフォールバック。
        """
        if self.reference_width and self.reference_width > 0:
            s0 = fw / self.reference_width
            r = np.arange(self.scale_window_low, self.scale_window_high + 1e-9, self.scale_step)
            S = list(np.unique(np.round(s0 * r, 6)))
        else:
            s0 = 1.0
            S = list(np.round(np.arange(0.5, 2.0 + 1e-9, self.scale_step), 6))
        # s0 を必ず候補に含む（丸め誤差対策）
        s0 = round(float(s0), 6)
        if s0 not in S:
            S = sorted(set(S + [s0]))
        return s0, S

    def detect(self, frame) -> list:
        """フレームからユーザカードを検出。上から順にソートした Card リストを返す。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = gray.shape[:2]

        # ---- スケール決定 ----
        s = 1.0
        badges = []
        if self.multi_scale:
            # キャッシュを再利用（フレームサイズが同じ場合）
            if self.scale_cache and self._scale is not None and \
               self._scale_fw == fw and self._scale_fh == fh:
                s = self._scale
                badges = self._detect_at_scale(gray, s)
            else:
                s0, S = self._estimate_scale_prior(fw)
                # 検出（Layer C: 粗→精、または Layer B: 直接）
                if self.coarse_to_fine:
                    approx = self._detect_coarse(gray, S, s0)
                    if approx:
                        badges_list, s_w = self._refine_fine(gray, approx, S, s0)
                    else:
                        badges_list, s_w = self._detect_badges_multiscale(gray, S, s0)
                else:
                    badges_list, s_w = self._detect_badges_multiscale(gray, S, s0)
                # フルスウィープ [0.5, 2.0] フォールバック
                if not badges_list:
                    S_full = list(np.round(np.arange(0.5, 2.0 + 1e-9, self.scale_step), 6))
                    badges_full, s_w_full = self._detect_badges_multiscale(gray, S_full, 1.0)
                    if badges_full:
                        badges_list = badges_full
                        s_w = s_w_full
                if badges_list:
                    s = s_w
                    badges = badges_list
                else:
                    s = 1.0
                    badges = self._detect_badges(gray)
                self._scale = s
                self._scale_fw = fw
                self._scale_fh = fh
        else:
            s = 1.0
            badges = self._detect_badges(gray)

        # ---- 端除外・ソート・ max_cards ----
        badges = [b for b in badges if self._not_touching_edge(b, fw, fh)]
        badges = sorted(badges, key=lambda b: b[1])[: self.max_cards]

        # ---- カード構築（スケール認識アンカー検索 + 名前/ファン数ボックス） ----
        cards = []
        for (bx, by, bw, bh, role) in badges:
            icon = self._find_anchor(
                gray, "i_icon", self.icon_threshold,
                x0=bx, y0=by - int(10 * s), x1=bx + int(600 * s), y1=by + bh + int(10 * s),
                s=s,
            )
            label = self._find_anchor(
                gray, "label", self.label_threshold,
                x0=bx, y0=by + bh - int(10 * s), x1=bx + int(400 * s), y1=by + int(150 * s),
                s=s,
            )

            # ユーザ名領域: バッジ右端 + name_margin 〜 i_icon 左端 (未検出なら max_name_width)
            # 縦はバッジより上下 name_v_margin 拡張（ユーザ名文字はバッジより高いため）
            name_right = icon[0] if icon else (bx + bw + int(self.max_name_width * s))
            name_x = bx + bw + int(self.name_margin * s)
            name_box = self._clamp(
                name_x, by - int(self.name_v_margin * s), name_right - name_x,
                bh + 2 * int(self.name_v_margin * s), fw, fh,
            )

            # ファン数領域: ラベル右端 + name_margin、幅 fan_width、高さ label.h*1.2、ラベル行中央
            fan_box = None
            if label:
                lx, ly, lw, lh = label
                fan_h = max(1, int(lh * 1.2))
                fan_y = ly + lh // 2 - fan_h // 2
                fan_box = self._clamp(lx + lw + int(self.name_margin * s), fan_y,
                                       int(self.fan_width * s), fan_h, fw, fh)

            cards.append(Card(role, (bx, by, bw, bh), name_box, fan_box))

        return cards

    def _collect_scale_cands(self, gray, S: list, ox: int = 0, oy: int = 0) -> list:
        """スケール S 各 s でバッジ候補 [x, y, s, score_m, score_l] を収集する。
        ox/oy は局所窓のグローバル座標オフセット（_refine_fine 用）。"""
        pyramid = self._build_scale_pyramid(S)
        cands = []
        for s in S:
            m_t = pyramid[s]["member"]
            l_t = pyramid[s]["leader"]
            res_m = cv2.matchTemplate(gray, m_t, cv2.TM_CCOEFF_NORMED)
            res_l = cv2.matchTemplate(gray, l_t, cv2.TM_CCOEFF_NORMED)
            m_peaks = self._find_peaks(res_m, self.badge_threshold, s)
            l_peaks = self._find_peaks(res_l, self.badge_threshold, s)
            for x, y, _sm in m_peaks:
                cands.append([ox + x, oy + y, s,
                              float(res_m[y, x]), self._sample(res_l, y, x)])
            for x, y, _sl in l_peaks:
                gx, gy = ox + x, oy + y
                if not self._close(cands, gx, gy, int(24 * s)):
                    cands.append([gx, gy, s,
                                  self._sample(res_m, y, x), float(res_l[y, x])])
        return cands

    @staticmethod
    def _sample(res, y: int, x: int) -> float:
        """別のテンプレートの結果マップを境界チェック付きで参照する。

        member(123px)/leader(124px) テンプレートの幅差で結果マップは
        幅が 0〜1px 異なり、右端列が片方のマップだけ存在し得る。
        範囲外の位置は 0.0（= そのテンプレートには非対応）として扱う。
        """
        if 0 <= y < res.shape[0] and 0 <= x < res.shape[1]:
            return float(res[y, x])
        return 0.0

    def _cands_to_badges(self, cands: list) -> list:
        """候補 [(x, y, bw, bh, role)] リストに変換（role: sm >= sl → member）。"""
        results = []
        for x, y, s, sm, sl in cands:
            role = "member" if sm >= sl else "leader"
            bw = int(124 * s)
            bh = int(37 * s)
            results.append((x, y, bw, bh, role))
        return results

    def _detect_at_scale(self, gray, s: float) -> list:
        """特定のスケール s でバッジを検出（キャッシュ再利用用）。"""
        cands = self._collect_scale_cands(gray, [s])
        if not cands:
            return []
        cands = self._nms_badges(cands)
        return self._cands_to_badges(cands)

    # ------------------------------------------------------------------ #
    # 内部ヘルパ
    # ------------------------------------------------------------------ #

    def _detect_badges(self, gray) -> list:
        """バッジ（単一スケール）を検出し (x, y, w, h, role) のリストを返す。"""
        cands = self._collect_scale_cands(gray, [1.0])
        cands = self._nms_badges(cands)
        return self._cands_to_badges(cands)

    # ------------------------------------------------------------------ #
    # Layer B: 多解像度バッジ検出（案 1）
    # ------------------------------------------------------------------ #

    def _build_scale_pyramid(self, S: list) -> dict:
        """スケール候補 S に対し、テンプレートのリサイズをキャッシュする。"""
        for s in S:
            if s not in self._pyramid:
                self._pyramid[s] = {}
                for name, tpl in self.templates.items():
                    w = max(1, int(tpl["w"] * s))
                    h = max(1, int(tpl["h"] * s))
                    self._pyramid[s][name] = cv2.resize(
                        tpl["gray"], (w, h), interpolation=cv2.INTER_AREA
                    )
        return self._pyramid

    def _detect_badges_multiscale(self, gray, S: list, s0: float):
        """各スケールで matchTemplate → ピーク → 候補 → 勝者スケール → NMS。
        返値: (badges, s_w) — badges は [(x, y, bw, bh, role)]、s_w は勝者スケール。"""
        cands = self._collect_scale_cands(gray, S)
        if not cands:
            return [], None
        s_w = self._pick_winning_scale(cands, s0)
        cands_w = [c for c in cands if c[2] == s_w]
        cands_w = self._nms_badges(cands_w)
        return self._cands_to_badges(cands_w), s_w

    def _pick_winning_scale(self, cands: list, s0: float) -> float:
        """score(s) = sum(max(sm, sl)) で argmax。同点 → s0 に近い方を優先。"""
        scores = {}
        for c in cands:
            s = c[2]
            sm, sl = c[3], c[4]
            scores[s] = scores.get(s, 0.0) + max(sm, sl)
        best_s = max(scores, key=lambda s: (scores[s], -abs(s - s0)))
        return best_s

    # ------------------------------------------------------------------ #
    # Layer C: 粗→精検出（案 3）
    # ------------------------------------------------------------------ #

    def _detect_coarse(self, gray, S: list, s0: float) -> list:
        """下サンプリングで高速バッジ検出。
        返値: [(x, y, s, role)] の近似バッジ（元解像度座標）。"""
        cs = self.coarse_scale
        h, w = gray.shape
        sh, sw = max(1, int(h * cs)), max(1, int(w * cs))
        small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_AREA)
        # スケール候補を補正（下サンプリング後の解像度に対するスケール）
        S_c = [s * cs for s in S]
        s0_c = s0 * cs
        badges, s_w = self._detect_badges_multiscale(small, S_c, s0_c)
        if not badges:
            return []
        approx = []
        for x, y, bw, bh, role in badges:
            approx.append([x / cs, y / cs, s_w / cs, role])
        return approx

    def _refine_fine(self, gray, approx: list, S: list, s0: float):
        """粗いバッジ位置からの局所窓内で精密検出。
        返値: (badges, s_w) — badges は [(x, y, bw, bh, role)]。"""
        cands = []
        for ax, ay, as_, role in approx:
            r = int(self.refine_radius)
            x0 = max(0, int(ax) - r)
            y0 = max(0, int(ay) - r)
            x1 = min(gray.shape[1], int(ax) + r)
            y1 = min(gray.shape[0], int(ay) + r)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            sub = gray[y0:y1, x0:x1]
            # 局所窓にテンプレートが収まるスケールのみ有効
            valid = [s for s in S
                     if sub.shape[1] >= max(1, int(self.templates["member"]["w"] * s))
                     and sub.shape[0] >= max(1, int(self.templates["member"]["h"] * s))]
            cands.extend(self._collect_scale_cands(sub, valid, ox=x0, oy=y0))
        if not cands:
            return [], None
        s_w = self._pick_winning_scale(cands, s0)
        cands_w = [c for c in cands if c[2] == s_w]
        cands_w = self._nms_badges(cands_w)
        return self._cands_to_badges(cands_w), s_w

    def _find_peaks(self, res, threshold: float, s: float = 1.0) -> list:
        """結果マップから閾値超えのピークを NMS しながら抽出。"""
        peaks = []
        res = res.copy()
        h, w = res.shape
        r = int(40 * s)
        while True:
            _mn, mx, _ml, xl = cv2.minMaxLoc(res)
            if mx < threshold:
                break
            x, y = xl
            peaks.append((x, y, float(mx)))
            res[max(0, y - r): min(h, y + r + 1), max(0, x - r): min(w, x + r + 1)] = -1
        return peaks

    def _find_anchor(self, gray, tpl_name: str, threshold: float,
                     x0: int, y0: int, x1: int, y1: int,
                     s: float = 1.0) -> Optional[tuple]:
        """領域 [x0:x1, y0:y1] 内で tpl の最良一致を検出。ヒットすれば (x, y, w, h) を返す。
        s != 1.0 かつピラミッドに存在する場合はスケール済みテンプレートを使う。"""
        if s != 1.0 and s in self._pyramid and tpl_name in self._pyramid[s]:
            t = self._pyramid[s][tpl_name]
        else:
            t = self.templates[tpl_name]["gray"]
        tw, th = t.shape[1], t.shape[0]
        x0 = max(0, int(x0)); y0 = max(0, int(y0))
        x1 = min(gray.shape[1], int(x1)); y1 = min(gray.shape[0], int(y1))
        if x1 - x0 < tw or y1 - y0 < th:
            return None
        sub = gray[y0:y1, x0:x1]
        res = cv2.matchTemplate(sub, t, cv2.TM_CCOEFF_NORMED)
        _mn, mx, _ml, xl = cv2.minMaxLoc(res)
        if mx < threshold:
            return None
        px, py = xl
        return (x0 + px, y0 + py, tw, th)

    def _nms_badges(self, cands: list) -> list:
        """バッジ候補を NMS (IoU>0.5 を抑制) し、高スコア順に整える。
        cands は [x, y, s, score_m, score_l] の5要素リスト。"""
        def score(c):
            return max(c[3], c[4])

        def box(c):
            return self._as_box(c, c[2])

        cands = sorted(cands, key=score, reverse=True)
        kept = []
        for c in cands:
            if all(self._iou(box(c), box(k)) <= 0.5 for k in kept):
                kept.append(c)
        return kept

    def _as_box(self, c, s: float = 1.0) -> tuple:
        """候補からバッジボックス (x, y, w, h) を生成。s でスケールする。"""
        # s=1.0 で現行の (124, 37) と一致
        w = int(124 * s) if s != 1.0 else 124
        h = int(37 * s) if s != 1.0 else 37
        return (c[0], c[1], w, h)

    @staticmethod
    def _iou(b1, b2) -> float:
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[0] + b1[2], b2[0] + b2[2]); y2 = min(b1[1] + b1[3], b2[1] + b2[3])
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        a1 = b1[2] * b1[3]; a2 = b2[2] * b2[3]
        return inter / max(a1 + a2 - inter, 1e-6)

    @staticmethod
    def _close(cands: list, x: int, y: int, r: int) -> bool:
        for cx, cy, *_ in cands:
            if abs(cx - x) <= r and abs(cy - y) <= r:
                return True
        return False

    def _not_touching_edge(self, badge, fw: int, fh: int) -> bool:
        """バッジがフレーム端に edge_margin 以内に接していないか (見切り除外)"""
        x, y, w, h, *_ = badge
        m = self.edge_margin
        return x >= m and y >= m and (x + w) <= (fw - m) and (y + h) <= (fh - m)

    @staticmethod
    def _clamp(x: int, y: int, w: int, h: int, fw: int, fh: int) -> Optional[tuple]:
        x = max(0, int(x)); y = max(0, int(y))
        w = int(w); h = int(h)
        if x + w > fw:
            w = fw - x
        if y + h > fh:
            h = fh - y
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)
