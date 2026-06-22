import re
import tkinter as tk
import tkinter.messagebox as mb
import tkinter.simpledialog as sd
import tkinter.ttk as ttk
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from src.domain.models import AppSettings, MemberEntry
from src.services.settings_service import SettingsService

class SettingsDialog:
    """ROIと各種オプションを設定するToplevelウィンドウ"""

    def __init__(self, parent, settings_service: SettingsService):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("設定")
        self.dialog.geometry("550x500")
        self.dialog.resizable(True, True)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        self.settings_service = settings_service
        self.current_settings = settings_service.load()

        # Scrollable content area with buttons fixed at bottom
        content_frame = tk.Frame(self.dialog)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)

        # Horizontal scroll frame
        h_scroll_frame = tk.Frame(content_frame)
        h_scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Vertical scrollbar frame
        v_scroll_frame = tk.Frame(content_frame)
        v_scroll_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(v_scroll_frame, highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(v_scroll_frame, orient="vertical", command=canvas.yview)
        h_scrollbar = ttk.Scrollbar(h_scroll_frame, orient="horizontal", command=canvas.xview)
        scrollable_area = tk.Frame(canvas)

        scrollable_area.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_area, anchor="nw")
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        h_scroll_frame.pack(fill=tk.BOTH, expand=True)
        v_scroll_frame.pack(fill=tk.BOTH, expand=True)
        canvas.pack(side="left", fill="both", expand=True)
        v_scrollbar.pack(side="right", fill="y", in_=v_scroll_frame)
        h_scrollbar.pack(side="bottom", fill="x", in_=h_scroll_frame)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", _on_mousewheel)
        canvas.bind("<Button-5>", _on_mousewheel)

        self.entry_y_start = self._create_roi_entry(scrollable_area, "Y 軸開始（上端比）", self.current_settings.roi_y_start)
        self.entry_y_end = self._create_roi_entry(scrollable_area, "Y 軸終了（下端比）", self.current_settings.roi_y_end)
        self.entry_x_start = self._create_roi_entry(scrollable_area, "X 軸開始（左端比）", self.current_settings.roi_x_start)
        self.entry_x_end = self._create_roi_entry(scrollable_area, "X 軸終了（右端比）", self.current_settings.roi_x_end)

        scale_frame = tk.LabelFrame(scrollable_area, text="画像スケーリング", font=("Meiryo UI", 9), padx=5, pady=5)
        scale_frame.pack(fill=tk.X, pady=(10, 5))
        self.var_img_scale = tk.StringVar(value=self.current_settings.img_scale or "")
        rb_none = tk.Radiobutton(scale_frame, text="なし", variable=self.var_img_scale, value="", font=("Meiryo UI", 9))
        rb_none.pack(anchor=tk.W)
        rb_gray = tk.Radiobutton(scale_frame, text="グレースケール変換後にOCR実行", variable=self.var_img_scale, value="gray", font=("Meiryo UI", 9))
        rb_gray.pack(anchor=tk.W)

        mode_frame = tk.LabelFrame(scrollable_area, text="認識モード", font=("Meiryo UI", 9), padx=5, pady=5)
        mode_frame.pack(fill=tk.X, pady=(5, 5))
        self.var_mode = tk.StringVar(value=self.current_settings.mode or "ocr")
        rb_ocr = tk.Radiobutton(mode_frame, text="WinRT OCR", variable=self.var_mode, value="ocr", font=("Meiryo UI", 9))
        rb_ocr.pack(anchor=tk.W)
        rb_vlm = tk.Radiobutton(mode_frame, text="VLM (Gemma-4)", variable=self.var_mode, value="vlm", font=("Meiryo UI", 9))
        rb_vlm.pack(anchor=tk.W)
        self.var_mode.trace_add("write", lambda *args: self._on_mode_change())

        vlm_frame = tk.Frame(scrollable_area)
        vlm_frame.pack(fill=tk.X, pady=(0, 5))
        self.var_vlm_enabled = tk.BooleanVar(value=self.current_settings.use_vlm)
        self.chk_vlm = tk.Checkbutton(vlm_frame, text="VLM 有効化（設定→VLMモード選択時に自動ON）", variable=self.var_vlm_enabled, font=("Meiryo UI", 9))
        self.chk_vlm.pack(side=tk.LEFT)
        self.entry_vlm_port = self._create_roi_entry(vlm_frame, "VLM ポート", self.current_settings.vlm_port)
        self.entry_vlm_port.pack(fill=tk.X)

        motion_frame = tk.LabelFrame(scrollable_area, text="モーション検知", font=("Meiryo UI", 9), padx=5, pady=5)
        motion_frame.pack(fill=tk.X, pady=(5, 5))
        self.var_motion_enabled = tk.BooleanVar(value=self.current_settings.motion_detection_enabled)
        self.chk_motion = tk.Checkbutton(motion_frame, text="有効", variable=self.var_motion_enabled, font=("Meiryo UI", 9))
        self.chk_motion.pack(side=tk.LEFT)

        self.entry_motion_threshold = self._create_roi_entry(motion_frame, "閾値 (0.01=1%)", self.current_settings.motion_threshold)
        self.entry_motion_threshold.configure(width=10) # Make it smaller

        chk_frame = tk.Frame(scrollable_area)
        chk_frame.pack(fill=tk.X, pady=(5, 10))
        self.var_debug = tk.BooleanVar(value=self.current_settings.debug)
        self.chk_debug = tk.Checkbutton(chk_frame, text="デバッグモード", variable=self.var_debug, font=("Meiryo UI", 9))
        self.chk_debug.pack(side=tk.LEFT)

        btn_frame = tk.Frame(self.dialog, padx=15, pady=10)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_ok = tk.Button(btn_frame, text="保存", font=("Meiryo UI", 10), command=self._on_save)
        self.btn_ok.pack(side=tk.RIGHT, padx=(5, 0))
        self.btn_cancel = tk.Button(btn_frame, text="キャンセル", font=("Meiryo UI", 10), command=self.dialog.destroy)
        self.btn_cancel.pack(side=tk.RIGHT)

    def _on_mode_change(self, *args):
        if self.var_mode.get() == "vlm":
            self.var_vlm_enabled.set(True)

    def _create_roi_entry(self, parent, label_text, default_value):
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        lbl = tk.Label(row, text=label_text, width=18, anchor=tk.W, font=("Meiryo UI", 9))
        lbl.pack(side=tk.LEFT)
        entry = tk.Entry(row, width=10, font=("Meiryo UI", 9))
        entry.insert(0, str(default_value))
        entry.pack(side=tk.LEFT)
        return entry

    def _get_values(self) -> Optional[AppSettings]:
        try:
            y_start = float(self.entry_y_start.get())
            y_end = float(self.entry_y_end.get())
            x_start = float(self.entry_x_start.get())
            x_end = float(self.entry_x_end.get())
            motion_threshold = float(self.entry_motion_threshold.get())
        except (ValueError, AttributeError):
            return None
        
        errors = self.settings_service.validate_roi(y_start, y_end, x_start, x_end)
        if errors:
            mb.showerror("入力エラー", "\n".join(errors))
            return None
        
        use_vlm = self.var_vlm_enabled.get() or self.var_mode.get() == "vlm"
        return AppSettings(
            roi_y_start=y_start, roi_y_end=y_end,
            roi_x_start=x_start, roi_x_end=x_end,
            img_scale=self.var_img_scale.get() or None,
            debug=self.var_debug.get(),
            use_vlm=use_vlm,
            mode=self.var_mode.get(),
            motion_detection_enabled=self.var_motion_enabled.get(),
            motion_threshold=motion_threshold,
            vlm_port=int(self.entry_vlm_port.get())
        )

    def _on_save(self):
        settings = self._get_values()
        if settings:
            self.settings_service.save(settings)
            self._saved_settings = settings
            self.dialog.destroy()

    def show(self) -> Optional[AppSettings]:
        self.dialog.wait_window()
        return getattr(self, '_saved_settings', None)

class MemberDetailDialog:
    """メンバー名と置換パターンを詳細に設定するToplevelダイアログ

    新規追加モードおよび既存メンバーの編集モードに対応。
    バリデーション（空文字チェック・重複チェック・正規表現検証）を含む。
    """

    def __init__(self, parent: tk.Tk, members: Dict[str, MemberEntry],
                 existing_name: Optional[str] = None):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("メンバー追加" if existing_name is None else "メンバー編集")
        self.dialog.geometry("420x520")
        self.dialog.resizable(True, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # 既存メンバーのデータを保持（重複チェック用）
        self.existing_members = {k: v for k, v in members.items()}
        self.is_edit_mode = existing_name is not None

        main_frame = tk.Frame(self.dialog, padx=12, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- 名前入力 ---
        name_frame = tk.LabelFrame(main_frame, text="メンバー名", font=("Meiryo UI", 9), padx=5, pady=5)
        name_frame.pack(fill=tk.X, pady=(0, 8))

        self.entry_name = tk.Entry(name_frame, font=("Meiryo UI", 10))
        self.entry_name.pack(fill=tk.X)
        if existing_name is not None:
            entry = self.existing_members.get(existing_name)
            if entry:
                self.entry_name.insert(0, entry.name)

        # --- 置換パターン ---
        pattern_frame = tk.LabelFrame(main_frame, text="OCR誤認識パターン", font=("Meiryo UI", 9), padx=5, pady=5)
        pattern_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # Listbox（既存パターン一覧）
        self.lst_patterns = tk.Listbox(pattern_frame, font=("Meiryo UI", 10), height=12)
        self.lst_patterns.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        if existing_name is not None:
            entry = self.existing_members.get(existing_name)
            if entry and entry.replace_patterns:
                for p in entry.replace_patterns:
                    self.lst_patterns.insert(tk.END, p)

        # パターン操作ボタン
        btn_frame = tk.Frame(pattern_frame)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Button(btn_frame, text="追加", width=6, font=("Meiryo UI", 9), command=self._add_pattern).pack(pady=(0, 4))
        self.entry_new_pattern = tk.Entry(btn_frame, width=12, font=("Meiryo UI", 9))
        self.entry_new_pattern.pack(fill=tk.X)
        tk.Button(btn_frame, text="削除", width=6, font=("Meiryo UI", 9), command=self._remove_pattern).pack(pady=(4, 0))

        # --- OK / キャンセル ---
        btn_row = tk.Frame(self.dialog, padx=12, pady=8)
        btn_row.pack(fill=tk.X, side=tk.BOTTOM)

        self.btn_ok = tk.Button(btn_row, text="保存", font=("Meiryo UI", 10), command=self._on_save, state=tk.DISABLED)
        self.btn_ok.pack(side=tk.RIGHT, padx=(5, 0))
        tk.Button(btn_row, text="キャンセル", font=("Meiryo UI", 10), command=self.dialog.destroy).pack(side=tk.RIGHT)

        # エラー表示ラベル
        self.lbl_error = tk.Label(main_frame, text="", fg="red", font=("Meiryo UI", 8), anchor=tk.W)
        self.lbl_error.pack(fill=tk.X, pady=(0, 4))

        # Entry変更時のバリデーション
        self.entry_name.bind("<KeyRelease>", lambda _: self._validate_form())

    def _add_pattern(self):
        val = self.entry_new_pattern.get().strip()
        if not val:
            return

        # 正規表現の妥当性チェック
        try:
            re.compile(val)
        except re.error as e:
            mb.showerror("エラー", f"不正な正規表現です:\n{val}\n\n{e}")
            return

        self.lst_patterns.insert(tk.END, val)
        self.entry_new_pattern.delete(0, tk.END)
        self._validate_form()

    def _remove_pattern(self):
        sel = self.lst_patterns.curselection()
        if sel:
            self.lst_patterns.delete(sel[0])

    def _get_patterns(self) -> List[str]:
        return list(self.lst_patterns.get(0, tk.END))

    def _validate_form(self) -> Optional[str]:
        """バリデーション実行。OK不可の理由を返す（OK可ならNone）"""
        self.lbl_error.config(text="")

        name = self.entry_name.get().strip()
        if not name:
            msg = "メンバー名は必須です"
            self.lbl_error.config(text=msg)
            self.btn_ok.config(state=tk.DISABLED)
            return msg

        # 重複チェック（編集モードでは自身のキーは除外）
        exclude_key = None if not self.is_edit_mode else name
        for k in self.existing_members:
            if k != exclude_key and k == name:
                msg = f"「{name}」は既に存在します"
                self.lbl_error.config(text=msg)
                self.btn_ok.config(state=tk.DISABLED)
                return msg

        # パターンの正規表現検証（既存含む）
        for p in self._get_patterns():
            try:
                re.compile(p)
            except re.error as e:
                msg = f"パターン「{p}」に不正な正規表現があります:\n{e}"
                self.lbl_error.config(text=msg)
                self.btn_ok.config(state=tk.DISABLED)
                return msg

        self.btn_ok.config(state=tk.NORMAL)
        return None

    def _on_save(self):
        error = self._validate_form()
        if error:
            mb.showerror("バリデーションエラー", error)
            return

        name = self.entry_name.get().strip()
        patterns = self._get_patterns()

        self.result_name = name
        self.result_patterns = patterns
        self.dialog.destroy()

    def show(self) -> Tuple[Optional[str], List[str]]:
        self.dialog.wait_window()
        if not hasattr(self, "result_name"):
            return None, []
        return self.result_name, self.result_patterns


class MemberEditorDialog:
    """メンバー一覧を管理するToplevelダイアログ

    追加・編集は `MemberDetailDialog` に委譲し、本クラスは一覧表示と削除を担当。
    """

    def __init__(self, parent: tk.Tk, members: Dict[str, MemberEntry]):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("メンバー管理")
        self.dialog.geometry("650x420")
        self.dialog.transient(parent)
        self.dialog.grab_set()

        # 変更後のメンバーデータを保持
        self.members: Dict[str, MemberEntry] = {k: v for k, v in members.items()}
        self.ok_pressed = False

        main_frame = tk.Frame(self.dialog, padx=10, pady=5)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Treeview（メンバー一覧）
        self.tv_members = ttk.Treeview(main_frame, columns=("名前", "パターン"), show="headings")
        self.tv_members.heading("名前", text="メンバー名")
        self.tv_members.heading("パターン", text="パターン数")
        self.tv_members.pack(fill=tk.BOTH, expand=True)

        # 操作ボタン
        btn_frame = tk.Frame(self.dialog, padx=10, pady=8)
        btn_frame.pack(fill=tk.X, side=tk.BOTTOM)

        tk.Button(btn_frame, text="➕ 新規追加", font=("Meiryo UI", 10), command=self._add_member).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(btn_frame, text="✏ 編集", font=("Meiryo UI", 10), command=self._edit_member).pack(side=tk.LEFT, padx=(0, 5))
        tk.Button(btn_frame, text="🗑 削除", font=("Meiryo UI", 10), command=self._remove_member).pack(side=tk.LEFT, padx=(0, 5))

        self.btn_ok = tk.Button(btn_frame, text="保存して閉じる", font=("Meiryo UI", 10), command=self._on_close_ok)
        self.btn_ok.pack(side=tk.RIGHT, padx=(5, 0))
        tk.Button(btn_frame, text="キャンセル", font=("Meiryo UI", 10), command=self.dialog.destroy).pack(side=tk.RIGHT)

        self._populate_treeview()

    def _populate_treeview(self):
        for item in self.tv_members.get_children():
            self.tv_members.delete(item)
        for entry in sorted(self.members.values(), key=lambda e: e.name):
            self.tv_members.insert("", tk.END, values=(entry.name, len(entry.replace_patterns)))

    def _get_selected_name(self) -> Optional[str]:
        """Treeviewで選択されているメンバー名を返す"""
        sel = self.tv_members.selection()
        if not sel:
            return None
        return self.tv_members.item(sel[0])["values"][0]

    # ----- MemberDetailDialog への委譲 -----

    def _add_member(self):
        """新規メンバー追加（MemberDetailDialog を呼び出す）"""
        result_name, patterns = MemberDetailDialog(self.dialog, self.members).show()
        if result_name:
            # 重複チェック（詳細ダイアログで既に行っているが防御的）
            if result_name in self.members:
                mb.showwarning("警告", f"「{result_name}」は既に存在します")
                return
            self.members[result_name] = MemberEntry(name=result_name, replace_patterns=patterns)
            self._populate_treeview()

    def _edit_member(self):
        """選択中のメンバーを編集（MemberDetailDialog を呼び出す）"""
        old_name = self._get_selected_name()
        if not old_name:
            mb.showinfo("情報", "まず一覧からメンバーを選択してください")
            return

        entry = self.members.get(old_name)
        if entry is None:
            return

        # 既存データで詳細ダイアログを開く
        result_name, patterns = MemberDetailDialog(self.dialog, self.members, existing_name=old_name).show()
        if result_name:
            # 名前の更新（変更がある場合）
            del self.members[old_name]
            self.members[result_name] = MemberEntry(name=result_name, replace_patterns=patterns)
            self._populate_treeview()

    def _remove_member(self):
        """選択中のメンバーを削除"""
        name = self._get_selected_name()
        if not name:
            mb.showinfo("情報", "まず一覧からメンバーを選択してください")
            return

        if mb.askyesno("確認", f"「{name}」を削除しますか？"):
            del self.members[name]
            self._populate_treeview()

    def _on_close_ok(self):
        """保存して閉じる"""
        self.ok_pressed = True
        self.dialog.destroy()

    def show(self) -> Optional[Dict[str, MemberEntry]]:
        self.dialog.wait_window()
        return dict(self.members) if self.ok_pressed else None
