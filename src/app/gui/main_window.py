import tkinter as tk
from tkinter import filedialog, ttk
import tkinter.messagebox as mb
import queue
import threading
import json
from pathlib import Path
import time

from src.domain.models import PipelineConfig, OCRResult, VLMConfig
from src.core.pipeline import PipelineRunner, PipelineWorker
from src.services.settings_service import SettingsService
from src.services.member_service import MemberService
from src.app.gui.dialogs import SettingsDialog, MemberEditorDialog

class AppWindow:
    def __init__(self, debug: bool = False, img_scale: str = None, mode: str = "ocr"):
        self.debug = debug
        self.img_scale = img_scale
        self.mode = mode
        self.is_processing = False

        self.settings_service = SettingsService()
        self.member_service = MemberService()
        
        self.root = tk.Tk()
        self.root.title("Umamusume Fan Count Extractor")
        self.root.geometry("700x550")
        
        self.result_queue = queue.Queue()
        self._setup_ui()

    def _setup_ui(self):
        # Top Area
        top_frame = tk.Frame(self.root, padx=10, pady=10)
        top_frame.pack(fill=tk.X)

        self.btn_select = tk.Button(top_frame, text="📂 動画ファイルを選択", command=self._on_select_video)
        self.btn_select.pack(side=tk.LEFT)

        self.lbl_path = tk.Label(top_frame, text="動画が選択されていません", fg="#666")
        self.lbl_path.pack(side=tk.LEFT, padx=10)

        # Settings/Members Buttons
        btn_frame = tk.Frame(top_frame)
        btn_frame.pack(side=tk.RIGHT)
        tk.Button(btn_frame, text="⚙ 設定", command=self._on_open_settings).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="メンバー管理", command=self._on_open_member_editor).pack(side=tk.LEFT)

        # Status Area
        status_frame = tk.LabelFrame(self.root, text="処理進捗", padx=5, pady=5)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.progress_bar = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress_bar.pack(fill=tk.X)
        
        self.lbl_eta = tk.Label(status_frame, text="")
        self.lbl_eta.pack(anchor=tk.W)
        
        self.txt_status = tk.Text(status_frame, height=8, state=tk.DISABLED)
        self.txt_status.pack(fill=tk.BOTH, expand=True)

        # Result Area
        result_frame = tk.Frame(self.root, padx=10, pady=5)
        result_frame.pack(fill=tk.X)
        
        self.btn_copy = tk.Button(result_frame, text="📋 コピー", command=self._on_copy_results)
        self.btn_copy.pack(side=tk.LEFT)
        
        self.tv_results = ttk.Treeview(self.root, columns=("メンバー名", "ファン数"), show="headings", height=5)
        self.tv_results.heading("メンバー名", text="メンバー名")
        self.tv_results.heading("ファン数", text="ファン数")
        self.tv_results.pack(fill=tk.X, padx=10, pady=5)

    def _add_status(self, msg: str):
        self.txt_status.configure(state=tk.NORMAL)
        self.txt_status.insert(tk.END, f"{msg}\n")
        self.txt_status.see(tk.END)
        self.txt_status.configure(state=tk.DISABLED)

    def _on_select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("MP4", "*.mp4")])
        if not file_path: return
        
        self.lbl_path.config(text=Path(file_path).name)
        
        # 先に設定を読み込む
        settings = self.settings_service.load()
        
        config = PipelineConfig(
            video_path=Path(file_path),
            debug=self.debug,
            img_scale=self.img_scale,
            mode=self.mode,
            motion_detection_enabled=settings.motion_detection_enabled,
            motion_threshold=settings.motion_threshold
        )
        
        config.vlm_config = VLMConfig(
            enabled=settings.use_vlm or settings.mode == "vlm",
            model_path="models/gemma-4-e2b-it-edited-q4_0.gguf",
            mmproj_path="models/mmproj-gemma-4-e2b-it-q4_0.gguf",
            debug=config.debug,
            max_workers=settings.vlm_max_workers,
        )
        config.roi_y_start = settings.roi_y_start
        config.roi_y_end = settings.roi_y_end
        config.roi_x_start = settings.roi_x_start
        config.roi_x_end = settings.roi_x_end
        config.img_scale = settings.img_scale
        config.debug = self.debug or settings.debug
        
        worker = PipelineWorker(config, self.result_queue)
        self.is_processing = True
        self.btn_select.config(state=tk.DISABLED)
        worker.start()
        self.root.after(100, self._check_queue)

    def _check_queue(self):
        try:
            while True:
                msg_type, data = self.result_queue.get_nowait()
                if msg_type == "progress_with_pct":
                    self.progress_bar["value"] = data["percent"] * 100
                    self._add_status(data["message"])
                elif msg_type == "result":
                    self._update_treeview(data.fan_counts)
                elif msg_type == "error":
                    mb.showerror("Error", data["message"])
                    self.is_processing = False
                    self.btn_select.config(state=tk.NORMAL)
                elif msg_type == "done":
                    self.is_processing = False
                    self.btn_select.config(state=tk.NORMAL)
                if self.result_queue.empty(): break
        except queue.Empty:
            pass
        
        if self.is_processing:
            self.root.after(100, self._check_queue)

    def _update_treeview(self, fan_counts):
        for item in self.tv_results.get_children():
            self.tv_results.delete(item)
        for member, count in fan_counts.items():
            self.tv_results.insert("", tk.END, values=(member, f"{count:,}"))

    def _on_copy_results(self):
        lines = [f"{i[0]}\t{i[1]}" for i in self.tv_results.get_children()]
        self.root.clipboard_clear()
        self.root.clipboard_append("\n".join(lines))

    def _on_open_settings(self):
        dlg = SettingsDialog(self.root, self.settings_service)
        dlg.show()

    def _on_open_member_editor(self):
        members = self.member_service.load(Path('input/memberList.txt'), Path('input/memberReplace.json'))
        dlg = MemberEditorDialog(self.root, members)
        result = dlg.show()
        if result:
            self.member_service.save(result, Path('input/memberList.txt'), Path('input/memberReplace.json'))

    def run(self):
        self.root.mainloop()
