import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

# 注意: 再生にはバックグラウンドスレッドを使わない。
# cv2.VideoCapture も Tk ウィジェットもメインスレッドのみがアクセスする
# （VideoCapture はスレッド非安全、Tkinter/Tcl もメインスレッド専用であり、
# 別スレッドからの並行アクセスはネイティブクラッシュを引き起こす）。


class VideoPreviewFrame(ctk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.current_frame_idx = 0
        self.on_process = None

        self._play_job_id = None  # 再生 tick の after() ジョブID

        self._setup_ui()

    def _setup_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)

        self.video_label = ctk.CTkLabel(self, text="動画プレビュー", anchor="center")
        self.video_label.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.progress = ctk.CTkProgressBar(self)
        self.progress.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        self.progress.set(0)

        self.btn_process = ctk.CTkButton(
            self, text="処理開始", command=self._on_process, state="disabled"
        )
        self.btn_process.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

        self.btn_play = ctk.CTkButton(self, text="再生", command=self._toggle_play)
        self.btn_play.grid(row=3, column=0, padx=5, pady=5, sticky="ew")

        self.btn_stop = ctk.CTkButton(self, text="停止", command=self._stop)
        self.btn_stop.grid(row=3, column=1, padx=5, pady=5, sticky="ew")

    def load_video(self, video_path: str):
        # 再生中のロード時はまず再生をキャンセル（tick の残存を防ぐ）
        self._cancel_play_tick()
        self.is_playing = False
        self.btn_play.configure(text="再生")

        self.video_path = video_path

        if self.cap:
            self.cap.release()

        self.cap = cv2.VideoCapture(video_path)
        if self.cap is None or not self.cap.isOpened():
            # 無効なファイル / 読込不能: 処理ボタンを無効化してエラーを表示
            if self.cap is not None:
                self.cap.release()
            self.cap = None
            self.video_path = None
            self.fps = 0.0
            self.frame_count = 0
            self.native_width = 0
            self.native_height = 0
            self.current_frame_idx = 0
            self.progress.set(0)
            self.btn_process.configure(state="disabled")
            self.video_label.configure(image="", text="動画を読み込めませんでした")
            return
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.native_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.native_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame_idx = 0

        self._compute_display_size()
        self._show_frame(0)
        self.btn_process.configure(state="normal")

    def _compute_display_size(self):
        base_w = self.winfo_width() if self.winfo_width() > 0 else 640
        base_h = self.winfo_height() if self.winfo_height() > 0 else 360
        if self.native_width <= 0 or self.native_height <= 0:
            self.disp_width = base_w
            self.disp_height = base_h
            return
        ratio = min(base_w / self.native_width, base_h / self.native_height)
        self.disp_width = max(1, int(self.native_width * ratio))
        self.disp_height = max(1, int(self.native_height * ratio))

    def _show_frame(self, frame_idx: int):
        if not self.cap:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        if not ret:
            return

        self._draw_frame(frame)

        self.progress.set(frame_idx / self.frame_count if self.frame_count > 0 else 0)

    def _draw_frame(self, frame):
        """フレームをプレビューに表示（シークなし・描画のみ）"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.disp_width, self.disp_height))
        image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image)

        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo  # GC されないようウィジェット側に保持

    # ---------- 再生制御（すべてメインスレッド） ----------

    def _playback_fps(self) -> float:
        if self.fps and self.fps > 0:
            return self.fps
        return 30.0  # fps が読めない場合のフォールバック

    def _toggle_play(self):
        if self.is_playing:
            self._stop()
        else:
            self._play()

    def _play(self):
        if not self.cap or not self.video_path or self.is_playing:
            return

        self.is_playing = True
        self.btn_play.configure(text="一時停止")
        self._play_tick()

    def _play_tick(self):
        # cv2 アクセスも Tk 操作もメインスレッドのみ（非スレッド安全のため）
        if not self.is_playing or self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            # 動画末尾到達 → メインスレッド上で自動停止（並行アクセスなし）
            self._stop()
            return
        self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self._draw_frame(frame)
        self.progress.set(
            self.current_frame_idx / self.frame_count if self.frame_count > 0 else 0
        )
        delay_ms = max(1, int(1000 / self._playback_fps()))
        self._play_job_id = self.after(delay_ms, self._play_tick)

    def _cancel_play_tick(self):
        if self._play_job_id is not None:
            self.after_cancel(self._play_job_id)
            self._play_job_id = None

    def _stop(self):
        self.is_playing = False
        self._cancel_play_tick()
        self.btn_play.configure(text="再生")
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_idx = 0
            self._show_frame(0)

    def _on_process(self):
        if self.on_process:
            self.on_process(self.video_path)

    def close(self):
        self.is_playing = False
        self._cancel_play_tick()
        self.btn_play.configure(text="再生")
        if self.cap:
            self.cap.release()
            self.cap = None
