# -*- coding: utf-8 -*-
"""
meeter_final_control_only_FULL.py
Windows + OpenCV + PySide6 + (optional) pyvirtualcam

你要求的重點（全部保留 / 不追蹤 / 小UI無內嵌螢幕）：
1) 小控制面板置頂（不嵌入預覽畫面）
2) 可選獨立預覽視窗（可拖曳疊字位置）
3) 重設背景：按鈕→倒數10秒→時間到自動拍背景（多幀平均）
4) 擷取遮罩：背景差分擷取「我 + 椅子」的 alpha（固定遮罩）
5) 消失/回復：粒子溶解特效，只作用在「前景層(我+椅子)」，背景不變 → 不會突然變暗
6) 文字疊加：支援換行、可開關、可調大小/粗細、可用按鈕/快捷鍵移動位置
7) 文字左右 flip（讓畫面鏡像時文字仍符合你的期望）
8) 背景/遮罩存檔與載入（下次不用重拍）
9) 輸出到 pyvirtualcam（Meet 選虛擬攝影機）
10) 椅子/人移動過大（遮罩特徵偏移）→ 自動重新擷取遮罩（連續多幀才觸發，避免手/頭小動就觸發）

注意：
- 此版本仍採「背景差分」路線，因此「背景」必須在環境相對穩定時拍。
- 但你提到椅子會轉，因此加入「自動重擷取遮罩」做自我修復。
"""

import sys
import time
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# pyvirtualcam 可能沒裝或失敗；程式會自動降級（仍可跑 UI）
try:
    import pyvirtualcam
except Exception:
    pyvirtualcam = None

from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer, QPoint
from PySide6.QtGui import QImage, QPixmap, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QPlainTextEdit, QGroupBox, QMessageBox, QCheckBox, QSizePolicy,
    QSpinBox, QFormLayout, QDialog, QComboBox
)
# 在檔案最上面（import 後）
EYE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# =========================
# 參數（你可以調）
# =========================
CAMERA_INDEX = 0              # 你的攝影機若 index=3，改 3
CAP_WIDTH = 1280
CAP_HEIGHT = 720
CAP_FPS = 30                  # 60 也可，但 pyvirtualcam/相機驅動不一定穩

# 遮罩生成參數（背景差分）
DIFF_THRESHOLD = 25           # 越小越敏感（容易噪點），越大越鈍（容易漏）
MORPH_KERNEL = 7              # 奇數，越大越平滑但可能吃掉細節
FEATHER_BLUR = 21             # 奇數，羽化邊緣

# 溶解特效
DISSOLVE_SPEED = 1.2          # 每秒 progress 變化
NOISE_SMOOTH_SIGMA = 2.0      # 噪聲模糊，越大越成團
NOISE_CONTRAST = 1.6          # 噪聲對比，越大越碎裂明顯

# 自動重擷取遮罩：判定門檻（重要）
AUTO_POS_THRESH = 0.05        # 重心位移（比例）> 0.08 視為大移動
AUTO_AREA_THRESH = 0.2       # 面積變化比例 > 0.35 視為椅子/姿態大改變
AUTO_CC_THRESH = 0.70         # 最大連通區比例 < 0.70 視為遮罩破碎
AUTO_NEED_FRAMES = 10         # 連續幀數才觸發（避免誤判）

# 文字 flip（你要求）
TEXT_FLIP_HORIZONTAL = True   # True: 文字左右翻（符合鏡像需求）

# 資料保存
SAVE_DIR = Path(__file__).resolve().parent / "meeter_data"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
BG_PATH = SAVE_DIR / "bg.png"
MASK_PATH = SAVE_DIR / "mask_alpha.png"      # 灰階 0..255 表示 alpha
CFG_PATH = SAVE_DIR / "config.json"


# =========================
# 工具：np/cv2 -> QImage
# =========================
def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


def ensure_odd(n: int) -> int:
    return n if (n % 2 == 1) else n + 1


# =========================
# 共享狀態（只用這一個，避免 state 不一致）
# =========================
@dataclass
class SharedState:
    mask_ref_centroid: tuple[float, float] | None = None  # normalized cx, cy
    mask_ref_area: float | None = None                    # area ratio
    mask_bad_counter: int = 0
    eye_black_bar_enabled: bool = False     # 是否顯示黑色遮眼框
    pupil_color_enabled: bool = False       # 是否啟用瞳孔換色
    pupil_color_bgr: tuple[int, int, int] = (0, 0, 255)  # 預設紅色
    lock: threading.Lock = field(default_factory=threading.Lock)

    # 背景 / 遮罩
    bg_image: np.ndarray | None = None
    mask_alpha: np.ndarray | None = None  # float32 0..1

    request_bg_capture: bool = False
    request_mask_capture: bool = False

    # UI 倒數
    bg_countdown_end_ts: float | None = None

    # 溶解（0=顯示, 1=消失）
    visible_target: bool = True
    dissolve_progress: float = 0.0

    # 文字疊加
    overlay_text: str = ""
    overlay_enabled: bool = True
    overlay_pos: tuple[int, int] = (20, 50)
    overlay_scale: float = 0.9
    overlay_thickness: int = 2

    # 無預覽時：文字移動步進
    overlay_step: int = 10

    # pyvirtualcam
    vcam_enabled: bool = True

    # 是否需要預覽（獨立視窗）
    preview_enabled: bool = False

    # 自動重擷取遮罩：參考特徵
    camera_index: int = 0
    # 狀態
    status: str = "就緒"


# =========================
# 可拖曳 Label（只在預覽視窗用）
# =========================
class DraggableVideoLabel(QLabel):
    dragged = Signal(int, int)  # dx, dy

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._dragging = False
        self._last_pos = QPoint(0, 0)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = True
            self._last_pos = event.position().toPoint()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._dragging:
            p = event.position().toPoint()
            dx = p.x() - self._last_pos.x()
            dy = p.y() - self._last_pos.y()
            self._last_pos = p
            self.dragged.emit(dx, dy)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._dragging = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)


# =========================
# 獨立預覽視窗（可選）
# =========================
class PreviewWindow(QWidget):
    def __init__(self, state: SharedState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("Meeter 預覽（可拖曳疊字）")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self.video_label = DraggableVideoLabel("等待影像…")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumSize(960, 540)
        self.video_label.dragged.connect(self.on_drag_overlay)

        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

    @Slot(int, int)
    def on_drag_overlay(self, dx: int, dy: int):
        with self.state.lock:
            x, y = self.state.overlay_pos
            x = int(np.clip(x + dx, 0, CAP_WIDTH - 10))
            y = int(np.clip(y + dy, 0, CAP_HEIGHT - 10))
            self.state.overlay_pos = (x, y)

    def set_frame(self, img: QImage):
        pix = QPixmap.fromImage(img)
        self.video_label.setPixmap(pix.scaled(
            self.video_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        ))


# =========================
# Worker：唯一攝影機讀取者（非常重要：UI 不再搶相機）
# =========================
class VideoWorker(QThread):
    frame_ready = Signal(QImage)   # 預覽用（可選）
    status_ready = Signal(str)
    virtualcam_failed = Signal()
    user_error = Signal(str)

    def __init__(self, state: SharedState, parent=None):
        super().__init__(parent)
        self.state = state
        self._running = True
        self._eye_frame_counter = 0
        self._cached_eyes = []


    def stop(self):
        self._running = False

    def _set_status(self, msg: str):
        with self.state.lock:
            self.state.status = msg
        self.status_ready.emit(msg)
    def _detect_eyes_safe(self, frame):
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = EYE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=6,
                minSize=(30, 30)
            )
            if eyes is None:
                return []
            return list(eyes)
        except Exception:
            return []


    # @staticmethod
    @staticmethod
    def draw_black_eye_bar(frame: np.ndarray, eyes):
        if eyes is None or len(eyes) == 0:
            return

        xs, ys, xe, ye = [], [], [], []
        for (x, y, w, h) in eyes:
            xs.append(x)
            ys.append(y)
            xe.append(x + w)
            ye.append(y + h)

        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xe), max(ye)

        pad_y = int((y2 - y1) * 0.6)
        pad_x = int((x2 - x1) * 0.2)

        y1 = max(0, y1 - pad_y)
        y2 = min(frame.shape[0], y2 + pad_y)
        x1 = max(0, x1 - pad_x)
        x2 = min(frame.shape[1], x2 + pad_x)

        frame[y1:y2, x1:x2] = (0, 0, 0)

    # 多幀平均拍背景（避免雜訊）
    def _capture_background_average(self, cap: cv2.VideoCapture, num_frames: int = 30) -> np.ndarray | None:
        acc = None
        got = 0
        for _ in range(num_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (CAP_WIDTH, CAP_HEIGHT), interpolation=cv2.INTER_AREA)
            f32 = frame.astype(np.float32)
            acc = f32 if acc is None else (acc + f32)
            got += 1
            time.sleep(0.005)
        if got == 0:
            return None
        return (acc / got).astype(np.uint8)

    # 遮罩生成（背景差分 + 形態學 + 羽化 + 填洞 + 最大連通區保留）
    def _build_mask_from_bg_diff(self, bg: np.ndarray, fg: np.ndarray) -> np.ndarray:
        diff = cv2.absdiff(fg, bg)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        _, th = cv2.threshold(gray, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

        k = ensure_odd(MORPH_KERNEL)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)

        # 填洞（避免洞洞）
        h, w = th.shape
        flood = th.copy()
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask, (0, 0), 255)
        holes = cv2.bitwise_not(flood)
        th = th | holes

        # 只保留最大連通區（通常是「我+椅子」）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
        if num_labels > 1:
            largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            th = np.where(labels == largest, 255, 0).astype(np.uint8)

        b = ensure_odd(FEATHER_BLUR)
        th = cv2.GaussianBlur(th, (b, b), 0)

        alpha = th.astype(np.float32) / 255.0
        return np.clip(alpha, 0.0, 1.0)

    # 溶解：只作用在前景 alpha（你+椅子）
    def _apply_dissolve(self, alpha: np.ndarray, progress: float, seed: int) -> np.ndarray:
        h, w = alpha.shape
        rng = np.random.default_rng(seed)
        noise = rng.random((h, w), dtype=np.float32)

        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=NOISE_SMOOTH_SIGMA, sigmaY=NOISE_SMOOTH_SIGMA)
        noise = np.clip((noise - 0.5) * NOISE_CONTRAST + 0.5, 0.0, 1.0)

        keep = (noise > progress).astype(np.float32)
        return alpha * keep

    # 文字：可換行 + 外框黑白字（清楚）
    def _draw_overlay_text(self, frame: np.ndarray, text: str, pos: tuple[int, int], scale: float, thickness: int):
        if not text.strip():
            return

        # 你要求文字左右 flip：做法是「在翻轉畫面上畫字，再翻回來」
        if TEXT_FLIP_HORIZONTAL:
            tmp = cv2.flip(frame, 1)
            self._draw_text_no_flip(tmp, text, pos, scale, thickness, flipped_canvas=True)
            frame[:] = cv2.flip(tmp, 1)
        else:
            self._draw_text_no_flip(frame, text, pos, scale, thickness, flipped_canvas=False)

    def _draw_text_no_flip(self, frame: np.ndarray, text: str, pos: tuple[int, int],
                           scale: float, thickness: int, flipped_canvas: bool):
        x, y = pos
        if flipped_canvas:
            x = int(CAP_WIDTH - x)

        font = cv2.FONT_HERSHEY_SIMPLEX
        line_h = int(36 * (scale / 0.9))
        lines = text.splitlines()

        for i, line in enumerate(lines[:8]):
            yy = y + i * line_h
            cv2.putText(frame, line, (x, yy), font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
            cv2.putText(frame, line, (x, yy), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # 遮罩特徵：重心 / 面積 / 最大連通區比例
    def _mask_features(self, alpha: np.ndarray):
        h, w = alpha.shape
        binary = (alpha > 0.15).astype(np.uint8)
        area = int(binary.sum())
        if area < 100:
            return None

        area_ratio = area / float(h * w)

        ys, xs = np.where(binary)
        cy = float(ys.mean()) / float(h)
        cx = float(xs.mean()) / float(w)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num <= 1:
            largest_ratio = 0.0
        else:
            largest = float(stats[1:, cv2.CC_STAT_AREA].max())
            largest_ratio = largest / float(area + 1e-6)

        return (cx, cy), float(area_ratio), float(largest_ratio)

    # 自動重擷取遮罩（連續多幀才觸發）
    def _auto_recover_mask_if_needed(self, alpha: np.ndarray):
        feat = self._mask_features(alpha)
        if feat is None:
            return

        (cx, cy), area_ratio, largest_cc = feat

        with self.state.lock:
            ref_c = self.state.mask_ref_centroid
            ref_a = self.state.mask_ref_area

            if ref_c is None or ref_a is None:
                self.state.mask_ref_centroid = (cx, cy)
                self.state.mask_ref_area = area_ratio
                self.state.mask_bad_counter = 0
                return

            bad = False

            if abs(cx - ref_c[0]) > AUTO_POS_THRESH or abs(cy - ref_c[1]) > AUTO_POS_THRESH:
                bad = True

            if abs(area_ratio - ref_a) / (ref_a + 1e-6) > AUTO_AREA_THRESH:
                bad = True

            if largest_cc < AUTO_CC_THRESH:
                bad = True

            if bad:
                self.state.mask_bad_counter += 1
            else:
                self.state.mask_bad_counter = 0

            if self.state.mask_bad_counter >= AUTO_NEED_FRAMES:
                self.state.request_mask_capture = True
                self.state.mask_bad_counter = 0
                need_msg = True
            else:
                need_msg = False

        if need_msg:
            self._set_status("偵測到移動過大/椅子大轉動：自動重新擷取遮罩")
    @staticmethod
    def recolor_pupil(frame: np.ndarray, eyes, color_bgr):
        if eyes is None or len(eyes) == 0:
            return

        for (x, y, w, h) in eyes:
            eye_roi = frame[y:y+h, x:x+w]

            hsv = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2HSV)

            mask = cv2.inRange(
                hsv,
                np.array([0, 0, 0]),
                np.array([180, 255, 60])
            )

            mask = cv2.GaussianBlur(mask, (15, 15), 0)

            color_layer = np.full_like(eye_roi, color_bgr)
            alpha = (mask / 255.0)[..., None]

            eye_roi[:] = (
                eye_roi * (1 - alpha) +
                color_layer * alpha
            ).astype(np.uint8)

    def _safe_detect_eyes(self, frame):
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            eyes = EYE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=6,
                minSize=(30, 30)
            )
            return eyes if eyes is not None else []
        except Exception:
            return []

    def run(self):
        cap = cv2.VideoCapture(self.state.camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            self.user_error.emit(
                "無法開啟攝影機。\n\n"
                "請檢查：\n"
                "1️⃣ 是否有插入攝影機\n"
                "2️⃣ 是否被其他程式（OBS / Meet / Zoom）佔用\n"
                "3️⃣ 攝影機是否選錯\n\n"
                "你可以在設定中更換攝影機來源。"
            )
            return


        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, CAP_FPS)

        self._set_status("攝影機已啟動（控制面板模式）")

        vcam = None
        try:
            with self.state.lock:
                vcam_enabled = bool(self.state.vcam_enabled)

            if vcam_enabled and (pyvirtualcam is not None):
                vcam = pyvirtualcam.Camera(
                    width=CAP_WIDTH,
                    height=CAP_HEIGHT,
                    fps=CAP_FPS
                )
                self._set_status("虛擬攝影機已啟動（Meet / Zoom 可選）")

            elif vcam_enabled and (pyvirtualcam is None):
                self._set_status("未偵測到虛擬攝影機（請安裝 OBS）")
                self.virtualcam_failed.emit()   # ★ 通知 UI

        except Exception as e:
            vcam = None
            self._set_status("虛擬攝影機啟動失敗")
            self.virtualcam_failed.emit()       # ★ 通知 UI


        last_ts = time.time()

        while self._running:
            ret, frame = cap.read()
            if not ret:
                self._set_status("讀取影格失敗（攝影機可能被占用）")
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (CAP_WIDTH, CAP_HEIGHT), interpolation=cv2.INTER_AREA)

            with self.state.lock:
                req_bg = self.state.request_bg_capture
                req_mask = self.state.request_mask_capture
                bg = None if self.state.bg_image is None else self.state.bg_image.copy()
                mask_alpha = None if self.state.mask_alpha is None else self.state.mask_alpha.copy()
                visible_target = bool(self.state.visible_target)
                progress = float(self.state.dissolve_progress)
                overlay_text = self.state.overlay_text
                overlay_enabled = bool(self.state.overlay_enabled)
                overlay_pos = tuple(self.state.overlay_pos)
                overlay_scale = float(self.state.overlay_scale)
                overlay_thickness = int(self.state.overlay_thickness)
                vcam_enabled = bool(self.state.vcam_enabled)
                preview_enabled = bool(self.state.preview_enabled)

            # ===== 背景拍攝 =====
            if req_bg:
                self._set_status("正在拍攝背景（多幀平均）…")
                bg_new = self._capture_background_average(cap, num_frames=30)
                with self.state.lock:
                    self.state.request_bg_capture = False
                if bg_new is not None:
                    with self.state.lock:
                        self.state.bg_image = bg_new
                        self.state.mask_ref_centroid = None
                        self.state.mask_ref_area = None
                        self.state.mask_bad_counter = 0
                    bg = bg_new.copy()
                    self._set_status("背景已更新（可存檔）")
                else:
                    self._set_status("背景拍攝失敗：請重試")

            # ===== 遮罩生成 / 重擷取 =====
            if req_mask:
                with self.state.lock:
                    self.state.request_mask_capture = False

                if bg is None:
                    self._set_status("遮罩擷取失敗：請先拍背景")
                else:
                    self._set_status("正在生成遮罩（背景差分：我+椅子）…")
                    alpha = self._build_mask_from_bg_diff(bg, frame)
                    with self.state.lock:
                        self.state.mask_alpha = alpha
                        feat = self._mask_features(alpha)
                        if feat is not None:
                            c, area_r, _ = feat
                            self.state.mask_ref_centroid = c
                            self.state.mask_ref_area = area_r
                            self.state.mask_bad_counter = 0
                    mask_alpha = alpha
                    self._set_status("遮罩已生成（我+椅子）")

            # ===== 自動重擷取監控 =====
            if mask_alpha is not None:
                self._auto_recover_mask_if_needed(mask_alpha)

            # ===== 溶解進度 =====
            now = time.time()
            dt = max(0.001, now - last_ts)
            last_ts = now

            target = 0.0 if visible_target else 1.0
            if abs(progress - target) > 1e-4:
                step = DISSOLVE_SPEED * dt
                if progress < target:
                    progress = min(target, progress + step)
                else:
                    progress = max(target, progress - step)
                with self.state.lock:
                    self.state.dissolve_progress = float(progress)

            # ===== 合成 =====
            out = frame.copy()
            if (bg is not None) and (mask_alpha is not None):
                seed = int((time.time() * 1000) % (2**31))
                alpha2 = self._apply_dissolve(mask_alpha, progress, seed=seed)
                a = alpha2[..., None].astype(np.float32)

                fg = frame.astype(np.float32)
                bgf = bg.astype(np.float32)
                out = (fg * a + bgf * (1.0 - a)).astype(np.uint8)
            # ===== 合成完成後 =====
            # out = (fg * a + bgf * (1.0 - a)).astype(np.uint8)

            # ===== 合成完成 =====

            # ===== 眼睛特效（由 UI 控制）=====
            with self.state.lock:
                eye_bar = self.state.eye_black_bar_enabled
                pupil_on = self.state.pupil_color_enabled
                pupil_color = self.state.pupil_color_bgr

            if eye_bar or pupil_on:
                self._eye_frame_counter += 1

                # 每 5 幀才重新偵測
                if self._eye_frame_counter % 5 == 0:
                    self._cached_eyes = self._detect_eyes_safe(out)

                eyes = self._cached_eyes
            else:
                eyes = []
            if eye_bar or pupil_on:
                self._eye_frame_counter += 1

                if self._eye_frame_counter % 5 == 0 or len(self._cached_eyes) == 0:
                    self._cached_eyes = self._detect_eyes_safe(frame)

                eyes = self._cached_eyes
            else:
                eyes = []


            if eye_bar:
                self.draw_black_eye_bar(out, eyes)
                
            if pupil_on:
                self.recolor_pupil(out, eyes, pupil_color)



            # ===== 疊字 =====
            if overlay_enabled:
                self._draw_overlay_text(out, overlay_text, overlay_pos, overlay_scale, overlay_thickness)

            # ===== 預覽 =====
            if preview_enabled:
                self.frame_ready.emit(bgr_to_qimage(out))

            # ===== 虛擬攝影機輸出 =====
            if (vcam is not None) and vcam_enabled:
                rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
                vcam.send(rgb)
                vcam.sleep_until_next_frame()

            time.sleep(0.001)

        cap.release()
        try:
            if vcam is not None:
                vcam.close()
        except Exception:
            pass
        self._set_status("已停止")

    



# =========================
# 設定對話框（維持小UI）
# =========================
class SettingsDialog(QDialog):
    def __init__(self, state: SharedState, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWindowTitle("設定")
        self.setModal(True)
        self.setFixedWidth(380)

        self.spin_font_scale = QSpinBox()
        self.spin_font_scale.setRange(50, 250)

        self.spin_thickness = QSpinBox()
        self.spin_thickness.setRange(1, 8)

        self.spin_step = QSpinBox()
        self.spin_step.setRange(1, 100)

        self.chk_vcam = QCheckBox("啟用 pyvirtualcam 輸出（Meet 用）")
        self.chk_preview = QCheckBox("開啟獨立預覽視窗（可拖曳疊字）")

        with state.lock:
            self.spin_font_scale.setValue(int(state.overlay_scale * 100))
            self.spin_thickness.setValue(int(state.overlay_thickness))
            self.spin_step.setValue(int(state.overlay_step))
            self.chk_vcam.setChecked(bool(state.vcam_enabled))
            self.chk_preview.setChecked(bool(state.preview_enabled))

        self.spin_font_scale.valueChanged.connect(self.on_font_scale_changed)
        self.spin_thickness.valueChanged.connect(self.on_thickness_changed)
        self.spin_step.valueChanged.connect(self.on_step_changed)
        self.chk_vcam.stateChanged.connect(self.on_toggle_vcam)
        self.chk_preview.stateChanged.connect(self.on_toggle_preview)

        form = QFormLayout()
        form.addRow("字體大小 (%)", self.spin_font_scale)
        form.addRow("筆畫粗細", self.spin_thickness)
        form.addRow("疊字移動步進", self.spin_step)

        box = QGroupBox("疊字設定")
        box.setLayout(form)

        layout = QVBoxLayout()
        layout.addWidget(box)
        layout.addWidget(self.chk_vcam)
        layout.addWidget(self.chk_preview)
        self.setLayout(layout)

    @Slot(int)
    def on_font_scale_changed(self, v: int):
        with self.state.lock:
            self.state.overlay_scale = float(v) / 100.0

    @Slot(int)
    def on_thickness_changed(self, v: int):
        with self.state.lock:
            self.state.overlay_thickness = int(v)

    @Slot(int)
    def on_step_changed(self, v: int):
        with self.state.lock:
            self.state.overlay_step = int(v)

    @Slot()
    def on_toggle_vcam(self):
        with self.state.lock:
            self.state.vcam_enabled = bool(self.chk_vcam.isChecked())

    @Slot()
    def on_toggle_preview(self):
        with self.state.lock:
            self.state.preview_enabled = bool(self.chk_preview.isChecked())

def detect_cameras(self):
    cameras = []
    for i in range(8):  # 一般不會超過 5
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            cameras.append(f"Camera {i}")
            cap.release()
    if not cameras:
        cameras.append("No camera found")
    return cameras

# =========================
# 主 UI（控制面板：無內嵌螢幕）
# =========================
class MainWindow(QWidget):
    def __init__(self):
        
        super().__init__()
        self.setWindowTitle("Meeter 控制面板")
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

        self.state = SharedState()
        self.preview_win: PreviewWindow | None = None

        self._load_config_and_assets()

        # --- UI 元件 ---
        # ===== 教學說明（UI 最上方）=====
        self.help_box = QPlainTextEdit()
        self.help_box.setReadOnly(True)
        self.help_box.setMinimumHeight(220)
        self.help_box.setMaximumHeight(200)

        self.help_box.setStyleSheet("""
            QPlainTextEdit {
                background-color: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 6px;
                padding: 6px;
                font-size: 12px;
            }
        """)

        self.help_box.setPlainText(
            "使用步驟（第一次）：\n\n"
            "1️1 先在下方選擇正確的「攝影機來源」\n"
            "   - 若畫面是黑的，請嘗試切換不同攝影機\n\n"
            "2️ 按「重設背景」\n"
            "   - 2～3 秒內離開畫面，讓鏡頭拍乾淨背景\n\n"
            "3️ 坐回位置，按「擷取遮罩」\n"
            "   - 系統會記住你與椅子的輪廓\n\n"
            "完成後即可開始使用（Meet / Zoom 請選擇虛擬攝影機）"
        )

        # ===== 攝影機選擇 UI =====
        self.combo_camera = QComboBox()
        self.combo_camera.addItems(self.detect_cameras())
        self.combo_camera.currentIndexChanged.connect(self.on_camera_changed)

        camera_box = QGroupBox("攝影機來源")
        camera_layout = QVBoxLayout()
        camera_layout.addWidget(self.combo_camera)
        camera_layout.addWidget(QLabel("⚠ 若沒有畫面，請嘗試切換攝影機"))
        camera_box.setLayout(camera_layout)
        


        self.status_label = QLabel("狀態：就緒")
        self.status_label.setWordWrap(True)

        self.countdown_label = QLabel("")
        f = QFont()
        f.setPointSize(11)
        f.setBold(True)
        self.countdown_label.setFont(f)
    
        self.btn_bg = QPushButton("重設背景2-3秒內離開畫面")
        self.btn_mask = QPushButton("擷取遮罩（人 + 近景）")
        self.btn_vanish = QPushButton("消失 / 回復（粒子溶解）")

        self.btn_settings = QPushButton("settings")

        self.btn_save_bg = QPushButton("Save Background")
        self.btn_load_bg = QPushButton("input Background")
        self.btn_save_mask = QPushButton("save Mask")
        self.btn_load_mask = QPushButton("input Mask")
        # ===== 眼睛特效 UI =====
        self.chk_eye_bar = QCheckBox("顯示黑色馬賽克")
        self.chk_pupil_color = QCheckBox("啟用血輪眼")

        self.chk_eye_bar.stateChanged.connect(self.on_toggle_eye_bar)
        self.chk_pupil_color.stateChanged.connect(self.on_toggle_pupil_color)

        eye_box = QGroupBox("眼睛特效")
        eye_layout = QVBoxLayout()
        eye_layout.addWidget(self.chk_eye_bar)
        eye_layout.addWidget(self.chk_pupil_color)
        eye_box.setLayout(eye_layout)

        self.btn_bg.clicked.connect(self.on_bg_reset)
        self.btn_mask.clicked.connect(self.on_capture_mask)
        self.btn_vanish.clicked.connect(self.on_toggle_vanish)
        self.btn_settings.clicked.connect(self.open_settings)

        self.btn_save_bg.clicked.connect(self.on_save_bg)
        self.btn_load_bg.clicked.connect(self.on_load_bg)
        self.btn_save_mask.clicked.connect(self.on_save_mask)
        self.btn_load_mask.clicked.connect(self.on_load_mask)
        
        # 文字輸入
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText("輸入要貼到畫面上的文字（支援換行）…")
        self.text_edit.setMaximumBlockCount(80)
        self.text_edit.textChanged.connect(self.on_text_changed)

        self.chk_text_enable = QCheckBox("顯示疊加文字")
        with self.state.lock:
            self.chk_text_enable.setChecked(bool(self.state.overlay_enabled))
        self.chk_text_enable.stateChanged.connect(self.on_toggle_text)

        self.btn_clear_text = QPushButton("清除疊字")
        self.btn_clear_text.clicked.connect(self.on_clear_text)
        # SharedState dataclass 內新增
        

        # 無預覽時：移動疊字
        self.btn_left = QPushButton("←")
        self.btn_right = QPushButton("→")
        self.btn_up = QPushButton("↑")
        self.btn_down = QPushButton("↓")
        self.btn_left.clicked.connect(lambda: self.move_overlay(-1, 0))
        self.btn_right.clicked.connect(lambda: self.move_overlay(1, 0))
        self.btn_up.clicked.connect(lambda: self.move_overlay(0, -1))
        self.btn_down.clicked.connect(lambda: self.move_overlay(0, 1))

        move_box = QGroupBox("疊字位置（無預覽時用按鈕/快捷鍵）")
        move_layout = QHBoxLayout()
        move_layout.addWidget(self.btn_left)
        move_layout.addWidget(self.btn_right)
        move_layout.addWidget(self.btn_up)
        move_layout.addWidget(self.btn_down)
        move_box.setLayout(move_layout)
        

        # 快捷鍵：方向鍵 + WASD
        QShortcut(QKeySequence("Left"), self, activated=lambda: self.move_overlay(-1, 0))
        QShortcut(QKeySequence("Right"), self, activated=lambda: self.move_overlay(1, 0))
        QShortcut(QKeySequence("Up"), self, activated=lambda: self.move_overlay(0, -1))
        QShortcut(QKeySequence("Down"), self, activated=lambda: self.move_overlay(0, 1))
        QShortcut(QKeySequence("A"), self, activated=lambda: self.move_overlay(-1, 0))
        QShortcut(QKeySequence("D"), self, activated=lambda: self.move_overlay(1, 0))
        QShortcut(QKeySequence("W"), self, activated=lambda: self.move_overlay(0, -1))
        QShortcut(QKeySequence("S"), self, activated=lambda: self.move_overlay(0, 1))
        QShortcut(QKeySequence("`"), self, activated=lambda: self.on_toggle_vanish())

        # 版面：小 + 集中
        box = QGroupBox("控制")
        r = QVBoxLayout()
        r.addWidget(camera_box)
        r.addWidget(self.btn_bg)
        r.addWidget(self.countdown_label)
        r.addWidget(self.btn_mask)
        r.addWidget(self.btn_vanish)
        r.addWidget(self.btn_settings)
        r.addSpacing(8)

        r.addWidget(eye_box)
        r.addSpacing(8)

        r.addWidget(QLabel("疊加文字(目前僅支援英文)："))
        r.addWidget(self.text_edit)
        r.addWidget(self.chk_text_enable)
        r.addWidget(self.btn_clear_text)
        r.addWidget(move_box)
        r.addSpacing(8)

        r.addWidget(QLabel("保存 / 載入："))
        row1 = QHBoxLayout()
        row1.addWidget(self.btn_save_bg)
        row1.addWidget(self.btn_load_bg)
        r.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(self.btn_save_mask)
        row2.addWidget(self.btn_load_mask)
        r.addLayout(row2)

        r.addSpacing(8)
        r.addWidget(self.status_label)
        box.setLayout(r)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.help_box)   # ← 教學在最上面
        main_layout.addWidget(box)
        self.setLayout(main_layout)

        # 倒數計時器
        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(200)
        self.countdown_timer.timeout.connect(self.on_countdown_tick)

        # Worker
        self.worker = VideoWorker(self.state)
        self.worker.status_ready.connect(self.on_status)
        self.worker.frame_ready.connect(self.on_preview_frame)
        self.worker.start()
        self.worker.virtualcam_failed.connect(self.on_virtualcam_failed)
        self._virtualcam_warning_shown = False
        self.set_status_ui("就緒：按「重設背景」→ 倒數拍背景 → 坐好按「擷取遮罩」")
        self._sync_preview_window()
        self.worker.user_error.connect(self.on_user_error)



        

    # ---------- config/load/save ----------
    def _load_config_and_assets(self):
        if CFG_PATH.exists():
            try:
                cfg = json.loads(CFG_PATH.read_text(encoding="utf-8"))
                with self.state.lock:
                    self.state.overlay_pos = tuple(cfg.get("overlay_pos", (20, 50)))
                    self.state.overlay_scale = float(cfg.get("overlay_scale", 0.9))
                    self.state.overlay_thickness = int(cfg.get("overlay_thickness", 2))
                    self.state.overlay_enabled = bool(cfg.get("overlay_enabled", True))
                    self.state.vcam_enabled = bool(cfg.get("vcam_enabled", True))
                    self.state.overlay_step = int(cfg.get("overlay_step", 10))
                    self.state.preview_enabled = bool(cfg.get("preview_enabled", False))
            except Exception:
                pass

        if BG_PATH.exists():
            bg = cv2.imread(str(BG_PATH))
            if bg is not None:
                bg = cv2.resize(bg, (CAP_WIDTH, CAP_HEIGHT), interpolation=cv2.INTER_AREA)
                with self.state.lock:
                    self.state.bg_image = bg

        if MASK_PATH.exists():
            m = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                m = cv2.resize(m, (CAP_WIDTH, CAP_HEIGHT), interpolation=cv2.INTER_AREA)
                alpha = np.clip(m.astype(np.float32) / 255.0, 0.0, 1.0)
                with self.state.lock:
                    self.state.mask_alpha = alpha

    def _save_config(self):
        with self.state.lock:
            cfg = {
                "overlay_pos": list(self.state.overlay_pos),
                "overlay_scale": self.state.overlay_scale,
                "overlay_thickness": self.state.overlay_thickness,
                "overlay_enabled": self.state.overlay_enabled,
                "vcam_enabled": self.state.vcam_enabled,
                "overlay_step": self.state.overlay_step,
                "preview_enabled": self.state.preview_enabled
            }
        CFG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

    def _sync_preview_window(self):
        with self.state.lock:
            want = bool(self.state.preview_enabled)
        if want and self.preview_win is None:
            self.preview_win = PreviewWindow(self.state, parent=None)
            self.preview_win.resize(1200, 700)
            self.preview_win.show()
            self.set_status_ui("已開啟獨立預覽視窗（可拖曳疊字）")
        elif (not want) and (self.preview_win is not None):
            try:
                self.preview_win.close()
            except Exception:
                pass
            self.preview_win = None
            self.set_status_ui("已關閉獨立預覽視窗")
    @Slot()
    def on_camera_changed(self):
        idx = self.combo_camera.currentIndex()

        cam_index = self._camera_index_map[idx]

        with self.state.lock:
            self.state.camera_index = cam_index

        self.restart_worker()


    @Slot(str)
    def on_user_error(self, msg: str):
        QMessageBox.critical(
            self,
            "攝影機錯誤",
            msg
        )

    # ---------- UI callbacks ----------
    @Slot()
    def on_virtualcam_failed(self):
        # 只顯示一次，避免每次啟動都跳
        if self._virtualcam_warning_shown:
            return

        self._virtualcam_warning_shown = True

        QMessageBox.information(
            self,
            "未偵測到虛擬攝影機",
            (
                "本程式需要「虛擬攝影機」才能輸出到 Google Meet / Zoom。\n\n"
                "請安裝以下其中一種（擇一即可）：\n"
                "1️1 OBS Studio（只需安裝，不必開啟）\n 連結: https://cdn-fastly.obsproject.com/downloads/OBS-Studio-32.0.4-Windows-x64-Installer.exe\n\n"
                "2️2 其他虛擬攝影機驅動（如 Unity Capture）\n\n"
                "安裝完成後，重新啟動本程式即可。"
            )
        )
    def detect_cameras(self):
        """
        回傳實際攝影機名稱 + index
        """
        try:
            from pygrabber.dshow_graph import FilterGraph
        except Exception:
            return ["Camera 0", "Camera 1"]

        graph = FilterGraph()
        devices = graph.get_input_devices()

        cameras = []
        self._camera_index_map = []

        for idx, name in enumerate(devices):
            cameras.append(name)
            self._camera_index_map.append(idx)

        if not cameras:
            cameras.append("No camera found")
            self._camera_index_map = [0]

        return cameras


    @Slot()
    def on_bg_reset(self):
        end_ts = time.time() + 1.0
        with self.state.lock:
            self.state.bg_countdown_end_ts = end_ts
        self.countdown_timer.start()
        self.set_status_ui("背景重設：10 秒後拍背景（請讓畫面只剩背景）")

    @Slot()
    def on_capture_mask(self):
        with self.state.lock:
            has_bg = self.state.bg_image is not None
        if not has_bg:
            QMessageBox.warning(self, "需要背景", "請先按「重設背景」並等倒數拍完背景，才能擷取遮罩。")
            return
        with self.state.lock:
            self.state.request_mask_capture = True
        self.set_status_ui("已要求擷取遮罩：請保持坐姿（我+椅子會被抓成遮罩）")
    @Slot()
    def on_toggle_eye_bar(self):
        with self.state.lock:
            self.state.eye_black_bar_enabled = self.chk_eye_bar.isChecked()
        self.set_status_ui(
            "已啟用黑色遮眼框" if self.chk_eye_bar.isChecked() else "已關閉黑色遮眼框"
        )


    @Slot()
    def on_toggle_pupil_color(self):
        with self.state.lock:
            self.state.pupil_color_enabled = self.chk_pupil_color.isChecked()
        self.set_status_ui(
            "已啟用瞳孔換色" if self.chk_pupil_color.isChecked() else "已關閉瞳孔換色"
        )

    @Slot()
    def on_toggle_vanish(self):
        with self.state.lock:
            self.state.visible_target = not self.state.visible_target
            target = self.state.visible_target
        self.set_status_ui("目標：顯示" if target else "目標：消失（溶解中）")

    @Slot()
    def on_text_changed(self):
        with self.state.lock:
            self.state.overlay_text = self.text_edit.toPlainText()

    @Slot()
    def on_clear_text(self):
        self.text_edit.setPlainText("")
        with self.state.lock:
            self.state.overlay_text = ""

    @Slot()
    def on_toggle_text(self):
        enabled = self.chk_text_enable.isChecked()
        with self.state.lock:
            self.state.overlay_enabled = bool(enabled)
        self._save_config()
        self.set_status_ui("疊字已開啟" if enabled else "疊字已關閉")

    def move_overlay(self, dx_sign: int, dy_sign: int):
        with self.state.lock:
            step = int(self.state.overlay_step)
            x, y = self.state.overlay_pos
            x = int(np.clip(x + dx_sign * step, 0, CAP_WIDTH - 10))
            y = int(np.clip(y + dy_sign * step, 0, CAP_HEIGHT - 10))
            self.state.overlay_pos = (x, y)
        self._save_config()
        self.set_status_ui(f"疊字位置：{x}, {y}（步進 {step}）")

    @Slot()
    def on_countdown_tick(self):
        with self.state.lock:
            end_ts = self.state.bg_countdown_end_ts

        if end_ts is None:
            self.countdown_label.setText("")
            self.countdown_timer.stop()
            return

        remain = end_ts - time.time()
        if remain > 0:
            self.countdown_label.setText(f"倒數：{remain:0.1f} 秒")
            return

        self.countdown_label.setText("倒數：0.0 秒（拍攝中）")
        with self.state.lock:
            self.state.bg_countdown_end_ts = None
            self.state.request_bg_capture = True
        self.countdown_timer.stop()
        self.set_status_ui("倒數結束：正在拍背景（多幀平均）…（拍完後可按擷取遮罩）")

    @Slot(str)
    def on_status(self, msg: str):
        self.status_label.setText(f"狀態：{msg}")

    @Slot(QImage)
    def on_preview_frame(self, img: QImage):
        if self.preview_win is not None:
            self.preview_win.set_frame(img)

    def set_status_ui(self, msg: str):
        self.status_label.setText(f"狀態：{msg}")
        with self.state.lock:
            self.state.status = msg

    # ---------- 保存 / 載入 ----------
    @Slot()
    def on_save_bg(self):
        with self.state.lock:
            bg = None if self.state.bg_image is None else self.state.bg_image.copy()
        if bg is None:
            QMessageBox.information(self, "沒有背景", "目前沒有背景可存。請先拍背景。")
            return
        ok = cv2.imwrite(str(BG_PATH), bg)
        if ok:
            self.set_status_ui(f"背景已存檔：{BG_PATH}")
        else:
            QMessageBox.warning(self, "存檔失敗", "背景存檔失敗，請確認資料夾權限。")

    @Slot()
    def on_load_bg(self):
        if not BG_PATH.exists():
            QMessageBox.information(self, "找不到檔案", f"找不到背景檔：{BG_PATH}")
            return
        bg = cv2.imread(str(BG_PATH))
        if bg is None:
            QMessageBox.warning(self, "載入失敗", "背景檔讀取失敗（檔案可能損壞）。")
            return
        bg = cv2.resize(bg, (CAP_WIDTH, CAP_HEIGHT), interpolation=cv2.INTER_AREA)
        with self.state.lock:
            self.state.bg_image = bg
            # 背景載入後，清掉遮罩參考（避免亂判）
            self.state.mask_ref_centroid = None
            self.state.mask_ref_area = None
            self.state.mask_bad_counter = 0
        self.set_status_ui("背景已載入（可按擷取遮罩）")

    @Slot()
    def on_save_mask(self):
        with self.state.lock:
            alpha = None if self.state.mask_alpha is None else self.state.mask_alpha.copy()
        if alpha is None:
            QMessageBox.information(self, "沒有遮罩", "目前沒有遮罩可存。請先擷取遮罩。")
            return
        m = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
        ok = cv2.imwrite(str(MASK_PATH), m)
        if ok:
            self.set_status_ui(f"遮罩已存檔：{MASK_PATH}")
        else:
            QMessageBox.warning(self, "存檔失敗", "遮罩存檔失敗，請確認資料夾權限。")

    @Slot()
    def on_load_mask(self):
        if not MASK_PATH.exists():
            QMessageBox.information(self, "找不到檔案", f"找不到遮罩檔：{MASK_PATH}")
            return
        m = cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)
        if m is None:
            QMessageBox.warning(self, "載入失敗", "遮罩檔讀取失敗（檔案可能損壞）。")
            return
        m = cv2.resize(m, (CAP_WIDTH, CAP_HEIGHT), interpolation=cv2.INTER_AREA)
        alpha = np.clip(m.astype(np.float32) / 255.0, 0.0, 1.0)
        with self.state.lock:
            self.state.mask_alpha = alpha
            # 載入遮罩後，重新建立參考特徵（避免一開始就觸發自動重擷取）
            self.state.mask_ref_centroid = None
            self.state.mask_ref_area = None
            self.state.mask_bad_counter = 0
        self.set_status_ui("遮罩已載入（會自動重新建立參考特徵）")

    # ---------- 設定 ----------
    @Slot()
    def open_settings(self):
        dlg = SettingsDialog(self.state, self)
        dlg.exec()
        self._save_config()
        self._sync_preview_window()
        self.set_status_ui("設定已更新")

    # ---------- 關閉事件：確實停掉 worker、釋放相機 ----------
    def closeEvent(self, event):
        try:
            self._save_config()
        except Exception:
            pass

        try:
            if self.preview_win is not None:
                self.preview_win.close()
        except Exception:
            pass

        try:
            if hasattr(self, "worker") and self.worker is not None:
                self.worker.stop()
                self.worker.wait(1500)
        except Exception:
            pass

        event.accept()

def restart_worker(self):
    if hasattr(self, "worker") and self.worker is not None:
        self.worker.stop()
        self.worker.wait(1000)

    self.worker = VideoWorker(self.state)
    self.worker.status_ready.connect(self.on_status)
    self.worker.frame_ready.connect(self.on_preview_frame)
    self.worker.user_error.connect(self.on_user_error)
    self.worker.virtualcam_failed.connect(self.on_virtualcam_failed)
    self.worker.start()

    self.set_status_ui("已切換攝影機")

def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
