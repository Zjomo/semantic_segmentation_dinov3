"""
ToupView 窗口语义分割 + 骨架识别集成  v3.0
==========================================
新增干净主路径显示（C 键切换）：
  - 灰色细线：完整骨架网络（所有分支）
  - 红色粗线：最长主路径（含环/分支，原样保留）
  - 黄色粗线：干净主路径（无环、无分支）[C 键切换]
  - 青色粗线：切割路径（从左上角开始，带方向箭头）[V 键切换]

新增切割模拟器（Snake 模式扫描完成后）：
  - 自动重置视角到初始点（左上角）
  - 沿切割路径控制 ToupView 视角移动
  - 模拟实际切割过程
  - 支持暂停/继续/重新开始/退出控制
"""

import os
import sys

script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import ctypes
import ctypes.wintypes as _wt
import time
import threading
import numpy as np
import cv2
import win32gui
import win32api
import win32con
import pyautogui
from pynput import keyboard
from typing import Tuple, Optional, List

from src.snake_scan import (
    ScanConfig, HardwareInterface, SegmentationInference,
    SnakeScanController, SEG_PALETTE, SegClass,
    PathSkeletonExtractor,
)
from src.skeleton_coordinates import (
    extract_skeleton_coordinates,
    save_skeleton_coordinates,
    visualize_skeleton_coordinates,
)

# ============================================================
# 模块级 ctypes 结构体定义
# ============================================================
class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", _wt.LONG), ("dy", _wt.LONG),
        ("mouseData", _wt.DWORD), ("dwFlags", _wt.DWORD),
        ("time", _wt.DWORD), ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", _wt.WORD), ("wScan", _wt.WORD),
        ("dwFlags", _wt.DWORD), ("time", _wt.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT)]

class _INPUT(ctypes.Structure):
    _fields_ = [("type", _wt.DWORD), ("_input", _INPUT_UNION)]

_INPUT_KEYBOARD  = 1
_KEYEVENTF_KEYUP = 0x0002
_VK_LEFT         = 0x25
_VK_RIGHT        = 0x27
_WHEEL_DELTA     = 120
_INPUT_SZ        = ctypes.sizeof(_INPUT)


# ============================================================
# 1. ToupView 硬件接口
# ============================================================
class PaintWindowHardware(HardwareInterface):

    def __init__(self, scan_cfg: ScanConfig,
                 window_title_contains: str = "ToupView",
                 capture_margin: int = 0):
        super().__init__(scan_cfg)
        self.title_kw    = window_title_contains
        self.margin      = capture_margin
        self.hwnd:        Optional[int] = None
        self.scroll_hwnd: Optional[int] = None
        self._lock = threading.Lock()
        self._connect()

    def _connect(self):
        windows = []
        def _cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if self.title_kw in title:
                    windows.append((hwnd, title))
            return True
        win32gui.EnumWindows(_cb, None)
        if not windows:
            raise RuntimeError(f"未找到包含 '{self.title_kw}' 的窗口")
        self.hwnd, title = windows[0]
        self.scroll_hwnd = self._find_scroll_child(self.hwnd) or self.hwnd
        print(f"[HW] 已连接: '{title}'  hwnd={self.hwnd}  scroll={self.scroll_hwnd}")

    def _find_scroll_child(self, parent: int) -> Optional[int]:
        candidates = []
        def _cb(hwnd, _):
            style = win32api.GetWindowLong(hwnd, win32con.GWL_STYLE)
            if (style & win32con.WS_HSCROLL) and (style & win32con.WS_VSCROLL):
                candidates.append(hwnd)
            return True
        win32gui.EnumChildWindows(parent, _cb, None)
        return candidates[-1] if candidates else None

    def grab_frame(self) -> np.ndarray:
        try:
            l, t, r, b = win32gui.GetWindowRect(self.hwnd)
        except Exception:
            return self._black_frame()
        window_width  = r - l
        window_height = b - t
        cap_x = l + 8; cap_y = t + 30
        cap_w = window_width - 16; cap_h = window_height - 40
        if cap_w <= 0 or cap_h <= 0:
            cap_x, cap_y, cap_w, cap_h = l, t, window_width, window_height
        shot  = pyautogui.screenshot(region=(cap_x, cap_y, cap_w, cap_h))
        frame = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
        fw, fh = self.cfg.frame_w, self.cfg.frame_h
        if self._roi is not None:
            rx, ry, rw, rh = self._roi
            h, w = frame.shape[:2]
            rx = max(0, min(rx, w - 1)); ry = max(0, min(ry, h - 1))
            rw = min(rw, w - rx);        rh = min(rh, h - ry)
            if rw > 0 and rh > 0:
                frame = cv2.resize(frame[ry:ry+rh, rx:rx+rw], (fw, fh),
                                   interpolation=cv2.INTER_LINEAR)
        if frame.shape[0] != fh or frame.shape[1] != fw:
            canvas = np.zeros((fh, fw, 3), dtype=np.uint8)
            h, w = min(frame.shape[0], fh), min(frame.shape[1], fw)
            canvas[:h, :w] = frame[:h, :w]
            frame = canvas
        return frame

    def _black_frame(self) -> np.ndarray:
        return np.zeros((self.cfg.frame_h, self.cfg.frame_w, 3), dtype=np.uint8)

    _roi: Optional[Tuple[int, int, int, int]] = None

    def select_roi(self):
        raw = self._grab_raw()
        win = "请框选识别区域（拖拽鼠标）  Enter确认  C/ESC取消"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(raw.shape[1], 1400), min(raw.shape[0], 900))
        hint = raw.copy()
        cv2.putText(hint, "拖拽选择识别区域  Enter=确认  C/ESC=取消全帧",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        if self._roi:
            rx, ry, rw, rh = self._roi
            cv2.rectangle(hint, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            cv2.putText(hint, f"当前ROI: ({rx},{ry}) {rw}x{rh}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        roi = cv2.selectROI(win, hint, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(win)
        x, y, w, h = roi
        if w > 10 and h > 10:
            self._roi = (int(x), int(y), int(w), int(h))
            print(f"[ROI] 已设置: x={x} y={y} w={w} h={h}")
        else:
            self._roi = None
            print("[ROI] 已清除，使用全帧")

    def _grab_raw(self) -> np.ndarray:
        try:
            l, t, r, b = win32gui.GetWindowRect(self.hwnd)
        except Exception:
            return self._black_frame()
        window_width  = r - l
        window_height = b - t
        if window_width <= 0 or window_height <= 0:
            return self._black_frame()
        cap_x = l + 8; cap_y = t + 30
        cap_w = window_width - 16; cap_h = window_height - 40
        if cap_w <= 0 or cap_h <= 0:
            cap_x, cap_y, cap_w, cap_h = l, t, window_width, window_height
        max_size = 1920
        if cap_w > max_size or cap_h > max_size:
            scale = max_size / max(cap_w, cap_h)
            cap_w = int(cap_w * scale); cap_h = int(cap_h * scale)
        shot = pyautogui.screenshot(region=(cap_x, cap_y, cap_w, cap_h))
        return cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)

    def _window_center_lparam(self) -> int:
        try:
            l, t, r, b = win32gui.GetWindowRect(self.hwnd)
            cx, cy = (l + r) // 2, (t + b) // 2
        except Exception:
            cx, cy = 400, 300
        return (cy & 0xFFFF) << 16 | (cx & 0xFFFF)

    def move_y(self, steps: int):
        ticks = abs(int(steps))
        if ticks == 0: return
        delta  = -_WHEEL_DELTA if steps > 0 else +_WHEEL_DELTA
        wparam = (delta & 0xFFFF) << 16
        lparam = self._window_center_lparam()
        with self._lock:
            for _ in range(ticks):
                win32api.SendMessage(self.scroll_hwnd, 0x020A, wparam, lparam)

    def move_x(self, steps: int):
        ticks = abs(int(steps))
        if ticks == 0: return
        vk = _VK_RIGHT if steps > 0 else _VK_LEFT
        win32gui.SetForegroundWindow(self.hwnd)
        time.sleep(0.04)
        for _ in range(ticks):
            inputs = (_INPUT * 2)()
            inputs[0].type = _INPUT_KEYBOARD
            inputs[0]._input.ki.wVk = vk
            inputs[1].type = _INPUT_KEYBOARD
            inputs[1]._input.ki.wVk     = vk
            inputs[1]._input.ki.dwFlags = _KEYEVENTF_KEYUP
            ctypes.windll.user32.SendInput(2, inputs, _INPUT_SZ)


# ============================================================
# 2. 骨架叠加渲染器（支持干净主路径）
# ============================================================
class SkeletonOverlay:
    """
    封装单帧骨架提取 + 叠加绘制逻辑。

    显示层次（由底到顶）：
      1. 原图 / 分割叠加
      2. 灰色细线  ── 完整骨架所有像素
      3. 红色粗线  ── 最长主路径（含环/分支，原样保留）
      4. 黄色粗线  ── 干净主路径（无环无分支）[show_clean=True 时显示]
      5. 青色粗线  ── 切割路径（从左上角开始，带方向箭头）[show_cutting=True 时显示]
      6. 文字统计信息
    """
    COLOR_ALL    = (160, 160, 160)   # 完整骨架：灰色
    COLOR_MAIN   = (0,   0,   255)   # 最长路径：红色（原样保留）
    COLOR_CLEAN  = (0,   255, 255)   # 干净路径：黄色
    COLOR_CUTTING = (255, 0, 255)    # 切割路径：青色
    RADIUS_ALL   = 1
    RADIUS_MAIN  = 3
    RADIUS_CLEAN = 3
    RADIUS_CUTTING = 3

    def __init__(self, scan_cfg: ScanConfig):
        self.extractor = PathSkeletonExtractor(scan_cfg)

    def apply(self, image: np.ndarray, mask: np.ndarray,
              show_stats: bool = True,
              show_clean: bool = True,
              show_cutting: bool = True) -> Tuple[np.ndarray, int, int, int, int]:
        """
        在 image 上叠加骨架并返回结果。

        Args:
            image        : BGR 图像
            mask         : 语义分割掩码
            show_stats   : 是否在左下角绘制统计文字
            show_clean   : 是否绘制黄色干净路径层（C 键控制）
            show_cutting : 是否绘制青色切割路径层（V 键控制）

        Returns:
            out     : 叠加结果图像
            n_all   : 完整骨架像素数
            n_main  : 最长路径像素数
            n_clean : 干净路径像素数
            n_cutting: 切割路径像素数
        """
        full_skel, main_skel, _ = self.extractor.extract_from_mask(mask)
        n_all  = int(np.sum(full_skel > 0))
        n_main = int(np.sum(main_skel > 0))

        if n_all == 0:
            return image.copy(), 0, 0, 0, 0

        # ① 完整骨架（灰色细线）
        out = self.extractor.draw_all_skeleton(
            image, full_skel, color=self.COLOR_ALL, radius=self.RADIUS_ALL)

        # ② 最长路径（红色粗线，原样保留）
        if n_main > 0:
            out = self.extractor.draw_all_skeleton(
                out, main_skel, color=self.COLOR_MAIN, radius=self.RADIUS_MAIN)

        # ③ 干净主路径（黄色粗线，无环无分支）
        n_clean = 0
        if show_clean:
            clean_mask, _ = self.extractor.extract_clean_main_path(
                mask, precomputed_skel=full_skel)
            n_clean = int(np.sum(clean_mask > 0))
            if n_clean > 0:
                out = self.extractor.draw_all_skeleton(
                    out, clean_mask, color=self.COLOR_CLEAN, radius=self.RADIUS_CLEAN)

        # ④ 切割路径（青色粗线，从左上角开始，带方向箭头）
        n_cutting = 0
        if show_cutting:
            cutting_mask, cutting_points, cutting_directions = self.extractor.extract_cutting_path(
                mask, precomputed_skel=full_skel)
            n_cutting = int(np.sum(cutting_mask > 0))
            if n_cutting > 0:
                out = self.extractor.draw_cutting_path_with_arrows(
                    out, cutting_mask, cutting_directions,
                    path_color=self.COLOR_CUTTING, arrow_color=(0, 255, 0),
                    arrow_spacing=30, arrow_size=15)

        # ⑤ 统计文字
        if show_stats:
            cv2.putText(out,
                        f"All:{n_all}  Main:{n_main}  Clean:{n_clean}  Cut:{n_cutting}",
                        (10, image.shape[0] - 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        return out, n_all, n_main, n_clean, n_cutting


# ============================================================
# 3. 实时分割 + 骨架预览
# ============================================================
class DisplayConfig:
    def __init__(self,
                 panel_scale: float = 1.0,
                 columns: int = 3,
                 window_w: Optional[int] = None,
                 window_h: Optional[int] = None):
        assert columns in (1, 2, 3), "columns 只能是 1 / 2 / 3"
        assert 0.1 <= panel_scale <= 4.0
        self.panel_scale = panel_scale
        self.columns     = columns
        self.window_w    = window_w
        self.window_h    = window_h

    def panel_size(self, frame_w: int, frame_h: int) -> Tuple[int, int]:
        return (max(1, int(frame_w * self.panel_scale)),
                max(1, int(frame_h * self.panel_scale)))

    def window_size(self, frame_w: int, frame_h: int) -> Tuple[int, int]:
        pw, ph = self.panel_size(frame_w, frame_h)
        return (self.window_w or pw * self.columns,
                self.window_h or ph)

    def resize_panel(self, img: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
        pw, ph = self.panel_size(frame_w, frame_h)
        if img.shape[1] == pw and img.shape[0] == ph:
            return img
        interp = cv2.INTER_AREA if self.panel_scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(img, (pw, ph), interpolation=interp)


class RealtimeSegViewer:
    """
    显示布局（列数由 DisplayConfig.columns 控制）：
    ┌──────────────┬──────────────┬──────────────┐
    │  原始画面     │  分割叠加    │ 分割+骨架     │
    └──────────────┴──────────────┴──────────────┘

    键位：W/S 上下  A/D 左右  K 切换骨架  C 切换干净路径  V 切换切割路径  +/- 缩放  Q 退出
    """
    ALPHA = 0.45
    STEP  = 5
    _SCALE_STEP = 0.1
    _SCALE_MIN  = 0.2
    _SCALE_MAX  = 3.0

    def __init__(self, hw: PaintWindowHardware,
                 seg: SegmentationInference,
                 scan_cfg: ScanConfig,
                 disp_cfg: Optional["DisplayConfig"] = None):
        self.hw            = hw
        self.seg           = seg
        self.skeleton      = SkeletonOverlay(scan_cfg)
        self.disp          = disp_cfg or DisplayConfig()
        self.running       = True
        self.show_skeleton = True
        self.show_clean    = True          # ★ C 键切换干净路径显示
        self.show_cutting  = True          # ★ V 键切换切割路径显示
        self._trigger_roi  = False
        self._x = self._y = self._steps = 0

    def _on_key(self, key):
        try:
            ch = key.char
        except AttributeError:
            return
        s = self.STEP
        ch_l = ch.lower()
        if   ch_l == 'w': self.hw.move_y(-s); self._y -= s; self._steps += 1
        elif ch_l == 's': self.hw.move_y(+s); self._y += s; self._steps += 1
        elif ch_l == 'a': self.hw.move_x(-s); self._x -= s; self._steps += 1
        elif ch_l == 'd': self.hw.move_x(+s); self._x += s; self._steps += 1
        elif ch_l == 'k': self.show_skeleton = not self.show_skeleton
        elif ch_l == 'c': self.show_clean    = not self.show_clean   # ★ 新增
        elif ch_l == 'v': self.show_cutting  = not self.show_cutting # ★ 新增
        elif ch_l == 'r': self._trigger_roi  = True
        elif ch in ('+', '='): self._scale_step(+self._SCALE_STEP)
        elif ch == '-':        self._scale_step(-self._SCALE_STEP)
        elif ch_l == 'q': self.running = False; return False

    def _scale_step(self, delta: float):
        new_scale = round(self.disp.panel_scale + delta, 2)
        new_scale = max(self._SCALE_MIN, min(self._SCALE_MAX, new_scale))
        self.disp.panel_scale = new_scale
        fw, fh = self.hw.cfg.frame_w, self.hw.cfg.frame_h
        ww, wh = self.disp.window_size(fw, fh)
        cv2.resizeWindow("语义分割+骨架 | WASD移动  K骨架  C干净路径  Q退出", ww, wh)
        print(f"[Display] scale={new_scale:.2f}  窗口={ww}x{wh}")

    @staticmethod
    def _colorize(mask: np.ndarray) -> np.ndarray:
        vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for cls, color in SEG_PALETTE.items():
            vis[mask == cls] = color
        return vis

    def _draw_legend(self, img: np.ndarray) -> np.ndarray:
        y = 20
        for cls, color in SEG_PALETTE.items():
            if cls == SegClass.BACKGROUND:
                continue
            cv2.rectangle(img, (10, y - 12), (28, y + 4), color, -1)
            cv2.putText(img, cls.name, (34, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 22
        return img

    def _blend_seg(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        overlay = self._colorize(mask)
        blended = cv2.addWeighted(frame, 1 - self.ALPHA, overlay, self.ALPHA, 0)
        return self._draw_legend(blended)

    def _draw_pos(self, img: np.ndarray) -> np.ndarray:
        cv2.putText(img,
                    f"X:{self._x:+5d}  Y:{self._y:+5d}  Steps:{self._steps}",
                    (10, img.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        return img

    def run(self):
        win_name = "语义分割+骨架 | WASD移动  K骨架  C干净路径  +/-缩放  Q退出"
        fw, fh   = self.hw.cfg.frame_w, self.hw.cfg.frame_h
        ww, wh   = self.disp.window_size(fw, fh)

        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, ww, wh)

        print("是否先框选识别区域？直接按 Enter 跳过，或按 R 键随时框选")
        self.hw.select_roi()

        listener = keyboard.Listener(on_press=self._on_key)
        listener.start()

        print("\n" + "=" * 60)
        print("🔬 实时语义分割 + 骨架识别  v2.1")
        print(f"  布局: {self.disp.columns}列  缩放:{self.disp.panel_scale:.2f}  窗口:{ww}x{wh}")
        print("  W/S→上下  A/D→左右  K→骨架  C→干净路径  V→切割路径  R→重选ROI  +/-→缩放  Q→退出")
        print("  颜色: 灰=完整骨架  红=最长路径(原样)  黄=干净路径(无环无分支)  青=切割路径(从左上角)")
        print("=" * 60 + "\n")

        while self.running:
            if self._trigger_roi:
                self._trigger_roi = False
                self.hw.select_roi()
                ww, wh = self.disp.window_size(fw, fh)
                cv2.resizeWindow(win_name, ww, wh)

            t0 = time.time()
            frame = self.hw.grab_frame()
            mask  = self.seg.predict(frame)

            frame_disp = frame.copy()
            if self.hw._roi:
                rx, ry, rw, rh = self.hw._roi
                cv2.putText(frame_disp, f"ROI({rx},{ry}) {rw}x{rh}  [R]重选",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            else:
                cv2.putText(frame_disp, "[R] 按R框选ROI",
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 200), 1)

            seg_view = self._blend_seg(frame.copy(), mask)

            if self.show_skeleton:
                # ★ apply() 现在返回 5 个值
                skel_view, n_all, n_main, n_clean, n_cutting = self.skeleton.apply(
                    seg_view.copy(), mask,
                    show_stats=True,
                    show_clean=self.show_clean,
                    show_cutting=self.show_cutting,
                )
                tag = (f"[K]Skel ON  "
                       f"[C]Clean:{'ON ' if self.show_clean else 'OFF'}"
                       f"({n_clean}px)  "
                       f"[V]Cut:{'ON ' if self.show_cutting else 'OFF'}"
                       f"({n_cutting}px)")
                cv2.putText(skel_view, tag,
                            (skel_view.shape[1] - 260, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            else:
                skel_view = seg_view.copy()
                cv2.putText(skel_view, "[K] Skel OFF",
                            (skel_view.shape[1] - 130, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            self._draw_pos(seg_view)
            self._draw_pos(skel_view)

            def rs(img): return self.disp.resize_panel(img, fw, fh)
            if   self.disp.columns == 1:
                panel = rs(skel_view)
            elif self.disp.columns == 2:
                panel = np.hstack([rs(seg_view), rs(skel_view)])
            else:
                panel = np.hstack([rs(frame_disp), rs(seg_view), rs(skel_view)])

            cv2.imshow(win_name, panel)

            fps = 1.0 / max(time.time() - t0, 1e-6)
            cv2.setWindowTitle(win_name,
                f"FPS:{fps:.1f}  X:{self._x}  Y:{self._y}"
                f"  scale:{self.disp.panel_scale:.2f}"
                f"  骨架:{'ON' if self.show_skeleton else 'OFF'}"
                f"  干净路径:{'ON' if self.show_clean else 'OFF'}"
                f"  切割路径:{'ON' if self.show_cutting else 'OFF'}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        listener.stop()
        cv2.destroyAllWindows()
        print(f"\n✅ 退出  位置:({self._x},{self._y})  步数:{self._steps}")


# ============================================================
# 4. 切割模拟器
# ============================================================
class CuttingSimulator:
    """
    切割模拟器：在骨架生成后，控制ToupView视角沿切割路径移动，模拟切割过程
    
    功能：
    1. 在生成"最优切割路径"后，控制局部视角
    2. 基于骨架的像素点位置，在规划好的路径上，模拟"切割"过程
    3. 在snake模式下，重新控制toupview的界面，回到初始点（左上角）
    4. 控制上下左右，让画面的中心始终沿"切割路径"移动
    """
    
    def __init__(self, hw: PaintWindowHardware, scan_cfg: ScanConfig):
        self.hw = hw
        self.cfg = scan_cfg
        self.running = True
        self.paused = False
        self.current_index = 0
        self.cutting_points = []
        self.directions = []
        self._listener = None
        
    def set_cutting_path(self, cutting_points: List[Tuple[int, int]], 
                        directions: List[Tuple[int, int, int]]):
        """
        设置切割路径
        
        Args:
            cutting_points: 切割路径点列表 [(y,x), ...]
            directions: 方向信息列表 [(y,x,dir_idx), ...]
        """
        self.cutting_points = cutting_points
        self.directions = directions
        self.current_index = 0
        print(f"[CuttingSim] 切割路径已设置，共 {len(cutting_points)} 个点")
        
    def reset_to_initial_position(self):
        """
        重置视角到初始点（左上角）
        通过发送大量向上和向左的移动指令回到左上角
        """
        print("[CuttingSim] 正在重置视角到初始点（左上角）...")
        
        # 发送大量向上和向左的移动指令
        reset_steps = 1000
        self.hw.move_y(-reset_steps)  # 向上
        time.sleep(0.1)
        self.hw.move_x(-reset_steps)  # 向左
        time.sleep(0.1)
        
        print("[CutSim] 视角已重置到初始点")
        
    def _on_key(self, key):
        """
        键盘事件处理
        """
        try:
            ch = key.char.lower()
            if ch == ' ':
                self.paused = not self.paused
                print(f"[CutSim] {'暂停' if self.paused else '继续'}切割模拟")
            elif ch == 'q':
                print("[CutSim] Q 键按下，退出切割模拟...")
                self.running = False
                return False
            elif ch == 'r':
                print("[CutSim] R 键按下，重新开始切割模拟...")
                self.current_index = 0
                self.paused = False
        except AttributeError:
            pass
            
    def _calculate_movement(self, current_pos: Tuple[int, int], 
                          target_pos: Tuple[int, int]) -> Tuple[int, int]:
        """
        计算从当前位置到目标位置的移动量
        
        Args:
            current_pos: 当前位置 (y, x)
            target_pos: 目标位置 (y, x)
            
        Returns:
            (dy, dx): y和x方向的移动步数
        """
        dy = target_pos[0] - current_pos[0]
        dx = target_pos[1] - current_pos[1]
        return dy, dx
        
    def run(self, speed: float = 0.05):
        """
        运行切割模拟
        
        Args:
            speed: 移动速度（秒），越小越快
        """
        if not self.cutting_points:
            print("[CutSim] 错误：切割路径未设置！")
            return
            
        print("\n" + "=" * 60)
        print("🔬 切割模拟器")
        print("=" * 60)
        print(f"  切割路径点数: {len(self.cutting_points)}")
        print(f"  移动速度: {speed:.3f}秒/步")
        print("  控制键:")
        print("    空格键 - 暂停/继续")
        print("    R 键   - 重新开始")
        print("    Q 键   - 退出")
        print("=" * 60 + "\n")
        
        # 重置视角到初始点
        self.reset_to_initial_position()
        
        # 启动键盘监听
        self._listener = keyboard.Listener(on_press=self._on_key)
        self._listener.start()
        
        try:
            # 获取第一帧作为当前位置
            frame = self.hw.grab_frame()
            h, w = frame.shape[:2]
            center_y, center_x = h // 2, w // 2
            
            # 假设初始位置在全局坐标系的(0, 0)
            current_global_y = 0
            current_global_x = 0
            
            while self.running and self.current_index < len(self.cutting_points):
                if self.paused:
                    time.sleep(0.1)
                    continue
                    
                # 获取当前目标点
                target_y, target_x = self.cutting_points[self.current_index]
                
                # 计算移动量
                dy = target_y - current_global_y
                dx = target_x - current_global_x
                
                # 移动视角
                if dy != 0:
                    self.hw.move_y(-dy)  # 注意：move_y的符号与y轴方向相反
                    time.sleep(0.01)
                if dx != 0:
                    self.hw.move_x(dx)
                    time.sleep(0.01)
                
                # 更新当前位置
                current_global_y = target_y
                current_global_x = target_x
                
                # 显示进度
                progress = (self.current_index + 1) / len(self.cutting_points) * 100
                if self.current_index % 10 == 0:
                    print(f"[CutSim] 进度: {progress:.1f}% ({self.current_index + 1}/{len(self.cutting_points)}) "
                          f"位置: ({target_x}, {target_y})")
                
                # 移动到下一个点
                self.current_index += 1
                
                # 控制速度
                time.sleep(speed)
                
            print("\n" + "=" * 60)
            print("✅ 切割模拟完成！")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("[CutSim] 用户中断切割模拟")
        finally:
            if self._listener:
                self._listener.stop()
            print("[CutSim] 切割模拟结束，清理资源...")


# ============================================================
# 5. 主程序
# ============================================================
def main():
    scan_cfg = ScanConfig(
        frame_w=1028, frame_h=776,
        um_per_pixel=1.0,               # 每个像素的微米数【越大，扫描范围越大】
        um_per_step=1.0,                # 每个步骤的微米数【越大，移动越快】      
        move_speed_steps=3,            # 移动速度的步骤数【越大，移动越快】
        row_overlap_ratio=0.1,          # 行重叠比例
        corner_roi_size=80,
        corner_ratio_threshold=0.25,
        corner_confirm_count=4,
        boundary_lock_margin_steps=10,
        skeleton_path_radius=1,
        skeleton_highlight_radius=3,
        skeleton_min_length=100000,      # 最小骨架长度【用于筛选最小的骨架 -- 过滤噪声】
        output_dir=r"F:/0_My_jupyter/0_Optical/scan_output/ToupView/test4",
        video_fps=10,
        save_debug_frames=True,
    )

    # 模式
    MODE = "realtime"
    MODE = "snake"
    
    # 切割模拟配置
    ENABLE_CUTTING_SIMULATION = True  # 是否启用切割模拟
    CUTTING_SIMULATION_SPEED = 0.05   # 切割模拟速度（秒/步），越小越快

    # 窗口名称
    window_title_contains = "ToupView"

    hw  = PaintWindowHardware(scan_cfg, window_title_contains=window_title_contains, capture_margin=0)
    seg = SegmentationInference()

    if MODE == "realtime":
        disp_cfg = DisplayConfig(panel_scale=0.4, columns=2)
        RealtimeSegViewer(hw, seg, scan_cfg, disp_cfg).run()

    elif MODE == "snake":
        print("\n" + "=" * 60)
        print("🔧 Snake 模式 - 区域限定")
        print("=" * 60)
        print("请框选需要扫描的芯片区域，排除工具栏等干扰区域")
        print("直接按 Enter 跳过，使用全帧扫描")
        print("=" * 60 + "\n")
        hw.select_roi()

        if hw._roi:
            rx, ry, rw, rh = hw._roi
            print(f"[Snake] ROI 已设置: x={rx} y={ry} w={rw} h={rh}")
        else:
            print(f"[Snake] 未设置 ROI，使用全帧扫描")

        class SnakeScanControllerWithROI(SnakeScanController):
            def __init__(self, scan_cfg, hw):
                super().__init__(scan_cfg, hw)
                self._trigger_roi = False
                self._listener    = None
                self._running     = True

            def _save_cutting_path_coordinates(self, cutting_points, cutting_directions, output_path):
                """
                保存切割路径坐标到 JSON 文件
                
                Args:
                    cutting_points: 切割路径点列表 [(y,x), ...]
                    cutting_directions: 方向信息列表 [(y,x,dir_idx), ...]
                    output_path: 输出 JSON 文件路径
                """
                import json
                import os
                
                # 确保路径存在
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # 转换坐标系统：从 (y, x) 转换为 (x, y)
                cutting_points_converted = [(int(x), int(y)) for y, x in cutting_points]
                
                # 方向名称映射
                direction_names = ['右', '下', '左', '上', '右下', '左下', '右上', '左上']
                
                # 准备数据
                cutting_data = {
                    'cutting_points': cutting_points_converted,
                    'cutting_count': len(cutting_points_converted),
                    'directions': [
                        {
                            'y': int(y),
                            'x': int(x),
                            'direction': direction_names[dir_idx] if dir_idx < len(direction_names) else f'方向{dir_idx}',
                            'direction_index': int(dir_idx)
                        }
                        for y, x, dir_idx in cutting_directions
                    ],
                    'direction_count': len(cutting_directions),
                    'start_point': cutting_points_converted[0] if cutting_points_converted else None,
                    'end_point': cutting_points_converted[-1] if cutting_points_converted else None
                }
                
                # 保存为 JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(cutting_data, f, indent=2, ensure_ascii=False)
                
                print(f"切割路径坐标已保存到: {output_path}")

            def _on_key(self, key):
                try:
                    ch = key.char.lower()
                    if ch == 'r':
                        self._trigger_roi = True
                        print("[Snake] R 键按下，准备重新选择 ROI...")
                    elif ch == 'q':
                        print("[Snake] Q 键按下，准备退出...")
                        self._running = False
                        return False
                except AttributeError:
                    pass

            def run(self):
                self._listener = keyboard.Listener(on_press=self._on_key)
                self._listener.start()

                try:
                    print("\n" + "=" * 60)
                    print("🔧 Snake 扫描模式  R=重选ROI  Q=退出")
                    print("=" * 60 + "\n")

                    import threading

                    def scan_thread():
                        try:
                            super(SnakeScanControllerWithROI, self).run()
                        except Exception as e:
                            print(f"[Snake] 扫描过程中发生错误: {e}")
                        finally:
                            self._running = False

                    t = threading.Thread(target=scan_thread, daemon=True)
                    t.start()

                    while self._running:
                        if self._trigger_roi:
                            self._trigger_roi = False
                            print("[Snake] 正在重新选择 ROI...")
                            self.hw.select_roi()
                            if self.hw._roi:
                                rx, ry, rw, rh = self.hw._roi
                                print(f"[Snake] 新 ROI 已设置: x={rx} y={ry} w={rw} h={rh}")
                        time.sleep(0.1)

                    t.join()

                    # ★ 扫描完成后提取骨架坐标
                    print("\n" + "=" * 60)
                    print("📍 提取骨架坐标")
                    print("=" * 60)

                    # ★ 解包 10 个返回值
                    (global_map_with_skeleton, global_mask, global_mask_color_with_skeleton,
                     full_skeleton, longest_path_mask, longest_path_points,
                     clean_path_mask, clean_path_points,
                     cutting_path_mask, cutting_path_points, cutting_directions) = self.stitcher.build_global_map()

                    coordinates = extract_skeleton_coordinates(
                        global_mask, self.cfg, use_physical_units=False)

                    print(f"✅ 骨架坐标提取完成:")
                    print(f"   - 完整骨架点数量: {coordinates['skeleton_count']}")
                    print(f"   - 最长路径点数量: {coordinates['longest_path_count']}")
                    print(f"   - 干净路径点数量: {len(clean_path_points)}")
                    print(f"   - 切割路径点数量: {len(cutting_path_points)}")

                    output_json = os.path.join(self.cfg.output_dir, "skeleton_coordinates.json")
                    save_skeleton_coordinates(coordinates, output_json)
                    print(f"✅ 骨架坐标已保存到: {output_json}")

                    # ★ 保存切割路径坐标
                    cutting_json = os.path.join(self.cfg.output_dir, "cutting_path_coordinates.json")
                    self._save_cutting_path_coordinates(cutting_path_points, cutting_directions, cutting_json)
                    print(f"✅ 切割路径坐标已保存到: {cutting_json}")

                    print("📊 可视化骨架坐标...")
                    visualized = visualize_skeleton_coordinates(
                        global_map_with_skeleton, coordinates,
                        point_radius=2,
                        skeleton_color=(0, 0, 255),
                        path_color=(0, 255, 0))

                    output_image = os.path.join(
                        self.cfg.output_dir, "skeleton_coordinates_visualization.png")
                    cv2.imwrite(output_image, visualized)
                    print(f"✅ 可视化结果已保存到: {output_image}")

                    print("\n📋 前10个骨架点坐标:")
                    for i, (x, y) in enumerate(coordinates['skeleton_points'][:10]):
                        print(f"   [{i+1}] ({x:.1f}, {y:.1f})")

                    print("\n📋 干净路径前10个点坐标 (y,x):")
                    for i, (y, x) in enumerate(clean_path_points[:10]):
                        print(f"   [{i+1}] ({x:.1f}, {y:.1f})")

                    print("\n📋 切割路径前10个点坐标 (y,x):")
                    for i, (y, x) in enumerate(cutting_path_points[:10]):
                        print(f"   [{i+1}] ({x:.1f}, {y:.1f})")

                    print("\n📋 切割路径方向统计:")
                    direction_names = ['右', '下', '左', '上', '右下', '左下', '右上', '左上']
                    dir_counts = {}
                    for _, _, dir_idx in cutting_directions:
                        dir_counts[dir_idx] = dir_counts.get(dir_idx, 0) + 1
                    for dir_idx, count in sorted(dir_counts.items()):
                        dir_name = direction_names[dir_idx] if dir_idx < len(direction_names) else f'方向{dir_idx}'
                        print(f"   {dir_name}: {count} 次")

                    print("\n" + "=" * 60)
                    print("✅ 所有任务完成！")
                    print("=" * 60)
                    
                    # ★ 启动切割模拟（如果启用）
                    if ENABLE_CUTTING_SIMULATION:
                        print("\n" + "=" * 60)
                        print("🔬 准备启动切割模拟...")
                        print("=" * 60)
                        
                        try:
                            input("\n按 Enter 键开始切割模拟，或 Ctrl+C 跳过...")
                            
                            # 创建切割模拟器
                            simulator = CuttingSimulator(hw, self.cfg)
                            
                            # 设置切割路径
                            simulator.set_cutting_path(cutting_path_points, cutting_directions)
                            
                            # 运行切割模拟
                            simulator.run(speed=CUTTING_SIMULATION_SPEED)
                        except KeyboardInterrupt:
                            print("\n[Snake] 用户跳过切割模拟")
                    else:
                        print("\n[Snake] 切割模拟已禁用（ENABLE_CUTTING_SIMULATION = False）")

                except KeyboardInterrupt:
                    print("[Snake] 用户中断扫描")
                finally:
                    if self._listener:
                        self._listener.stop()
                    print("[Snake] 扫描结束，清理资源...")

        snake_controller = SnakeScanControllerWithROI(scan_cfg, hw)
        snake_controller.run()


if __name__ == "__main__":
    main()
