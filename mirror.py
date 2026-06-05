#!/usr/bin/env python3
"""
Airheads — Smart Mirror Experience
Trigger: Space / Enter / USB button  OR  pressure sensor via serial (Pi Pico)
"""
import sys
import os
import time
import math
import glob
import threading
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pygame
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
WIG_PATH              = "assets/wig.png"
FACE_LANDMARKER_MODEL = "face_landmarker.task"
FACE_LANDMARKER_URL   = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

WIG_WIDTH_SCALE = 1.5
WIG_RING_BOTTOM = 0.9   # fraction of wig height where forehead anchor lands
                         # increase → wig moves up, decrease → wig moves down
                         # tune with ↑↓ arrow keys while running

# Serial trigger (pressure sensor via Pi Pico)
# Set to a specific port string to skip auto-detect, e.g. "/dev/cu.usbmodem101"
# Leave as None to auto-detect the first connected Pico.
SERIAL_PORT = None
SERIAL_BAUD = 9600

# Meditation text sequence — (text, display_seconds)
TEXT_PHASES = [
    ("Let the worm look inside your mind.",                                   6),
    ("Direct the worm towards the thoughts you want to let go of.",           7),
    ("Visualize the worm devouring those thoughts.",                          6),
    ("Really crunching on them, gnawing on them, messily slurping them up.",  7),
    ("Close your eyes and take two deep breaths as the worm cleans up.",     10),
    ("You have now reached Airvana.",                                          6),
]
# Wig appears instantly at this phase index ("You have now reached Airvana.")
WIG_FADE_PHASE  = 5
WIG_HOLD_SECS   = 12    # hold at full opacity after last text card
TEXT_FADE_SECS  = 0.8   # per-text fade-in / fade-out duration

IDLE_TEXT       = "press the button to begin"
IDLE_TEXT_ALPHA = 150   # subtle — out of 255
# ─────────────────────────────────────────────────────────────────────────────

FOREHEAD_TOP = 10
LEFT_TEMPLE  = 234
RIGHT_TEMPLE = 454

# Cumulative phase timeline: all text phases + a final hold after last card
_PHASE_DURS           = [d for _, d in TEXT_PHASES] + [WIG_HOLD_SECS]
_PHASE_STARTS         = [sum(_PHASE_DURS[:i]) for i in range(len(_PHASE_DURS))]
TOTAL_MEDITATION_SECS = sum(_PHASE_DURS)
WIG_HOLD_PHASE        = len(TEXT_PHASES)   # hold phase index (after all text)

TRIGGER_KEYS = {pygame.K_SPACE, pygame.K_RETURN}


# ── Serial trigger (pressure sensor) ─────────────────────────────────────────
def find_pico_port() -> str | None:
    """Return the first likely Pi Pico serial port, or None."""
    for pattern in ("/dev/cu.usbmodem*", "/dev/ttyACM*", "/dev/ttyUSB*"):
        matches = glob.glob(pattern)
        if matches:
            return matches[0]
    return None


class SerialListener(threading.Thread):
    """
    Reads lines from the Pi Pico over USB serial.
    Any line containing 'TRIGGER' sets the shared event.
    Silently does nothing if pyserial isn't installed or no device is found.
    """
    def __init__(self, trigger: threading.Event):
        super().__init__(daemon=True)
        self.trigger = trigger
        self._stop   = threading.Event()

    def run(self) -> None:
        try:
            import serial
        except ImportError:
            print("[serial] pyserial not installed — serial trigger disabled")
            return

        port = SERIAL_PORT or find_pico_port()
        if not port:
            print("[serial] no device found — serial trigger disabled")
            return

        try:
            conn = serial.Serial(port, SERIAL_BAUD, timeout=1)
        except Exception as e:
            print(f"[serial] could not open {port}: {e}")
            return

        print(f"[serial] listening on {port}")
        try:
            while not self._stop.is_set():
                line = conn.readline().decode("utf-8", errors="ignore").strip()
                if "TRIGGER" in line:
                    self.trigger.set()
        finally:
            conn.close()

    def stop(self) -> None:
        self._stop.set()


# ── Face tracking ─────────────────────────────────────────────────────────────
class WigPhysics:
    """Exponential smoothing for wig position and head-tilt angle."""
    def __init__(self):
        self.x = self.y = self.angle = None

    def update(self, x: float, y: float, angle: float) -> None:
        if self.x is None:
            self.x, self.y, self.angle = float(x), float(y), float(angle)
            return
        self.x     += (x     - self.x)     * 0.25
        self.y     += (y     - self.y)     * 0.25
        self.angle += (angle - self.angle) * 0.20

    def reset(self) -> None:
        self.x = self.y = self.angle = None


# ── Assets ────────────────────────────────────────────────────────────────────
def ensure_model() -> str:
    if not os.path.exists(FACE_LANDMARKER_MODEL):
        print("Downloading face landmarker model…")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_MODEL)
        print("Done.")
    return FACE_LANDMARKER_MODEL


def load_wig(path: str) -> tuple[np.ndarray, float]:
    """Load wig PNG and auto-compute where its base sits for head anchoring."""
    if not os.path.exists(path):
        sys.exit(f"[error] wig not found at {path!r} — add assets/wig.png and restart")
    img  = Image.open(path).convert("RGBA")
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    arr   = np.array(img)
    alpha = arr[:, :, 3]

    # Find the bottom-most row where at least 5% of pixels are opaque.
    # That row is treated as the hat base — the point that lands on the forehead.
    row_fill      = (alpha > 50).mean(axis=1)
    opaque_rows   = np.where(row_fill > 0.05)[0]
    ring_bottom   = float(opaque_rows[-1]) / arr.shape[0] if len(opaque_rows) else 0.9
    print(f"[wig] auto ring_bottom = {ring_bottom:.3f}")
    return arr, ring_bottom


# ── Rendering ─────────────────────────────────────────────────────────────────
def overlay_wig(frame_bgr: np.ndarray, wig_rgba: np.ndarray,
                landmarks, img_w: int, img_h: int,
                physics: WigPhysics, opacity: float,
                width_scale: float, ring_bottom: float) -> np.ndarray:
    lm     = landmarks
    raw_x  = lm[FOREHEAD_TOP].x * img_w
    raw_y  = lm[FOREHEAD_TOP].y * img_h
    lx, ly = lm[LEFT_TEMPLE].x  * img_w, lm[LEFT_TEMPLE].y  * img_h
    rx, ry = lm[RIGHT_TEMPLE].x * img_w, lm[RIGHT_TEMPLE].y * img_h

    face_width = abs(rx - lx)
    if face_width < 10:
        return frame_bgr

    physics.update(raw_x, raw_y, math.atan2(ry - ly, rx - lx))
    px, py, pa = physics.x, physics.y, physics.angle

    wig_w = int(face_width * width_scale)
    wig_h = int(wig_rgba.shape[0] * wig_w / wig_rgba.shape[1])
    if wig_w < 2 or wig_h < 2:
        return frame_bgr

    wig_r        = cv2.resize(wig_rgba, (wig_w, wig_h), interpolation=cv2.INTER_AREA)
    acx, acy     = wig_w / 2.0, wig_h * ring_bottom
    cos_a, sin_a = math.cos(pa), math.sin(pa)

    M = np.float32([
        [cos_a, -sin_a, px - cos_a * acx + sin_a * acy],
        [sin_a,  cos_a, py - sin_a * acx - cos_a * acy],
    ])
    warped = cv2.warpAffine(wig_r, M, (img_w, img_h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))

    alpha   = warped[:, :, 3:4].astype(np.float32) / 255.0 * opacity
    wig_bgr = cv2.cvtColor(warped[:, :, :3], cv2.COLOR_RGB2BGR)
    return (wig_bgr.astype(np.float32) * alpha +
            frame_bgr.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def frame_to_surface(frame_bgr: np.ndarray, w: int, h: int) -> pygame.Surface:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (w, h))
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))


def draw_text(screen: pygame.Surface, font: pygame.font.Font,
              text: str, center_x: int, y: int, alpha: int) -> None:
    """Draw text with a black background bar, positioned by top-left y."""
    alpha   = max(0, min(255, alpha))
    surf    = font.render(text, True, (255, 255, 255))
    tw, th  = surf.get_size()
    pad_x, pad_y = 36, 14

    bg = pygame.Surface((tw + pad_x * 2, th + pad_y * 2), pygame.SRCALPHA)
    bg.fill((0, 0, 0, int(210 * alpha / 255)))
    screen.blit(bg, (center_x - tw // 2 - pad_x, y - pad_y))

    surf.set_alpha(alpha)
    screen.blit(surf, (center_x - tw // 2, y))


def text_alpha(elapsed: float, duration: float) -> int:
    fade = TEXT_FADE_SECS
    if elapsed < fade:
        return int(255 * elapsed / fade)
    if elapsed > duration - fade:
        return max(0, int(255 * (duration - elapsed) / fade))
    return 255


def phase_at(meditation_elapsed: float) -> tuple[int, float]:
    """Return (phase_index, elapsed_within_phase). Returns (N, 0) when done."""
    for i, (start, dur) in enumerate(zip(_PHASE_STARTS, _PHASE_DURS)):
        if meditation_elapsed < start + dur:
            return i, meditation_elapsed - start
    return len(_PHASE_DURS), 0.0


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    # ── Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("[error] camera not found — check USB connection and try again")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── Assets + face model
    wig, auto_ring_bottom = load_wig(WIG_PATH)
    physics               = WigPhysics()
    fl_opts    = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    landmarker = mp_vision.FaceLandmarker.create_from_options(fl_opts)

    # ── Serial listener (starts silently, no-ops if hardware absent)
    trigger_event  = threading.Event()
    serial_listener = SerialListener(trigger_event)
    serial_listener.start()

    # ── Display
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    SW, SH = screen.get_size()
    pygame.display.set_caption("Airheads")
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    font_body = pygame.font.SysFont("helveticaneue", 44)
    font_idle = pygame.font.SysFont("helveticaneue", 26)

    # ── State
    state            = "idle"   # "idle" | "meditation"
    meditation_start = 0.0

    # Staff-only calibration (hidden from participants)
    # H toggles the on-screen HUD; arrow keys adjust; values print to terminal
    ring_bottom  = auto_ring_bottom   # auto-detected from PNG
    width_scale  = WIG_WIDTH_SCALE
    show_hud     = False

    def start_meditation() -> None:
        nonlocal state, meditation_start
        state            = "meditation"
        meditation_start = time.time()
        physics.reset()

    def reset_to_idle() -> None:
        nonlocal state
        state = "idle"
        physics.reset()

    # ── Loop
    try:
        while True:
            clock.tick(30)
            now = time.time()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    if event.key in TRIGGER_KEYS:
                        if state == "idle":
                            trigger_event.set()
                        else:
                            reset_to_idle()
                    # Staff calibration — H shows/hides HUD, arrows adjust
                    if event.key == pygame.K_h:
                        show_hud = not show_hud
                    if event.key == pygame.K_UP:
                        ring_bottom = round(min(ring_bottom + 0.02, 0.99), 2)
                        print(f"WIG_RING_BOTTOM = {ring_bottom}")
                    if event.key == pygame.K_DOWN:
                        ring_bottom = round(max(ring_bottom - 0.02, 0.01), 2)
                        print(f"WIG_RING_BOTTOM = {ring_bottom}")
                    if event.key == pygame.K_RIGHT:
                        width_scale = round(min(width_scale + 0.05, 3.0), 2)
                        print(f"WIG_WIDTH_SCALE = {width_scale}")
                    if event.key == pygame.K_LEFT:
                        width_scale = round(max(width_scale - 0.05, 0.3), 2)
                        print(f"WIG_WIDTH_SCALE = {width_scale}")

            # Serial or keyboard trigger → start meditation
            if state == "idle" and trigger_event.is_set():
                trigger_event.clear()
                start_meditation()

            # Auto-return to idle when sequence completes
            if state == "meditation" and (now - meditation_start) >= TOTAL_MEDITATION_SECS:
                reset_to_idle()

            # ── Camera frame
            screen.fill((0, 0, 0))
            ok, frame = cap.read()
            if not ok:
                pygame.display.flip()
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            # Face detection + wig — fades in at WIG_FADE_PHASE, holds after
            wig_opacity = 0.0
            if state == "meditation":
                elapsed              = now - meditation_start
                phase, phase_elapsed = phase_at(elapsed)
                if phase >= WIG_FADE_PHASE:
                    wig_opacity = 1.0

            if wig_opacity > 0:
                rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_img)
                if result.face_landmarks:
                    frame = overlay_wig(
                        frame, wig, result.face_landmarks[0], w, h,
                        physics, wig_opacity, width_scale, ring_bottom,
                    )

            screen.blit(frame_to_surface(frame, SW, SH), (0, 0))

            # ── Text overlay
            cx = SW // 2
            cy = SH // 2

            if state == "idle":
                draw_text(screen, font_idle, IDLE_TEXT, cx, SH - 100, IDLE_TEXT_ALPHA)

            elif state == "meditation":
                elapsed              = now - meditation_start
                phase, phase_elapsed = phase_at(elapsed)
                if phase < len(TEXT_PHASES):
                    text, dur = TEXT_PHASES[phase]
                    draw_text(screen, font_body, text, cx, SH - 120,
                              text_alpha(phase_elapsed, dur))

            # Staff HUD — press H to show/hide, invisible to participants
            if show_hud:
                hud = font_idle.render(
                    f"↑↓ anchor:{ring_bottom:.2f}  ←→ scale:{width_scale:.2f}",
                    True, (180, 180, 60))
                hud.set_alpha(220)
                screen.blit(hud, (20, 20))

            pygame.display.flip()

    finally:
        cap.release()
        serial_listener.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
