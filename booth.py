import sys
import os
import subprocess
import time
import math
from pathlib import Path

import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import pygame
from PIL import Image

# ── Config ────────────────────────────────────────────────────────────────────
WINDOW_W, WINDOW_H = 1024, 768
PREVIEW_W, PREVIEW_H = 800, 600
WIG_PATH = "wig.png"          # transparent PNG — swap for any wig asset
SAVE_DIR = Path("captures")
COUNTDOWN_SECS = 3
PRINTER_NAME = None            # None = system default; set to "DNP_DS620" etc.

# How wide the wig renders relative to detected face width.
WIG_WIDTH_SCALE = 1.5

# Where the ring-bottom sits in the PNG as a fraction of PNG height.
# The ring base is ~55% down; pigtails fill the lower 45%.
# If the wig floats too high or low, adjust this number.
WIG_RING_BOTTOM = 0.55

FACE_LANDMARKER_MODEL = "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
# ─────────────────────────────────────────────────────────────────────────────

FOREHEAD_TOP = 10       # crown of forehead
LEFT_TEMPLE  = 234      # left side of face
RIGHT_TEMPLE = 454      # right side of face


class WigPhysics:
    """Smoothed head position and tilt angle for stable wig placement."""
    def __init__(self):
        self.x = self.y = self.angle = None

    def update(self, x: float, y: float, angle: float) -> None:
        if self.x is None:
            self.x, self.y, self.angle = float(x), float(y), float(angle)
            return
        self.x     += (x     - self.x)     * 0.25
        self.y     += (y     - self.y)     * 0.25
        self.angle += (angle - self.angle) * 0.20


def ensure_model() -> str:
    if not os.path.exists(FACE_LANDMARKER_MODEL):
        print(f"Downloading face landmarker model…")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_MODEL)
        print("Done.")
    return FACE_LANDMARKER_MODEL


def load_wig(path: str) -> np.ndarray | None:
    if not os.path.exists(path):
        print(f"[warn] wig not found at {path} — running without overlay")
        return None
    img = Image.open(path).convert("RGBA")
    # Crop to the bounding box of non-transparent pixels so anchor math isn't
    # thrown off by empty padding in the PNG.
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    return np.array(img)  # H x W x 4


def overlay_wig(frame_bgr: np.ndarray, wig_rgba: np.ndarray,
                landmarks, img_w: int, img_h: int,
                physics: WigPhysics,
                width_scale: float = WIG_WIDTH_SCALE,
                ring_bottom: float = WIG_RING_BOTTOM) -> np.ndarray:
    lm = landmarks

    raw_x = lm[FOREHEAD_TOP].x * img_w
    raw_y = lm[FOREHEAD_TOP].y * img_h
    lx    = lm[LEFT_TEMPLE].x  * img_w;  ly = lm[LEFT_TEMPLE].y  * img_h
    rx    = lm[RIGHT_TEMPLE].x * img_w;  ry = lm[RIGHT_TEMPLE].y * img_h

    face_width = abs(rx - lx)
    if face_width < 10:
        return frame_bgr

    # Smooth position and head-tilt angle
    physics.update(raw_x, raw_y, math.atan2(ry - ly, rx - lx))
    px, py, pangle = physics.x, physics.y, physics.angle

    wig_w = int(face_width * width_scale)
    wig_h = int(wig_rgba.shape[0] * wig_w / wig_rgba.shape[1])
    if wig_w < 2 or wig_h < 2:
        return frame_bgr

    wig_resized = cv2.resize(wig_rgba, (wig_w, wig_h), interpolation=cv2.INTER_AREA)

    # Anchor in wig-image coords (where the forehead landmark should land)
    acx, acy = wig_w / 2.0, wig_h * ring_bottom
    cos_a, sin_a = math.cos(pangle), math.sin(pangle)

    # Affine: rotate wig around its anchor, translate anchor to (px, py)
    M = np.float32([
        [cos_a, -sin_a, px - cos_a * acx + sin_a * acy],
        [sin_a,  cos_a, py - sin_a * acx - cos_a * acy],
    ])
    warped = cv2.warpAffine(wig_resized, M, (img_w, img_h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(0, 0, 0, 0))

    alpha   = warped[:, :, 3:4].astype(np.float32) / 255.0
    wig_bgr = cv2.cvtColor(warped[:, :, :3], cv2.COLOR_RGB2BGR)
    return (wig_bgr.astype(np.float32) * alpha +
            frame_bgr.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def frame_to_surface(frame_bgr: np.ndarray) -> pygame.Surface:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (PREVIEW_W, PREVIEW_H))
    return pygame.surfarray.make_surface(rgb.swapaxes(0, 1))


def save_capture(frame_bgr: np.ndarray) -> Path:
    SAVE_DIR.mkdir(exist_ok=True)
    filename = SAVE_DIR / f"booth_{int(time.time())}.jpg"
    cv2.imwrite(str(filename), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return filename


def print_photo(path: Path) -> None:
    if sys.platform == "linux":
        cmd = ["lpr"]
        if PRINTER_NAME:
            cmd += ["-P", PRINTER_NAME]
        cmd.append(str(path))
        subprocess.run(cmd)
    elif sys.platform == "darwin":
        cmd = ["lpr", str(path)]
        if PRINTER_NAME:
            cmd = ["lpr", "-P", PRINTER_NAME, str(path)]
        subprocess.run(cmd)
    else:
        os.startfile(str(path), "print")  # Windows fallback


def draw_button(surface: pygame.Surface, rect: pygame.Rect,
                text: str, font: pygame.font.Font,
                color=(60, 180, 80), hover=False) -> None:
    c = tuple(min(v + 30, 255) for v in color) if hover else color
    pygame.draw.rect(surface, c, rect, border_radius=12)
    label = font.render(text, True, (255, 255, 255))
    surface.blit(label, label.get_rect(center=rect.center))


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Wig Booth")
    clock = pygame.time.Clock()
    font_lg = pygame.font.SysFont("sans", 80, bold=True)
    font_md = pygame.font.SysFont("sans", 36)
    font_sm = pygame.font.SysFont("sans", 24)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    wig         = load_wig(WIG_PATH)
    wig_physics = WigPhysics()
    fl_options = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        running_mode=mp_vision.RunningMode.IMAGE,
    )
    face_landmarker = mp_vision.FaceLandmarker.create_from_options(fl_options)

    btn_capture = pygame.Rect(WINDOW_W // 2 - 120, PREVIEW_H + 90, 240, 60)
    btn_print   = pygame.Rect(WINDOW_W // 2 - 120, PREVIEW_H + 90, 240, 60)

    state = "preview"   # preview | countdown | captured
    countdown_start = 0.0
    last_frame_bgr = None
    captured_surface = None
    captured_path = None
    status_msg = ""

    # Live-tuning state (arrow keys adjust while running)
    ring_bottom  = WIG_RING_BOTTOM
    width_scale  = WIG_WIDTH_SCALE

    while True:
        dt = clock.tick(30)
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    cap.release()
                    pygame.quit()
                    sys.exit()
                elif event.key == pygame.K_UP:
                    ring_bottom = round(min(ring_bottom + 0.02, 0.99), 2)
                    print(f"WIG_RING_BOTTOM = {ring_bottom}")
                elif event.key == pygame.K_DOWN:
                    ring_bottom = round(max(ring_bottom - 0.02, 0.01), 2)
                    print(f"WIG_RING_BOTTOM = {ring_bottom}")
                elif event.key == pygame.K_RIGHT:
                    width_scale = round(min(width_scale + 0.05, 3.0), 2)
                    print(f"WIG_WIDTH_SCALE = {width_scale}")
                elif event.key == pygame.K_LEFT:
                    width_scale = round(max(width_scale - 0.05, 0.3), 2)
                    print(f"WIG_WIDTH_SCALE = {width_scale}")

            if event.type == pygame.MOUSEBUTTONDOWN:
                if state == "preview" and btn_capture.collidepoint(mouse_pos):
                    state = "countdown"
                    countdown_start = time.time()

                elif state == "captured":
                    if btn_print.collidepoint(mouse_pos) and captured_path:
                        print_photo(captured_path)
                        status_msg = "Sent to printer!"
                    # Tap anywhere else to retake
                    elif not btn_print.collidepoint(mouse_pos):
                        state = "preview"
                        status_msg = ""

        screen.fill((20, 20, 20))

        # ── PREVIEW / COUNTDOWN ───────────────────────────────────────────────
        if state in ("preview", "countdown"):
            ok, frame = cap.read()
            if ok:
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                if wig is not None:
                    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = face_landmarker.detect(mp_img)
                    if result.face_landmarks:
                        frame = overlay_wig(frame, wig,
                                            result.face_landmarks[0], w, h,
                                            wig_physics,
                                            width_scale=width_scale,
                                            ring_bottom=ring_bottom)

                last_frame_bgr = frame.copy()
                surf = frame_to_surface(frame)
                screen.blit(surf, ((WINDOW_W - PREVIEW_W) // 2, 10))

            # Countdown overlay
            if state == "countdown":
                elapsed = time.time() - countdown_start
                remaining = COUNTDOWN_SECS - int(elapsed)
                if remaining > 0:
                    num = font_lg.render(str(remaining), True, (255, 220, 0))
                    screen.blit(num, num.get_rect(
                        center=(WINDOW_W // 2, PREVIEW_H // 2 + 10)))
                else:
                    # Take the shot
                    if last_frame_bgr is not None:
                        captured_path = save_capture(last_frame_bgr)
                        captured_surface = frame_to_surface(last_frame_bgr)
                    state = "captured"
                    status_msg = ""
            else:
                hover = btn_capture.collidepoint(mouse_pos)
                draw_button(screen, btn_capture, "TAKE PHOTO", font_md, hover=hover)

            hud = font_sm.render(
                f"↑↓ anchor:{ring_bottom:.2f}  ←→ scale:{width_scale:.2f}",
                True, (200, 200, 80))
            screen.blit(hud, (10, PREVIEW_H + 15))

        # ── CAPTURED ──────────────────────────────────────────────────────────
        elif state == "captured":
            if captured_surface:
                screen.blit(captured_surface, ((WINDOW_W - PREVIEW_W) // 2, 10))

            hover = btn_print.collidepoint(mouse_pos)
            draw_button(screen, btn_print, "PRINT", font_md, color=(30, 120, 200), hover=hover)

            retake = font_sm.render("tap anywhere else to retake", True, (160, 160, 160))
            screen.blit(retake, retake.get_rect(center=(WINDOW_W // 2, PREVIEW_H + 170)))

            if status_msg:
                msg = font_md.render(status_msg, True, (100, 220, 100))
                screen.blit(msg, msg.get_rect(center=(WINDOW_W // 2, PREVIEW_H + 50)))

        pygame.display.flip()

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()
