"""Кадры с реальным временем (VFR): позиция декодера в мс, а не индекс/fps."""
import cv2
import numpy as np


def video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
    end_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
    cap.release()
    return end_ms / 1000.0 if end_ms > 0 else n / fps


def sample_frames(path: str, fps: float):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {path}")
    out, next_t, step = [], 0.0, 1.0 / fps
    while True:
        ok = cap.grab()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t + 1e-6 >= next_t:
            ok, bgr = cap.retrieve()
            if not ok:
                break
            out.append((round(t, 3), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            next_t += step
    cap.release()
    if not out:
        raise ValueError(f"no frames decoded: {path}")
    return out


def make_windows(frames, window_sec, stride_sec, k):
    if not frames:
        return
    t_last = frames[-1][0]
    t0 = 0.0
    while True:
        t1 = t0 + window_sec
        inside = [f for f in frames if t0 <= f[0] < t1]
        if inside:
            idx = np.linspace(0, len(inside) - 1, num=min(k, len(inside))).round().astype(int)
            yield t0, min(t1, t_last), [inside[i] for i in idx]
        if t1 >= t_last:
            break
        t0 += stride_sec


def encode_jpeg_b64(rgb, quality=80, max_side=848):
    import base64
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1.0:
        rgb = cv2.resize(rgb, (int(w * s), int(h * s)))
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf.tobytes()).decode("ascii")
