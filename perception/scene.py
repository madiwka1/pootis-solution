"""Открытая дверь кассы по пикселям: зона двери против первого кадра, люди вырезаны по рамкам YOLO.
Открытая створка меняет ~30% пикселей зоны, закрытая < 10%.
"""
import cv2
import numpy as np
import pandas as pd

DOOR_REGION = (0.0, 0.0, 0.40, 0.50)   # доли кадра
WALL_REGION = (0.62, 0.05, 0.95, 0.40)  # меняется и стена - это свет, не дверь
PIX_THR = 25


def _gray(f):
    return cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (9, 9), 0).astype(np.int16)


def door_change(frames, det: pd.DataFrame | None, step_s: float = 1.0):
    """-> секунда: доля изменившихся пикселей зоны двери."""
    if not frames:
        return pd.Series(dtype=float)
    H, W = frames[0][1].shape[:2]
    x0, y0, x1, y1 = DOOR_REGION
    ys, xs = slice(int(y0 * H), int(y1 * H)), slice(int(x0 * W), int(x1 * W))
    a0, b0, a1, b1 = WALL_REGION
    wy, wx = slice(int(b0 * H), int(b1 * H)), slice(int(a0 * W), int(a1 * W))
    ref = _gray(frames[0][1])
    P = det[det.cls == "person"] if det is not None and len(det) else None
    out, nxt = {}, 0.0
    for t, f, _ in frames:
        if t + 1e-6 < nxt:
            continue
        nxt = t + step_s
        m = np.ones((H, W), bool)
        if P is not None:
            for r in P[(P.t - t).abs() < 0.11].itertuples():
                m[max(0, int(r.y1) - 10):int(r.y2) + 10, max(0, int(r.x1) - 10):int(r.x2) + 10] = False
        diff = np.abs(_gray(f) - ref)
        d = diff[ys, xs][m[ys, xs]]
        w = diff[wy, wx][m[wy, wx]]
        wall = float(np.mean(w > PIX_THR)) if w.size > 500 else 0.0
        out[int(t)] = np.nan if wall > 0.15 else (float(np.mean(d > PIX_THR)) if d.size > 500 else np.nan)
    return pd.Series(out).sort_index()


def door_events(change: pd.Series, n_person: pd.Series | None = None, open_thr=0.25, min_open_s=6):
    """Таймкод: кто-то вошёл при открытой двери, иначе начало + 2 с."""
    s = change.fillna(0) >= open_thr
    ev, start = [], None
    idx = list(s.index)
    for i, sec in enumerate(idx + [None]):
        on = sec is not None and s[sec]
        if on and start is None:
            start = sec
        if not on and start is not None:
            end = idx[i - 1]
            if end - start + 1 >= min_open_s:
                t = start + 2
                if n_person is not None and len(n_person):
                    w = n_person.loc[start:end]
                    rise = [k for k in w.index[1:] if w[k] > w[k - 1] and w[k] >= 2] if len(w) > 1 else []
                    if rise:
                        t = rise[0]
                ev.append(("DOOR_LEFT_OPEN", start, end, t))
            start = None
    return ev


def main():
    """python scene.py --videos ../videos --det data -> data/scene_events.csv"""
    import argparse
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import model as M
    from service.pipeline import read_frames
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(Path(__file__).resolve().parents[1] / "videos"))
    ap.add_argument("--det", default=str(Path(__file__).resolve().parent / "data"))
    a = ap.parse_args()
    dets = {M.norm(f.stem): f for f in Path(a.det).glob("*-detections.csv")}
    rows = []
    for v in sorted(Path(a.videos).glob("*.mov")):
        vid = M.norm(v.stem)
        det = M.load_detections(dets[vid]) if vid in dets else None
        frames = read_frames(v, 2.0)
        n_person = M.features(det).n_person if det is not None and len(det) else None
        for ty, s, e, t in door_events(door_change(frames, det), n_person):
            rows.append((vid, ty, s, e, t, "strong"))
        print(vid, [r for r in rows if r[0] == vid], flush=True)
    out = Path(a.det) / "scene_events.csv"
    pd.DataFrame(rows, columns=["vid", "type_id", "start", "end", "t", "tier"]).to_csv(out, index=False)
    print("->", out)


if __name__ == "__main__":
    main()
