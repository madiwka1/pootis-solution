"""YOLO: обучение, оценка, прогон по видео.

  python train_yolo.py train   [--epochs 40] [--imgsz 960] [--from runs/fraud/v2/weights/best.pt]
  python train_yolo.py eval    [--weights ...]
  python train_yolo.py predict [--weights ...] [--videos videos] [--only X-26,X-17] [--fps 5] [--save-video]

Выход predict: data/<video_id>-detections.csv и .jsonl (поля = perception/contract.py DETECTION_FIELDS).
"""
import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import cv2
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

BASE = Path(__file__).resolve().parent
DATA_YAML = BASE / "data.yaml"
RUNS = BASE / "runs" / "fraud"
RUN_NAME = "v2"
DEFAULT_WEIGHTS = RUNS / RUN_NAME / "weights" / "best.pt"
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def video_id(p) -> str:
    """Кириллическая Х -> латинская."""
    s = unicodedata.normalize("NFC", Path(p).stem).strip()
    return s.replace("Х", "X").replace("х", "x")


def runtime_yaml() -> Path:
    cfg = yaml.safe_load(open(DATA_YAML, encoding="utf-8"))
    cfg["path"] = str(BASE)
    out = BASE / "data_runtime.yaml"
    yaml.safe_dump(cfg, open(out, "w", encoding="utf-8"), sort_keys=False, allow_unicode=True)
    return out


def check_classes(model: YOLO):
    """Не дать молча использовать старые веса с другим списком классов."""
    want = yaml.safe_load(open(DATA_YAML, encoding="utf-8"))["names"]
    want = list(want.values()) if isinstance(want, dict) else list(want)
    have = [model.names[i] for i in sorted(model.names)]
    if have != want:
        sys.exit(f"Классы весов не совпадают с data.yaml!\n  веса: {have}\n  data: {want}\n"
                 f"Переобучите: python train_yolo.py train")


def cmd_train(a):
    start = a.init if a.init else "yolo11s.pt"
    if a.init and not Path(a.init).exists():
        sys.exit(f"Нет весов для дообучения: {a.init}")
    print(f"device={DEVICE}  start={start}  imgsz={a.imgsz}  epochs={a.epochs}")
    YOLO(start).train(
        data=str(runtime_yaml()),
        epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=DEVICE,
        project=str(RUNS), name=RUN_NAME, exist_ok=True, seed=0,
        val=True,
        patience=10,
        cache="ram",
        workers=4,
        freeze=a.freeze,  # мало данных
        cos_lr=True, close_mosaic=5,
        # без flipud: камера всегда сверху
        degrees=10, perspective=0.0005, fliplr=0.5, scale=0.5, hsv_v=0.4, mosaic=1.0,
        cls=1.0,
        plots=True,
    )
    print(f"\nГотово: {DEFAULT_WEIGHTS}\nДальше: python train_yolo.py eval")


def cmd_eval(a):
    m = YOLO(str(a.weights)); check_classes(m)
    r = m.val(data=str(runtime_yaml()), imgsz=a.imgsz, device=DEVICE, conf=0.001, verbose=False)
    print(f"\nALL  mAP50={r.box.map50:.3f}  mAP50-95={r.box.map:.3f}")
    for i, c in enumerate(r.box.ap_class_index):
        flag = "  <-- мало находит" if r.box.r[i] < 0.7 else ""
        print(f"  {m.names[c]:17s} mAP50={r.box.ap50[i]:.3f}  P={r.box.p[i]:.3f}  R={r.box.r[i]:.3f}{flag}")
    print("\nГлавное — R (recall) по cash / person / drink / id_card: пропуск нарушения недопустим.")


def predict_video(m: YOLO, path: Path, a, out_dir: Path):
    vid = video_id(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[skip] не открылось: {path}"); return
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if a.save_video:
        writer = cv2.VideoWriter(str(out_dir / f"{vid}-annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    step, next_t, rows, idx, n_proc, t0 = 1.0 / a.fps, 0.0, [], -1, 0, time.time()
    first = True
    while True:
        if not cap.grab():
            break
        idx += 1
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # VFR
        if t + 1e-6 < next_t:
            continue
        next_t += step
        ok, frame = cap.retrieve()
        if not ok:
            break
        # сброс трекера между видео
        res = m.track(frame, persist=not first, conf=a.conf, imgsz=a.imgsz, device=DEVICE,
                      tracker=a.tracker, verbose=False)[0]
        first = False; n_proc += 1
        b = res.boxes
        ids = b.id.int().tolist() if b.id is not None else [-1] * len(b)
        for (x1, y1, x2, y2), c, cf, tid in zip(b.xyxy.tolist(), b.cls.int().tolist(), b.conf.tolist(), ids):
            rows.append({"video_id": vid, "t": round(t, 3), "frame_idx": idx, "track_id": tid,
                         "cls": m.names[c], "cls_id": c, "conf": round(cf, 4),
                         "x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
                         "img_w": W, "img_h": H})
        if writer is not None:
            writer.write(res.plot())
    cap.release()
    if writer is not None:
        writer.release()
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{vid}-detections.csv", index=False)
    with open(out_dir / f"{vid}-detections.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            r2 = {k: r[k] for k in ("video_id", "t", "frame_idx", "track_id", "cls", "cls_id", "conf", "img_w", "img_h")}
            r2["bbox"] = [r["x1"], r["y1"], r["x2"], r["y2"]]
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    cnt = df["cls"].value_counts().to_dict() if len(df) else {}
    print(f"{vid:6s} {n_proc:4d} кадров  {len(df):5d} детекций  {dt:5.1f} с  {cnt}")


def cmd_predict(a):
    m = YOLO(str(a.weights)); check_classes(m)
    out_dir = BASE / "data"; out_dir.mkdir(parents=True, exist_ok=True)
    vids = sorted(Path(a.videos).glob("*.mov"), key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0))
    only = {video_id(x) for x in a.only.split(",")} if a.only else None
    print(f"device={DEVICE}  weights={a.weights}  fps={a.fps}  conf={a.conf}")
    for p in vids:
        if only and video_id(p) not in only:
            continue
        predict_video(m, p, a, out_dir)


def main():
    # без аргументов: train (если нет весов), eval, predict
    if len(sys.argv) == 1:
        steps = ([] if DEFAULT_WEIGHTS.exists() else [["train"]]) + [["eval"], ["predict"]]
        print("Без аргументов: " + " -> ".join(s[0] for s in steps))
        for s in steps:
            sys.argv = [sys.argv[0]] + s
            main()
        return
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--epochs", type=int, default=40); t.add_argument("--imgsz", type=int, default=960)
    t.add_argument("--batch", type=int, default=-1); t.add_argument("--freeze", type=int, default=10)
    t.add_argument("--from", dest="init", default=None, help="дообучить от прошлых весов (быстрее)")
    e = sp.add_parser("eval")
    e.add_argument("--weights", default=str(DEFAULT_WEIGHTS)); e.add_argument("--imgsz", type=int, default=960)
    p = sp.add_parser("predict")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS)); p.add_argument("--videos", default=str(BASE / "videos"))
    p.add_argument("--only", default=None); p.add_argument("--fps", type=float, default=5.0)
    p.add_argument("--conf", type=float, default=0.15, help="низкий порог: лучше лишнее, чем пропуск")
    p.add_argument("--imgsz", type=int, default=960); p.add_argument("--tracker", default="bytetrack.yaml")
    p.add_argument("--save-video", action="store_true")
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
