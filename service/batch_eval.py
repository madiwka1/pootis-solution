"""Прогон полной цепочки (YOLO -> правила -> Qwen) по нескольким видео без HTTP + сводка против разметки.

  python -m service.batch_eval --videos videos --only X-1,X-13,X-15,X-34,X-44,X-5,X-8,X-14,X-19,X-35,X-39,X-48
  python -m service.batch_eval --videos videos              # все .mov в папке

Результат: service/eval_out/<video>.json (полный ответ, с полями _vlm/_rule) и eval_out/summary.csv.
Видео не копируются и не сохраняются.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perception"))

from service import pipeline          # noqa: E402
from model import norm                # noqa: E402  perception/model.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=str(ROOT / "videos"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--verdicts", default=str(ROOT / "open_labels.xlsx"))
    ap.add_argument("--labels", default=str(ROOT / "labels" / "annotations_draft.csv"))
    ap.add_argument("--out", default=str(ROOT / "service" / "eval_out"))
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    vids = sorted(Path(a.videos).glob("*.mov"), key=lambda p: int("".join(ch for ch in p.stem if ch.isdigit()) or 0))
    if a.only:
        want = {norm(x) for x in a.only.split(",")}
        vids = [v for v in vids if norm(v.stem) in want]

    gt = {}
    if Path(a.verdicts).exists():
        x = pd.read_excel(a.verdicts)
        gt = {norm(r.iloc[0]): str(r.iloc[2]).strip() == "нарушение" for _, r in x.iterrows()}
    lab = {}
    if Path(a.labels).exists():
        e = pd.read_csv(a.labels, encoding="utf-8-sig")
        lab = {norm(r.video_id): (r.type_id, float(r.gt_timecode_sec)) for r in e.itertuples()}

    rows = []
    for v in vids:
        vid = norm(v.stem)
        t0 = time.time()
        try:
            res = pipeline.analyze(v, job_id=vid)
        except Exception as ex:                       # одно видео не должно ронять весь прогон
            print(f"{vid}: ошибка {ex}"); rows.append({"video": vid, "error": str(ex)[:200]}); continue
        (out / f"{vid}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        pred = res["verdict"] == "violation"
        g = gt.get(vid)
        res_cell = "?" if g is None else {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}[(int(g), int(pred))]
        ev = res.get("violations", [])
        gt_type, gt_t = lab.get(vid, ("", None))
        hit = any(x.get("_type_id") == gt_type and gt_t is not None and abs(x["timecode_sec"] - gt_t) <= 2 for x in ev)
        rows.append({"video": vid, "gt": {None: "?", True: "violation", False: "clean"}[g], "pred": res["verdict"],
                     "res": res_cell, "gt_type": gt_type, "gt_t": gt_t, "event_hit": hit,
                     "events": "; ".join(f"{x.get('_type_id')}@{x['timecode_sec']}[{x.get('_vlm')}]" for x in ev),
                     "sec": round(time.time() - t0, 1)})
        print(rows[-1], flush=True)

    S = pd.DataFrame(rows)
    S.to_csv(out / "summary.csv", index=False, encoding="utf-8-sig")
    if "res" in S:
        c = S.res.value_counts().to_dict()
        n = sum(c.get(k, 0) for k in ("TP", "FP", "FN", "TN")) or 1
        print(f"\nTP={c.get('TP', 0)} FP={c.get('FP', 0)} FN={c.get('FN', 0)} TN={c.get('TN', 0)}  "
              f"accuracy={(c.get('TP', 0) + c.get('TN', 0)) / n:.3f}   макс. время на видео: {S.sec.max()} с")
    print(f"Сохранено: {out}")


if __name__ == "__main__":
    main()
