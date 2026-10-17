"""Run the VLM over videos and evaluate against open_labels.xlsx.

  python -m vlm.cli run  --videos videos --out preds            [--backend mock] [--only X-1,Х-4]
  python -m vlm.cli eval --pred preds --labels open_labels.xlsx [--tol 2]
"""
import argparse
import glob
import json
import os
import re
import time
import unicodedata

from .analyze import analyze_video
from .config import VLMConfig


def norm_name(s: str) -> str:
    """'X-1' (Latin) and 'Х-1' (Cyrillic) -> 'X-1'."""
    s = unicodedata.normalize("NFC", os.path.splitext(os.path.basename(str(s)))[0]).strip()
    return s.replace("Х", "X").replace("х", "x")


def parse_ts(s):
    if not s:
        return None
    m = re.findall(r"(\d+)\s*мин", str(s))
    sec = re.findall(r"(\d+)\s*сек", str(s))
    return (int(m[0]) * 60 if m else 0) + (int(sec[0]) if sec else 0)


def load_labels(path):
    import openpyxl
    ws = openpyxl.load_workbook(path).active
    lab = {}
    for name, ts, v in list(ws.iter_rows(values_only=True))[1:]:
        if name:
            lab[norm_name(name)] = {"violation": str(v).strip() == "нарушение", "t": parse_ts(ts)}
    return lab


def cmd_run(a):
    cfg = VLMConfig()
    if a.backend:
        cfg.backend = a.backend
    os.makedirs(a.out, exist_ok=True)
    only = {norm_name(x) for x in a.only.split(",")} if a.only else None
    for p in sorted(glob.glob(os.path.join(a.videos, "*.mov"))):
        vid = norm_name(p)
        if only and vid not in only:
            continue
        t0 = time.time()
        res, dbg = analyze_video(p, job_id=vid, cfg=cfg, return_debug=True)
        dt = time.time() - t0
        res["_latency_sec"] = round(dt, 1)
        with open(os.path.join(a.out, f"{vid}.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False)
        with open(os.path.join(a.out, f"{vid}.windows.json"), "w", encoding="utf-8") as f:
            json.dump(dbg, f, ensure_ascii=False, indent=1)
        print(f"{vid:6s} {res['verdict']:13s} {[v['timecode_sec'] for v in res['violations']]} {dt:.1f}s")


def cmd_eval(a):
    lab = load_labels(a.labels)
    tp = fp = fn = tn = 0
    errs, rows = [], []
    for vid, g in sorted(lab.items(), key=lambda kv: int(re.sub(r"\D", "", kv[0]) or 0)):
        p = os.path.join(a.pred, f"{vid}.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p, encoding="utf-8"))
        pv = r["verdict"] == "violation"
        if g["violation"] and pv:
            tp += 1
            d = min(abs(v["timecode_sec"] - g["t"]) for v in r["violations"]) if g["t"] is not None else None
            if d is not None:
                errs.append(d)
        elif g["violation"]:
            fn += 1; d = None
        elif pv:
            fp += 1; d = None
        else:
            tn += 1; d = None
        rows.append((vid, g["violation"], g["t"], pv, [v["timecode_sec"] for v in r["violations"]], d))
    for row in rows:
        mark = "OK " if row[1] == row[3] else ("FN!" if row[1] else "FP ")
        print(f"{mark} {row[0]:6s} gt={'V' if row[1] else '-'}@{row[2]}  pred={'V' if row[3] else '-'}{row[4]}  dt={row[5]}")
    n = tp + fp + fn + tn
    print(f"\nvideos={n}  TP={tp} FP={fp} FN={fn} TN={tn}  acc={(tp+tn)/max(n,1):.3f}")
    if errs:
        within = sum(e <= a.tol for e in errs) / len(errs)
        print(f"timecode |dt|: median={sorted(errs)[len(errs)//2]:.2f}s  within {a.tol}s={within:.2%}")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--videos", default="videos"); r.add_argument("--out", default="preds")
    r.add_argument("--backend", choices=["hf", "mock"]); r.add_argument("--only")
    e = sp.add_parser("eval"); e.add_argument("--pred", default="preds"); e.add_argument("--labels", default="open_labels.xlsx")
    e.add_argument("--tol", type=float, default=2.0)
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_eval(a)


if __name__ == "__main__":
    main()
