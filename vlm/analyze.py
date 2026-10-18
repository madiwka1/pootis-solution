"""Запасной путь без YOLO: VLM по скользящим окнам -> ответ API."""
import json
import time

from .config import VLMConfig
from .frames import sample_frames, make_windows, encode_jpeg_b64
from .model import get_vlm
from .prompts import window_prompt, parse_json


def _load_taxonomy(cfg):
    with open(cfg.taxonomy_path, encoding="utf-8") as f:
        return json.load(f)


def scan_windows(path: str, cfg: VLMConfig):
    tax = _load_taxonomy(cfg)
    vlm = get_vlm(cfg)
    fps = max(1.0, cfg.frames_per_window / cfg.window_sec)
    frames = sample_frames(path, fps=fps)
    windows = list(make_windows(frames, cfg.window_sec, cfg.stride_sec, cfg.frames_per_window))
    started, answers = time.time(), []
    for i in range(0, len(windows), cfg.batch_size):
        if time.time() - started > cfg.time_budget_sec:
            break
        chunk = windows[i:i + cfg.batch_size]
        batch = [([f for _, f in w[2]], window_prompt(tax, [t for t, _ in w[2]])) for w in chunk]
        for (t0, t1, wf), raw in zip(chunk, vlm.generate(batch)):
            a = parse_json(raw)
            fi = a.get("frame_index", -1)
            fi = fi if isinstance(fi, int) and 0 <= fi < len(wf) else len(wf) // 2
            answers.append({"t_start": t0, "t_end": t1, "type": str(a.get("type", "NONE")).upper(),
                            "confidence": float(a.get("confidence", 0) or 0),
                            "t_peak": wf[fi][0], "description": a.get("description_ru", ""), "raw": raw})
    return frames, answers, tax


def merge_events(answers, cfg: VLMConfig):
    hits = sorted([a for a in answers if a["type"] != "NONE" and a["confidence"] >= cfg.conf_threshold],
                  key=lambda a: a["t_start"])
    events = []
    for a in hits:
        last = events[-1] if events else None
        if last and last["type"] == a["type"] and a["t_start"] - last["t_end"] <= cfg.merge_gap_sec:
            last["t_end"] = max(last["t_end"], a["t_end"])
            last["n"] += 1
            if a["confidence"] > last["confidence"]:
                last.update(confidence=a["confidence"], t_peak=a["t_peak"], description=a["description"])
        else:
            events.append(dict(a, n=1))
    events.sort(key=lambda e: -e["confidence"])
    return sorted(events[:cfg.max_events], key=lambda e: e["t_peak"])


def _nearest(frames, t):
    return min(frames, key=lambda f: abs(f[0] - t))


def build_result(job_id, frames, events, tax, cfg: VLMConfig):
    by_id = {t["id"]: t for t in tax["types"]}
    by_id[tax["other"]["id"]] = tax["other"]
    out = []
    for e in events:
        rule = by_id.get(e["type"], tax["other"])
        times = [e["t_peak"], e["t_peak"] - 1.0, e["t_peak"] + 1.0][:cfg.evidence_frames]
        ev = []
        for t in times:
            ft, img = _nearest(frames, t)
            ev.append({"timestamp_sec": round(ft, 2), "image_base64": encode_jpeg_b64(img, cfg.jpeg_quality)})
        desc = e["description"] or rule["name_ru"]
        if rule.get("doc"):
            desc += f" Возможное нарушение: {rule['doc']}, {rule['clause']}."
        out.append({"type": rule["name_ru"], "timecode_sec": round(e["t_peak"], 2), "description": desc,
                    "evidence_frames": ev,
                    "type_id": e["type"], "confidence": round(e["confidence"], 3),
                    "rule": {k: rule.get(k) for k in ("doc", "clause", "url")}})
    return {"job_id": job_id, "status": "done", "verdict": "violation" if out else "no_violation", "violations": out}


def analyze_video(path: str, job_id: str = "local", cfg: VLMConfig = None, return_debug=False):
    cfg = cfg or VLMConfig()
    frames, answers, tax = scan_windows(path, cfg)
    events = merge_events(answers, cfg)
    res = build_result(job_id, frames, events, tax, cfg)
    return (res, answers) if return_debug else res
