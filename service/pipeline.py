"""Конвейер анализа одного видео -> результат строго по схеме API.

  YOLO (5 к/с, трекинг) -> правила perception/model.py -> кандидаты
  -> Qwen-VL (Ollama) подтверждает и описывает -> вердикт, таймкод, 1–2 кадра, пункт закона из taxonomy.json

Если весов YOLO нет — запасной путь: VLM по скользящим окнам (vlm.analyze_video).
Если VLM недоступна — события правил возвращаются без проверки (лучше лишнее, чем пропуск).
"""
import base64
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perception"))

from vlm.config import VLMConfig                       # noqa: E402
from vlm.prompts import verify_prompt, left_checklist_prompt, manual_checklist_prompt, parse_json  # noqa: E402

TAX = json.load(open(ROOT / "vlm" / "taxonomy.json", encoding="utf-8"))
TYPES = {t["id"]: t for t in TAX["types"]}
OTHER = TAX["other"]

YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", str(ROOT / "weights" / "best.pt"))
YOLO_FPS = float(os.getenv("YOLO_FPS", "5"))
MAX_FRAMES = int(os.getenv("MAX_FRAMES", "900"))
MAX_SIDE = int(os.getenv("MAX_SIDE", "960"))                     # длинная сторона кадра в памяти                 # ~30-60 с YOLO на 3050; 3 мин видео = 5 к/с
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.15"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "960"))
YOLO_MODE = os.getenv("YOLO_MODE", "predict")                    # predict | track
DEADLINE_SEC = float(os.getenv("DEADLINE_SEC", "240"))          # лимит клиента 300 с, оставляем запас
VERIFY_DROP_CONF = float(os.getenv("VERIFY_DROP_CONF", "0.6"))  # VLM отклонила с такой уверенностью -> убрать
MAX_EVENTS = int(os.getenv("MAX_EVENTS", "5"))
MAX_WEAK_CHECKS = int(os.getenv("MAX_WEAK_CHECKS", "3"))       # вызовов Qwen на слабые события за видео (время!)
SKIP_TYPES = set(os.getenv("SKIP_TYPES", "CASH_MIXED").split(","))  # смешение пачек: Qwen подтверждает почти всё, + время
VLM_CALL_S = float(os.getenv("VLM_CALL_S", "35"))
VERIFY_FRAMES = int(os.getenv("VERIFY_FRAMES", "3"))               # запас на один вызов Qwen перед дедлайном
RULES_CFG = ROOT / "perception" / "data" / "best_cfg.json"      # из `python model.py --tune`, если есть

COLORS = {"person": (255, 80, 0), "cash": (0, 0, 255), "cash_bag": (0, 160, 0), "phone": (255, 0, 255),
          "safe_open": (0, 200, 255), "safe_closed": (160, 160, 160), "counting_machine": (255, 140, 0),
          "handbag": (0, 255, 0), "drink": (0, 140, 255), "id_card": (255, 255, 0), "folder": (128, 0, 128)}

_yolo = None


def _log(job, msg):
    print(f"[{job}] {msg}", flush=True)


# ------------------------------------------------------------------ видео
def read_frames(path, fps):
    """Кадры с реальным временем (PTS, переменный FPS) с шагом 1/fps. -> [(t, bgr, frame_idx)]"""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("не удалось открыть видео")
    out, nxt, idx = [], 0.0, -1
    while True:
        if not cap.grab():
            break
        idx += 1
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t + 1e-6 < nxt:
            continue
        ok, f = cap.retrieve()
        if not ok:
            break
        h, w = f.shape[:2]
        if max(h, w) > MAX_SIDE:                       # 4K/1080p -> экономим память; наши ролики 848 px не трогаем
            k = MAX_SIDE / max(h, w)
            f = cv2.resize(f, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        out.append((round(t, 3), f, idx)); nxt += 1.0 / fps
    cap.release()
    if not out:
        raise ValueError("в видео нет кадров")
    return out


def nearest(frames, t):
    return min(frames, key=lambda x: abs(x[0] - t))


def jpeg_b64(bgr, q=85):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()


# ------------------------------------------------------------------ YOLO
def yolo_available():
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return Path(YOLO_WEIGHTS).exists()


def run_yolo(frames, vid):
    global _yolo
    from ultralytics import YOLO
    import torch
    if _yolo is None:
        _yolo = YOLO(YOLO_WEIGHTS)
        need = {"person", "cash", "safe_open", "safe_closed", "counting_machine"}
        if not need <= set(_yolo.names.values()):            # базовые веса COCO (yolo26n.pt, yolo11s.pt) — не наши
            _yolo = None
            raise RuntimeError(f"{YOLO_WEIGHTS}: в весах нет классов {sorted(need)} — это не обученная модель "
                               f"(нужен runs/.../weights/best.pt после train_yolo.py)")
    dev = 0 if torch.cuda.is_available() else "cpu"
    rows, first = [], True
    for t, f, idx in frames:
        if YOLO_MODE == "track":
            r = _yolo.track(f, persist=not first, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, device=dev,
                            tracker="bytetrack.yaml", verbose=False)[0]
        else:   # трекинг на 5 к/с теряет слабые рамки денег (ByteTrack не связывает их между редкими кадрами);
                # правила track_id не используют — берём все рамки >= YOLO_CONF
            r = _yolo.predict(f, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, device=dev, verbose=False)[0]
        first = False
        b = r.boxes
        ids = b.id.int().tolist() if b.id is not None else [-1] * len(b)
        for (x1, y1, x2, y2), c, cf, tid in zip(b.xyxy.tolist(), b.cls.int().tolist(), b.conf.tolist(), ids):
            rows.append(dict(video_id=vid, t=t, frame_idx=idx, track_id=tid, cls=_yolo.names[c], cls_id=c,
                             conf=cf, x1=x1, y1=y1, x2=x2, y2=y2, img_w=f.shape[1], img_h=f.shape[0]))
    try:                       # освободить VRAM перед VLM (4 ГБ: вместе не помещаются)
        torch.cuda.empty_cache()
    except Exception:
        pass
    return pd.DataFrame(rows)


def draw(bgr, det, t, classes=None):
    img = bgr.copy()
    if det is None or det.empty:
        return img
    d = det[(det.t - t).abs() < 0.11]
    for _, r in d.iterrows():
        if r.cls == "person" or (classes and r.cls not in classes):
            continue
        c = COLORS.get(r.cls, (0, 255, 255))
        cv2.rectangle(img, (int(r.x1), int(r.y1)), (int(r.x2), int(r.y2)), c, 2)
        cv2.putText(img, r.cls, (int(r.x1), max(12, int(r.y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    return img


# ------------------------------------------------------------------ VLM
def get_vlm_safe(job):
    cfg = VLMConfig()
    if cfg.backend == "none":
        return None, cfg
    try:
        from vlm.model import get_vlm
        return get_vlm(cfg), cfg
    except Exception as e:
        _log(job, f"VLM недоступна ({e}) — события правил без проверки")
        return None, cfg


def desk_crop(bgr, anchor, side=448):
    """Крупный план стола кассира (ноутбук, сейф, столешница) вокруг точки рабочего места."""
    H, W = bgr.shape[:2]
    cx, cy, h = anchor
    x0, x1 = int(max(0, cx - 1.5 * h)), int(min(W, cx + 1.5 * h))
    y0, y1 = int(max(0, cy - 1.0 * h)), int(min(H, cy + 0.9 * h))
    c = bgr[y0:y1, x0:x1]
    if c.size == 0:
        return bgr
    k = side / max(c.shape[:2])
    return cv2.resize(c, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)


def verify_left(vlm, frames, ev, anchor):
    """LEFT_UNATTENDED: кадры ПОСЛЕ ухода, крупно стол; решение по чек-листу, а не по общему «нарушение?»."""
    end = frames[-1][0]
    ts = [min(end, ev["t"] + d) for d in (2.0, 3.5, 5.0)]
    sel = [nearest(frames, t) for t in ts]
    full = sel[1][1]
    k = 448 / max(full.shape[:2])
    rgb = [cv2.cvtColor(cv2.resize(full, None, fx=k, fy=k), cv2.COLOR_BGR2RGB)]          # [0] вся комната
    rgb += [cv2.cvtColor(desk_crop(f[1], anchor), cv2.COLOR_BGR2RGB) for f in sel[1:]]   # [1..] крупно стол
    sel = [sel[1]] + sel[1:]
    a = parse_json(vlm.generate([(rgb, left_checklist_prompt([f[0] for f in sel]))])[0])
    screen = str(a.get("laptop_screen", "")).lower()
    left_things = screen == "open_program" or a.get("key_in_safe") is True or a.get("valuables_on_desk") is True
    # кассир на месте (правило ошиблось: ушёл клиент, а не кассир — X-5, X-12, X-19) -> не нарушение
    a["confirmed"] = left_things and a.get("person_at_desk") is not True
    a["confidence"] = 0.8 if a["confirmed"] else 0.7
    fi = a.get("frame_index", -1)
    a["t_best"] = sel[fi][0] if isinstance(fi, int) and 0 <= fi < len(sel) else ev["t"]
    _log("vlm", f"LEFT check @{ev['t']}: at_desk={a.get('person_at_desk')} screen={screen} key={a.get('key_in_safe')} "
              f"valuables={a.get('valuables_on_desk')}"
         + (f" | ответ модели: {str(a.get('error') or a.get('_raw'))[:200]}" if not screen else ""))
    return a


def verify(vlm, cfg, frames, ev, anchor=None):
    t0, t1 = max(0.0, ev["t"] - 1.5), ev["t"] + 1.5
    win = [f for f in frames if t0 <= f[0] <= t1] or [nearest(frames, ev["t"])]
    idx = np.linspace(0, len(win) - 1, num=min(VERIFY_FRAMES, len(win))).round().astype(int)
    sel = [win[i] for i in idx]
    # крупный план рабочего места: руки, деньги, машинка видны лучше, а токенов меньше -> быстрее
    rgb = [cv2.cvtColor(desk_crop(f[1], anchor) if anchor is not None else f[1], cv2.COLOR_BGR2RGB) for f in sel]
    if ev["type_id"] == "MANUAL_COUNT_NO_CHECK":
        raw = vlm.generate([(rgb, manual_checklist_prompt([f[0] for f in sel]))])[0]
        a = parse_json(raw)
        if "confirmed" not in a and ("cash_in_hands" in a or "counting_by_hand" in a):
            # все три ответа должны сходиться: деньги в руках И пересчёт пальцами И без машинки
            # («counting=True при hands=False» — противоречие, на чистых X-14, X-39 это давало ложные тревоги)
            a["confirmed"] = (a.get("cash_in_hands") is True and a.get("counting_by_hand") is True
                              and a.get("machine_used") is not True)
            a["confidence"] = 0.8
        _log("vlm", f"MANUAL check @{ev['t']}: hands={a.get('cash_in_hands')} counting={a.get('counting_by_hand')} "
                    f"machine={a.get('machine_used')}")
    else:
        raw = vlm.generate([(rgb, verify_prompt(TAX, ev["type_id"], [f[0] for f in sel]))])[0]
        a = parse_json(raw)
    _log("vlm", f"{ev['type_id']} @{ev['t']}: confirmed={a.get('confirmed')} conf={a.get('confidence')}"
         + (f" | ответ модели: {raw[:200]}" if "confirmed" not in a or a.get("error") else ""))
    if "confirmed" not in a:                               # mock / старый формат ответа
        a["confirmed"] = str(a.get("type", "NONE")).upper() not in ("NONE", "ERROR")
    fi = a.get("frame_index", -1)
    if isinstance(fi, int) and 0 <= fi < len(sel):
        a["t_best"] = sel[fi][0]
    return a


# ------------------------------------------------------------------ сборка ответа
def describe(ev):
    rule = TYPES.get(ev["type_id"], OTHER)
    text = (ev.get("vlm_desc") or "").strip() or f"Обнаружены признаки: {rule['name_ru'].lower()}."
    if rule.get("doc"):
        text += f" Может указывать на нарушение: {rule['doc']}, {rule['clause']}."
    else:
        text += " Требует проверки человеком."
    return text


def build(job_id, frames, det, events):
    out = []
    events = sorted(events, key=lambda e: e["t"])
    for ev in events[:MAX_EVENTS]:
        rule = TYPES.get(ev["type_id"], OTHER)
        t = ev.get("t_best", ev["t"])
        ft, f, _ = nearest(frames, t)
        ev_frames = [{"timestamp_sec": round(ft, 2), "image_base64": jpeg_b64(draw(f, det, ft))}]
        ft2, f2, _ = nearest(frames, t + 1.0)
        if abs(ft2 - ft) > 0.4:
            ev_frames.append({"timestamp_sec": round(ft2, 2), "image_base64": jpeg_b64(draw(f2, det, ft2))})
        out.append({"type": rule["name_ru"], "timecode_sec": round(ft, 2), "description": describe(ev),
                    "evidence_frames": ev_frames,
                    "_type_id": ev["type_id"], "_rule": {k: rule.get(k) for k in ("doc", "clause", "quote", "url")},
                    "_source": ev.get("source", "rules"), "_vlm": ev.get("vlm_status", "n/a")})
    return {"job_id": job_id, "status": "done", "verdict": "violation" if out else "no_violation", "violations": out}


def dedup(evs, gap=3.0):
    evs = sorted(evs, key=lambda e: e["t"]); res = []
    for e in evs:
        if res and res[-1]["type_id"] == e["type_id"] and e["t"] - res[-1]["t"] <= gap:
            continue
        res.append(e)
    return res


def analyze(path, job_id="local", submitted_at=None):
    """Главная функция: путь к видео -> dict по схеме API (+ поля с '_' для интерфейса).
    submitted_at — время загрузки: клиент ждёт не больше 5 минут С МОМЕНТА ЗАГРУЗКИ, поэтому ожидание в очереди
    тоже съедает бюджет. Если очередь длинная, Qwen проверяет меньше событий (сильные правила остаются)."""
    start = submitted_at or time.time()
    vid = Path(path).stem
    # длинное видео: снижаем частоту кадров, чтобы YOLO + Qwen уложились в 5 минут (не больше MAX_FRAMES кадров)
    cap = cv2.VideoCapture(str(path))
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)) if cap.isOpened() else 0
    cap.release()
    fps = min(YOLO_FPS, MAX_FRAMES / dur) if dur > 0 else YOLO_FPS
    if fps < YOLO_FPS:
        _log(job_id, f"длинное видео {dur:.0f} с -> {fps:.2f} к/с")
    frames = read_frames(path, fps)
    _log(job_id, f"{len(frames)} кадров, {frames[-1][0]:.1f} с")
    det, events = None, []

    if yolo_available():
        import model as rules                                   # perception/model.py
        cache = Path(os.getenv("DET_CACHE", "")) / f"{vid}-{YOLO_MODE}-detections.csv" if os.getenv("DET_CACHE") else None
        if cache is not None and cache.exists() and cache.stat().st_mtime > Path(YOLO_WEIGHTS).stat().st_mtime:
            det = pd.read_csv(cache)                            # повторный прогон: YOLO не пересчитываем
            _log(job_id, f"детекции из кэша {cache.name}")
        else:
            det = run_yolo(frames, vid)
            if cache is not None:
                cache.parent.mkdir(parents=True, exist_ok=True); det.to_csv(cache, index=False)
        if os.getenv("DUMP_DET"):                               # отладка: сохранить детекции сервиса для сравнения
            Path(os.getenv("DUMP_DET")).mkdir(parents=True, exist_ok=True)
            det.to_csv(Path(os.getenv("DUMP_DET")) / f"{vid}-service-detections.csv", index=False)
        cfg = dict(rules.CFG)
        if RULES_CFG.exists():
            cfg.update(json.load(open(RULES_CFG)))
        ev = rules.detect(det, cfg) if not det.empty else pd.DataFrame(columns=["type_id", "start", "end", "t"])
        events = [{"type_id": r.type_id, "t": float(r.t), "source": "yolo+rules",
                   "tier": getattr(r, "tier", "strong")} for r in ev.itertuples()]
        try:                                                    # открытая дверь — по пикселям (perception/scene.py)
            import scene
            n_person = rules.features(det, cfg).n_person if not det.empty else None
            for ty, s, e, t in scene.door_events(scene.door_change(frames, det), n_person):
                events.append({"type_id": ty, "t": float(t), "source": "scene", "tier": "strong"})
        except Exception as ex:
            _log(job_id, f"scene: {ex}")
        _log(job_id, f"YOLO {len(det)} детекций, правил сработало {len(events)} ({time.time() - start:.0f} с)")
    else:
        _log(job_id, "весов YOLO нет — запасной путь: VLM по окнам")
        from vlm.analyze import scan_windows, merge_events
        vcfg = VLMConfig(); vcfg.time_budget_sec = DEADLINE_SEC - 20
        try:
            _, answers, _ = scan_windows(str(path), vcfg)
            for e in merge_events(answers, vcfg):
                events.append({"type_id": e["type"] if e["type"] in TYPES else OTHER["id"], "t": float(e["t_peak"]),
                               "vlm_desc": e["description"], "source": "vlm-windows", "vlm_status": "window"})
        except Exception as e:
            _log(job_id, f"VLM по окнам не сработала: {e}")

    events = [e for e in events if e["type_id"] not in SKIP_TYPES]
    events = dedup(events, gap=10.0)
    if events and events[0].get("source") in ("yolo+rules", "scene"):
        # порядок проверки: сильные -> «ушла» -> ручной пересчёт -> прочее; слабых не больше MAX_WEAK_CHECKS
        prio = {"LEFT_UNATTENDED": 0, "MANUAL_COUNT_NO_CHECK": 1}
        events.sort(key=lambda e: (e.get("tier") == "weak", prio.get(e["type_id"], 2), e["t"]))
        n_weak = 0
        for e in events:
            if e.get("tier") == "weak":
                n_weak += 1
                if n_weak > MAX_WEAK_CHECKS:
                    e["tier"] = "dropped"
        vlm, vcfg = get_vlm_safe(job_id)
        anchor = None
        try:
            anchor = rules.features(det, cfg).attrs.get("anchor") if det is not None and not det.empty else None
        except Exception:
            pass
        kept = []
        for ev in events:
            if ev.get("source") == "scene":                # по пикселям надёжно, Qwen дверь различает хуже
                ev["vlm_status"] = "not_needed"; kept.append(ev); continue
            if ev.get("tier") == "dropped":                 # лимит проверок: слабое без подтверждения не выдаём
                continue
            if vlm is None or time.time() - start > DEADLINE_SEC - VLM_CALL_S:
                if ev.get("tier") == "weak":                # не успели проверить слабое — не выдаём (ложные тревоги)
                    continue
                # без проверки оставляем всё: пропуск хуже лишнего срабатывания
                ev["vlm_status"] = "skipped"; kept.append(ev); continue
            try:
                if ev["type_id"] == "LEFT_UNATTENDED" and anchor is not None:
                    a = verify_left(vlm, frames, ev, anchor)
                else:
                    a = verify(vlm, vcfg, frames, ev, anchor)
            except Exception as e:
                ev["vlm_status"] = f"error: {e}"[:80]; kept.append(ev); continue
            conf = float(a.get("confidence", 0) or 0)
            weak = ev.get("tier") == "weak"      # ручной пересчёт / смешение пачек: по детекциям не отличить от нормы
            if (weak and not a["confirmed"]) or (not a["confirmed"] and conf >= VERIFY_DROP_CONF):
                _log(job_id, f"VLM отклонила {ev['type_id']}@{ev['t']} ({ev.get('tier')})"); continue
            ev["vlm_status"] = "confirmed" if a["confirmed"] else "unsure"
            ev["vlm_desc"] = a.get("description_ru", "")
            if "t_best" in a:
                ev["t_best"] = a["t_best"]
            kept.append(ev)
        events = kept

    res = build(job_id, frames, det, events)
    res["_elapsed_sec"] = round(time.time() - start, 1)
    _log(job_id, f"готово: {res['verdict']}, {len(res['violations'])} событий, {res['_elapsed_sec']} с")
    return res


def warmup():
    """Загрузить YOLO и Qwen заранее, чтобы первое видео не ждало загрузки моделей."""
    t0 = time.time()
    try:
        if yolo_available():
            run_yolo([(0.0, np.zeros((848, 464, 3), np.uint8), 0)], "warmup")
    except Exception as e:
        _log("warmup", f"YOLO: {e}")
    try:
        vlm, _ = get_vlm_safe("warmup")
        if vlm is not None:
            vlm.generate([([np.zeros((64, 64, 3), np.uint8)], 'Answer with JSON {"ok": true}')])
    except Exception as e:
        _log("warmup", f"VLM: {e}")
    _log("warmup", f"модели загружены за {time.time() - t0:.0f} с")


def public(res):
    """Ровно схема API: убрать служебные поля с '_'."""
    keep = ("type", "timecode_sec", "description", "evidence_frames")
    return {"job_id": res["job_id"], "status": res["status"], "verdict": res["verdict"],
            "violations": [{k: x[k] for k in keep} for x in res.get("violations", [])]}
