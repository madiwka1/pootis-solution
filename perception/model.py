"""
Правила поверх детекций YOLO -> события-нарушения + оценка hit/miss и вердикта по видео.

Вход : data/*-detections.csv или *.jsonl  (новый формат train_yolo.py: video_id,t,frame_idx,track_id,cls,conf,x1..y2
                                          или старый test.py: video,time_sec,frame,class_name,...)
       video_labels.csv  (video_id, gt_timecode_sec, type_id, ...)        — события
       open_labels.xlsx  (наименование файла, таймстемп, нарушение/без)   — вердикт по всем 42 видео
Выход: data/predicted_events.csv   все события (type_id, start, end, t = таймкод)
       data/candidates.jsonl       окна для проверки Qwen/VLM (t_from, t_to, type_id, t)
       data/evaluation.csv         по каждому размеченному событию: HIT / WRONG_TYPE / MISS
       data/verdicts.csv           по каждому видео: вердикт, TP/FP/FN/TN
Запуск:
  python model.py                  оценка с текущими порогами
  python model.py --tune           перебор порогов: 0 пропусков по видео -> минимум ложных -> максимум HIT
Правила двух уровней (колонка tier):
  strong — надёжные, сами дают вердикт (на 42 видео: 0 ложных тревог);
  weak   — ручной пересчёт и смешение пачек: по рамкам не отличить от нормы, в сервисе их подтверждает Qwen.
  python model.py --tol 3          допуск таймкода, с
"""
import argparse
import itertools
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ALIAS = {"X-47": "X-4", "X-48": "X-5"}       # одинаковые файлы (md5): детекции берём у пары

# ---- Пороги. Ставятся на полноту: лишнее отсеет Qwen, пропуск не вернуть ----
CFG = {
    "person_conf": 0.30,
    "person_y2_min": 0.39,     # низ рамки человека ниже 39% высоты кадра; выше — отражение в стекле / за дверью
    "obj_conf": 0.20,          # cash, cash_bag, phone, drink, ...
    "present_frac": 0.40,      # доля обработанных кадров секунды, чтобы считать объект присутствующим
    "gap_fill_s": 1,           # заливать мигание детектора короче этого
    "absent_min_s": 2,         # у стола никого >= N с
    "safe_open_alone_s": 3,
    "safe_near_r": 0.5,        # человек «у сейфа»: центр ближе safe_near_r * рост к центру сейфа
    "phone_min_s": 1,
    "personal_min_s": 2,
    "entry_persons": 3,        # людей одновременно (кассир + клиент = 2)
    "entry_min_s": 3,          # 3-й человек >= N с (1-2 с — проход/отражение)
    "near_px": 40,             # «рядом» (сейф без людей рядом)
    "machine_px": 10,          # деньги «в машинке» — рамки пересекаются (с запасом N px)
    "desk_r": 0.35,             # кассир «на месте»: центр человека в радиусе desk_r * рост от точки рабочего места
    "entry_offset_s": 2,       # таймкод входа: через N с после появления 3-го человека (дошёл до стола)
    "manual_min_s": 2,         # деньги в руках не у машинки
    "machine_grace_s": 6,      # если за N с деньги попали в машинку — это нормальный пересчёт
    "static_px": 12,           # пачка «лежит» — центр сдвигается меньше
    "static_min_s": 3,
    "rejoin_s": 8,             # новый клиент пришёл в течение N с после ухода прошлого
    "merge_window_s": 4,
    "bag_persons": 2,
    "bag_leave_s": 6,          # сдатчик ушёл в течение N с после того, как сумка у кассира
    "bag_at_shift_s": 2,       # таймкод: за N с до «сумка у кассира» (момент, когда её ставят на стол)
    "personal_cash_s": 2,      # напиток считается «при деньгах», если деньги были в пределах N с
    "own_bag_min_s": 3,
    "leave_look_s": 3,         # деньги в работе в последние N с перед уходом клиента        # своя сумка/пальто на столе >= N с
}
TYPES = ["LEFT_UNATTENDED", "UNAUTHORIZED_ENTRY", "MANUAL_COUNT_NO_CHECK", "PERSONAL_ITEMS",
         "PHONE_AT_WORKPLACE", "COUNT_WITHOUT_CLIENT", "BAG_NO_CHECK", "SAFE_LEFT_OPEN",
         "PREV_CASH_ON_DESK", "CASH_MIXED",
         "DOOR_LEFT_OPEN", "CLIENT_NO_ID", "SAFE_KEY_LEFT"]


# ============================================================ загрузка
def norm(v):
    s = unicodedata.normalize("NFC", Path(str(v)).stem).strip()
    s = s.replace("Х", "X").replace("х", "x").upper().replace("-DETECTIONS", "")
    return s


def load_detections(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True) if path.suffix == ".jsonl" else pd.read_csv(path)
    if df.empty:
        return df
    ren = {"time_sec": "t", "frame": "frame_idx", "class_name": "cls", "video": "video_id"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns and v not in df.columns})
    if "bbox" in df.columns and "x1" not in df.columns:
        df[["x1", "y1", "x2", "y2"]] = pd.DataFrame(df.bbox.tolist(), index=df.index)
    if "img_h" not in df.columns:
        df["img_h"] = 848
    df["video_id"] = df.video_id.map(norm)
    df["cls"] = df.cls.replace({"bottle": "drink", "cup": "drink", "cell phone": "phone"})
    return subsample(df, EVAL_FPS)


EVAL_FPS = 5.0     # сервис читает видео с 5 к/с — оцениваем так же (и в 6 раз быстрее, чем на 30 к/с)


def subsample(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Оставить по одному кадру на шаг 1/fps (как service/pipeline.read_frames)."""
    if fps <= 0 or df.empty:
        return df
    fr = df.groupby("frame_idx").t.first().sort_index()
    keep, nxt = [], 0.0
    for idx, t in fr.items():
        if t + 1e-6 >= nxt:
            keep.append(idx); nxt += 1.0 / fps
            while nxt <= t:
                nxt += 1.0 / fps
    return df[df.frame_idx.isin(keep)].reset_index(drop=True)


def load_labels(events_csv: Path, verdict_xlsx: Path):
    ev = pd.DataFrame(columns=["vid", "type_id", "gt_t"])
    if events_csv.exists():
        e = pd.read_csv(events_csv, encoding="utf-8-sig")
        e["vid"] = e.video_id.map(norm)
        e = e[e.type_id.isin(TYPES)]
        ev = pd.DataFrame({"vid": e.vid, "type_id": e.type_id, "gt_t": e.gt_timecode_sec.astype(float),
                           "conf": e.get("confidence", "")})
    verdict = {}
    if verdict_xlsx.exists():
        x = pd.read_excel(verdict_xlsx)
        for _, r in x.iterrows():
            verdict[norm(r.iloc[0])] = str(r.iloc[2]).strip() == "нарушение"
    else:   # без xlsx: размеченные события = нарушение, остальное неизвестно
        verdict = {v: True for v in ev.vid}
    return ev, verdict


# ============================================================ признаки по секундам
def close(s: pd.Series, k: int) -> pd.Series:
    if k <= 0 or s.empty:
        return s
    w = 2 * k + 1
    d = s.astype(int).rolling(w, center=True, min_periods=1).max()
    return d.rolling(w, center=True, min_periods=1).min().astype(bool)


def runs(flags: pd.Series, min_len: float):
    out, start, prev = [], None, None
    for sec, f in flags.items():
        if f and start is None:
            start = sec
        if not f and start is not None:
            if prev - start + 1 >= min_len:
                out.append((int(start), int(prev)))
            start = None
        prev = sec
    if start is not None and prev - start + 1 >= min_len:
        out.append((int(start), int(prev)))
    return out


def near(a, b, m):
    return not (a[2] < b[0] - m or a[0] > b[2] + m or a[3] < b[1] - m or a[1] > b[3] + m)


def inside(a, b, m):
    """центр рамки a внутри рамки b (с запасом m px): деньги именно В машинке, а не рядом с ней"""
    cx, cy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    return b[0] - m <= cx <= b[2] + m and b[1] - m <= cy <= b[3] + m


def ctr(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def features(df: pd.DataFrame, c=CFG):
    df = df.copy()
    df["sec"] = df.t.astype(int)
    secs = pd.RangeIndex(0, int(df.sec.max()) + 1)
    # число ОБРАБОТАННЫХ кадров в секунде (а не fps видео) — работает и для 5 к/с, и для 30 к/с
    nfr = df.groupby("sec").frame_idx.nunique().reindex(secs, fill_value=0).replace(0, np.nan)

    is_person = (df.cls == "person") & (df.conf >= c["person_conf"]) & (df.y2 >= c["person_y2_min"] * df.img_h)
    obj = df[(df.cls != "person") & (df.conf >= c["obj_conf"])]
    P = df[is_person]

    def frac(sub):
        return (sub.groupby("sec").frame_idx.nunique().reindex(secs, fill_value=0) / nfr).fillna(0)

    def present(classes, thr=None):
        return close(frac(obj[obj.cls.isin(classes)]) >= (thr or c["present_frac"]), c["gap_fill_s"])

    n_person = (P.groupby(["sec", "frame_idx"]).size().groupby("sec").median()
                .reindex(secs, fill_value=0).round())
    # рабочее место кассира = самая «обжитая» точка, где чаще всего стоит/сидит человек (без лиц, по позиции)
    F = pd.DataFrame(index=secs)
    F["at_desk"] = False
    anchor = None
    if len(P):
        cx, cy = (P.x1 + P.x2) / 2, (P.y1 + P.y2) / 2
        cell = (cx // 40).astype(int).astype(str) + "_" + (cy // 40).astype(int).astype(str)
        top = cell.value_counts().index[0]
        sel = P[cell == top]
        anchor = (float(((sel.x1 + sel.x2) / 2).median()), float(((sel.y1 + sel.y2) / 2).median()),
                  float((sel.y2 - sel.y1).median()))
        R = c["desk_r"] * anchor[2]
        at = P[np.hypot(cx - anchor[0], cy - anchor[1]) <= R]
        F["at_desk"] = close(frac(at) >= c["present_frac"], c["gap_fill_s"])
    F.attrs["anchor"] = anchor
    F["n_person"] = n_person
    F["person"] = close(frac(P) >= c["present_frac"], c["gap_fill_s"])
    for k, cl in {"cash": ["cash"], "bag": ["cash_bag"], "safe_open": ["safe_open"], "phone": ["phone"],
                  "personal": ["drink", "handbag"], "docs": ["folder", "id_card"]}.items():
        F[k] = present(cl)

    # по кадрам: деньги не у машинки, деньги у машинки, телефон у человека, сейф без людей рядом, пачки
    off, onm, ph, safe_alone = {}, {}, {}, {}
    cash_boxes, bag_desk, left_alone, in_hands = {}, {}, {}, {}
    for (sec, fr), g in df.groupby(["sec", "frame_idx"]):
        Bg = g[(g.cls == "cash_bag") & (g.conf >= c["obj_conf"])][["x1", "y1", "x2", "y2"]].values
        if anchor is not None:
            bag_desk.setdefault(sec, []).append(any(np.hypot(*np.subtract(ctr(b), anchor[:2])) <= 1.5 * c["desk_r"] * anchor[2] for b in Bg))
        C = g[(g.cls == "cash") & (g.conf >= c["obj_conf"])][["x1", "y1", "x2", "y2"]].values
        M = g[(g.cls == "counting_machine") & (g.conf >= c["obj_conf"])][["x1", "y1", "x2", "y2"]].values
        Pp = g[is_person.loc[g.index]][["x1", "y1", "x2", "y2"]].values
        Ph = g[(g.cls == "phone") & (g.conf >= c["obj_conf"])][["x1", "y1", "x2", "y2"]].values
        S = g[(g.cls == "safe_open") & (g.conf >= c["obj_conf"])][["x1", "y1", "x2", "y2"]].values
        at_m = [any(inside(a, b, c["machine_px"]) for b in M) for a in C]
        off.setdefault(sec, []).append(len(C) > 0 and not all(at_m))
        onm.setdefault(sec, []).append(any(at_m))
        ph.setdefault(sec, []).append(any(near(a, b, 0) for a in Ph for b in Pp))
        safe_alone.setdefault(sec, []).append(len(S) > 0 and not any(
            np.hypot(*np.subtract(ctr(a), ctr(b))) < c["safe_near_r"] * (b[3] - b[1]) for a in S for b in Pp))
        in_hands.setdefault(sec, []).append(any(near(a, b, 0) and not any(inside(a, m, c["machine_px"]) for m in M)
                                                for a in C for b in Pp))
        cash_boxes.setdefault(sec, []).append([tuple(x) for x in C])
        # ценности, рядом с которыми нет ни одного человека (оставлены): деньги, сумка, документы
        V = g[g.cls.isin(["cash", "cash_bag", "folder", "id_card"]) & (g.conf >= c["obj_conf"])][["x1", "y1", "x2", "y2"]].values
        left_alone.setdefault(sec, []).append(any(not any(near(a, b, c["near_px"]) for b in Pp) for a in V))

    def sec_mean(d):
        return pd.Series({k: float(np.mean(v)) for k, v in d.items()}).reindex(secs, fill_value=0)

    F["left_alone"] = close(sec_mean(left_alone) >= c["present_frac"], c["gap_fill_s"])
    F["bag_at_desk"] = sec_mean(bag_desk) >= 0.3 if bag_desk else False
    F["cash_in_hands"] = close(sec_mean(in_hands) >= c["present_frac"], c["gap_fill_s"])
    F["cash_off"] = close(sec_mean(off) >= c["present_frac"], c["gap_fill_s"])
    F["cash_on_machine"] = close(sec_mean(onm) >= 0.2, c["gap_fill_s"])
    F["phone_in_hand"] = close(sec_mean(ph) >= 0.3, c["gap_fill_s"])
    F["safe_alone"] = close(sec_mean(safe_alone) >= c["present_frac"], c["gap_fill_s"])
    # число пачек и «неподвижная пачка» (одна и та же точка >= static_min_s)
    F["n_cash"] = pd.Series({k: np.median([len(x) for x in v]) for k, v in cash_boxes.items()}).reindex(secs, fill_value=0)
    first = {k: v[len(v) // 2] for k, v in cash_boxes.items()}
    static = pd.Series(False, index=secs)
    for s in secs:
        if s not in first or not first[s]:
            continue
        for b in first[s]:
            cb, ok = ctr(b), 0
            for s2 in range(s, s + c["static_min_s"]):
                if s2 in first and any(np.hypot(*np.subtract(ctr(x), cb)) <= c["static_px"] for x in first[s2]):
                    ok += 1
            if ok >= c["static_min_s"]:
                static.loc[s:s + c["static_min_s"] - 1] = True
    F["cash_static"] = close(static, c["gap_fill_s"])
    return F


# ============================================================ правила
def detect(df: pd.DataFrame, c=CFG) -> pd.DataFrame:
    F = features(df, c)
    ev = []

    def add(t, s, e, at=None, weak=False):
        ev.append((t, s, e, s if at is None else at, "weak" if weak else "strong"))

    seen = F.at_desk.cummax()
    n = F.n_person
    # ценности ОСТАЛИСЬ без людей рядом (деньги в руках у уходящего кассира — не нарушение)
    valuable = F.left_alone | F.safe_alone
    for s, e in runs(~F.at_desk & seen, c["absent_min_s"]):     # кассир ушёл с рабочего места (не обязательно из кадра)
        if valuable.loc[s:e].any():
            add("LEFT_UNATTENDED", s, e)
        else:
            # ценностей YOLO не видит, но нарушение может быть в том, что не детектируется:
            # незаблокированный компьютер, ключ в сейфе. Решает Qwen по кадрам.
            add("LEFT_UNATTENDED", s, e, weak=True)
    # сейф открыт, у сейфа никого И кассир не за столом. Открытый сейф при кассире на месте —
    # обычная работа
    for s, e in runs(F.safe_alone & ~F.at_desk & seen, c["safe_open_alone_s"]):
        add("SAFE_LEFT_OPEN", s, e)
    for s, e in runs(F.phone_in_hand | F.phone, c["phone_min_s"]):
        add("PHONE_AT_WORKPLACE", s, e)
    # личные вещи в зоне работы с деньгами: напиток/сумка, пока на столе деньги
    cash_near = close(F.cash, c["personal_cash_s"])
    for s, e in runs(F.personal & cash_near, c["personal_min_s"]):
        add("PERSONAL_ITEMS", s, e)
    # «сумка» у стола, когда кассир один (клиента нет) — это её вещи (пальто/сумка), а не инкассация
    bag_desk = F.bag_at_desk if not isinstance(F.bag_at_desk, bool) else pd.Series(False, index=F.index)
    for s, e in runs(F.bag & (n <= 1), c["own_bag_min_s"]):
        bd = bag_desk.loc[s:e]
        add("PERSONAL_ITEMS", s, e, at=int(bd[bd].index[0]) if bd.any() else s)

    # люди: уход клиента (2+ -> 1) и приход (1 -> 2+)
    drops = [s for s in n.index[1:] if n[s - 1] >= 2 and n[s] < 2]
    rises = [s for s in n.index[1:] if n[s - 1] < 2 and n[s] >= 2]
    for s, e in runs(n >= c["entry_persons"], c["entry_min_s"]):
        add("UNAUTHORIZED_ENTRY", s, e, at=min(s + c["entry_offset_s"], e))

    # следующий клиент пришёл, а пачка прошлого лежит на столе
    for d in drops:
        for r in rises:
            if d < r <= d + c["rejoin_s"] and F.cash_static.loc[max(0, d - 3):r + 3].any():
                add("PREV_CASH_ON_DESK", r, min(r + 3, F.index[-1]), at=r)
                break
    # клиент ушёл, пока деньги ещё в обработке
    # клиент ушёл, пока деньги ещё в обработке. Уход начинается раньше, чем человек пропал из кадра
    for d in drops:
        w = F.loc[max(0, d - c["leave_look_s"]):d]
        if (w.cash | w.cash_on_machine).any():
            add("COUNT_WITHOUT_CLIENT", max(0, d - 3), d, at=max(0, d - 2))
    # инкассаторская сумка: принесли (>= 2 людей), кассир забрал, и сдатчик ушёл почти сразу — проверки пломбы не было.
    # Если сдатчик остаётся — идёт приём с проверкой.
    for s, e in runs(F.bag & (n >= c["bag_persons"]), 1):
        bd = bag_desk.loc[s:]
        t_desk = int(bd[bd].index[0]) if bd.any() else s
        left = [d for d in drops if s < d <= t_desk + c["bag_leave_s"]]
        if left:
            add("BAG_NO_CHECK", s, left[0], at=max(s, t_desk - c["bag_at_shift_s"]))

    # ---- слабые признаки: по детекциям не отличить от нормы, решает Qwen (tier=weak) ----
    nc = F.n_cash
    for s in nc.index[1:]:
        # чужие деньги кладут на пачку: только при клиенте (кассир одна раскладывает свои пачки — норма)
        if nc[s - 1] >= 2 and nc[s] < nc[s - 1] and n[s] >= 2 and F.cash_static.loc[max(0, s - c["merge_window_s"]):s].any() \
                and not F.cash_on_machine.loc[max(0, s - 2):s].any():
            add("CASH_MIXED", s - 1, s, at=s - 1, weak=True)
    for s, e in runs(F.cash_in_hands, c["manual_min_s"]):         # деньги в руках, не у машинки
        if not F.cash_on_machine.loc[s:e + c["machine_grace_s"]].any():
            add("MANUAL_COUNT_NO_CHECK", s, e, at=(s + e) // 2, weak=True)   # середина эпизода: сам пересчёт, а не момент взятия денег
    return pd.DataFrame(ev, columns=["type_id", "start", "end", "t", "tier"])


# ============================================================ оценка
def evaluate(pred: pd.DataFrame, ev: pd.DataFrame, verdict: dict, have: set, tol: float, quiet=False, strong_only=True):
    """strong_only=True — вердикт только по сильным правилам (худший случай: Qwen отклонила все слабые)."""
    rows = []
    for _, g in ev.iterrows():
        vid = ALIAS.get(g.vid, g.vid)
        if vid not in have:
            rows.append({"video": g.vid, "type": g.type_id, "gt": g.gt_t, "result": "no detections", "dt": None}); continue
        p = pred[pred.vid == vid]
        hit_t = p[(p.type_id == g.type_id) & (p.start - tol <= g.gt_t) & (p.end + tol >= g.gt_t)]
        any_t = p[(p.start - tol <= g.gt_t) & (p.end + tol >= g.gt_t)]
        if len(hit_t):
            res, dt = "HIT", float((hit_t.t - g.gt_t).abs().min())
        elif len(any_t):
            res, dt = "WRONG_TYPE(" + ",".join(sorted(set(any_t.type_id))) + ")", float((any_t.t - g.gt_t).abs().min())
        else:
            res, dt = "MISS", None
        rows.append({"video": g.vid, "type": g.type_id, "gt": g.gt_t, "result": res, "dt": dt})
    E = pd.DataFrame(rows)

    vr = []
    for vid, is_v in verdict.items():
        src = ALIAS.get(vid, vid)
        if src not in have:
            continue
        p = pred[pred.vid == src]
        if strong_only and "tier" in p:
            p = p[p.tier == "strong"]
        pv = len(p) > 0
        vr.append({"video": vid, "gt": "violation" if is_v else "clean", "pred": "violation" if pv else "clean",
                   "res": {(1, 1): "TP", (0, 1): "FP", (1, 0): "FN", (0, 0): "TN"}[(int(is_v), int(pv))],
                   "types": ",".join(sorted(set(p.type_id)))})
    V = pd.DataFrame(vr)
    m = {k: int((V.res == k).sum()) if len(V) else 0 for k in ("TP", "FP", "FN", "TN")}
    hits = int((E.result == "HIT").sum()) if len(E) else 0
    if not quiet:
        print("\n=== События (тип + время ±%gs) ===" % tol)
        print(E.to_string(index=False))
        if len(E):
            print("\nHIT по типам:")
            print(E.groupby("type").result.apply(lambda r: f"{(r == 'HIT').sum()}/{len(r)}").to_string())
            d = E.dt.dropna()
            if len(d):
                print(f"\nОшибка таймкода у найденных: медиана {d.median():.1f} с, max {d.max():.1f} с")
        print("\n=== Вердикт по видео (%s) ===" % ("только сильные правила" if strong_only else "сильные + слабые"))
        print(V.sort_values("res").to_string(index=False))
        n = max(1, sum(m.values()))
        print(f"\nTP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}  accuracy={(m['TP'] + m['TN']) / n:.3f}"
              f"   events HIT={hits}/{len(E)}")
        if m["FN"]:
            print("ПРОПУЩЕНЫ видео с нарушением:", ", ".join(V[V.res == "FN"].video))
    return E, V, m, hits


def run_all(c, files):
    preds = []
    for vid, df in files.items():
        e = detect(df, c); e["vid"] = vid; preds.append(e)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(columns=["type_id", "start", "end", "t", "vid"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det", default=str(BASE / "data"))
    ap.add_argument("--labels", default=str(BASE / "video_labels.csv"))
    ap.add_argument("--verdicts", default=str(BASE / "open_labels.xlsx"))
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--tune", action="store_true")
    a = ap.parse_args()

    files = {}
    for f in sorted(Path(a.det).glob("*-detections.*")):
        if f.suffix not in (".csv", ".jsonl"):
            continue
        df = load_detections(f)
        if df.empty:
            print(f"{f.name}: пусто, пропуск"); continue
        vid = df.video_id.iloc[0]
        if vid in files and f.suffix == ".csv":
            continue                                   # есть и .jsonl, и .csv — берём один
        files[vid] = df
    print(f"Видео с детекциями: {len(files)}")
    ev, verdict = load_labels(Path(a.labels), Path(a.verdicts))
    cfg = dict(CFG)

    if a.tune:
        grid = {"present_frac": [0.3, 0.4, 0.5], "desk_r": [0.25, 0.35, 0.5],
                "absent_min_s": [2, 3], "near_px": [25, 40]}      # 36 комбинаций; ~30 с каждая на CPU
        best = None
        for vals in itertools.product(*grid.values()):
            c = dict(cfg, **dict(zip(grid, vals)))
            _, _, m, hits = evaluate(run_all(c, files), ev, verdict, set(files), a.tol, quiet=True)
            key = (m["FN"], m["FP"], -hits)
            if best is None or key < best[0]:
                best = (key, c)
        print(f"\nЛучшие пороги: FN={best[0][0]} FP={best[0][1]} HIT={-best[0][2]}")
        print({k: best[1][k] for k in grid})
        cfg = best[1]
        json.dump(cfg, open(BASE / "data" / "best_cfg.json", "w"), indent=1)

    pred = run_all(cfg, files)
    sc = Path(a.det) / "scene_events.csv"          # события по пикселям (scene.py: открытая дверь), если посчитаны
    if sc.exists():
        se = pd.read_csv(sc)
        se["vid"] = se.vid.map(norm)
        pred = pd.concat([pred, se[se.vid.isin(files)]], ignore_index=True)
        print(f"+ события сцены: {len(se)} ({', '.join(sorted(set(se.type_id)))})")
    out = BASE / "data"; out.mkdir(exist_ok=True)
    pred.to_csv(out / "predicted_events.csv", index=False)
    with open(out / "candidates.jsonl", "w", encoding="utf-8") as f:     # окна для Qwen
        for _, r in pred.iterrows():
            f.write(json.dumps({"video_id": r.vid, "type_id": r.type_id, "t": float(r.t), "tier": r.tier,
                                "t_from": max(0.0, float(r.start) - 1.5), "t_to": float(r.end) + 1.5}) + "\n")
    E, V, m, hits = evaluate(pred, ev, verdict, set(files), a.tol)
    _, V2, m2, _ = evaluate(pred, ev, verdict, set(files), a.tol, quiet=True, strong_only=False)
    n2 = max(1, sum(m2.values()))
    print(f"Если Qwen подтвердит ВСЕ слабые (ручной пересчёт, смешение пачек): TP={m2['TP']} FP={m2['FP']} "
          f"FN={m2['FN']} TN={m2['TN']}  accuracy={(m2['TP'] + m2['TN']) / n2:.3f}")
    print("  слабые сработали на чистых:", ", ".join(V2[V2.res == "FP"].video) or "нет",
          "| вытаскивают пропуски:", ", ".join(sorted(set(V[V.res == "FN"].video) & set(V2[V2.res == "TP"].video))) or "нет")
    E.to_csv(out / "evaluation.csv", index=False)
    V.to_csv(out / "verdicts.csv", index=False)
    print("\nСобытий найдено по типам:")
    print(pred.type_id.value_counts().to_string() if len(pred) else "нет")


if __name__ == "__main__":
    main()
