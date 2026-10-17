import json

SYSTEM = (
    "You are a bank cash-room compliance assistant. You look at frames from an elevated CCTV camera in a small "
    "cash room: the cashier sits at a desk with a banknote counting machine, a laptop and a phone; a grey safe "
    "stands next to the desk; a client may sit opposite the cashier; the room has a glass door. "
    "People: the cashier, clients, cash collectors, other staff. "
    "Report ONLY what is clearly visible. Never guess about things that cannot be seen "
    "(accounting entries, reconciliation, documents' content). Do not identify people."
)


def window_prompt(taxonomy: dict, frame_times) -> str:
    types = "\n".join(f"- {t['id']}: {t['visual']}" for t in taxonomy["types"])
    times = ", ".join(f"[{i}] t={t:.1f}s" for i, t in enumerate(frame_times))
    return f"""The images are consecutive frames of one short window: {times}.

Possible violation types:
{types}
- {taxonomy['other']['id']}: a clearly visible cash-discipline risk that is not in the list above
- NONE: normal work, nothing suspicious

Answer with ONE JSON object and nothing else:
{{"type": "<one id from the list or NONE>",
  "confidence": <0.0-1.0>,
  "frame_index": <index of the frame where it is most clearly visible, or -1>,
  "description_ru": "<1-2 sentences in Russian: what exactly is visible (who, what action, where)>"}}

Rules: choose NONE unless the action is clearly visible. confidence>0.7 only if there is no doubt."""


def verify_prompt(taxonomy: dict, type_id: str, frame_times) -> str:
    """Проверка кандидата от правил YOLO: подтвердить / отклонить + описание того, что видно."""
    t = next((x for x in taxonomy["types"] if x["id"] == type_id), None)
    what = t["visual"] if t else "a cash-discipline risk"
    times = ", ".join(f"[{i}] t={tt:.1f}s" for i, tt in enumerate(frame_times))
    return f"""The images are consecutive frames: {times}.
An automatic detector suspects this violation: {type_id} — {what}.

Look carefully and answer with ONE JSON object and nothing else:
{{"confirmed": <true if this is clearly visible, else false>,
  "confidence": <0.0-1.0>,
  "frame_index": <index of the frame that shows it best, or -1>,
  "description_ru": "<1-2 sentences in Russian: what exactly is visible (who, what action, where). No legal claims.>"}}
Do not explain your reasoning. Output only the JSON. /no_think"""


def left_checklist_prompt(frame_times) -> str:
    """Кассир ушла с рабочего места: что осталось. Конкретные вопросы — маленькой модели проще, чем «нарушение?».
    Кадр [0] — вся комната, остальные — крупный план стола."""
    times = ", ".join(f"[{i}] t={tt:.1f}s" for i, tt in enumerate(frame_times))
    return f"""Image [0] shows the whole cash room. The other images are close-up crops of the cashier's desk: {times}.
Answer each question only from what is visible.

0. person_at_desk: is anybody sitting or standing AT the cashier's desk / in the cashier's chair in image [0]? (true/false)
1. laptop_screen: what does the laptop screen show?
   "open_program" = a document, table, spreadsheet or application window (usually bright white with rows/text);
   "lock_screen" = a lock/login screen or wallpaper (usually blue/dark picture, big clock);
   "off" = black or lid closed; "not_visible" = laptop not visible.
2. key_in_safe: is a key sticking out of the lock of the grey safe? (true/false)
3. valuables_on_desk: are banknotes, a cash bag or documents left lying on the desk? (true/false)

Answer with ONE JSON object and nothing else:
{{"person_at_desk": <true|false>, "laptop_screen": "<open_program|lock_screen|off|not_visible>",
  "key_in_safe": <true|false>, "valuables_on_desk": <true|false>, "frame_index": <best frame index or -1>,
  "description_ru": "<1 sentence in Russian: what is left at the workplace>"}}
Do not explain your reasoning. Output only the JSON. /no_think"""


def manual_checklist_prompt(frame_times) -> str:
    """Ручной пересчёт: конкретные вопросы вместо «нарушение?» (на общий вопрос Qwen3 начинает «размышлять»)."""
    times = ", ".join(f"[{i}] t={tt:.1f}s" for i, tt in enumerate(frame_times))
    return f"""The images are close-up crops of the cashier's workplace, consecutive frames: {times}.
Answer each question only from what is visible.

1. cash_in_hands: does the cashier hold banknotes in her hands? (true/false)
2. counting_by_hand: is she counting / flipping / fanning the banknotes with her fingers? (true/false)
3. machine_used: are the banknotes put into or taken out of the banknote counting machine in these frames? (true/false)

Answer with ONE JSON object and nothing else:
{{"cash_in_hands": <true|false>, "counting_by_hand": <true|false>, "machine_used": <true|false>,
  "frame_index": <best frame index or -1>, "description_ru": "<1 sentence in Russian: what the cashier does with the money>"}}
Do not explain your reasoning. Output only the JSON. /no_think"""


def parse_json(text: str) -> dict:
    """JSON из ответа модели. Qwen3 может сначала «думать» (<think>…</think>) — берём ПОСЛЕДНИЙ корректный
    JSON-объект после размышлений; если JSON не дописан, вытаскиваем поля чек-листа из текста."""
    import re
    raw = text or ""
    body = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    if "<think>" in body:                                   # размышление оборвалось — JSON нет, остался только текст
        body = body.split("<think>", 1)[0]
    found = []
    for src in (body, raw):
        depth, start = 0, None
        for i, ch in enumerate(src):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}" and depth:
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        found.append(json.loads(src[start:i + 1]))
                    except json.JSONDecodeError:
                        pass
                    start = None
        if found:
            break
    keys = ("confirmed", "laptop_screen", "person_at_desk", "cash_in_hands", "counting_by_hand", "type", "what")
    good = [d for d in found if isinstance(d, dict) and any(k in d for k in keys) and "<" not in str(d.get("laptop_screen", ""))]
    if good:
        return good[-1]
    # запасной путь: модель рассуждала текстом — ищем явные формулировки
    low = raw.lower()
    out = {"type": "NONE", "confidence": 0.0, "frame_index": -1, "description_ru": "", "_raw": raw[:300]}
    m = re.search(r'laptop_screen"?\s*[:=]\s*"?(open_program|lock_screen|off|not_visible)', low)
    if m:
        out["laptop_screen"] = m.group(1)
    elif re.search(r"screen (is )?(showing|shows) (a |an )?(program|document|table|spreadsheet)", low):
        out["laptop_screen"] = "open_program"
    elif "lock screen" in low or "lock_screen" in low:
        out["laptop_screen"] = "lock_screen"
    for k in ("person_at_desk", "key_in_safe", "valuables_on_desk", "cash_in_hands", "counting_by_hand",
              "machine_used", "confirmed"):
        m = re.search(k + r'"?\s*[:=]\s*(true|false)', low)
        if m:
            out[k] = m.group(1) == "true"
    return out
