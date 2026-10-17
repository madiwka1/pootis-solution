"""Единый контракт имён между YOLO, правилами и VLM. Менять имена только здесь."""

# YOLO-классы: id -> имя. Ровно такой порядок в perception/data.yaml и perception/cvat_labels.json.
CLASSES = {
    0: "person",
    1: "cash",
    2: "cash_bag",
    3: "phone",
    4: "safe_open",
    5: "safe_closed",
    6: "counting_machine",
    7: "handbag",
    8: "drink",            # стакан / кофе / бутылка / банка
    9: "id_card",          # удостоверение / карточка клиента
    10: "folder",          # папка / планшет с документами
}
CLASS_IDS = {v: k for k, v in CLASSES.items()}

# Предразметка готовой COCO-моделью (yolo11m.pt): имя COCO -> наше имя.
# Используется, чтобы не обводить людей / телефоны / сумки вручную. Ложные срабатывания проверять глазами.
COCO_MAP = {
    "person": "person",
    "cell phone": "phone",
    "handbag": "handbag",
    "cup": "drink",
    "bottle": "drink",
}
# laptop из COCO — не класс разметки, но нужен на инференсе как якорь рабочего места кассира.

# perception/<video_id>.jsonl — одна строка = одна детекция
DETECTION_FIELDS = ["video_id", "t", "frame_idx", "track_id", "cls", "cls_id", "conf", "bbox", "img_w", "img_h"]

# perception/<video_id>.signals.csv — одна строка = один шаг (0.5 с)
SIGNAL_COLUMNS = [
    "video_id", "t", "n_person", "person_entered", "person_left",
    "cashier_present", "client_present",
    "cash_visible", "cash_unattended", "cash_in_hands",
    "bag_visible", "phone_visible", "phone_in_hand",
    "safe_open", "personal_items_on_desk",
    "cash_static_on_desk", "cash_merged", "next_client_arrived",
]

# Типы нарушений — те же id, что в vlm/taxonomy.json
VIOLATION_TYPES = [
    "LEFT_UNATTENDED", "UNAUTHORIZED_ENTRY", "MANUAL_COUNT_NO_CHECK", "PERSONAL_ITEMS",
    "PHONE_AT_WORKPLACE", "COUNT_WITHOUT_CLIENT", "BAG_NO_CHECK", "SAFE_LEFT_OPEN",
    "PREV_CASH_ON_DESK", "CASH_MIXED",
    "DOOR_LEFT_OPEN", "CLIENT_NO_ID", "SAFE_KEY_LEFT",
]


def norm_video_id(name: str) -> str:
    """'Х-26.mov' (кириллица) / 'X-26' (латиница) -> 'X-26'."""
    import os, unicodedata
    s = unicodedata.normalize("NFC", os.path.splitext(os.path.basename(str(name)))[0]).strip()
    return s.replace("Х", "X").replace("х", "x")
