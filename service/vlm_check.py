"""Проверка Qwen на одном кадре: ответ и время.

  python -m service.vlm_check --video C:\\...\\videos\\X-1.mov --t 10
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vlm.config import VLMConfig                      # noqa: E402
from vlm.model import get_vlm                         # noqa: E402
from vlm.prompts import left_checklist_prompt         # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--t", type=float, default=10.0)
a = ap.parse_args()

cap = cv2.VideoCapture(a.video)
cap.set(cv2.CAP_PROP_POS_MSEC, a.t * 1000)
ok, f = cap.read()
if not ok:
    sys.exit("не удалось прочитать кадр")
cfg = VLMConfig()
print("модель:", cfg.model_id, "| адрес:", cfg.ollama_url)
vlm = get_vlm(cfg)
for name, prompt in [("простой вопрос", 'Answer with JSON {"what": "<what is on the desk, in English, 1 sentence>"}'),
                     ("чек-лист стола", left_checklist_prompt([a.t]))]:
    t0 = time.time()
    out = vlm.generate([([cv2.cvtColor(f, cv2.COLOR_BGR2RGB)], prompt)])[0]
    print(f"\n--- {name}: {time.time() - t0:.1f} с ---\n{out[:600]}")
