"""VLM backends. `ollama` = локальный Ollama (qwen3-vl). `hf` = transformers на GPU. `mock` = без модели."""
import json
import os
import random
import zlib
from typing import List

from PIL import Image

from .config import VLMConfig
from .prompts import SYSTEM


class MockVLM:
    """Deterministic fake answers so the pipeline/API can be built and tested on any laptop."""

    def __init__(self, cfg: VLMConfig):
        self.cfg = cfg

    def generate(self, batch: List[tuple]) -> List[str]:
        outs = []
        for images, prompt in batch:
            rnd = random.Random(zlib.crc32(prompt.encode()) + int(images[0].mean()))
            if rnd.random() < 0.1:
                outs.append(json.dumps({"type": "LEFT_UNATTENDED", "confidence": 0.8,
                                        "frame_index": len(images) // 2,
                                        "description_ru": "MOCK: кассир отошёл от рабочего места при клиенте"}))
            else:
                outs.append('{"type": "NONE", "confidence": 0.9, "frame_index": -1, "description_ru": ""}')
        return outs


class HFVLM:
    """Qwen2.5-VL / Qwen3-VL (or any image-text-to-text model) via transformers, fully local."""

    def __init__(self, cfg: VLMConfig):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText

        self.cfg = cfg
        kw = dict(torch_dtype=getattr(torch, cfg.dtype), device_map="auto", attn_implementation=cfg.attn_impl)
        if cfg.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=getattr(torch, cfg.dtype))
        self.model = AutoModelForImageTextToText.from_pretrained(cfg.model_id, **kw).eval()
        self.processor = AutoProcessor.from_pretrained(cfg.model_id, min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels)
        self.processor.tokenizer.padding_side = "left"
        self.torch = torch

    def generate(self, batch: List[tuple]) -> List[str]:
        texts, all_images = [], []
        for images, prompt in batch:
            content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}]
            msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": content}]
            texts.append(self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            all_images.append([Image.fromarray(im) for im in images])
        inputs = self.processor(text=texts, images=all_images, return_tensors="pt", padding=True).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.cfg.max_new_tokens, do_sample=False)
        out = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(out, skip_special_tokens=True)


class OllamaVLM:
    """Qwen3-VL через локальный Ollama (http://127.0.0.1:11434). Данные не уходят из машины."""

    def __init__(self, cfg: VLMConfig):
        import requests
        self.cfg, self.rq = cfg, requests
        try:
            tags = self.rq.get(cfg.ollama_url + "/api/tags", timeout=5).json().get("models", [])
        except Exception as e:
            raise RuntimeError(f"Ollama недоступен по {cfg.ollama_url}: {e}. Запустите `ollama serve`.")
        if not any(m.get("name", "").startswith(cfg.model_id) for m in tags):
            raise RuntimeError(f"Модели {cfg.model_id} нет в Ollama. Выполните: ollama pull {cfg.model_id}")

    def _b64(self, im):
        import base64, io
        h, w = im.shape[:2]; s = min(1.0, self.cfg.max_side / max(h, w))
        pil = Image.fromarray(im)
        if s < 1:
            pil = pil.resize((int(w * s), int(h * s)))
        buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()

    def generate(self, batch: List[tuple]) -> List[str]:
        outs = []
        for images, prompt in batch:
            body = {"model": self.cfg.model_id, "stream": False, "format": "json", "think": False,  # без «размышлений» — сразу JSON
                    "options": {"temperature": 0, "num_predict": self.cfg.max_new_tokens,
                                "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "4096"))},   # <=3 кадра (~1k токенов каждый) + промпт; больше -> модель вылезает из 4 ГБ на CPU;
                    "keep_alive": "30m",                                                # меньше контекст -> вся модель в 4 ГБ
                    "messages": [{"role": "system", "content": SYSTEM},
                                 {"role": "user", "content": prompt, "images": [self._b64(i) for i in images]},
                                 ]}
            if prompt.rstrip().endswith("/no_think") and "counting_by_hand" in prompt:
                # «подсказка» начала ответа только для вопроса о пересчёте: иначе модель уходит в <think>.
                # Для «ушла» не нужна: с ней Qwen отвечает быстрее, но хуже различает экран блокировки
                body["messages"].append({"role": "assistant", "content": "{"})
            try:
                r = self.rq.post(self.cfg.ollama_url + "/api/chat", json=body, timeout=self.cfg.ollama_timeout)
                j = r.json()
                if "error" in j:                                   # Ollama вернула ошибку (память, модель, параметр)
                    raise RuntimeError(j["error"])
                msg = j.get("message", {})
                txt = msg.get("content") or msg.get("thinking") or ""
                if not txt.strip():
                    raise RuntimeError("пустой ответ модели")
                if not txt.lstrip().startswith("{") and "<think>" not in txt[:20]:
                    txt = "{" + txt                                    # продолжение после подсказки "{"
                outs.append(txt)
            except Exception as e:
                outs.append(json.dumps({"type": "ERROR", "confirmed": False, "error": str(e)[:200]}))
        return outs


_CACHE = {}


def get_vlm(cfg: VLMConfig):
    key = (cfg.backend, cfg.model_id)
    if key not in _CACHE:
        _CACHE[key] = {"mock": MockVLM, "hf": HFVLM, "ollama": OllamaVLM}[cfg.backend](cfg)
    return _CACHE[key]
