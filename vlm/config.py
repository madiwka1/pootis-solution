from dataclasses import dataclass, field
import os


@dataclass
class VLMConfig:
    # --- model ---
    # "mock" runs without GPU/torch (for pipeline + API development).
    backend: str = os.getenv("VLM_BACKEND", "ollama")        # "ollama" | "hf" | "mock"
    # ollama: qwen3-vl:2b (≈1.9 ГБ, влезает в 4 ГБ) или qwen3-vl:4b; hf: Qwen/Qwen3-VL-2B-Instruct. Apache-2.0.
    model_id: str = os.getenv("VLM_MODEL", "qwen3-vl:2b")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "90"))
    dtype: str = os.getenv("VLM_DTYPE", "bfloat16")           # bfloat16 | float16
    load_in_4bit: bool = os.getenv("VLM_4BIT", "0") == "1"    # for 12-16 GB GPUs (needs bitsandbytes)
    attn_impl: str = os.getenv("VLM_ATTN", "sdpa")            # "flash_attention_2" if installed
    max_new_tokens: int = int(os.getenv("VLM_MAX_TOKENS", "1024"))   # Qwen3 может «думать» перед JSON
    # pixels per image fed to the model (controls tokens & speed). 464x848 ~= 0.39 MP
    max_pixels: int = 448 * 448
    min_pixels: int = 128 * 128

    # --- sampling ---
    window_sec: float = 4.0          # length of one analysed window
    stride_sec: float = 3.0          # step between windows
    frames_per_window: int = 4       # frames sent to the VLM per window
    batch_size: int = 1              # windows per generate() call (ollama: 1)
    max_side: int = 448              # длинная сторона кадра для VLM, px

    # --- decision ---
    conf_threshold: float = 0.5      # min confidence for an event to count
    merge_gap_sec: float = 3.0       # merge same-type events closer than this
    max_events: int = 5
    evidence_frames: int = 2         # 1..3 per TZ
    jpeg_quality: int = 80

    # --- files ---
    taxonomy_path: str = os.path.join(os.path.dirname(__file__), "taxonomy.json")
    time_budget_sec: float = 200.0   # stop adding windows after this (TZ limit is 300 s incl. queue)
