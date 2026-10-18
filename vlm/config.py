from dataclasses import dataclass, field
import os


@dataclass
class VLMConfig:
    backend: str = os.getenv("VLM_BACKEND", "ollama")  # ollama | hf | mock
    model_id: str = os.getenv("VLM_MODEL", "qwen3-vl:2b")
    ollama_url: str = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    ollama_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "90"))
    dtype: str = os.getenv("VLM_DTYPE", "bfloat16")
    load_in_4bit: bool = os.getenv("VLM_4BIT", "0") == "1"
    attn_impl: str = os.getenv("VLM_ATTN", "sdpa")
    max_new_tokens: int = int(os.getenv("VLM_MAX_TOKENS", "1024"))
    max_pixels: int = 448 * 448
    min_pixels: int = 128 * 128

    window_sec: float = 4.0
    stride_sec: float = 3.0
    frames_per_window: int = 4
    batch_size: int = 1
    max_side: int = 448

    conf_threshold: float = 0.5
    merge_gap_sec: float = 3.0
    max_events: int = 5
    evidence_frames: int = 2
    jpeg_quality: int = 80

    taxonomy_path: str = os.path.join(os.path.dirname(__file__), "taxonomy.json")
    time_budget_sec: float = 200.0
