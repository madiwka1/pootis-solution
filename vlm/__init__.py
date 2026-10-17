"""Local VLM module: sliding-window video analysis with a local vision-language model."""
from .config import VLMConfig
from .analyze import analyze_video

__all__ = ["VLMConfig", "analyze_video"]
