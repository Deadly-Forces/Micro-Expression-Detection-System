"""
microex — Micro-Expression Detection System
============================================

A CPU-friendly (with optional GPU acceleration) pipeline for detecting,
tracking, and classifying micro-expressions in video streams using OpenCV.

Public API
----------
- ``SystemConfig`` / ``load_config`` / ``save_config`` — configuration
- ``FrameSource`` / ``WebcamSource`` / ``VideoSource`` / ``DatasetSource`` — capture
- ``FrameResult`` — per-frame data envelope

Usage
-----
>>> from microex import SystemConfig, load_config, VideoSource
>>> cfg = load_config("my_config.json")
>>> with VideoSource(cfg.video_path) as src:
...     for result in src:
...         process(result.frame)
"""

from __future__ import annotations

__version__: str = "0.1.0"
__author__: str = "Micro-Expression Detection Team"

# Re-export public symbols from sub-modules so callers can do:
#     from microex import SystemConfig, VideoSource, ...
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import SystemConfig, load_config, save_config
from src.microex.capture import (
    FrameResult,
    FrameSource,
    WebcamSource,
    VideoSource,
    DatasetSource,
)

__all__: list[str] = [
    # version
    "__version__",
    "__author__",
    # config
    "SystemConfig",
    "load_config",
    "save_config",
    # capture
    "FrameResult",
    "FrameSource",
    "WebcamSource",
    "VideoSource",
    "DatasetSource",
]
