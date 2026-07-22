"""GPT-SoVITS text-to-speech sidecar for the ASMR voice toolkit.

GPT-SoVITS' ONNX export is only partial, so TTS runs here in Python (invoked by
the Rust `asmr tts` command) rather than as pure-Rust ONNX. This package is a
thin, fully-typed orchestration layer over a GPT-SoVITS checkout: `finetune`
few-shot adapts the model to a target voice, and `speak` synthesizes from text
using a reference clip.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
