"""RVC training and ONNX-export pipeline for the ASMR voice toolkit.

The package is a thin, fully-typed orchestration layer around the reference RVC
training code. It handles corpus preprocessing (mp3 -> 16 kHz mono segments),
drives feature/F0 extraction and generator training, and exports the trained
generator to ONNX with the exact input/output contract the Rust `asmr-vc` crate
expects.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
