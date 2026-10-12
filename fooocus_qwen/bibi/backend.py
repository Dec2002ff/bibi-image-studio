"""Backend contracts for Bibi Image Studio.

The product layer talks to this protocol rather than importing a concrete
Diffusers or ComfyUI implementation.  Keep this module free of torch/Gradio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class LoraSpec:
    path: str
    weight: float = 1.0
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("LoRA path is required")
        if not -4.0 <= float(self.weight) <= 4.0:
            raise ValueError("LoRA weight must be between -4 and 4")


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    negative_prompt: str = ""
    references: list[str] = field(default_factory=list)
    mask: str | None = None
    width: int = 2048
    height: int = 2048
    seed: int = -1
    steps: int = 40
    true_cfg_scale: float = 1.0
    loras: list[LoraSpec] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt is required")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if len(self.references) > 10:
            raise ValueError("Qwen-Image-2.1 supports at most 10 reference images")

    @property
    def is_edit(self) -> bool:
        return bool(self.references or self.mask)


@dataclass(frozen=True)
class GenerationResult:
    images: list[Path]
    seed: int
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageBackend(Protocol):
    """Minimal backend interface used by the future product/UI layer."""

    name: str

    def available(self) -> bool:
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...
