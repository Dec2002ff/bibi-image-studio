"""Read-only model manifest utilities for the Bibi product layer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelEntry:
    id: str
    role: str
    source: str
    required: bool
    license: str | None = None
    install: str | None = None
    commercial_use: str | None = None


@dataclass(frozen=True)
class ModelManifest:
    models: list[ModelEntry]
    optional_backends: list[dict[str, Any]]
    policies: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "ModelManifest":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("schema_version") != 1:
            raise ValueError("unsupported model manifest schema")
        models = [
            ModelEntry(
                id=item["id"],
                role=item["role"],
                source=item["source"],
                required=bool(item.get("required", False)),
                license=item.get("license"),
                install=item.get("install"),
                commercial_use=item.get("commercial_use"),
            )
            for item in raw.get("models", [])
        ]
        return cls(
            models=models,
            optional_backends=list(raw.get("optional_backends", [])),
            policies=dict(raw.get("policies", {})),
        )

    def by_id(self, model_id: str) -> ModelEntry | None:
        return next((item for item in self.models if item.id == model_id), None)
