"""Backend-independent request schemas for the Jev HTTP API."""

from typing import Any

from pydantic import BaseModel, Field


class ChoiceRequest(BaseModel):
    state: Any
    question: str
    options: list[str] = Field(min_length=2, max_length=255)
    temperature: float | None = Field(default=None, gt=0.0, allow_inf_nan=False)
    cache_salt: str | None = None
    use_prefix_cache: bool = True
    priority: int = 0


class BatchChoiceRequest(BaseModel):
    requests: list[ChoiceRequest] = Field(min_length=1, max_length=64)


class SystemOneRequest(BaseModel):
    state: Any
    questions: dict[str, dict[str, Any]]
    model: str | None = None
    cache_salt: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        pattern=r"\S",
        strict=True,
        description="Shared prefix-cache namespace for requests from a trusted caller.",
    )
    # Clef takes media beside the state, as its model card documents.
    images: list[Any] | None = None
    videos: list[Any] | None = None
    # JevAny preserves its training wire format: [{type: "image", uri: "..."}].
    media: list[Any] | None = None
