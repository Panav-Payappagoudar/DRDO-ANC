from collections.abc import Callable
from dataclasses import dataclass

from .base import Enhancer


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for instantiating and benchmarking an enhancer."""

    name: str
    streaming_delay_samples: int
    factory: Callable[[], Enhancer]


_MODEL_REGISTRY: dict[str, ModelConfig] = {}


def register_model(config: ModelConfig) -> None:
    """Register a model configuration by name."""

    key = config.name.strip()

    if not key:
        raise ValueError("Model name must not be empty.")

    if key in _MODEL_REGISTRY:
        raise ValueError(f"Model {key!r} is already registered.")

    _MODEL_REGISTRY[key] = config


def get_model_config(name: str) -> ModelConfig:
    """Return the registered configuration for a model name."""

    key = name.strip()

    try:
        return _MODEL_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(_MODEL_REGISTRY))
        raise KeyError(
            f"Unknown model {key!r}. "
            f"Available models: {available or '(none)'}."
        ) from exc


def list_models() -> tuple[str, ...]:
    """Return registered model names in sorted order."""

    return tuple(sorted(_MODEL_REGISTRY))


def create_enhancer(
    name: str,
    *,
    load: bool = True,
) -> Enhancer:
    """Instantiate a registered enhancer, optionally loading it."""

    config = get_model_config(name)
    enhancer = config.factory()

    if load:
        enhancer.load()

    return enhancer


def _register_builtin_models() -> None:
    from .deepfilternet import DeepFilterNetEnhancer
    from .finetuned import (
        FINETUNED_MODEL_NAME,
        FINETUNED_STREAMING_DELAY_SAMPLES,
        FineTunedDeepFilterNetEnhancer,
    )

    register_model(
        ModelConfig(
            name="DeepFilterNet3",
            streaming_delay_samples=1440,
            factory=DeepFilterNetEnhancer,
        ),
    )
    register_model(
        ModelConfig(
            name=FINETUNED_MODEL_NAME,
            streaming_delay_samples=FINETUNED_STREAMING_DELAY_SAMPLES,
            factory=FineTunedDeepFilterNetEnhancer,
        ),
    )


_register_builtin_models()
