from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from django.conf import settings

from .inference import TORInference as EfficientNetInference
from .inference_efficientnet_topk import TORInference as EfficientNetTopKInference
from .inference_resnet50_mean import TORInference as ResNet50MeanInference


DEFAULT_MODEL_KEY = 'efficientnet_b0'
EFFICIENTNET_TOPK_MODEL_KEY = 'efficientnet_b0_topk'
RESNET50_MEAN_MODEL_KEY = 'resnet50_mean'


@dataclass(frozen=True)
class DetectorConfig:
    key: str
    label: str
    inference_class: type
    weights_path: str
    threshold: float
    device: str
    inference_kwargs: dict | None = None


def model_options() -> dict[str, DetectorConfig]:
    return {
        DEFAULT_MODEL_KEY: DetectorConfig(
            key=DEFAULT_MODEL_KEY,
            label='EfficientNet-B0 baseline',
            inference_class=EfficientNetInference,
            weights_path=settings.TOR_MODEL_WEIGHTS_PATH,
            threshold=settings.TOR_INFERENCE_THRESHOLD,
            device=settings.TOR_INFERENCE_DEVICE,
        ),
        EFFICIENTNET_TOPK_MODEL_KEY: DetectorConfig(
            key=EFFICIENTNET_TOPK_MODEL_KEY,
            label='EfficientNet-B0 top-k aggregation',
            inference_class=EfficientNetTopKInference,
            weights_path=settings.TOR_EFFICIENTNET_TOPK_MODEL_WEIGHTS_PATH,
            threshold=settings.TOR_EFFICIENTNET_TOPK_INFERENCE_THRESHOLD,
            device=settings.TOR_EFFICIENTNET_TOPK_INFERENCE_DEVICE,
            inference_kwargs={
                'aggregation': settings.TOR_EFFICIENTNET_TOPK_AGGREGATION,
                'top_k': settings.TOR_EFFICIENTNET_TOPK_TOP_K,
            },
        ),
        RESNET50_MEAN_MODEL_KEY: DetectorConfig(
            key=RESNET50_MEAN_MODEL_KEY,
            label='ResNet50 mean aggregation',
            inference_class=ResNet50MeanInference,
            weights_path=settings.TOR_RESNET50_MODEL_WEIGHTS_PATH,
            threshold=settings.TOR_RESNET50_INFERENCE_THRESHOLD,
            device=settings.TOR_RESNET50_INFERENCE_DEVICE,
        ),
    }


def normalize_model_key(model_key: str | None) -> str:
    key = (model_key or DEFAULT_MODEL_KEY).strip()

    return key or DEFAULT_MODEL_KEY


def get_model_config(model_key: str | None) -> DetectorConfig:
    key = normalize_model_key(model_key)
    options = model_options()

    if key not in options:
        raise ValueError(f'Unknown model_key: {key}.')

    return options[key]


def model_metadata(model_key: str | None) -> dict:
    config = get_model_config(model_key)

    return {
        'model_key': config.key,
        'model_label': config.label,
        'model_threshold': config.threshold,
    }


@lru_cache(maxsize=None)
def get_detector(model_key: str):
    config = get_model_config(model_key)
    device = config.device or None

    if not Path(config.weights_path).exists():
        raise FileNotFoundError(f'Model weights not found for {config.key}: {config.weights_path}')

    return config.inference_class(
        weights_path=config.weights_path,
        threshold=config.threshold,
        device=device,
        **(config.inference_kwargs or {}),
    )
