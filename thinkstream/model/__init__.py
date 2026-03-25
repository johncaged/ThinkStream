from transformers import (
    PretrainedConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
)

MODEL_CLS = {
    "qwen2.5vl": Qwen2_5_VLForConditionalGeneration,
    "qwen3vl": Qwen3VLForConditionalGeneration,
}

DEFAULT_VIDEO_FLEX_WINDOW_SIZE = 20


def get_text_config(config: PretrainedConfig) -> PretrainedConfig:
    """Return the text backbone sub-config, handling both flat and nested layouts."""
    tc = getattr(config, "text_config", None)
    return tc if tc is not None else config
