from transformers import (
    LlamaForCausalLM,
    WhisperConfig,
    GemmaForCausalLM,
    WhisperForConditionalGeneration,
    Qwen2ForCausalLM,
)
from transformers.utils import logging

from .whisper_generation_mixin import CustomWhisperGenerationMixin
from .base_generation_mixin import CustomGenerationMixin

logger = logging.get_logger(__name__)


class CustomLlamaForCausalLM(CustomGenerationMixin, LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)


class CustomQwen2ForCausalLM(CustomGenerationMixin, Qwen2ForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)


class CustomGemmaForCausalLM(CustomGenerationMixin, GemmaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)


class CustomWhisperForConditionalGeneration(
    CustomWhisperGenerationMixin, WhisperForConditionalGeneration
):
    base_model_prefix = "model"
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, config: WhisperConfig):
        super().__init__(config)
