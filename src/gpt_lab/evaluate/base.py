from lighteval.models.abstract_model import LightevalModel
from lighteval.models.model_output import ModelResponse
from lighteval.tasks.requests import Doc, SamplingMethod
from lighteval.utils.cache_management import SampleCache, cached
from typing import List

class EvaluationBaseModel(LightevalModel):
    """
    Base class for evaluation models.
    
    Implementation based on Lighteval template: 
    https://huggingface.co/docs/lighteval/main/en/evaluating-a-custom-model.
    """
    def __init__(self, model, config):
        super().__init__(config)
        self.model = model
        self.config = config

        # Enable caching (recommended)
        self._cache = SampleCache(config)

    @cached(SamplingMethod.GENERATIVE)
    def greedy_until(self, docs: List[Doc]) -> List[ModelResponse]:
        # Implement generation logic
        raise NotImplementedError

    @cached(SamplingMethod.LOGPROBS)
    def loglikelihood(self, docs: List[Doc]) -> List[ModelResponse]:
        # Implement loglikelihood computation
        raise NotImplementedError

    @cached(SamplingMethod.PERPLEXITY)
    def loglikelihood_rolling(self, docs: List[Doc]) -> List[ModelResponse]:
        # Implement rolling loglikelihood computation
        raise NotImplementedError
    
    def greedy_until(self, docs: list[Doc]) -> list[ModelResponse]:
        """
        Generate text until stop sequence or max tokens.

        Args:
            docs: list of documents containing prompts and generation parameters

        Returns:
            list of model responses with generated text
        """
        raise NotImplementedError
    
    def loglikelihood(self, docs: list[Doc]) -> list[ModelResponse]:
        """
        Compute log probabilities of continuations.

        Args:
            docs: list of documents containing context and continuation pairs

        Returns:
            list of model responses with log probabilities
        """
        raise NotImplementedError
    
    def loglikelihood_rolling(self, docs: list[Doc]) -> list[ModelResponse]:
        """
        Compute rolling log probabilities of sequences.

        Args:
            docs: list of documents containing text sequences

        Returns:
            list of model responses with rolling log probabilities
        """
        raise NotImplementedError