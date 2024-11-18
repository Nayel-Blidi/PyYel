import os, sys

from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from accelerate import init_empty_weights

from .LLM import LLM


class LLMTextSummarizationPegasus(LLM):
    """
    A collection of pretrained models based on the Microsoft's DeBERTaV3 backbone, fine-tuned for zero-shot text classification.
    """
    def __init__(self, weights_path: str = None, version: str = "base") -> None:
        """
        Initializes a pretrained model based on the Microsoft's DeBERTaV3 backbone, fine-tuned for zero-shot text classification.

        Parameters
        ----------
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Versions
        --------
        - ``'base'`` _(default)_ : The base 86 million parameters version of DeBERTaV3. 
            - Initializes the model with ``'MoritzLaurer/deberta-v3-base-zeroshot-v2.0'`` weights for zero-shot classification.
            - For the full float32 model, requires 0.5Go of RAM/VRAM. 

        - ``'large'``: The large 304 million parameters version of DeBERTaV3.
            - Initializes the model with ``'MoritzLaurer/deberta-v3-large-zeroshot-v2.0'`` for zero-shot classification.
            - For the full float32 model, requires 1Go of RAM/VRAM. 

        - ``'xsum'``: The base 86 million parameters version of DeBERTaV3 fine-tuned over the MNLI dataset.
            - Initializes the model with ``'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'`` for zero-shot classification.
            - For the full float32 model, requires 0.5Go of RAM/VRAM. 
            - This version is fine-tuned on the MNLI dataset.

        Note
        ----
        - Multiple tasks may be supported. See ``load_model()``.
        - Quantization isn't supported. See ``load_model()``.
        """

        self.version = version
        if version == "base": 
            super().__init__(model_name="MoritzLaurer/deberta-v3-base-zeroshot-v2.0", weights_path=weights_path)
        elif version == "large": 
            super().__init__(model_name="MoritzLaurer/deberta-v3-large-zeroshot-v2.0", weights_path=weights_path)
        elif version == "base-mnli": 
            super().__init__(model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", weights_path=weights_path)
        else:
            print("LLMZeroShotClassificationDeBERTaV3 >> Warning: Invalid model version, model 'base' will be used instead.")
            self.version = "base"
            super().__init__(model_name="MoritzLaurer/deberta-v3-base-zeroshot-v2.0", weights_path=weights_path)

        return None


    def load_model(self, display: bool = False):
        """
        Loads the selected model for zero-shot classification.

        Parameters
        ----------
        display: bool, False
            Prints the model's device mapping if ``True``.
        
        Note
        ----
        - Quantization is not available.
            - Reason: TODO
        """

        with init_empty_weights(include_buffers=True):
            empty_model = AutoModelForSequenceClassification.from_pretrained(self.model_folder)
            self._device_map(model=empty_model, dtype_correction=1, display=display)
            del empty_model

        # MODEL SETUP (loading)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_folder, 
                                                                        trust_remote_code=True, 
                                                                        device_map=self.device_map)
        
        # TOKENIZER SETUP (loading)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_folder, 
                                                       clean_up_tokenization_spaces=True)
        
        # PIPELINE SETUP (init)
        self.pipe = pipeline(task="summarization", 
                             model=self.model,
                             tokenizer=self.tokenizer)
                
        return None
    
    
    def evaluate_model(self, prompts: list[str], **kwargs) -> list[str]:
        """
        Summarizes the prompts.

        Parameters
        ----------
            prompts: list[str]
                The list of prompts to summarize.

        Returns
        -------
            summarization_results: list[str]
                The summarization_results results as a list.
        """

        prompts = self._preprocess(prompts=prompts)

        results = []
        for prompt in prompts:
            results.append(self.pipe(prompt))
        
        return self._postprocess(results=results)


    def _preprocess(self, prompts: list[str], **kwargs):
        """
        Preprocesses the pipeline inputs.
        """

        if isinstance(prompts, str): prompts = [prompts]
        if not isinstance(prompts, list): 
            print(f"LLMTextSummarizationT5 >> Error: Model's input should be of type 'list[str]', got '{type(prompts)}' instead.")

        return prompts


    def _postprocess(self, results: list[dict], **kwargs):
        """
        Postprocesses the pipeline output to make it directly readable.

        Parameters
        ----------
        results: list[dict]
            The pipeline outputs that will be cherry-picked to only return the predicted label or output.
        """

        summarization_results = []
        for result in results:
            # TODO
            summarization_results.append(result)

        return summarization_results



