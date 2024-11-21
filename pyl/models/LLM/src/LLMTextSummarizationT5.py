import os, sys

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from accelerate import init_empty_weights

from .LLM import LLM


class LLMTextSummarizationT5(LLM):
    """
    A collection of pretrained models based on the Google's T5 backbone, fine-tuned for text summarization.
    """
    def __init__(self, weights_path: str = None, version: str = "small") -> None:
        """
        Initializes a pretrained model based on the Google's T5 backbone, fine-tuned for text summarization.

        Parameters
        ----------
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Versions
        --------
        - ``'small'`` _(default)_ : The smallest 60.5 million parameters version of T5. 
            - Initializes the model with ``'google-t5/t5-small'`` weights for text summarization.
            - For the full float32 model, requires TODO of RAM/VRAM. 

        - ``'base'``: The 223 million parameters version of T5. 
            - Initializes the model with ``'google-t5/t5-base'`` weights for text summarization.
            - For the full float32 model, requires TODO of RAM/VRAM. 

        - ``'large'``: The 738 million parameters version of T5. 
            - Initializes the model with ``'google-t5/t5-large'`` weights for text summarization.
            - For the full float32 model, requires TODO of RAM/VRAM. 

        - ``'falconsai'``: The Falconsai's fine-tuned small 60.5 million parameters version of T5. 
            - Initializes the model with ``'Falconsai/text_summarization'`` weights for text summarization.
            - For the full float32 model, requires TODO of RAM/VRAM. 

        Note
        ----
        - Multiple tasks may be supported. See ``load_model()``.
        - Quantization isn't supported. See ``load_model()``.
        """

        self.version = version
        if version == "small": 
            super().__init__(model_name="google-t5/t5-small", weights_path=weights_path)
        elif version == "base": 
            super().__init__(model_name="google-t5/t5-base", weights_path=weights_path)
        elif version == "large": 
            super().__init__(model_name="google-t5/t5-large", weights_path=weights_path)
        elif version == "falconsai": 
            super().__init__(model_name="Falconsai/text_summarization", weights_path=weights_path)
        else:
            print("LLMTextSummarizationT5 >> Warning: Invalid model version, model 'small' will be used instead.")
            self.version = "base"
            super().__init__(model_name="google-t5/t5-small0", weights_path=weights_path)

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
            empty_model = AutoModelForSeq2SeqLM.from_pretrained(self.model_folder)
            self._device_map(model=empty_model, dtype_correction=1, display=display)
            del empty_model

        # MODEL SETUP (loading)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_folder, 
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
    
    
    def evaluate_model(self, 
                       prompts: list[str], 
                       min_length: int = 10,
                       max_length: int = 200,
                       **kwargs) -> list[str]:
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
            results.append(self.pipe(prompt, min_length=min_length, max_length=max_length))
        
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
            summarization_results.append(result[0]["summary_text"])

        return summarization_results



