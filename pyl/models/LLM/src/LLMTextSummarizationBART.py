import os, sys

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from accelerate import init_empty_weights

from .LLM import LLM


class LLMTextSummarizationBART(LLM):
    """
    A collection of pretrained models based on the Facebook AI Research's BART backbone, fine-tuned for text summarization.
    """
    def __init__(self, weights_path: str = None, version: str = "large-cnn") -> None:
        """
        Initializes a pretrained model based on the Facebook AI Research's BART backbone, fine-tuned for text summarization.

        Parameters
        ----------
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Versions
        --------
        - ``'large-cnn'`` _(default)_ : The large 406 million parameters version of BART fine-tuned on the CNN Daily Mail dataset. 
            - Initializes the model with ``'facebook/bart-large-cnn'`` weights for text summarization.
            - For the full float32 model, requires 0.5Go of RAM/VRAM. 

        Note
        ----
        - Multiple tasks may be supported. See ``load_model()``.
        - Quantization isn't supported. See ``load_model()``.
        """

        self.version = version
        if version == "large-cnn": 
            super().__init__(model_name="facebook/bart-large-cnn", weights_path=weights_path)
        else:
            print("LLMTextSummarizationBART >> Warning: Invalid model version, model 'large-cnn' will be used instead.")
            self.version = "large-cnn"
            super().__init__(model_name="facebook/bart-large-cnn", weights_path=weights_path)

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



