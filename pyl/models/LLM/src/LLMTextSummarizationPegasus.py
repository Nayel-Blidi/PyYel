import os, sys

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from accelerate import init_empty_weights

from .LLM import LLM


class LLMTextSummarizationPegasus(LLM):
    """
    A collection of pretrained models based on the Google's Pegasus backbone, fine-tuned for text summarization.
    """
    def __init__(self, weights_path: str = None, version: str = "xsum") -> None:
        """
        Initializes a pretrained models based on the Google's Pegasus backbone, fine-tuned for text summarization.

        Parameters
        ----------
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Versions
        --------
        - ``'large'``: The original 568 million parameters large version of Pegasus. 
            - Initializes the model with ``'google/pegasus-large'`` weights for text summarization.
            - For the full float32 model, requires 2Go of RAM/VRAM. 

        - ``'xsum'`` _(default)_: The 568 million parameters large version of Pegasus, fine-tuned on the XSUM dataset. 
            - Initializes the model with ``'google/pegasus-xsum'`` weights for text summarization.
            - For the full float32 model, requires 2Go of RAM/VRAM. 

        - ``'cnn'``: The 568 million parameters large version of Pegasus, fine-tuned on the CNN Daily News dataset. 
            - Initializes the model with ``'google/pegasus-cnn_dailymail'`` weights for text summarization.
            - For the full float32 model, requires 2Go of RAM/VRAM. 

        - ``'arxiv'`` : The 568 million parameters large version of Pegasus, fine-tuned on a scientifical Arxiv articles dataset. 
            - Initializes the model with ``'google/pegasus-arxiv'`` weights for text summarization.
            - For the full float32 model, requires 2Go of RAM/VRAM. 

        Note
        ----
        - The extended Pegasus-x models are not supported due to their very large RAM requirements.
        - Multiple tasks may be supported. See ``load_model()``.
        - Quantization isn't supported. See ``load_model()``.
        """

        self.version = version
        if version == "base": 
            super().__init__(model_name="google/pegasus-base", weights_path=weights_path)
        elif version == "large": 
            super().__init__(model_name="google/pegasus-large", weights_path=weights_path)
        elif version == "cnn": 
            super().__init__(model_name="google/pegasus-cnn_dailymail", weights_path=weights_path)
        elif version == "xsum": 
            super().__init__(model_name="google/pegasus-xsum", weights_path=weights_path)
        elif version == "arxiv": 
            super().__init__(model_name="google/pegasus-arxiv", weights_path=weights_path)
        else:
            print("LLMTextSummarizationPegasus >> Warning: Invalid model version, model 'xsum' will be used instead.")
            self.version = "xsum"
            super().__init__(model_name="google/pegasus-xsum", weights_path=weights_path)

        return None


    def load_model(self, display: bool = False):
        """
        Loads the selected model for text summarization.

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
            print(f"LLMTextSummarizationPegasus >> Error: Model's input should be of type 'list[str]', got '{type(prompts)}' instead.")

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



