import os, sys
import json

import torch
import numpy as np
from tqdm import tqdm

from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, pipeline
from accelerate import init_empty_weights, infer_auto_device_map

LOCAL_DIR = os.path.dirname(__file__)
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(LOCAL_DIR)))

from .LLM import LLM


class LLMEncodingDeBERTaV3BaseMNLI(LLM):
    """
    An implementation of the HuggingFace MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli transformer for zero-shot classification
    """
    def __init__(self, weights_path: str = None) -> None:
        """
        Initializes the model with ``'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'`` for zero-shot classification.

        Parameters
        ----------
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Note
        ----
        - For the full float32 model, requires 0.5Go of RAM/VRAM. 
        - This version is fine-tuned on the MNLI dataset.
        - Multiple encoding tasks may be supported. See ``load_model()``.
        """
        super().__init__(model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli", weights_path=weights_path)

        return None


    def load_model(self, task: str = "zero-shot-classification", display: bool = False):
        """
        Loads the MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli model for zero-shot classification.

        Parameters
        ----------
        task: str, 'zero-shot-classification'
            The task to use this encoder for. Default is zero-shoot-classification.
        
        Note
        ----
        - Quantization is not available.
        """

        supported_tasks = ["zero-shot-classification"]
        if task not in supported_tasks:
            print("LLMEncoderBARTLargeMNLI >> Task not supported, the pipeline will likely break. "
                  "Supported tasks are:", *supported_tasks)
        
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
        self.pipe = pipeline(task, 
                             model=self.model,
                             tokenizer=self.tokenizer)
        
        if display: print(torch.cuda.memory_summary(device=torch.device('cuda')))
        
        return None
        
        
    def sample_model(self):
        pass

    def train_model(self):
        pass

    def test_model(self):
        pass


    def evaluate_model(self, 
                       prompts: list[str], 
                       candidate_labels: list[str], 
                       multi_label: bool = False,
                       display: bool = False,
                       **kwargs) -> list[dict[str, float]]:
        """
        Classifies the prompts using zero-shot classification.

        Parameters
        ----------
            prompts: List[str]
                The list of prompts to classify.
            candidate_labels: List[str] 
                The list of candidate labels for classification.
            multi_label: bool
                Whether to perform multi-label classification. Default is False.
                If ``multi_label==True``, returns every label logit, otherwise returns the most likely label.
            display: bool
                Whether to print the model output. Default is True.

        Returns
        -------
            classification_results: List[Dict[str, float]]
                The classification results as a list of sorted dictionaries. Each dictionary structure is {label: prob} where label is a string, prob is a float between 0 and 1.
                If ``multi_label==False`` returns a one element list for each prompt.
        """
        if isinstance(prompts, str): prompts = [prompts]

        results = []
        for prompt in prompts:
            results.append(self.pipe(prompt, candidate_labels=candidate_labels, multi_label=multi_label))
        
        return self._postprocess(results=results, multi_label=multi_label, display=display)


    def _preprocess(self, prompts: list[str], **kwargs):
        """
        Preprocessing not required.
        """
        return prompts


    def _postprocess(self, results: list[dict], multi_label: bool = False, display: bool = False, **kwargs):
        """
        Postprocesses the pipeline output to make it directly readable.
        
        Parameters
        ----------
        results: list[dict]
            The pipeline outputs that will be cherry-picked to only return the predicted label or output.
        multi_label: bool
            Whereas returning the top prediction, or all the predictions, sorted by score.
        display: bool
            Whereas to display the processing bar or not.
        """

        classification_results = []
        if display: results = tqdm(results, postfix="Postprocessing") 
        for result in results:
            
            scores = result["scores"]
            labels = result["labels"]

            classification_result = dict(sorted(dict(zip(labels, scores)).items(), key=lambda item: item[1], reverse=True)) # ensures output is sorted
            if not multi_label:
                key = next(iter(classification_result)) # retrieves first key
                classification_result = {key: classification_result[key]} # 'truncates' the dictionary to keep the first key/value pair only

            classification_results.append(classification_result)

        return classification_results
    