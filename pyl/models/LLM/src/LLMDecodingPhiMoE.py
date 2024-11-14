import os, sys
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import init_empty_weights, infer_auto_device_map

LOCAL_DIR = os.path.dirname(__file__)
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(LOCAL_DIR))))

from .LLM import LLM


class LLMDecodingPhiMoE(LLM):
    """
    An implementation of the public HuggingFace Phi-3.5 MoE transformer.
    """
    def __init__(self, weights_path: str = None) -> None:
        """
        Initializes the model with ``'microsoft/Phi-3.5-moe'`` for text-to-text generation.

        Parameters
        ----------
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Note
        ----
        - For the full float32 model, requires 42Go of RAM/VRAM. If quantization is possible, it can be acheived 
        when loading the model in ``load_model()``.
        """
        super().__init__(model_name="microsoft/Phi-3.5-moe", weights_path=weights_path)

        return None


    def load_model(self, quantization: str = None, display: bool = False):
        """
        Loads the Phi-3.5 MoE model from the HuggingFace public weights at 'microsoft/Phi-3.5-moe'

        Parameters
        ----------
        quantization: str, None
            Quantisizes a model to reduce its memory usage and improve speed. Quantization can only be done
            on GPU be in 4-bits (``quantization='4b'``) or 8-bits (``quantization='8b'``).  

        Note
        ----
        - Make sure you have a combinaison of devices that has enough RAM/VRAM to host the whole model. Extra weights will be sent to CPU RAM, that will
        greatly reduce the computing speed, additionnal memory needs offloaded to scratch disk (default disk).
        - If you lack memory, try quantisizing the models for important performances improvements. Although may break some models or lead to more hallucinations.
        - Quantization in 8-bits requires roughly / of VRAM
        - Quantization in 4-bits requires roughly / of VRAM
        """

        if quantization in ["8b", "4b"] and self.device != "cpu":
            if quantization == "8b":
                load_in_8bit = True
                load_in_4bit = False
            if quantization == "4b":
                load_in_8bit = False
                load_in_4bit = True
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,  
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            quantization_config = None

        self._device_dispatch(model=AutoModelForCausalLM.from_pretrained(self.model_folder, 
                                                                         trust_remote_code=True, 
                                                                         quantization_config=quantization_config),
                              display=display)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_folder, clean_up_tokenization_spaces=True)

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map=self.device_map
            )
        
        return None

    
    def evaluate_model(self, prompt: str, context: str = "", display=True):
        """
        Evaluates a prompt and returns the model answer.

        Parameters
        ----------
        prompt: str
            The model querry
        display: bool
            Whereas printing the model answer or not. Default is 'True'

        Returns
        -------
        output: str
            The model's response to the input.

        Example
        -------
        >>> prompt = "Synthesize this conversation"
        >>> context = f'{conversation}'
        # The model input will be formatted as:
        >>> model_input = context + prompt
        """

        generation_args = {
            "max_new_tokens": 1000,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        message = self._preprocess(prompt=prompt, context=context)

        result = self.pipe(message, **generation_args)

        return self._postprocess(result=result)


    def _preprocess(self, prompt: str, context: str, **kwargs):
        """
        Preprocessing not required.
        """
        return context + '\n' + prompt


    def _postprocess(self, result: list[dict], **kwargs):
        """
        Postprocesses the pipeline output to make it directly readable.
        """
        return result[0]["generated_text"]
