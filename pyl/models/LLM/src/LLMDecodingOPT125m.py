import os, sys
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from accelerate import init_empty_weights, infer_auto_device_map

LOCAL_DIR = os.path.dirname(__file__)
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(LOCAL_DIR)))

from .LLM import LLM


class LLMDecodingOPT125m(LLM):
    """
    An implementation of the public HuggingFace OPT-125M transformer.
    """
    def __init__(self, weights_path: str = None) -> None:
        """
        Initializes the model with ``'facebook/opt-125m'`` for text-to-text generation.

        Args
        ----
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Note
        ----
        - For the full / model, requires / of RAM/VRAM. If quantization is possible, it can be acheived 
        when loading the model in ``load_model()``.
        """
        super().__init__(model_name="facebook/opt-125m", weights_path=weights_path)
        
        return None


    def load_model(self, quantization: str = None, display: bool = False):
        """
        Loads the OPT-125M model from the HuggingFace public weights at 'facebook/opt-125m'.

        Args
        ----
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

        self._dispatch_device(model=AutoModelForCausalLM.from_pretrained(self.model_folder, 
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

        return True


    def sample_model(self):
        pass

    
    def train_model(self):
        pass

    
    def test_model(self):
        pass

    
    def evaluate_model(self, prompt: str, context: str = "", max_tokens: int = 1000, display: bool = False):
        """
        Evaluates a prompt and returns the model answer.

        Args
        ----
        prompt: str
            The model querry.
        context: str
            Enhances a prompt by concatenating a string content beforehand. Default is '', adding the context
            is equivalent to enhancing the prompt input directly.
        display: bool
            Whereas printing the model answer or not. Default is 'True'.

        Returns
        -------
        output: str
            The model response.
        """

        generation_args = {
            "max_new_tokens": max_tokens,
            "return_full_text": False,
            # "temperature": 0.0,
            "do_sample": False,
            # "stream":True
        }

        # Model enhanced prompting
        messages = context + '\n' + prompt
        output: str = self.pipe(messages, **generation_args)[0]["generated_text"]
        if display: print(output)
        
        return output

