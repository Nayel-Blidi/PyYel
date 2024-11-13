import os, sys
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from accelerate import init_empty_weights, infer_auto_device_map

LOCAL_DIR = os.path.dirname(__file__)
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(LOCAL_DIR)))

from .LLM import LLM


class LLMDecodingPhi(LLM):
    """
    An implementation of the public HuggingFace Phi-3.5 Mini Instruct transformer.
    """
    def __init__(self, weights_path: str = None) -> None:
        """
        Initializes the model with ``'microsoft/Phi-3.5-mini-instruct'`` for text-to-text generation.

        Parameters
        ----------
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Note
        ----
        - For the full bfloat16 model, requires 7.7Go of RAM/VRAM. If quantization is possible, it can be acheived 
        when loading the model in ``load_model()``.
        """
        super().__init__(model_name="microsoft/Phi-3.5-mini-instruct", weights_path=weights_path)

        return None


    def load_model(self, quantization: str = None, display: bool = False):
        """
        Loads the Phi-3.5 Mini-Instruct model from the HuggingFace public weights 
        at 'microsoft/Phi-3.5-mini-instruct'.

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
        - Quantization in 8-bits requires roughly 16Go of VRAM
        - Quantization in 4-bits requires roughly 11Go of VRAM
        """
        
        if quantization in ["8b", "4b"] and torch.cuda.is_available():
            if quantization == "8b":
                load_in_8bit = True
                load_in_4bit = False
                dtype_correction = 2.0
            if quantization == "4b":
                load_in_8bit = False
                load_in_4bit = True
                dtype_correction = 4.0
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit, 
                llm_int8_threshold=6.0,
                llm_int8_enable_fp32_cpu_offload=True,)
                # bnb_4bit_compute_dtype=torch.bfloat16)
            low_cpu_mem_usage = True
        else:
            low_cpu_mem_usage = None
            quantization_config = None
            dtype_correction = 2.0
        
        with init_empty_weights(include_buffers=True):
            empty_model = AutoModelForCausalLM.from_pretrained(self.model_folder, quantization_config=quantization_config)
            self._device_map(model=empty_model, dtype_correction=dtype_correction, display=display)
            del empty_model

        self.model = AutoModelForCausalLM.from_pretrained(self.model_folder, 
                                                          trust_remote_code=True, 
                                                          quantization_config=quantization_config, 
                                                          low_cpu_mem_usage=low_cpu_mem_usage,
                                                          device_map=self.device_map,
                                                        #   attn_implementation="flash_attention_2",
                                                          torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_folder, 
                                                       clean_up_tokenization_spaces=True)
        self.pipe = pipeline("text-generation",
                             model=self.model,
                             tokenizer=self.tokenizer,
                             torch_dtype=torch.bfloat16,
                             device_map=self.device_map)
        
        if display: print(torch.cuda.memory_summary(device=torch.device('cuda')))
        
        return None


    def sample_model(self):
        pass

    
    def train_model(self):
        pass

    
    def test_model(self):
        pass

    
    def evaluate_model(self, prompt: str, context: str = "", max_tokens: int = 1000, display: bool = False):
        """
        Evaluates a prompt and returns the model answer.

        Parameters
        ----------
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
            "temperature": 0.1,
            "do_sample": True,
            # "stream":True
        }

        messages = context + '\n' + prompt
        output: str = self.pipe(messages, **generation_args)[0]["generated_text"]
        if display: print(output)
        
        return output

