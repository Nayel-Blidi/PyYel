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


    def load_model(self, quantization: str = None):
        """
        Loads the OPT-125M model from the HuggingFace public weights at 'facebook/opt-125m'.

        Args
        ----
        quantization: str, None
            Quantisizes a model to reduce its memory usage and improve speed. Quantization can only be done
            on GPU be in 4-bits (``quantization='4b'``) or 8-bits (``quantization='8b'``).  

        Note
        ----
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
            with init_empty_weights():
                self.model = AutoModelForCausalLM.from_pretrained(self.model_folder, quantization_config=quantization_config)
            # device_map = infer_auto_device_map(model, max_memory={0: "6GB", "cpu": "12GB"}, no_split_module_classes=["GPTNeoXLayer"])
        else:
            quantization_config = None


        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_folder, 
            trust_remote_code=True, 
            quantization_config=quantization_config, 
            # device_map=device_map
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_folder, clean_up_tokenization_spaces=True)

        return None


    def sample_model(self):
        pass

    
    def train_model(self):
        pass

    
    def test_model(self):
        pass

    
    def evaluate_model(self, prompt: str, log_content: str, display: bool = False):
        """
        Evaluates a prompt and returns the model answer.

        Args
        ----
        prompt: str
            The model querry
        display: bool
            Whereas printing the model answer or not. Default is 'True'
        """

        # Model settings
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            # model_kwargs={"quantization_config":quantization_config}
        )
        generation_args = {
            "max_new_tokens": 1000,
            "return_full_text": False,
            # "temperature": 0.0,
            "do_sample": False,
            # "stream":True
        }

        # Model enhanced prompting
        messages = "You are a bot designed to assist a team that maintains virtual machines, and github CI/CD pipelines. You analyze logs and suggest causes that may have lead to such a log track." \
                    + log_content \
                    + prompt

        output: str = pipe(messages, **generation_args)[0]["generated_text"]
        print(output)
        
        return output

