
import os, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from accelerate import init_empty_weights, infer_auto_device_map
import json

LOCAL_DIR = os.path.dirname(__file__)
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(LOCAL_DIR))))

from .LLM import LLM


class LLMDecodingMistral7b(LLM):
    """
    An implementation of the public HuggingFace Mistral7b transformer.
    """
    def __init__(self, weights_path: str = None) -> None:
        """
        Initializes the model with ``'mistralai/Mistral-7B-v0.1'`` for zero-shot classification.

        Args
        ----
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Note
        ----
        - For the full float32 model, requires 28Go of RAM/VRAM. If quantization is possible, it can be acheived 
        when loading the model in ``load_model()``.
        """
        super().__init__(model_name="mistralai/Mistral-7B-v0.1", weights_path=weights_path)
        
        return None


    def load_model(self, quantization: str = None, display: bool = False):
        """
        Loads the Mistral7b model from the HuggingFace public weights at 'mistralai/Mistral-7B-v0.1'

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
        - Quantization in 8-bits requires roughly 16Go of VRAM
        - Quantization in 4-bits requires roughly 11Go of VRAM
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
        context: str, ''
            Enhances a prompt by concatenating a string content beforehand. Default is '', adding the context
            is equivalent to enhancing the prompt input directly.
        display: bool, False
            Whereas printing the model answer or not. Default is 'False'.

        Returns
        -------
        output: str
            The model response.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device="cuda")

        # Generate text
        print("LLMMistral7b >> Evaluating prompt.")
        output = self.model.generate(**inputs, max_length=max_tokens)

        # Decode and print the generated text
        print("LLMMistral7b >> Decoding prompt.")
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        if display:
            print("LLMMistral7b >> Prompt was:", prompt)
            print("LLMMistral7b >> Answer is:", generated_text[len(prompt)+1:])

        return generated_text[len(prompt)+1:]

