import os, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from accelerate import init_empty_weights

from .LLM import LLM


class LLMTextGenerationMistral(LLM):
    """
    A collection of pretrained models based on MistralAI's backbones, fine-tuned for text generation.
    """
    def __init__(self, weights_path: str = None, version: str = "7b") -> None:
        """
        Initializes a pretrained model based on MistralAI's backbones, fine-tuned for text generation.
        
        Parameters
        ----------
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Versions
        --------
        - ``'7b'`` _(default)_ : The 7 billion parameters v1.0 version of Mistral7b.
            - Initializes the model with ``'mistralai/Mistral-7B-v0.1'`` weights for text generation.
            - For the full float32 model, requires 28Go of RAM/VRAM.

        Note
        ----
        - Quantization may be supported. See ``load_model()``.
        """

        self.verison = version
        if version == "7b": 
            super().__init__(model_name="mistralai/Mistral-7B-v0.1", weights_path=weights_path)
        else:
            print("LLMTextGenerationMistral >> Warning: Invalid model version, model '7b' will be used instead.")
            self.version = "7b"
            super().__init__(model_name="mistralai/Mistral-7B-v0.1", weights_path=weights_path)
        
        return None


    def load_model(self, quantization: str = None, display: bool = False):
        """
        Loads the selected model for text generation.

        Parameters
        ----------
        quantization: str, None
            Quantisizes a model to reduce its memory usage and improve speed. Quantization can only be done
            on compatible GPUs, either in 4-bits (``quantization='4b'``) or 8-bits (``quantization='8b'``).  
        display: bool, False
            Prints the model's device mapping if ``True``.

        Note
        ----
        - Make sure you have a combinaison of devices that has enough RAM/VRAM to host the whole model. Extra weights will be sent to CPU RAM, which will
        greatly reduce the computing speed. Additionnal memory needs offloaded to scratch disk (default disk, not recommended).
        - If you lack memory, try quantisizing the models for important performances improvements. This may break some models or lead to more hallucinations.
            - Quantization in 8-bits requires roughly TODO of VRAM
            - Quantization in 4-bits requires roughly TODO of VRAM
        """
        
        dtype_correction = 1.0
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
        
        with init_empty_weights(include_buffers=True):
            empty_model = AutoModelForCausalLM.from_pretrained(self.model_folder, quantization_config=quantization_config)
            self._device_map(model=empty_model, dtype_correction=dtype_correction, display=display)
            del empty_model

        self.model = AutoModelForCausalLM.from_pretrained(self.model_folder, 
                                                          trust_remote_code=True, 
                                                          quantization_config=quantization_config, 
                                                          device_map=self.device_map)
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_folder, 
                                                       clean_up_tokenization_spaces=True)
        
        self.pipe = pipeline("text-generation",
                             model=self.model,
                             tokenizer=self.tokenizer,
                             device_map=self.device_map)
                
        return None

    
    def evaluate_model(self, prompt: str, context: str = "", max_tokens: int = 1000, display: bool = False):
        """
        Evaluates a prompt and returns the model answer.

        Parameters
        ----------
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
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

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


    def _postprocess(self):
        pass

    def _preprocess(self):
        pass
