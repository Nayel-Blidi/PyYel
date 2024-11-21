import os, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from accelerate import init_empty_weights

from .LLM import LLM


class LLMTextGenerationOPT(LLM):
    """
    A collection of pretrained models based on the Facebook AI Research's OPT backbone, fine-tuned for text generation.
    """
    def __init__(self, weights_path: str = None, version: str = "125m") -> None:
        """
        Initializes a pretrained model based on the Facebook AI Research's OPT backbone, fine-tuned for text generation.

        Parameters
        ----------
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Versions
        --------
        - ``'125m'`` _(default)_ : The smallest 125m parameters version among the OPTs models. 
            - Initializes the model with ``'facebook/opt-125m'`` weights for text generation.
            - For the full model, requires 1Go of RAM/VRAM. 

        - ``'350m'``: The 350 million parameters version of the OPTs models. 
            - Initializes the model with ``'facebook/opt-350m'`` weights for text generation.
            - For the full model, requires X of RAM/VRAM. 

        - ``'1.3b'``: The 1.3 billion parameters version of the OPTs models. 
            - Initializes the model with ``'facebook/opt-1.3b'`` weights for text generation.
            - For the full model, requires X of RAM/VRAM. 

        - ``'2.7b'``: The 2.7 billion parameters version of the OPTs models. 
            - Initializes the model with ``'facebook/opt-2.7b'`` weights for text generation.
            - For the full model, requires X of RAM/VRAM. 

        - ``'6.7b'``: The 6.7 billion parameters version of the OPTs models. 
            - Initializes the model with ``'facebook/opt-6.7b'`` weights for text generation.
            - For the full model, requires X of RAM/VRAM. 

        - ``'13b'``: The 13 billion parameters version of the OPTs models. 
            - Initializes the model with ``'facebook/opt-13b'`` weights for text generation.
            - For the full model, requires X of RAM/VRAM. 

        - ``'1.3b-iml'``: The Instruction Meta-Learning (IML) 1.3 billion parameters version of the OPTs models. 
            - Initializes the model with ``'facebook/opt-iml-1.3b'`` weights for text generation.
            - For the full model, requires X of RAM/VRAM. 

        Note
        ----
        - Quantization may be supported. See ``load_model()``.
        """

        self.version = version
        if version == "125m": 
            super().__init__(model_name="facebook/opt-125m", weights_path=weights_path)
        elif version == "350m": 
            super().__init__(model_name="facebook/opt-350m", weights_path=weights_path)
        elif version == "1.3b": 
            super().__init__(model_name="facebook/opt-1.3b", weights_path=weights_path)
        elif version == "2.7b": 
            super().__init__(model_name="facebook/opt-2.7b", weights_path=weights_path)
        elif version == "6.7b": 
            super().__init__(model_name="facebook/opt-6.7b", weights_path=weights_path)
        elif version == "13b": 
            super().__init__(model_name="facebook/opt-13b", weights_path=weights_path)
        elif version == "1.3b-iml": 
            super().__init__(model_name="facebook/opt-iml-1.3b", weights_path=weights_path)
        else:
            print("LLMTextGenerationOPT >> Warning: Invalid model version, model '125m' will be used instead.")
            self.version = "mini"
            super().__init__(model_name="facebook/opt-125m", weights_path=weights_path)
        
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


    def _postprocess(self):
        pass

    def _preprocess(self):
        pass
