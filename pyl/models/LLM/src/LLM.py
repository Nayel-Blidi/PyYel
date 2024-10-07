
import os, sys
from abc import ABC, abstractmethod
import shutil
import psutil

import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model

from huggingface_hub import login
from huggingface_hub import snapshot_download

LOCAL_DIR = os.path.dirname(__file__)
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(LOCAL_DIR)))


class LLM(ABC):
    """
    Base LLM class
    """
    def __init__(self, model_name: str, weights_path: str = None) -> None:
        """
        Inits a LLM base class, and loads it from a local checkpoint or from HF using 
        the transformers API.

        Args
        ----
        model_name: str
            The name of the model to use. The folder where the weights will saved will have the same name.
        weights_path: str, None
            The path to the folder where the models weights should be saved. If None, the current working 
            directory path will be used instead.

        Note
        ----
        - Make sure to save you HuggingFace API token into the os env variables:
        >>> os.environ["HF_TOKEN"] = "hf_your_token"
        """
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_token = os.getenv("HF_TOKEN")

        self.model_name = model_name
        if weights_path is None: weights_path = os.getcwd()
        self.model_folder = os.path.join(weights_path, model_name.split(sep='/')[-1])

        if not os.path.exists(self.model_folder):
            self._init_model()

        return None
    

    def _init_model(self):
        """
        Initializes a model locally by loging to the HuggingFace API and creating a snapshot of its body.
        """

        login(token=self.hf_token)

        os.mkdir(path=self.model_folder)
        self._add_gitignore(folder_path=self.model_folder)

        allow_patterns = [
            "pytorch_model.bin",        # PyTorch weights
            "*.safetensors",            # Optional: If the model uses safetensors
            "config.json",              # Model configuration
            "tokenizer.json",           # Tokenizer files
            "vocab.txt",                # Optional tokenizer vocab file
            "merges.txt",               # For BPE tokenizers (e.g., GPT)
            "special_tokens_map.json",  # Special tokens file (if any)
            "tokenizer_config.json"     # Tokenizer configuration
        ]
        ignore_patterns = [
            "tf_model.h5",              # TensorFlow weights
            "flax_model.msgpack",        # JAX/Flax weights
            "*.onnx",                    # Optional: Exclude ONNX files if present
            "*.tflite",                  # Optional: Exclude TensorFlow Lite files
        ]

        snapshot_download(repo_id=self.model_name, local_dir=self.model_folder, ignore_patterns=ignore_patterns, allow_patterns=None)

        return True


    def _repair_model(self):
        """
        Repairs a model by deleting the currrent model's snapshot and 'downloading' a new one.
        """

        self._empty_folder(self.model_folder)
        os.rmdir(self.model_folder)
        self._init_model()

        return True


    def _add_gitignore(self, folder_path: str):
        """
        Creates a local .gitignore file to ignore the weights and model's content.
        """
        gitignore_content = "*"
        gitignore_path = os.path.join(folder_path, '.gitignore')
        with open(gitignore_path, 'w') as gitignore_file:
            gitignore_file.write(gitignore_content)
        
        return True
    

    def _map_device(self, display: bool = False):
        """
        Auto maps the device weights repartition across PC GPU/CPU.

        Args
        ----
        model_ram: float
            The RAM required by the weights to be loaded into the VRAM/RAM.
        display: bool, False
            Whereas printing verbose or not, debug util. 

        Note
        ----
        - Make sure you have a combinaison of devices that has enough RAM/VRAM to host the whole model. Extra weights will be sent to CPU RAM, that will
        greatly reduce the computing speed, additionnal memory needs offloaded to scratch disk (default disk).
        - If you lack memory, try quantisizing the models for important performances improvements. Although may break some models or lead to more hallucinations.
        """

        self.device_memory = {}
        if torch.cuda.is_available():
            self.device = "cuda"

            gpus_memory = 0
            gpus = torch.cuda.device_count()
            for gpu_id in range(gpus):

                gpu_name = torch.cuda.get_device_name(gpu_id)
                gpu_memory = round(torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3), ndigits=0)
                gpus_memory += gpu_memory
                if display: print("LLM >> GPU ID:", gpu_id, "| GPU name:", gpu_name, "| GPU VRAM:", gpu_id)

                # Device mapping, accelerate library max_memory format
                self.device_memory[gpu_id] = f"{gpu_memory}GB" 

            # If additionnal memory is needed, it will be allocated to CPU
            # self.device_memory["cpu"] = f"{max(model_ram - gpus_memory, 0)}GB"
            self.device_memory["cpu"] = f"{round(psutil.virtual_memory().available / (1024 ** 3), ndigits=0)}GB"

            if display: print("LLM >> Model sent to GPUs mapped as:", self.device_memory)

        else:
            if display: print("LLM >> Model sent to CPU mapped as:", f"cpu:{round(psutil.virtual_memory().total / (1024 ** 3), ndigits=0)}GB")
            self.device = "cpu"
            self.device_map = None

        return True


    def _dispatch_device(self, model, display: bool = False):
        """
        Takes a model and loads it across the available devices. Prioritizes GPUs first, in their natural 
        order (unless gpu:0 is the CPU chipset, which will be skipped).

        Args
        ----
        model: Any
            The loaded model. Can be inialized with empty weights to improve performances, and only retreive the device_map. 
        display: bool, False
            Whereas printing verbose or not, debug util. 
        """

        self._map_device(display=display)
        if torch.cuda.is_available() :
            self.device_map = infer_auto_device_map(model)
            self.model = dispatch_model(model, self.device_map)
        else:
            self.model = model.to(self.device)

        return True



    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def sample_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def test_model(self):
        pass

    @abstractmethod
    def evaluate_model(self, prompt: str, **kwargs):
        pass

    def save_output(self, string_data: str, save_path: str = None):
        """
        Saves a string output into a txt file.

        Args
        ----
        string_data: str
            The content to save into a text file.
        save_path: str, None
            The path to the file to the save the content into. If it does not exist, a file will be created.
            If already existing, content will be added to the same file. If None, will create a ``llm_output.txt``
            file into the current working directory.

        """

        if save_path is None: save_path = os.path.join(os.getcwd(), "llm_output.txt")

        with open(save_path, 'w') as raw_file:
            raw_file.write(string_data)

        return True 


    def parallelized_evaluate_model(self, prompts: list[str], max_workers: int = os.cpu_count(), **kwargs):
        """
        Parallelizes the inference of a list of prompts by dividing the inference into multiple subprocesses.
        By default, assigns one task per logical core. This will likely be too demanding if the inference is
        done on CPU, or if the GPU input requires a hefty preprocessing.

        Args
        ----
        prompts: list[str]
            The list of inputs to infer. This list will be divided between the workers.
        max_workers: int
            The number of processes to divide the work into. By default, is equal to the number of
            CPU processors.

        Returns
        -------
        sorted_outputs: dict
            The inferred outputs, in the same key order as given in the prompts list. Keys are the input prompts.
            Values are the output results.
        """

        failures = []
        outputs = {}
        with tqdm(total=len(prompts)) as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.evaluate_model, prompt=prompt, **kwargs): i 
                           for i, prompt in enumerate(prompts)}

                for future in as_completed(futures):
                    try:
                        # appends (prompt_idx, prompt, output(prompt))
                        outputs[futures[future]] = {"input": prompts[futures[future]], "output": future.result()}
                    except Exception as e:
                        failures.append(future)
                    pbar.update(1)

        if failures:
            print(f"S3Client >> {len(failures)} failures happened during model inference: ", *failures, sep="\n")

        return dict(sorted(outputs.items()))

    
    def _empty_folder(self, path: str = None):
        """
        Empties a folder.

        Args
        ----
        path: str
            The path to the folder to empty from all its content
        """
        
        if os.path.exists(path):
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                try:
                    # Check if it is a file and delete it
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    # Check if it is a directory and delete it and its contents
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"LLM >> Failed to empty {path}: {e}")

        return True