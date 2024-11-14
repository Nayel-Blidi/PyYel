
import os, sys
from abc import ABC, abstractmethod
import shutil
import psutil

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model

from huggingface_hub import login
from huggingface_hub import snapshot_download
from datasets import Dataset
from transformers import Pipeline, AutoModel, AutoTokenizer

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

        Parameters
        ----------
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
    

    def _device_map(self, model, dtype_correction: float = 1, display: bool = False):
        """
        Auto maps the device weights repartition across PC GPU/CPU.

        Parameters
        ----------
        model_ram: float
            The RAM required by the weights to be loaded into the VRAM/RAM.
        dtype_correction: float, 1.0
            The accelerate library used expects float32 types. When using smaller types, the weights would be incorrectly
            divided across the devices. To fix this, a coefficent can be added to the available VRAM count, to simulate 
            smaller weights bit-wise.
        display: bool, False
            Whereas printing verbose or not, debug util. 

        Note
        ----
        - Make sure you have a combinaison of devices that has enough RAM/VRAM to host the whole model. Extra weights will be sent to CPU RAM, that will
        greatly reduce the computing speed, additionnal memory needs offloaded to scratch disk (default disk).
        - If you lack memory, try quantisizing the models for important performances improvements. Although may break some models or lead to more hallucinations.
        """

        device_memory = {}
        if torch.cuda.is_available():

            gpus_memory = 0
            gpus = torch.cuda.device_count()
            for gpu_id in range(gpus):

                gpu_name = torch.cuda.get_device_name(gpu_id)
                gpu_memory = round(torch.cuda.get_device_properties(gpu_id).total_memory / (1024 ** 3), ndigits=0)
                gpus_memory += gpu_memory
                if display: print("LLM >> GPU ID:", gpu_id, "| GPU name:", gpu_name, "| GPU VRAM:", gpu_memory)

                # Device mapping, accelerate library max_memory format
                device_memory[gpu_id] = f"{gpu_memory*dtype_correction}GB" 

            # If additionnal memory is needed, it will be allocated to CPU
            # self.device_memory["cpu"] = f"{max(model_ram - gpus_memory, 0)}GB"
            # device_memory["npu"] = "8GB"
            device_memory["cpu"] = f"{round(psutil.virtual_memory().available / (1024 ** 3), ndigits=0)}GB"

            if display and dtype_correction != 1.0: print(f"LLM >> Model memory corrected to simulate dtypes {dtype_correction} times",
                                                          f"smaller bits-wise (VRAM multiplied by {dtype_correction}x)")
            if display: print("LLM >> Model sent to GPUs mapped as:", device_memory)

            self.device = "cuda" # to easily send small tensors to the GPU
            # self.device_map = infer_auto_device_map(model=model, max_memory=device_memory, no_split_module_classes=["GPTNeoXLayer"], verbose=display)
            self.device_map = infer_auto_device_map(model=model, max_memory=device_memory, verbose=display)

        else:
            if display: print("LLM >> Model sent to CPU mapped as:", f"cpu: {round(psutil.virtual_memory().total / (1024 ** 3), ndigits=0)}GB")
            self.device = "cpu" # for compatibility when sending tensors to device
            self.device_map = None

        return True


    def _device_dispatch(self, model, dtype_correction: float = 1.0, display: bool = False):
        """
        TODO: replace with torch.nn.module input and init_empty_weights for dispatching of CNN and FCN
        Takes a model and loads it across the available devices. Prioritizes GPUs first, in their natural 
        order (unless gpu:0 is the CPU chipset, which will be skipped).

        Parameters
        ----------
        model: Any
            The loaded model. Can be inialized with empty weights to improve performances, and only retreive the device_map.
        dtype_correction: float, 1.0
            The accelerate library used expects float32 types. When using smaller types, the weights would be incorrectly
            divided across the devices. To fix this, a coefficent can be added to the available VRAM count, to simulate 
            smaller weights bit-wise.
        display: bool, False
            Whereas printing verbose or not, debug util. 
        """

        self._device_map(model=model, dtype_correction=dtype_correction, display=display)
        if torch.cuda.is_available() :
            self.model = dispatch_model(model, self.device_map)
        else:
            self.model = model.to(self.device)

        return True


    @abstractmethod
    def load_model(self):
        self.tokenizer: AutoTokenizer = None
        self.model: AutoModel = None
        self.pipe: Pipeline = None
        pass

    @abstractmethod
    def evaluate_model(self, prompt: str, **kwargs):
        pass

    @abstractmethod
    def _postprocess(self):
        pass

    @abstractmethod
    def _preprocess(self):
        pass


    def save_output(self, string_data: str, save_path: str = None):
        """
        Saves a string output into a txt file.

        Parameters
        ----------
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


    def parallelize_evaluate_model(self, prompts: list[str], batch_size: int = 2**3, **model_kwargs):
        """
        Optimizes an inference process by distributing the workload:
        - Dataloader preprocesses batches of data before sending it to GPU using CPU subprocesses
        - Mixed-precision distributes workload across GPU and CPU (if GPU is available)
        - Data is regrouped as a single batch before post process

        Parameters
        ----------
        prompts: list[str]
            The list of inputs to infer on. Should be identical to the expected ``evaluate_model()`` method args.
        batch_size: int, 8
            The size of the batch of data to send to the model at once. Larger batches require more VRAM.
        model_kwargs: dict
            The kwargs to pass to the model. Should be similar to the expected ``evaluate_model()`` method args.

        Returns
        -------
        results: dict
            The postprocessed model predictions.
        """

        dataset = Dataset.from_dict({"text": self._preprocess(prompts)})
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size)#, num_workers=10*max_workers)


        results = []
        self.pipe._batch_size = batch_size
        with torch.no_grad():
            for batch in tqdm(dataloader, colour="green", postfix="Model Inference"):
                
                self.pipe.call_count = 0 # Removes sequential call warning

                if self.device != "cpu":
                    with autocast(device_type="cuda"):
                        outputs = self.pipe(batch["text"], **model_kwargs)
                        # outputs = self.evaluate_model(prompts=batch["text"], **model_kwargs)
                        results.extend(outputs)

                else:
                    outputs = self.pipe(batch["text"], **model_kwargs)
                    results.extend(outputs)

        results = self._postprocess(results=results, **model_kwargs)

        return results

    
    def _empty_folder(self, path: str, include: str | list[str] = [], exclude: str | list[str] = []):
        """
        Empties a folder from all its content, files and folders.

        Parameters
        ----------
        path: str
            The path to the folder to empty
        include: str or list[str], []
            A list of extensions or files to include to the deletion list. Extensions must be of format '.ext'. If empty, 
            will try to delete any file but those whose extensions are excluded (cf. ``exclude``).
        exclude: str or list[str], []
            A list of extensions of files to exclude from deletion. Will only be used if include is empty. 
        """

        if type(include) is str:
            include = [include]
        if type(exclude) is str:
            exclude = [exclude]

        # Corrects the missing '.' in from of the extension
        for extension in include:
            if not extension.startswith('.'):
                # In the case extension is actually a file, the original include pattern is kept, and the .include pattern is added
                include.append(f".{extension}")
        for extension in exclude:
            if not extension.startswith('.'):
                # In the case extension is actually a file, the original include pattern is kept, and the .include pattern is added
                exclude.append(f".{extension}")
        
        if os.path.exists(path):
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                file_extension = os.path.splitext(filename)[-1]
                if file_extension in exclude or filename in exclude:
                    None
                elif (os.path.splitext(filename)[-1] in include) or (include == []):
                    try:
                        # Check if it is a file and delete it
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        # Check if it is a directory and delete it and its contents
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f"Pipeline >> Failed to delete {file_path}: {e}")

        return True
    

