
from .CNN import CNN

class CNNKeypoint(CNN):

    def __init__(self, model_name: str, weights_path: str = None) -> None:
        super().__init__(model_name, weights_path)