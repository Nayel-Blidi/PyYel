

from .CNN import CNN

class CNNDetectionRetinaNet(CNN):
    """
    Class of method to labelize (bbox detection + label) images using SSD model
    """

    def __init__(self, model_name: str, weights_path: str = None) -> None:
        super().__init__(model_name, weights_path)