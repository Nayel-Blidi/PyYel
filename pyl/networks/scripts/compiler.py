import os
import sys

PRELABELLING_DIR_PATH = os.path.dirname(os.path.dirname(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.dirname(PRELABELLING_DIR_PATH))

from prelabelling.models.torchvision.classificationRESNET import ClassificationRESNET
from prelabelling.models.torchvision.detectionSSD import DetectionSSD
from prelabelling.models.torchvision.segmentationDEEPLAB import segmentationDEEPLAB

from prelabelling.models.modelsabstract import ModelsAbstract

class Compiler():
    """
    The compiler runs the Active Learning steps evaluates datapoints to help the labellisation
    """

    def __init__(self, model:ModelsAbstract, weights:str="", version:str="18") -> None:

        if not isinstance(model(), ModelsAbstract):
            raise TypeError("Model isn't a subclass of ModelsAbstract")

        self.model = model(weights=weights, version=version)
        self.weights = weights
        
        return None
    
    def _assert_compatibility():
        pass

    def _assign_model():
        pass

    
    def loader(self, name:str=None):
        self.model.load()

    def trainer(self):
        self.model.train()
        return None
    
    def tester(self):
        self.model.test()
        return None

    def evluate(self):
        self.model.evaluate()

if __name__ == "__main__":
    comp = Compiler(model=ClassificationRESNET, version="34")
    comp.loader()
    comp.trainer()