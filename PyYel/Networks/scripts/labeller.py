import os
import sys

PRELABELLING_DIR_PATH = os.path.dirname(os.path.dirname(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.dirname(PRELABELLING_DIR_PATH))


class labeller():
    """
    The labeller suggests labels to an annotation task to help the user.
    It is not part of the Active Learning loop, but rather its purpose.
    """

    def __init__(self, model:str, weights:str) -> None:
        self.model = model
        self.weights = weights