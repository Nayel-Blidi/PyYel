import os, sys

PYYEL_DIR = os.path.dirname(os.path.dirname(__file__))
if __name__ == "__main__":
    sys.path.append(PYYEL_DIR)

from pyl.models.LLM import LLMBARTLargeMNLI


LLMBARTLargeMNLI()