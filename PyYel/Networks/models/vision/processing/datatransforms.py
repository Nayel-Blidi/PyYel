from abc import ABC, abstractmethod

class CustomTransform(ABC):
    @abstractmethod
    def __call__(self, data):
        return data
