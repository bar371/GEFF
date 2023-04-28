from abc import ABC, abstractmethod

import torch.nn


class ReIDModel(ABC):
    def __init__(self, device):
        self.model = None
        self.device = device

    @abstractmethod
    def init_model(self, config):
        pass

    @abstractmethod
    def extract_features(self, *argv):
        pass

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
