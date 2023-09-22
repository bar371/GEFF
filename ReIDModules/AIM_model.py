import torch

from ReIDModules.AIM_CCReID.models import build_model
from ReIDModules.AIM_CCReID.models.img_resnet import ResNet50
from ReIDModules.AIM_CCReID.test import extract_img_feature
from ReIDModules.ReID_model import ReIDModel


class AIMModel(ReIDModel):
    def __init__(self, device):
        super().__init__(device)

    def init_model(self, config, checkpoint_path=None):
        self.model = ResNet50(config)
        if checkpoint_path:
            print(f'Loading model from checkpoint:{checkpoint_path}')
            self.load_checkpoint(checkpoint_path=checkpoint_path)

    def extract_features(self, dataloader):
        return self._extract_img_features(dataloader)

    def _extract_img_features(self, dataloader):
        self.model.eval()
        if self.device != 'cpu':
            self.model.cuda(self.device)
        with torch.no_grad():
            return extract_img_feature(self.model, dataloader)
