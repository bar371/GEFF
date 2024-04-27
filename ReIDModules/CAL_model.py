import torch
from ReIDModules.CAL.data.dataset_loader import Street42Dataset
from ReIDModules.ReID_model import ReIDModel
from ReIDModules.CAL.test import extract_img_feature, extract_img_feature_street42
from ReIDModules.CAL.models.img_resnet import ResNet50
from ReIDModules.CAL.models.vid_resnet import C2DResNet50, I3DResNet50, AP3DResNet50, NLResNet50, AP3DNLResNet50


factory = {
    'resnet50': ResNet50,
    'c2dres50': C2DResNet50,
    'i3dres50': I3DResNet50,
    'ap3dres50': AP3DResNet50,
    'nlres50': NLResNet50,
    'ap3dnlres50': AP3DNLResNet50,
}


class CALModel(ReIDModel):
    def __init__(self, device):
        super().__init__(device)

    def init_model(self, config, checkpoint_path=None):
        self.model = factory[config.MODEL.NAME](config)
        if checkpoint_path:
            print(f'Loading model from checkpoint:{checkpoint_path}')
            self.load_checkpoint(checkpoint_path=checkpoint_path)

    def extract_features(self, dataloader):
        return self._extract_img_features(dataloader)

    def _extract_img_features(self, dataloader):
        self.model.eval()
        if self.device != 'cpu':
            self.model.cuda(self.device)
        if type(dataloader.dataset) == Street42Dataset:
            with torch.no_grad():
                print("Extracting features for 42Street dataset")
                return extract_img_feature_street42(self.model, dataloader, self.device)
        with torch.no_grad():
            return extract_img_feature(self.model, dataloader, self.device)
