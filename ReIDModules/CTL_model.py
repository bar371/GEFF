import torch

from ReIDModules.CAL.data.dataset_loader import Street42Dataset
from ReIDModules.centroids_reid.inference.inference_utils import _inference, _inference_42street
from ReIDModules.centroids_reid.train_ctl_model import CTLModel as CTL
from ReIDModules.ReID_model import ReIDModel

CTL_MODEL_PRETRAINED_PATH = "./checkpoints/CTL/resnet50-19c8e357.pth"

class CTLModel(ReIDModel):
    def __init__(self, device):
        super().__init__(device)

    def init_model(self, config, checkpoint_path=None):
        checkpoint = torch.load(checkpoint_path)
        checkpoint['hyper_parameters']['MODEL']['PRETRAIN_PATH'] = CTL_MODEL_PRETRAINED_PATH
        self.model = CTL._load_model_state(checkpoint)
        self.config = config
        print(f'Model loaded with checkpoint according to path: {checkpoint_path}')

    def extract_features(self, dataloader):
        if type(dataloader.dataset) == Street42Dataset:
            return self._extract_42street_img_features(dataloader)
        return self._extract_img_features(dataloader)

    def _extract_img_features(self, dataloader):
        extract_id = lambda x: x.rsplit("/", 1)[1].rsplit("_")[0]
        self.model.eval()
        if self.device != 'cpu':
            device = int(self.device.split(':')[1])
            self.model = self.model.cuda(device=device)
        with torch.no_grad():
            feats, pids  ,camids, clothes_ids = self._run_inference(dataloader=dataloader, device=self.device)
        return feats, pids, camids, clothes_ids

    def _run_inference(self, dataloader, device=None):
        """
        Inference method taken from ./centroids_reid/inference/inference_utils.py and modified to support device selection.
        The function receives a data loader and creates embeddings for all the images in the data loader.
        """

        embeddings = []
        pids = []
        camids = []
        clothes_ids = []
        for x in dataloader:
            b_embeddings, b_pids, b_camids, b_clothes_ids = _inference(self.model, x, device)
            for embedding, pid, cam_id, clothes_id in zip(b_embeddings, b_pids, b_camids, b_clothes_ids):
                pids.append(pid)
                embeddings.append(embedding.detach().cpu().numpy())
                camids.append(cam_id)
                clothes_ids.append(clothes_id)

        embeddings = torch.tensor(embeddings)
        pids = torch.tensor(pids)
        camids = torch.tensor(camids)
        clothes_ids = torch.tensor(clothes_ids)

        return embeddings, pids, camids, clothes_ids

    def _extract_42street_img_features(self, dataloader):
        self.model.eval()
        if self.device != 'cpu':
            device = int(self.device.split(':')[1])
            self.model = self.model.cuda(device=device)
        with torch.no_grad():
            feats, pids = self._run_inference_42street(dataloader=dataloader, device=self.device)
        return feats, pids

    def _run_inference_42street(self, dataloader, device=None):
        """
        Inference method taken from ./centroids_reid/inference/inference_utils.py and modified to support device selection.
        The function receives a data loader and creates embeddings for all the images in the data loader.
        """

        embeddings = []
        pids = []
        for x in dataloader:
            b_embeddings, b_pids = _inference_42street(self.model, x, device)
            for embedding, pid in zip(b_embeddings, b_pids):
                pids.append(pid)
                embeddings.append(embedding.detach().cpu().numpy())

        embeddings = torch.tensor(embeddings)
        pids = torch.tensor(pids)

        return embeddings, pids