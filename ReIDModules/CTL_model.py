import torch

from ReIDModules.centroids_reid.inference.inference_utils import create_pid_path_index, \
    calculate_centroids, _inference
from ReIDModules.centroids_reid.train_ctl_model import CTLModel as CTL
from ReIDModules.ReID_model import ReIDModel
import numpy as np

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
        return self._extract_img_features(dataloader)

    def _extract_img_features(self, dataloader):
        extract_id = lambda x: x.rsplit("/", 1)[1].rsplit("_")[0]
        self.model.eval()
        if self.device != 'cpu':
            device = int(self.device.split(':')[1])
            self.model = self.model.cuda(device=device)
        with torch.no_grad():
            print('Starting to create gallery feature vectors')
            feats, pids  ,camids, clothes_ids = self._run_inference(dataloader=dataloader, device=self.device)
            if self.config.MODEL.USE_CENTROIDS:
                # pid_path_index = create_pid_path_index(paths=paths_gallery, func=extract_id)
                # g_feat, paths_gallery = calculate_centroids(g_feat, pid_path_index)
                print('Created gallery feature vectors using centroids.')
            else:
                # paths_gallery = np.array([pid.split('/')[-1].split('_')[0] for pid in
                #                           paths_gallery])  # need to be only the string id of a person ('0015' etc.)
                print('Did not use centroids for gallery feature vectors.')
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
