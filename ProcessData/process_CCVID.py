import platform
import tqdm
from typing import List

from ProcessData.process_data_constants import CCVID
from ProcessData.process_dataset import ProcessDataset
import pandas as pd
import os
import numpy as np

SESSION_ID = "session_id"
LABEL = "label"
CLOTHES_TAG = "clothes_tag"


class ProcessCCVID(ProcessDataset):
    def __init__(self, data_base_path):
        super().__init__(data_base_path)
        self.clothes_ids_gallery = []
        self.clothes_ids_query = []
        self.session2clothes = {}
        self.dataset = CCVID

    def get_text_by_split(self, split):
        return os.path.join(self.data_base_path, split + '.txt')

    def create_imgs_paths(self, split) -> []:
        df = pd.read_csv(self.get_text_by_split(split), sep="\t", header=None)
        assert not df.empty
        df.columns = [SESSION_ID, LABEL, CLOTHES_TAG]
        imgs_paths = []
        print(f"Reading {split} images paths for CCVID dataset.")
        for index, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            row_ret = row[SESSION_ID]
            if platform.system() == "Windows":
                row_ret = row[SESSION_ID].replace("/", "\\")
            cur_session = os.path.join(self.data_base_path, row_ret)
            clothes_label = f'{row[LABEL]}_{row[CLOTHES_TAG]}'
            self.session2clothes[row_ret.replace(os.sep, '_')] = clothes_label
            for img in list(reversed(sorted(os.listdir(cur_session)))):
                imgs_paths.append(os.path.join(cur_session, img))
                if split == 'gallery':
                    self.clothes_ids_gallery.append(clothes_label)
                if split == 'query':
                    self.clothes_ids_query.append(clothes_label)

        print(f"Done loading {len(imgs_paths)} images paths from {split}.")
        return imgs_paths

    def create_unique_name_from_img_path(self, img_path: str) -> str:
        """
        :param img_path: example "/CCVID/session3/001_01/00306.jpg"
        :return: session3_001_01_00306.jpg
        """
        return "_".join(img_path.split(os.sep)[-3:])

    def convert_imgs_path_to_labels(self, img_paths: [list, np.array]) -> List[str]:
        """
        :param img_paths: a list with the paths for which the label should be extracted.
        :return: a list with the matching label for every input image path
        Example: given ["/CCVID/session3/001_01/00306.jpg"]
                 return ["001"]
        """
        labels = []
        for img_path in img_paths:
            labels.append(img_path.split(os.sep)[-2][:3])
        return labels

    def convert_img_path_to_sequence(self, img_path):
        """
        Given an image path, return the sequence to which the image belongs.
        Example: given "/CCVID/session3/001_01/00306.jpg"
                 return "session3_001_01"
        """
        img_path = os.path.normpath(img_path)
        session_num, sequence_num = img_path.split(os.sep)[-3: -1]
        return f'{session_num}_{sequence_num}'

    def extract_camids(self, imgs_paths):
        """
        Given an image path, return the camid of the image.
        Example: given "/CCVID/session2/001_01/00306.jpg"
                 return 1
        Note: for session3, we add +12 following the procedure of CAL: https://github.com/guxinqian/Simple-CCReID/issues/10.
        """
        camids = []
        for img_path in imgs_paths:
            session, tracklet_path = os.path.normpath(img_path).split(os.sep)[-3:-1]
            cam = tracklet_path.split('_')[1]
            if session == 'session3':
                camid = int(cam) + 12
            elif session == 'session4':
                camid = int(cam) + 12
            else:
                camid = int(cam)
            camids.append(camid)
        return np.array(camids)
