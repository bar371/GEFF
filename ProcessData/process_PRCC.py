from glob import glob
import tqdm
from ProcessData.process_data_constants import PRCC
from ProcessData.process_dataset import ProcessDataset
import os
from typing import List
import numpy as np

FOLDER_TO_CAMID = {'A': 1, 'B': 2, 'C': 3}


class ProcessPRCC(ProcessDataset):
    def __init__(self, data_base_path):
        super().__init__(data_base_path)
        self.dataset = PRCC

    def create_imgs_paths(self, split) -> []:
        assert split in ['A', 'B', 'C'], "PRCC split of query (test) folder must be either A,B or C"
        imgs_paths = []
        print(f"Loading split {split} for PRCC dataset..")
        glob_paths = glob(os.path.join(self.data_base_path, split) + "**/**", recursive=True)
        glob_paths.sort()
        for img in tqdm.tqdm(glob_paths):
            suffix = img[-3:]
            if suffix == 'jpg' and os.path.isfile(img):
                imgs_paths.append(img)
        print(f'Done. {len(imgs_paths)} loaded.')
        return imgs_paths

    def create_unique_name_from_img_path(self, img_path:str) -> str:
        """
        :param img_path: example "./prcc/rgb/test/A/001/cropped_rgb001.jpg"
        :return: session3_001_01_00306.jpg
        """
        img_path = os.path.normpath(img_path)
        return "_".join(img_path.split(os.sep)[-3:])

    def convert_imgs_path_to_labels(self, img_paths: list) -> List[str]:
        """
        :param img_paths: a list with the paths for which the label should be extracted.
        :return: a list with the matching label for every input image path
        Example: given ["./prcc/rgb/test/A/001/cropped_rgb001.jpg"]
                 return ["001"]
        """
        labels = []
        for img_path in img_paths:
            img_path = os.path.normpath(img_path)
            labels.append(img_path.split(os.sep)[-2])
        return labels

    def convert_img_path_to_sequence(self, img_path):
        """
        Given an image path, return the sequence to which the image belongs.
        Example: given "./prcc/rgb/test/A/001/cropped_rgb001.jpg"
                 return "A_001"
        """
        img_path = os.path.normpath(img_path)
        session_num, sequence_num = img_path.split(os.sep)[-3: -1]
        return f'{session_num}_{sequence_num}'

    def extract_camids(self, imgs_paths):
        """
        Given an image path, return the camid of the image.
        Example: given "/prcc/rgb/test/A/001/cropped_rgb001.jpg" return 1
        """
        camids = []
        for img_path in imgs_paths:
            camid = FOLDER_TO_CAMID[os.path.normpath(img_path).split(os.sep)[-3]]
            camids.append(camid)
        return np.array(camids)

