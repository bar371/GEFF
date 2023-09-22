import os
from glob import glob
import tqdm
import numpy as np
from ProcessData.process_data_constants import LTCC
from ProcessData.process_dataset import ProcessDataset

ID = 0
CLOTHES_COUNTER = 1


class ProcessLTCC(ProcessDataset):
    def __init__(self, data_base_path):
        super().__init__(data_base_path)
        self.dataset = LTCC

    def create_imgs_paths(self, split) -> []:
        assert split in ['query', 'test'], "LTCC split must be one of ['query', 'test']"
        imgs_paths = []
        print(f"Loading split {split} for LTCC dataset..")
        glob_paths = glob(os.path.join(self.data_base_path, split) + "**/**", recursive=True)
        glob_paths.sort()
        for img in tqdm.tqdm(glob_paths):
            suffix = img[-3:]
            if suffix in ['jpg', 'png'] and os.path.isfile(img):
                imgs_paths.append(img)
                path_split = img.split(os.sep)[-1].split('_')
                clothes_label = f'{path_split[ID]}_{path_split[CLOTHES_COUNTER]}'
                if split == 'test':
                    self.clothes_ids_gallery.append(clothes_label)
                elif split == 'query':
                    self.clothes_ids_query.append(clothes_label)
        print(f'Done. {len(imgs_paths)} loaded.')
        return imgs_paths

    def convert_imgs_path_to_labels(self, img_paths):
        """
        :param img_paths: a list with the paths for which the label should be extracted.
        :return: a list with the matching label for every input image path
        Example: given ["./LTCC_ReID/query/148_1_c3_014811.png"]
                 return ["148"]
        """
        labels = []
        for img_path in img_paths:
            img_path = os.path.normpath(img_path)
            labels.append(img_path.split(os.sep)[-1].split('_')[0])
        return labels

    def convert_img_path_to_sequence(self, img_path):
        pass

    def create_unique_name_from_img_path(self, img_path):
        """
        :param img_path: example "./LTCC_ReID/query/148_1_c3_014811.png"
        :return: query_148_1_c3_014811.png
        """
        img_path = os.path.normpath(img_path)
        return "_".join(img_path.split(os.sep)[-2:])

    def extract_camids(self, imgs_paths):
        """
        Given an image path, return the camid of the image.
        :param img_path: example "/LTCC_ReID/query/148_1_c3_014811.png"
        :return: "3"
        """
        camids = []
        for img_path in imgs_paths:
            camid = os.path.normpath(img_path).split(os.sep)[-1].split('_')[2][1:]
            camids.append(camid)
        return np.array(camids)
