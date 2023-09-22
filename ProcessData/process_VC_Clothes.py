import os
from glob import glob
import tqdm
import numpy as np
from ProcessData.process_data_constants import VCCLOTHES
from ProcessData.process_dataset import ProcessDataset
import re

ID = 0
CLOTHES_COUNTER = 2


class ProcessVCClothes(ProcessDataset):
    def __init__(self, data_base_path):
        super().__init__(data_base_path)
        self.dataset = VCCLOTHES

    def create_imgs_paths(self, split, mode='all') -> []:
        pattern = re.compile(r'(\d+)-(\d+)-(\d+)-(\d+)')
        assert split in ['query', 'gallery'], "VC-Clothes split must be one of ['query', 'gallery']"
        imgs_paths = []
        print(f"Loading split {split} for VC-Clothes dataset..")
        glob_paths = glob(os.path.join(self.data_base_path, split) + "**/**", recursive=True)
        glob_paths.sort()
        for img in tqdm.tqdm(glob_paths):
            suffix = img[-3:]
            if suffix in ['jpg', 'png'] and os.path.isfile(img):
                pid, camid, clothes, _ = pattern.search(img).groups()
                camid = int(camid)
                if mode == 'sc' and camid not in [2, 3]:
                    continue
                if mode == 'cc' and camid not in [3, 4]:
                    continue
                imgs_paths.append(img)
                path_split = img.split(os.sep)[-1].split('-')
                clothes_label = f'{path_split[ID]}_{path_split[CLOTHES_COUNTER]}'
                if split == 'gallery':
                    self.clothes_ids_gallery.append(clothes_label)
                elif split == 'query':
                    self.clothes_ids_query.append(clothes_label)
        print(f'Done. {len(imgs_paths)} loaded.')
        return imgs_paths

    def convert_imgs_path_to_labels(self, img_paths):
        """
        :param img_paths: a list with the paths for which the label should be extracted.
        :return: a list with the matching label for every input image path
        Example: given ["./VC-Clothes/query/0001-02-03-04.jpg"]
                 return ["0001"]
        """
        labels = []
        for img_path in img_paths:
            img_path = os.path.normpath(img_path)
            labels.append(img_path.split(os.sep)[-1].split('-')[0])
        return labels

    def convert_img_path_to_sequence(self, img_path):
        pass

    def create_unique_name_from_img_path(self, img_path):
        """
        :param img_path: example "./VC-Clothes/query/0001-02-03-04.jpg"
        :return: query-0001-02-03-04.jpg
        """
        img_path = os.path.normpath(img_path)
        return "-".join(img_path.split(os.sep)[-2:])

    def extract_camids(self, imgs_paths):
        """
        Given an image path, return the camid of the image.
        :param img_path: example "./VC-Clothes/query/0001-02-03-04.jpg"
        :return: "2"
        """
        camids = []
        for img_path in imgs_paths:
            camid = os.path.normpath(img_path).split(os.sep)[-1].split('-')[1][1:]
            camids.append(camid)
        return np.array(camids)
