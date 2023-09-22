import os
from glob import glob
import tqdm
import numpy as np
from ProcessData.process_data_constants import LAST
from ProcessData.process_dataset import ProcessDataset


class ProcessLaST(ProcessDataset):
    def __init__(self, data_base_path):
        super().__init__(data_base_path)
        self.dataset = LAST

    def create_imgs_paths(self, split) -> []:
        assert split in ['query', 'gallery'], "LaST split must be one of ['query', 'gallery']"
        imgs_paths = []
        print(f"Loading {split} for LaST dataset..")
        glob_paths = glob(os.path.join(self.data_base_path,'test', split) + "**/**", recursive=True)
        glob_paths.sort()
        for img in tqdm.tqdm(glob_paths):
            suffix = img[-3:]
            if suffix in ['jpg', 'png'] and os.path.isfile(img):
                imgs_paths.append(img)
        print(f'Done. {len(imgs_paths)} loaded.')
        return imgs_paths

    def convert_imgs_path_to_labels(self, img_paths):
        """
        :param img_paths: a list with the paths for which the label should be extracted.
        :return: a list with the matching label for every input image path
        Example: given ["./last/test/query/010903_012_01066_1075_01_000.jpg"]
                 return ["010903"]
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
        :param img_path: example "./last/test/query/010903_012_01066_1075_01_000.jpg"
        :return: query_010903_012_01066_1075_01_000.jpg
        """
        img_path = os.path.normpath(img_path)
        return "_".join(img_path.split(os.sep)[-2:])

    def extract_camids(self, imgs_paths):
        """
        Given an image path, return the camid of the image.
        In this dataset, no camid is provided, hence we set the camid to 1 for query images and 2 for gallery images to
        avoid filtering out images due to camids.
        """
        camids = []
        for img_path in imgs_paths:
            query_folder = os.path.normpath(img_path).split(os.sep)[-2]
            gallery_folder = os.path.normpath(img_path).split(os.sep)[-3]
            if query_folder == 'query':
                camids.append(1)
            elif gallery_folder == 'gallery':
                camids.append(2)
            else:
                raise Exception(f'Unrecognized data folder: {query_folder}, {gallery_folder}')
        return np.array(camids)
