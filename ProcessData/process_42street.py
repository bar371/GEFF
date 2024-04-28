import glob
import os
import os.path as osp
from pathlib import Path
from typing import List

from ProcessData.process_data_constants import STREET42, GALLERY, TRACKLETS, EXTRA_DATA
from ProcessData.process_dataset import ProcessDataset


class Process42Street(ProcessDataset):
    def __init__(self, data_base_path):
        super().__init__(data_base_path)
        self.dataset = STREET42
        self.dataset_dir = data_base_path
        self.query_dir_val = osp.join(self.dataset_dir, 'val', TRACKLETS)
        self.query_dir_test = osp.join(self.dataset_dir, 'test', TRACKLETS)
        self.gallery_dir = osp.join(self.dataset_dir, GALLERY)
        self.extra_data_dir = osp.join(self.dataset_dir, EXTRA_DATA)
        self.gpids = []
        self.qpids = []
        self.enriched_dir = None
        if osp.isdir(osp.join(self.dataset_dir, 'enriched_gallery')):
            self.enriched_dir = osp.join(self.dataset_dir, 'enriched_gallery')

    def create_imgs_paths(self, split) -> []:
        if split == TRACKLETS:
            img_paths, self.qpids = self._process_tracklets_to_query_paths(
                os.path.join(self.data_base_path, 'test', TRACKLETS))

        elif split == EXTRA_DATA:
            img_paths = self._create_extra_data_paths()

        elif split == GALLERY:
            img_paths, self.gpids = self._create_gallery_paths()

        else:
            raise NotImplementedError("Invalid split. Options: tracklets, gallery, extra_data")

        return img_paths

    def _process_tracklets_to_query_paths(self, path):
        videos = os.listdir(path)
        query_paths = []
        qpids = []
        print(f"Loading query for dataset..")
        for vid in videos:
            video_tracks = os.listdir(os.path.join(path, vid))
            for tracklet_path in video_tracks:
                img_paths = glob.glob(osp.join(path, vid, tracklet_path, '*.png'))
                img_paths.sort()
                qpids.extend([int(tracklet_path.split('_')[0])] * len(img_paths))
                query_paths.extend(img_paths)
        print(f'Done. {len(query_paths)} loaded.')
        return query_paths, qpids

    def _create_gallery_paths(self) -> []:
        imgs_paths = []
        gpids = []
        print(f"Loading gallery for dataset..")
        for img in glob.glob(self.gallery_dir + "/*"):
            suffix = img[-3:]
            if suffix != 'jpg' and suffix != 'png':
                continue
            if os.path.isfile(img):
                gpid = int(Path(img).name.split('_')[0])
                gpids.append(gpid)
                imgs_paths.append(img)
        print(f'Done. {len(imgs_paths)} loaded.')
        return imgs_paths, gpids

    def _create_extra_data_paths(self) -> []:
        imgs_paths = []
        print(f"Loading extra data for dataset..")
        for img in glob.glob(self.extra_data_dir + "/*"):
            suffix = img[-3:]
            if suffix != 'jpg' and suffix != 'png':
                continue
            if os.path.isfile(img):
                imgs_paths.append(img)
        print(f'Done. {len(imgs_paths)} loaded.')
        return imgs_paths

    def create_unique_name_from_img_path(self, img_path: str) -> str:
        """
        :param img_path: example "42street/test/tracklets/part5_s13000_e13501/001_000/v_part5_s13000_e13501_f0_bbox_733_179_974_1125.png"
        :return: /part5_s13000_e13501/001_000/v_part5_s13000_e13501_f0_bbox_733_179_974_1125.png
        """
        return "_".join(img_path.split(os.sep)[-3:])

    def convert_imgs_path_to_labels(self, img_paths) -> List[str]:
        """
        :param img_paths: a list with the paths for which the label should be extracted.
        :return: a list with the matching label for every input image path
        Example: given ["42street/test/tracklets/part5_s13000_e13501/001_000/v_part5_s13000_e13501_f0_bbox_733_179_974_1125.png"]
                 return ["001"]
        """
        assert  len(img_paths) > 0 , "img_paths seems to be empty"
        if TRACKLETS in img_paths[0]:
            return self.qpids

        elif GALLERY in img_paths[0]:
            return self.gpids

        elif EXTRA_DATA in img_paths[0]:
            return self.qpids

        else:
            raise Exception(f'Invalid type_flag. Options: {TRACKLETS}, {GALLERY}')

    def convert_img_path_to_sequence(self, img_path):
        """
        Given an image path, return the sequence to which the image belongs.
        Example: given "42street/test/tracklets/part5_s13000_e13501/001_000/v_part5_s13000_e13501_f0_bbox_733_179_974_1125.png"
                 return "part5_s13000_e13501_001_000"
        """
        img_path = os.path.normpath(img_path)
        session_num, sequence_num = img_path.split(os.sep)[-3: -1]
        return f'{session_num}_{sequence_num}'

    def extract_camids(self, imgs_paths):
        return None
