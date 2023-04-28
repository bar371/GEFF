import os
from typing import List
import shutil

from GalleryEnrichment.gallery_builder import GalleryBuilder

CLOTHES_LABEL = '0'  # should be a unique label that does not exist in the original dataset.
QUERY_CAM_ID = 'c0'  # should be a unique camera id that does not exist in the original dataset.
UNDETECTED_FACE = '-01'


class GalleryBuilderLTCC(GalleryBuilder):
    def __init__(self, dataset_path, query_imgs_paths: List, predicted_labels, gallery_folder):
        super().__init__(query_imgs_paths, predicted_labels)
        self.dataset_path = dataset_path
        self.predicted_labels = [f'{int(pred_label):03d}' for pred_label in self.predicted_labels]
        self.id_img_counter = {}
        self.gallery_folder = gallery_folder

    def _rename_query_for_gallery(self):
        for i, query_img_path in enumerate(self.query_imgs_paths):
            predicted_label = self.predicted_labels[i]
            if predicted_label == UNDETECTED_FACE:  # this query wasn't relabeled, add ''
                self.renamed_query.append('')
            else:
                img_num = self._get_id_num_img(predicted_label)
                query_new_name = os.path.join(self.dataset_path, self.gallery_folder,
                                              f'{int(predicted_label):03d}_{CLOTHES_LABEL}_{QUERY_CAM_ID}_1{img_num:05d}.png')
                self.renamed_query.append(query_new_name)

    def _get_id_num_img(self, predicted_label):
        self.id_img_counter[predicted_label] = self.id_img_counter.get(predicted_label, 0) + 1
        return self.id_img_counter[predicted_label]

    def _create_sequence_folders(self):
        os.makedirs(os.path.join(self.dataset_path, self.gallery_folder), exist_ok=True)

    def _move_predicted_query_to_test(self):
        print(f'Copying images from {os.path.join(self.dataset_path, self.gallery_folder)} to {os.path.join(self.dataset_path, "test")}')
        relabeled_query_path = os.path.join(self.dataset_path, self.gallery_folder)
        for img_path in os.listdir(relabeled_query_path):
            shutil.copy(f'{os.path.join(relabeled_query_path, img_path)}', f"{os.path.join(self.dataset_path, 'test')}")