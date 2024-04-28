import os
import shutil
from typing import List

from GalleryEnrichment.gallery_builder import GalleryBuilder

UNDETECTED_FACE = '-01'
QUERY_CAM_ID = 'c0'  # should be a unique camera id that does not exist in the original dataset.


class GalleryBuilder42Street(GalleryBuilder):
    def __init__(self, dataset_path, query_imgs_paths: List, predicted_labels, enriched_gallery_folder):
        super().__init__(query_imgs_paths, predicted_labels)
        self.predicted_labels = [f'{int(pred_label):03d}' for pred_label in self.predicted_labels]
        self.dataset_path = dataset_path
        self.enriched_gallery_folder = enriched_gallery_folder

    def _rename_query_for_gallery(self):
        current_gallery_idx = 0
        for i, query_img_path in enumerate(self.query_imgs_paths):
            predicted_label = self.predicted_labels[i]
            if predicted_label == UNDETECTED_FACE:  # this query wasn't relabeled, add ''
                self.renamed_query.append('')
            else:
                query_new_name = os.path.join(self.dataset_path, self.enriched_gallery_folder,
                                              f'{int(predicted_label):03d}_{QUERY_CAM_ID}_1{current_gallery_idx:07d}.png')
                self.renamed_query.append(query_new_name)
                current_gallery_idx += 1

    def _create_sequence_folders(self):
        os.makedirs(os.path.join(self.dataset_path, self.enriched_gallery_folder), exist_ok=True)

    def _move_predicted_query_to_test(self):
        print(f'Copying images from {os.path.join(self.dataset_path, self.enriched_gallery_folder)} to {os.path.join(self.dataset_path, "gallery")}')
        relabeled_query_path = os.path.join(self.dataset_path, self.enriched_gallery_folder)
        for img_path in os.listdir(relabeled_query_path):
            shutil.copy(f'{os.path.join(relabeled_query_path, img_path)}', f"{os.path.join(self.dataset_path, 'gallery')}")