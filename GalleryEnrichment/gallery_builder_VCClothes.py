import os
from typing import List
import shutil
from pathlib import Path

from GalleryEnrichment.gallery_builder import GalleryBuilder

CLOTHES_LABEL = '00'  # should be a unique label that does not exist in the original dataset.
QUERY_CAM_ID = '00'  # should be a unique camera id that does not exist in the original dataset.
IMG_NUM_ADD = 30
UNDETECTED_FACE = '-001'


class GalleryBuilderVCClothes(GalleryBuilder):
    def __init__(self, dataset_path, query_imgs_paths: List, predicted_labels, gallery_folder, gallery_references):
        super().__init__(query_imgs_paths, predicted_labels)
        self.dataset_path = dataset_path
        self.predicted_labels = [f'{int(pred_label):04d}' for pred_label in self.predicted_labels]
        self.id_img_counter = {}
        self.gallery_folder = gallery_folder
        self.gallery_references = gallery_references

    def _rename_query_for_gallery(self):
        for i, query_img_path in enumerate(self.query_imgs_paths):
            predicted_label = self.predicted_labels[i]
            if predicted_label == UNDETECTED_FACE:  # this query wasn't relabeled, add ''
                self.renamed_query.append('')
            else:
                if self.gallery_references is not None:
                    query_new_name = os.path.join(self.dataset_path, self.gallery_folder,
                                                  f'{Path(self.gallery_references[i]).stem}_{i}.jpg')
                else:
                    img_num = self._get_id_num_img(predicted_label)
                    query_new_name = os.path.join(self.dataset_path, self.gallery_folder,
                                                  f'{int(predicted_label):04d}-{CLOTHES_LABEL}-{QUERY_CAM_ID}-{img_num:03d}.jpg')
                self.renamed_query.append(query_new_name)

    def _get_id_num_img(self, predicted_label):
        self.id_img_counter[predicted_label] = self.id_img_counter.get(predicted_label, 0) + 1
        return self.id_img_counter[predicted_label]

    def _create_sequence_folders(self):
        os.makedirs(os.path.join(self.dataset_path, self.gallery_folder), exist_ok=True)

    def _move_predicted_query_to_test(self):
        if self.gallery_references is None:  # when using the references we don't want to move the images to the original gallery
            print(f'Copying images from {os.path.join(self.dataset_path, self.gallery_folder)} to {os.path.join(self.dataset_path, "gallery")}')
            relabeled_query_path = os.path.join(self.dataset_path, self.gallery_folder)
            for img_path in os.listdir(relabeled_query_path):
                shutil.copy(f'{os.path.join(relabeled_query_path, img_path)}', f"{os.path.join(self.dataset_path, 'gallery')}")