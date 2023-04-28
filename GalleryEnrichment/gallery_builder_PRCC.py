import os
from typing import List
import shutil

from GalleryEnrichment.gallery_builder import GalleryBuilder

NUM_QUERY_SAME = 3873
NUM_QUERY_DIFF = 3543
UNDETECTED_FACE = '-01'


class GalleryBuilderPRCC(GalleryBuilder):
    def __init__(self, dataset_path, query_imgs_paths: List, predicted_labels, folder_char):
        super().__init__(query_imgs_paths, predicted_labels)
        self.predicted_labels = [f'{int(pred_label):03d}' for pred_label in self.predicted_labels]
        self.dataset_path = dataset_path
        self.id_img_counter = {}
        self.folder_char = folder_char

    def _rename_query_for_gallery(self):
        for i, query_img_path in enumerate(self.query_imgs_paths):
            predicted_label = self.predicted_labels[i]
            if predicted_label == UNDETECTED_FACE:  # this query wasn't relabeled, add ''
                self.renamed_query.append('')
            else:
                img_num = self._get_id_num_img(predicted_label)
                query_new_name = os.path.join(self.dataset_path, self.folder_char, f'{int(predicted_label):03d}', f'D_{img_num:04d}.jpg')
                self.renamed_query.append(query_new_name)

    def _get_id_num_img(self, predicted_label):
        self.id_img_counter[predicted_label] = self.id_img_counter.get(predicted_label, 0) + 1
        return self.id_img_counter[predicted_label]

    def _create_sequence_folders(self):
        for id in self.id_img_counter.keys():
            os.makedirs(os.path.join(self.dataset_path, self.folder_char, f'{int(id):03d}'), exist_ok=True)

    def _move_predicted_query_to_test(self):
        print(f'Copying images from {os.path.join(self.dataset_path, self.folder_char)} to {os.path.join(self.dataset_path, "A")}')
        id_folders = os.path.join(self.dataset_path, self.folder_char)
        for id_folder in os.listdir(id_folders):
            a_path = os.path.join(self.dataset_path, "A", id_folder)
            for d_file in os.listdir(os.path.join(id_folders, id_folder)):
                shutil.copy(os.path.join(id_folders, id_folder, d_file), a_path)

