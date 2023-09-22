import shutil
from typing import List
import os
from pathlib import Path

from GalleryEnrichment.gallery_builder import GalleryBuilder


CLOTHES_LABEL = '999'  # should be a unique label that does not exist in the original dataset.
QUERY_CAM_ID = '30000'  # needs to be a number larger than the number of query images. (see line 41 in ./Simple-CCReID/data/datasets/last.py)
IMAGE_ID = '999'
VIDEO_ID = '99999'
BBOX = '99'
UNDETECTED_FACE = '-00001'


class GalleryBuilderLaST(GalleryBuilder):

    def __init__(self, dataset_path, query_imgs_paths: List, predicted_labels, gallery_folder, gallery_references):
        super().__init__(query_imgs_paths, predicted_labels)
        self.dataset_path = dataset_path
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
                    query_new_name = os.path.join(self.dataset_path, 'test', self.gallery_folder,
                                                  f'{Path(self.gallery_references[i]).stem}_{i}.jpg')
                else:
                    frame_num = self._get_id_num_img(predicted_label)
                    query_new_name = os.path.join(self.dataset_path, 'test', self.gallery_folder,
                                                  f'{predicted_label}_{IMAGE_ID}_{VIDEO_ID}_{frame_num:04d}_{BBOX}_{CLOTHES_LABEL}.jpg')
                self.renamed_query.append(query_new_name)

    def _get_id_num_img(self, predicted_label):
        if self.id_img_counter.get(predicted_label):
            self.id_img_counter[predicted_label] += 1
        else:
            self.id_img_counter[predicted_label] = 1
        return self.id_img_counter[predicted_label]

    def _create_sequence_folders(self):
        os.makedirs(os.path.join(self.dataset_path, 'test', self.gallery_folder), exist_ok=True)

    def _move_predicted_query_to_test(self):
        if self.gallery_references is None:  # when using the references we don't want to move the images to the original gallery
            relabeled_query_path = os.path.join(self.dataset_path, 'test', self.gallery_folder)
            enriched_gallery_path = os.path.join(self.dataset_path, 'test', 'gallery')
            print(f'Copying images from {relabeled_query_path} to {enriched_gallery_path}')
            for img_path in os.listdir(relabeled_query_path):
                predicted_id = img_path.split('_')[0]
                shutil.copy(f'{os.path.join(relabeled_query_path, img_path)}',
                            f"{os.path.join(enriched_gallery_path, predicted_id)}")
