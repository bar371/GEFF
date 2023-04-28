import os
from typing import List

from GalleryEnrichment.gallery_builder import GalleryBuilder

SESSION_NUM = 'session4'
CLOTHES_LABEL = 'u5_l5_s5_c5_a5'  # should be a unique label that does not exist in the original dataset.
FIRST_SEQ = 13  # In evaluation time the model filters all sequences that have the same cam_id as the current query
                # sequence. To avoid this on the relabeled query sequences we add to the gallery, we set the initial
                # sequence counter to be bigger than the biggest sequence in the original dataset.
UNDETECTED_FACE = '-01'


class GalleryBuilderCCVID(GalleryBuilder):
    def __init__(self, dataset_path, query_imgs_paths: List, predicted_labels, gallery_file):
        super().__init__(query_imgs_paths, predicted_labels)
        self.predicted_labels = [f'{int(pred_label):03d}' for pred_label in self.predicted_labels]
        self.dataset_path = dataset_path
        self.gallery_file = gallery_file
        self.seq_mapper = {}
        self.predicted_seq_counter = None  # initialized in 'init_seq_counter'
        self._init_seq_counter()

    def _init_seq_counter(self):
        predicted_ids = set(self.predicted_labels)
        self.predicted_seq_counter = {predicted_id: FIRST_SEQ for predicted_id in predicted_ids}

    def _rename_query_for_gallery(self):
        for i, query_img_path in enumerate(self.query_imgs_paths):
            predicted_label = self.predicted_labels[i]
            if predicted_label == UNDETECTED_FACE:  # this query wasn't relabeled, add ''
                self.renamed_query.append('')
            else:
                seq_num = self._get_seq_num(query_img_path, predicted_label)
                query_new_name = os.path.join(self.dataset_path, SESSION_NUM, seq_num, query_img_path.split(os.sep)[-1])
                self.renamed_query.append(query_new_name)

    def _get_seq_num(self, query_img_path, predicted_label):
        session_num, orig_seq, im_num = query_img_path.split(os.sep)[-3:]
        seq_map = f'{session_num}_{orig_seq}_{predicted_label}'
        if not self.seq_mapper.get(seq_map):
            self.predicted_seq_counter[predicted_label] += 1
            seq_num = self.predicted_seq_counter.get(predicted_label)
            self.seq_mapper[seq_map] = f'{predicted_label}_{seq_num:02d}'
        return self.seq_mapper.get(seq_map)

    def _create_sequence_folders(self):
        with open(os.path.join(self.dataset_path, f'{self.gallery_file}.txt'), 'w') as f:
            for sequence in self.seq_mapper.values():
                os.makedirs(os.path.join(self.dataset_path, SESSION_NUM, sequence), exist_ok=True)
                f.write(f'{SESSION_NUM}/{sequence}\t{sequence.split("_")[0]}\t{CLOTHES_LABEL}\n')

    def _move_predicted_query_to_test(self):
        with open(os.path.join(self.dataset_path, f'{self.gallery_file}.txt'), 'r') as enriched_gallery, \
                open(os.path.join(self.dataset_path, f'gallery.txt'), 'a') as orig_gallery:
            for line in enriched_gallery:
                orig_gallery.write(line)
