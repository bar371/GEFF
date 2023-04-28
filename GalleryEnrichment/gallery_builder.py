from abc import ABC, abstractmethod
from typing import List
import shutil


class GalleryBuilder(ABC):
    def __init__(self, query_imgs_paths: List, predicted_labels):
        self.query_imgs_paths = query_imgs_paths
        self.predicted_labels = predicted_labels
        self.dataset_path = None
        self.renamed_query = []

    @abstractmethod
    def _rename_query_for_gallery(self):
        pass

    @abstractmethod
    def _create_sequence_folders(self):
        pass

    @abstractmethod
    def _move_predicted_query_to_test(self):
        pass

    def create_query_based_gallery(self):
        print(f'Creating new gallery in {self.dataset_path}')

        # Rename the query images to match the gallery format:
        self._rename_query_for_gallery()
        assert self.renamed_query, "query gallery is empty"

        # Add all predicted images to the enriched gallery folder:
        self._create_sequence_folders()
        for i, new_query in enumerate(self.renamed_query):
            if new_query:  # Face was detected
                shutil.copy(self.query_imgs_paths[i], new_query)

        # Move all the enriched samples to the original folder of the gallery:
        self._move_predicted_query_to_test()


