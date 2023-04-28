from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, SequentialSampler
from ProcessData.process_data_constants import PRCC
from ReIDModules.CAL.data import build_dataset, build_img_transforms, ImageDataset


class ProcessDataset(ABC):
    def __init__(self, data_base_path):
        self.data_base_path = data_base_path
        self.dataset = None
        self.train_paths = None
        self.gallery_paths = None
        self.query_path = None
        self.clothes_ids_gallery = []
        self.clothes_ids_query = []
        self.session2clothes = {}

    @abstractmethod
    def create_imgs_paths(self, split) -> []:
        pass

    @abstractmethod
    def convert_imgs_path_to_labels(self, img_paths):
        pass

    @abstractmethod
    def convert_img_path_to_sequence(self, img_path):
        pass

    @abstractmethod
    def create_unique_name_from_img_path(self, img_path):
        pass

    @abstractmethod
    def extract_camids(self, imgs_paths):
        pass


def build_dataloader(config):
    dataset = build_dataset(config)

    _, transform_test = build_img_transforms(config)
    galleryloader = DataLoader(dataset=ImageDataset(dataset.gallery, transform=transform_test),
                               sampler=SequentialSampler(dataset.gallery),
                               batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                               pin_memory=True, drop_last=False, shuffle=False)

    if config.DATA.DATASET == PRCC:
        queryloader_same = DataLoader(dataset=ImageDataset(dataset.query_same, transform=transform_test),
                                 sampler=SequentialSampler(dataset.query_same),
                                 batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=False, shuffle=False)
        queryloader_diff = DataLoader(dataset=ImageDataset(dataset.query_diff, transform=transform_test),
                                 sampler=SequentialSampler(dataset.query_diff),
                                 batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=False, shuffle=False)

        return queryloader_same, queryloader_diff, galleryloader, dataset
    else:
        queryloader = DataLoader(dataset=ImageDataset(dataset.query, transform=transform_test),
                                 sampler=SequentialSampler(dataset.query),
                                 batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                 pin_memory=True, drop_last=False, shuffle=False)

        return queryloader, galleryloader, dataset
