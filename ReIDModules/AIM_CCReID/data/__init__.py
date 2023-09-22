from torch.utils.data import SequentialSampler, DataLoader

import ReIDModules.AIM_CCReID.data.img_transforms as T
from ReIDModules.AIM_CCReID.data.dataloader import DataLoaderX
from ReIDModules.AIM_CCReID.data.dataset_loader import ImageDataset
from ReIDModules.AIM_CCReID.data.samplers import DistributedRandomIdentitySampler, DistributedInferenceSampler
from ReIDModules.AIM_CCReID.data.datasets.ltcc import LTCC
from ReIDModules.AIM_CCReID.data.datasets.prcc import PRCC

__factory = {
    'ltcc': LTCC,
    'prcc': PRCC,
}

def get_names():
    return list(__factory.keys())


def build_dataset(config):
    if config.DATA.DATASET not in __factory.keys():
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(config.DATA.DATASET, __factory.keys()))

    dataset = __factory[config.DATA.DATASET](root=config.DATA.ROOT)

    return dataset


def build_img_transforms(config):
    transform_train = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.RandomCroping(p=config.AUG.RC_PROB),
        T.RandomHorizontalFlip(p=config.AUG.RF_PROB),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.RandomErasing(probability=config.AUG.RE_PROB)
    ])
    transform_test = T.Compose([
        T.Resize((config.DATA.HEIGHT, config.DATA.WIDTH)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return transform_train, transform_test


def build_dataloader(config):
    dataset = build_dataset(config)
    transform_train, transform_test = build_img_transforms(config)

    galleryloader = DataLoader(dataset=ImageDataset(dataset.gallery, transform=transform_test),
                                sampler=SequentialSampler(dataset.gallery),
                                batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                pin_memory=True, drop_last=False, shuffle=False)

    enrichedloader = None
    if hasattr(dataset, 'enriched_gallery'):
        enrichedloader = DataLoader(dataset=ImageDataset(dataset.enriched_gallery, transform=transform_test),
                               sampler=SequentialSampler(dataset.enriched_gallery),
                               batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                               pin_memory=True, drop_last=False, shuffle=False)

    if config.DATA.DATASET == 'prcc':
        queryloader_same = DataLoader(dataset=ImageDataset(dataset.query_same, transform=transform_test),
                                    sampler=SequentialSampler(dataset.query_same),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False)
        queryloader_diff = DataLoader(dataset=ImageDataset(dataset.query_diff, transform=transform_test),
                                    sampler=SequentialSampler(dataset.query_diff),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False)

        return queryloader_same, queryloader_diff, galleryloader, dataset, enrichedloader
    else:
        queryloader = DataLoader(dataset=ImageDataset(dataset.query, transform=transform_test),
                                    sampler=SequentialSampler(dataset.query),
                                    batch_size=config.DATA.TEST_BATCH, num_workers=config.DATA.NUM_WORKERS,
                                    pin_memory=True, drop_last=False, shuffle=False)

        return queryloader, galleryloader, dataset, enrichedloader