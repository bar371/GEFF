import os

from ProcessData.process_CCVID import ProcessCCVID
from ProcessData.process_LTCC import ProcessLTCC
from ProcessData.process_LaST import ProcessLaST
from ProcessData.process_PRCC import ProcessPRCC
from ProcessData.process_data_constants import CCVID, PRCC, PRCC_TEST_PATH, LTCC, LAST, DATASETS


def prepare_dataset(args):
    if args.dataset == CCVID:
        dataset_processor = ProcessCCVID(args.dataset_path)
        gallery_paths = dataset_processor.create_imgs_paths(split='gallery')
        query_paths = dataset_processor.create_imgs_paths(split='query')

    elif args.dataset == PRCC:
        dataset_path = os.path.join(args.dataset_path, PRCC_TEST_PATH)
        dataset_processor = ProcessPRCC(dataset_path)
        gallery_paths = dataset_processor.create_imgs_paths(split='A')
        query_paths = dataset_processor.create_imgs_paths(split='B')
        query_paths.extend(dataset_processor.create_imgs_paths(split='C'))

    elif args.dataset == LTCC:
        dataset_processor = ProcessLTCC(args.dataset_path)
        gallery_paths = dataset_processor.create_imgs_paths(split='test')
        query_paths = dataset_processor.create_imgs_paths(split='query')

    elif args.dataset == LAST:
        dataset_processor = ProcessLaST(args.dataset_path)
        gallery_paths = dataset_processor.create_imgs_paths(split='gallery')
        query_paths = dataset_processor.create_imgs_paths(split='query')
    else:
        raise Exception(f'Invalid dataset. Options: {DATASETS}')

    return dataset_processor, gallery_paths, query_paths
