import os
import argparse
import os.path as osp

import torch
import torch.nn as nn
from torch import distributed as dist

from configs.default_img import get_img_config
from data import build_dataloader
from models import build_model
from tools.utils import set_seed, get_logger
from test import test, test_prcc

def parse_option():
    parser = argparse.ArgumentParser(description='Train clothes-changing re-id model with clothes-based adversarial loss')
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    # Datasets
    parser.add_argument('--root', type=str, help="your root path to data directory")
    parser.add_argument('--dataset', type=str, default='ltcc', help="ltcc, prcc")
    # Miscs
    parser.add_argument('--output', type=str, help="your output path to save model and logs")
    parser.add_argument('--resume', type=str, metavar='PATH')
    parser.add_argument('--eval', action='store_true', help="evaluation only")
    parser.add_argument('--tag', type=str, help='tag for log file')
    parser.add_argument('--gpu', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')

    args, unparsed = parser.parse_known_args()
    config = get_img_config(args)

    return config

def main(config):
    # Build dataloader
    if config.DATA.DATASET == 'prcc':
        trainloader, queryloader_same, queryloader_diff, galleryloader, dataset, train_sampler = build_dataloader(config)
    else:
        trainloader, queryloader, galleryloader, dataset, train_sampler = build_dataloader(config)

    # Build model
    model, model2, fusion, classifier, clothes_classifier, clothes_classifier2 = build_model(config, dataset.num_train_pids, dataset.num_train_clothes)

    if config.MODEL.RESUME:
        logger.info("Loading checkpoint from '{}'".format(config.MODEL.RESUME))
        checkpoint = torch.load(config.MODEL.RESUME, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['model_state_dict'])
        
    local_rank = dist.get_rank()
    model = model.cuda(local_rank)
    torch.cuda.set_device(local_rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    if config.EVAL_MODE:
        logger.info("Evaluate only")
        with torch.no_grad():
            if config.DATA.DATASET == 'prcc':
                test_prcc(model, queryloader_same, queryloader_diff, galleryloader, dataset)
            else:
                test(config, model, queryloader, galleryloader, dataset)
        return

if __name__ == '__main__':
    config = parse_option()
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
    # Init dist
    dist.init_process_group(backend="nccl", init_method='env://')
    local_rank = dist.get_rank()
    # Set random seed
    set_seed(config.SEED + local_rank)
    # get logger
    if not config.EVAL_MODE:
        output_file = osp.join(config.OUTPUT, 'log_train.log')
    else:
        output_file = osp.join(config.OUTPUT, 'log_test.log')
    logger = get_logger(output_file, local_rank, 'reid')
    logger.info("Config:\n-----------------------------------------")
    logger.info(config)
    logger.info("-----------------------------------------")

    main(config)