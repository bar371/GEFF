import sys
from collections import defaultdict

sys.path.append('.')

from GalleryEnrichment.gallery_builder_PRCC import NUM_QUERY_SAME
from ProcessData.process_data_constants import *
from argparse import ArgumentParser
from ReIDModules.CAL.configs.default_img import _C
from ReIDModules.factory import ReID_module_factory
from evaluate import evalute_wrapper, evaluate_performance_ccvid
from ProcessData.process_dataset import build_custom_dataloader
from ProcessData.utils import prepare_dataset
import torch
from FaceModules.insightface import InsightFace
import torch.nn.functional as F
import numpy as np
from ReIDModules.CAL.tools.utils import set_seed
from pathlib import Path


def get_args():
    parser = ArgumentParser()
    parser.add_argument('dataset', help='On which dataset to run?', choices=DATASETS)
    parser.add_argument('reid_model', help='Which ReID module to use', choices=['CAL','AIM', 'CTL'])
    parser.add_argument('--dataset_path',
                        help='Full path to the dataset.')
    parser.add_argument('--detection_threshold',
                        help='The threshold that should be used for face detection in the gallery images', default=0.0)
    parser.add_argument('--similarity_threshold',
                        help='The threshold that should be used for similarity between query and gallery images',
                        default=0.0)
    parser.add_argument('--device', help='CPU or Cuda device num to use.', default='cpu')
    parser.add_argument('--reid_config', help='Config file for the ReID model.')
    parser.add_argument('--reid_checkpoint', help='Checkpoint for pre-trained ReID module.')
    parser.add_argument('--alpha', help='alpha combining reid and face modules', default=0.75)
    parser.add_argument('--CC', help='Discard images that have same clothes id between query and gallery samples ',
                        action='store_true')
    return parser.parse_args()


def get_config(args):
    config = None
    if args.reid_model == 'CTL':
        from ReIDModules.centroids_reid.config import cfg
        config = cfg.clone()
        config.defrost()
        config.merge_from_file(args.reid_config)
        if args.dataset:
            config.DATASETS.NAMES = args.dataset
        if args.dataset_path:
            config.DATASETS.ROOT_DIR = args.dataset_path

    elif args.reid_model in ['AIM','CAL']:
        config = _C.clone()
        config.defrost()
        config.merge_from_file(args.reid_config)
        if args.dataset:
            config.DATA.DATASET = args.dataset
        if args.dataset_path:
            config.DATA.ROOT = args.dataset_path
    config.freeze()
    return config


def get_cosine_distmat(qf, gf, device):
    if args.device != 'cpu':
        qf, gf = qf.cuda(device), gf.cuda(device)
    qf = F.normalize(qf, p=2, dim=1)
    gf = F.normalize(gf, p=2, dim=1)
    m, n = qf.size(0), gf.size(0)
    distmat = torch.zeros((m, n))
    qf, gf = qf.cuda(), gf.cuda()
    for i in range(m):
        distmat[i] = (- torch.mm(qf[i:i + 1], gf.t())).cpu()
    distmat = distmat.numpy()
    return distmat


def convert_ccvid_to_img_dataset(dataloader):
    temp_dataset = []
    track_to_imgs_mapper = defaultdict(list)
    img_counter = 0
    for track in dataloader.dataset.dataset:
        for img_path in track[0]:
            temp_dataset.append((img_path, track[1], track[2], track[3]))
            track_id = create_unique_ccvid_track(img_path, dataset_origin=CAL_DATALOADER)
            track_to_imgs_mapper[track_id].append(img_counter)
            img_counter += 1

    dataloader.dataset.dataset = temp_dataset
    dataloader.sampler.data_source = temp_dataset
    return dataloader, track_to_imgs_mapper


def create_unique_ccvid_track(img_path, dataset_origin):
    if dataset_origin == CAL_DATALOADER:
        path_split = img_path.split(os.sep)
        track_id = path_split[-2: -1][0]
        if "/" in track_id:
            track_id = track_id.replace('/', '_')
        else:
            track_id = f"{path_split[-3]}_{track_id}"
    else:
        session_num, sequence_num = img_path.split(os.sep)[-3: -1]
        track_id = f'{session_num}_{sequence_num}'

    return track_id


def get_score_all_ids_full_gallery(simmat, g_pids):
    ids_score = {i: 0 for i in set(g_pids)}
    for pid in set(g_pids):
        id_indices = np.where(g_pids == int(pid))
        if len(id_indices[0] > 0):
            id_matrix = simmat[:, id_indices[0]]
            if len(id_matrix.shape) == 2:
                ids_score[pid] += np.max(np.nan_to_num(id_matrix), axis=1).mean()
            # something went wrong
            else:
                raise ValueError('ID cosine similarity matrix is not in the correct dims.')
    return ids_score


def find_best_match(q_feat, q_camid, g_feats, g_pids, g_camids, q_clothes_id, g_clothes_id):
    """
    Given feature vectors of the query images, return the ids of the images that are most similar in the test gallery
    """
    features = F.normalize(q_feat, p=2, dim=1)
    others = F.normalize(g_feats.type(torch.float32), p=2, dim=1)
    simmat = torch.mm(features, others.t()).cpu().numpy()
    same_camid_indices = np.where(g_camids == q_camid[0])[0]  # Ignore prediction with the same camid
    simmat[:, same_camid_indices] = -np.inf
    if args.CC:
        # Filter out appearances with the same clothes id
        same_clothes_indices = np.where(g_clothes_id == q_clothes_id[0])[0]
        simmat[:, same_clothes_indices] = -np.inf
    ids_score = get_score_all_ids_full_gallery(simmat, g_pids)
    return ids_score


def reid_inference_on_ccvid(args, config):
    queryloader, galleryloader, dataset = build_custom_dataloader(config, model_name=args.model)
    queryloader, q_track_mapper = convert_ccvid_to_img_dataset(queryloader)
    galleryloader, _ = convert_ccvid_to_img_dataset(galleryloader)
    reid_model = ReID_module_factory(args.reid_model, device=args.device)
    reid_model.init_model(config, args.reid_checkpoint)
    print(f'Extracting reid query features in {args.dataset}')
    qf, q_pids, q_camids, q_clothes_ids = reid_model.extract_features(queryloader)
    print(f'Extracting reid gallery features in {args.dataset}')
    gf, g_pids, g_camids, g_clothes_ids = reid_model.extract_features(galleryloader)

    q_score_vectors = defaultdict(dict)
    # go over all tracks and create a feature vector for each track (with the size of #identities)
    for track_id, q_track_indices in q_track_mapper.items():
        track_scores = find_best_match(qf[q_track_indices], q_camids[q_track_indices], gf, np.array(g_pids, dtype=int),
                                       np.array(g_camids),
                                       q_clothes_ids[q_track_indices], g_clothes_ids)
        q_score_vectors[track_id] = track_scores
    return q_score_vectors, q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids


def reid_inference_on_prcc(args, config):
    queryloader_same, queryloader_diff, galleryloader, dataset, enrichedloader = build_custom_dataloader(config, model_name=args.reid_model)
    reid_model = ReID_module_factory(model_name=args.reid_model, device=args.device)
    reid_model.init_model(config, args.reid_checkpoint)
    print('Extracting query features of same clothes samples in PRCC')
    qsf, qs_pids, qs_camids, qs_clothes_ids = reid_model.extract_features(queryloader_same)
    print('Extracting query features of different clothes samples in PRCC')
    qdf, qd_pids, qd_camids, qd_clothes_ids = reid_model.extract_features(queryloader_diff)

    print('Extracting gallery features in PRCC')
    gf, g_pids, g_camids, g_clothes_ids = reid_model.extract_features(galleryloader)

    print('Computing distance matrix')
    distmat_same = get_cosine_distmat(qf=qsf, gf=gf, device=reid_model.device)
    distmat_diff = get_cosine_distmat(qf=qdf, gf=gf, device=reid_model.device)

    if enrichedloader:
        print(f'Extracting reid enriched gallery features in {args.dataset}')
        egf, _, _, _ = reid_model.extract_features(enrichedloader)

        print(f'Computing reid enriched distmat_same in {args.dataset}')
        enriched_distmat_same = get_cosine_distmat(qf=qsf, gf=egf, device=reid_model.device)

        # create unique paths:
        gallery_paths = [("_".join(img_path[0].split(os.sep)[-3:]), ) for img_path in dataset.gallery]
        enriched_paths = [("_".join(img_path[0].split(os.sep)[-3:]), ) for img_path in dataset.enriched_gallery]
        distmat_same = apply_enriched_gallery(distmat_same, enriched_distmat_same, gallery_paths, enriched_paths)

        print(f'Computing reid enriched distmat_diff in {args.dataset}')
        enriched_distmat_diff = get_cosine_distmat(qf=qdf, gf=egf, device=reid_model.device)
        distmat_diff = apply_enriched_gallery(distmat_diff, enriched_distmat_diff, gallery_paths, enriched_paths)

    qs_pids, qs_camids, qs_clothes_ids = qs_pids.numpy(), qs_camids.numpy(), qs_clothes_ids.numpy()
    qd_pids, qd_camids, qd_clothes_ids = qd_pids.numpy(), qd_camids.numpy(), qd_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()

    return distmat_diff, \
           distmat_same, \
           qs_pids, qs_camids, qs_clothes_ids, \
           qd_pids, qd_camids, qd_clothes_ids, \
           g_pids, g_camids, g_clothes_ids


def reid_inference(args, config):
    queryloader, galleryloader, dataset, enrichedloader = build_custom_dataloader(config, model_name=args.reid_model)
    reid_model = ReID_module_factory(args.reid_model, device=args.device)
    reid_model.init_model(config, args.reid_checkpoint)
    print(f'Extracting reid query features in {args.dataset}')
    qf, q_pids, q_camids, q_clothes_ids = reid_model.extract_features(queryloader)
    print(f'Extracting reid gallery features in {args.dataset}')
    gf, g_pids, g_camids, g_clothes_ids = reid_model.extract_features(galleryloader)
    print(f'Computing reid distmat in {args.dataset}')
    distmat = get_cosine_distmat(qf=qf, gf=gf, device=reid_model.device)

    if enrichedloader:
        print(f'Extracting reid enriched gallery features in {args.dataset}')
        egf, _, _, _ = reid_model.extract_features(enrichedloader)

        print(f'Computing reid enriched distmat in {args.dataset}')
        enriched_distmat = get_cosine_distmat(qf=qf, gf=egf, device=reid_model.device)
        if args.dataset == LAST:
            gallery_paths = [(img_path[0].split(os.sep)[-1], ) for img_path in dataset.gallery]
            enriched_paths = [(img_path[0].split(os.sep)[-1], ) for img_path in dataset.enriched_gallery]
        else:
            gallery_paths = dataset.gallery
            enriched_paths = dataset.enriched_gallery
        distmat = apply_enriched_gallery(distmat, enriched_distmat, gallery_paths, enriched_paths)

    q_pids, q_camids, q_clothes_ids = q_pids.numpy(), q_camids.numpy(), q_clothes_ids.numpy()
    g_pids, g_camids, g_clothes_ids = g_pids.numpy(), g_camids.numpy(), g_clothes_ids.numpy()
    return distmat, q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids


def apply_enriched_gallery(distmat, enriched_distmat, gallery_paths, enriched_gallery_paths):
    enriched_to_orig_mapping = {}
    for e_idx, enriched_sample in enumerate(enriched_gallery_paths):
        orig_gallery_name = ('_').join((Path(enriched_sample[0]).name.split('_')[:-1]))
        for o_idx, gallery_sample in enumerate(gallery_paths):
            if orig_gallery_name in gallery_sample[0]:
                enriched_to_orig_mapping[e_idx] = o_idx
    for q_idx in range(len(distmat)):
        replaced_imgs = 0
        for e_idx in range(len(enriched_gallery_paths)):
            if enriched_distmat[q_idx][e_idx] < distmat[q_idx][enriched_to_orig_mapping[e_idx]]:
                distmat[q_idx][enriched_to_orig_mapping[e_idx]] = enriched_distmat[q_idx][e_idx]
                replaced_imgs += 1
        # print(f'Replaced the distance of {replaced_imgs} gallery samples for query {q_idx}')
    return distmat



def face_inference(args):
    dataset_processor, gallery_paths, query_paths = prepare_dataset(args)

    # Initialize the face model:
    face_model = InsightFace(gallery_paths, query_paths, dataset_processor, device=args.device,
                             detection_threshold=float(args.detection_threshold),
                             similarity_threshold=args.similarity_threshold,
                             CC=args.CC)
    face_model.init()

    # Run the model (detect faces in query and gallery and compute similarities)
    face_model.compute_feature_vectors()
    face_model.compute_similarities(gallery_enrichment=False)
    distmat = 1 - face_model.similarities
    return distmat


def ccvid_face_inference(args):
    dataset_processor, gallery_paths, query_paths = prepare_dataset(args)

    # Initialize the face model used for gallery enrichment:
    face_model = InsightFace(gallery_paths, query_paths, dataset_processor, device=args.device,
                             detection_threshold=float(args.detection_threshold),
                             similarity_threshold=args.similarity_threshold,
                             CC=args.CC)
    face_model.init()
    face_model.compute_feature_vectors()
    face_tracks = defaultdict(list)
    for i, img_path in enumerate(face_model.query_imgs_paths):
        current_track = create_unique_ccvid_track(img_path=img_path, dataset_origin=FACE_DATALOADER)
        face_tracks[current_track].append(i)

    face_track_scores = defaultdict(dict)
    g_face_numpy = np.zeros((len(face_model.gallery_features), face_model.gallery_features[0].shape[-1]))
    for i, vector in enumerate(face_model.gallery_features):
        g_face_numpy[i, :] = vector
    g_face_tensor = torch.tensor(g_face_numpy, dtype=torch.float32, device=args.device)
    for track, indices in face_tracks.items():
        q_face_numpy = np.zeros((len(face_model.query_features), face_model.query_features[0].shape[-1]))
        for i, vector in enumerate(face_model.query_features):
            q_face_numpy[i, :] = vector
        q_face_tensor = torch.tensor(q_face_numpy, dtype=torch.float32, device=args.device)[indices]
        face_track_scores[track] = find_best_match(q_feat=q_face_tensor,
                                                   q_camid=face_model.q_camids,
                                                   g_feats=g_face_tensor,
                                                   g_pids=np.array(face_model.g_pids, dtype=int),
                                                   g_camids=face_model.g_camids,
                                                   q_clothes_id=dataset_processor.clothes_ids_query,
                                                   g_clothes_id=dataset_processor.clothes_ids_gallery)
    return face_track_scores


def run_inference(args, config):
    alpha = float(args.alpha)
    if args.dataset == PRCC:
        reid_distmat_diff, reid_distmat_same, qs_pids, qs_camids, qs_clothes_ids, \
        qd_pids, qd_camids, qd_clothes_ids, g_pids, g_camids, g_clothes_ids = reid_inference_on_prcc(args, config)
        if float(alpha) == 1.0:
            distmat_same = reid_distmat_same
            distmat_diff = reid_distmat_diff
        else:
            face_distmat = face_inference(args)
            distmat_same = alpha * reid_distmat_same + (1 - alpha) * face_distmat[:NUM_QUERY_SAME, :]
            distmat_diff = alpha * reid_distmat_diff + (1 - alpha) * face_distmat[NUM_QUERY_SAME:, :]

        evalute_wrapper(args.dataset, distmat_same, qs_pids, g_pids, qs_camids, g_camids, g_clothes_ids=None,
                        q_clothes_ids=None, extra_msg="Computing CMC and mAP for the same clothes setting")

        evalute_wrapper(args.dataset, distmat_diff, qd_pids, g_pids, qd_camids, g_camids, g_clothes_ids=None,
                        q_clothes_ids=None, extra_msg="Computing CMC and mAP only for clothes changing")
    elif args.dataset == CCVID:
        reid_score_vectors, q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids = reid_inference_on_ccvid(
            args, config)
        g_pids = np.array(g_pids, dtype=int)
        face_score_vectors = ccvid_face_inference(args)
        all_tracks_results = []
        for track in face_score_vectors.keys():
            track_scores = {i: 0 for i in set(g_pids)}
            for pid in track_scores.keys():
                track_scores[pid] = reid_score_vectors[track][pid] * alpha + (1 - alpha) * face_score_vectors[track][
                    pid]
            all_tracks_results.append({'final_scores': track_scores, 'track_id': track})

        evaluate_performance_ccvid(all_tracks_results, args.alpha)

    else:
        # LTCC, LAST, VC-Clothes
        if float(alpha) == 1.0:  # only ReID module should be used
            distmat, q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids = reid_inference(args, config)
        elif float(alpha) == 0.0:  # only Face module should be used
            distmat = face_inference(args)
            _, q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids = reid_inference(args, config)
        else:  # combination of both Face and ReID modules.
            face_distmat = face_inference(args)
            reid_distmat, q_pids, q_camids, q_clothes_ids, g_pids, g_camids, g_clothes_ids = reid_inference(args, config)
            distmat = alpha * reid_distmat + (1 - alpha) * face_distmat

        evalute_wrapper(args.dataset, distmat, q_pids, g_pids, q_camids, g_camids,
                        q_clothes_ids=q_clothes_ids,
                        g_clothes_ids=g_clothes_ids,
                        extra_msg='')


def main(args):
    config = get_config(args)
    if args.reid_model == 'CTL':
        print(f'Setting seed to: {config.REPRODUCIBLE_SEED}')
        set_seed(config.REPRODUCIBLE_SEED)
    else:
        print(f'Setting seed to: {config.SEED}')
        set_seed(config.SEED)
    run_inference(args, config=config)


if __name__ == '__main__':
    args = get_args()
    main(args)
