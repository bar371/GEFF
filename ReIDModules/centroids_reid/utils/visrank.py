from __future__ import print_function, absolute_import
import numpy as np
import shutil
import cv2
import os

import sys

sys.path.append("..")
from ReIDModules.centroids_reid.datasets.transforms.build import ReidTransforms
from ReIDModules.centroids_reid.datasets.bases import BaseDatasetLabelled


__all__ = ["visualize_ranked_results"]

GRID_SPACING = 2
QUERY_EXTRA_SPACING = 8
BW = 3  # border width
GREEN = (0, 255, 0)
RED = (0, 0, 255)


def visualize_ranked_results(
    distmat,
    dataset,
    data_type,
    cfg,
    width=128,
    height=256,
    save_dir="",
    topk=10,
):
    """Visualizes ranked results.

    Supports both image-reid and video-reid.

    For image-reid, ranks will be plotted in a single figure. For video-reid, ranks will be
    saved in folders each containing a tracklet.

    Args:
        distmat (numpy.ndarray): distance matrix of shape (num_query, num_gallery).
        dataset (tuple): a 2-tuple containing (query, gallery), each of which contains
            tuples of (img_path(s), pid, camid).
        data_type (str): "image" or "video".
        width (int, optional): resized image width. Default is 128.
        height (int, optional): resized image height. Default is 256.
        save_dir (str): directory to save output images.
        topk (int, optional): denoting top-k images in the rank list to be visualized.
            Default is 10.
    """
    num_q, num_g = distmat.shape
    os.makedirs(save_dir, exist_ok=True)
    trans = ReidTransforms(cfg)
    val_transforms = trans.build_transforms(is_train=False)
    query = BaseDatasetLabelled(dataset[:num_q], val_transforms, return_paths=True)

    # Only for centroid eval mode it makes sense
    respect_camids = cfg.MODEL.USE_CENTROIDS & cfg.MODEL.KEEP_CAMID_CENTROIDS

    if cfg.MODEL.USE_CENTROIDS:
        gallery_list = np.asarray(dataset[num_q:])
        labels_gallery = np.asarray([int(item[1]) for item in gallery_list])
        camids = np.asarray(
            [int(item[2]) for item in [*dataset[:num_q], *gallery_list]]
        )

        labels_query = np.asarray([int(item[1]) for item in dataset[:num_q]])

        from collections import defaultdict
        import random

        random.seed(0)

        labels2idx = defaultdict(list)
        for idx, label in enumerate(labels_gallery):
            labels2idx[int(label)].append(idx)

        labels2idx_q = defaultdict(list)
        for idx, label in enumerate(labels_query):
            labels2idx_q[int(label)].append(idx)

        unique_labels = sorted(np.unique(list(labels2idx.keys())))
        centroids = []

        # Create centroids for each pid seperately
        for label in unique_labels:
            cmaids_combinations = set()
            inds = labels2idx[label]
            inds_q = labels2idx_q[label]
            if respect_camids:
                selected_camids_g = camids[inds]

                selected_camids_q = camids[inds_q]
                unique_camids = sorted(np.unique(selected_camids_q))

                for current_camid in unique_camids:
                    # We want to select all gallery images that comes from DIFFERENT cameraId
                    camid_inds = np.where(selected_camids_g != current_camid)[0]
                    if camid_inds.shape[0] == 0:
                        continue
                    used_camids = tuple(
                        sorted(
                            np.unique(
                                [
                                    cid
                                    for cid in selected_camids_g
                                    if cid != current_camid
                                ]
                            )
                        )
                    )
                    if used_camids not in cmaids_combinations:
                        cmaids_combinations.add(used_camids)
                        centroid = list(random.choice(gallery_list[inds][camid_inds]))

                        centroid[2] = used_camids
                        centroids.append(centroid)

        gallery_input = centroids
    else:
        gallery_input = dataset[num_q:]

    gallery = BaseDatasetLabelled(gallery_input, val_transforms, return_paths=True)

    assert num_q == len(query)
    assert num_g == len(
        gallery
    )  ## Not checked when using aproximated visualization for centroids

    indices = np.argsort(distmat, axis=1)

    def _cp_img_to(src, dst, rank, prefix, matched=False):
        """
        Args:
            src: image path or tuple (for vidreid)
            dst: target directory
            rank: int, denoting ranked position, starting from 1
            prefix: string
            matched: bool
        """
        if isinstance(src, (tuple, list)):
            if prefix == "gallery":
                suffix = "TRUE" if matched else "FALSE"
                dst = (
                    os.path.join(dst, prefix + "_top" + str(rank).zfill(3))
                    + "_"
                    + suffix
                )
            else:
                dst = os.path.join(dst, prefix + "_top" + str(rank).zfill(3))
            os.makedirs(dst, exist_ok=True)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = os.path.join(
                dst,
                prefix + "_top" + str(rank).zfill(3) + "_name_" + os.path.basename(src),
            )
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        qimg, qpid, qcamid, qimg_path = query[q_idx]
        qimg_path_name = (
            qimg_path[0] if isinstance(qimg_path, (tuple, list)) else qimg_path
        )

        if data_type == "image":
            qimg = cv2.imread(qimg_path)
            qimg = cv2.resize(qimg, (width, height))
            qimg = cv2.copyMakeBorder(
                qimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            # resize twice to ensure that the border width is consistent across images
            qimg = cv2.resize(qimg, (width, height))
            num_cols = topk + 1
            grid_img = 255 * np.ones(
                (
                    height,
                    num_cols * width + topk * GRID_SPACING + QUERY_EXTRA_SPACING,
                    3,
                ),
                dtype=np.uint8,
            )
            grid_img[:, :width, :] = qimg
        else:
            qdir = os.path.join(
                save_dir, os.path.basename(os.path.splitext(qimg_path_name)[0])
            )
            os.makedirs(qdir, exist_ok=True)
            _cp_img_to(qimg_path, qdir, rank=0, prefix="query")

        rank_idx = 1
        for g_idx in indices[q_idx, :]:
            gimg, gpid, gcamid, gimg_path = gallery[g_idx]

            if respect_camids:
                # invalid = (gpid == qpid) & (gcamid != qcamid)
                invalid = (int(gpid) == int(qpid)) & (qcamid in gcamid)
            else:
                invalid = (qpid == gpid) & (qcamid == gcamid)

            if not invalid:
                matched = int(gpid) == int(qpid)
                if data_type == "image":
                    border_color = GREEN if matched else RED
                    gimg = cv2.imread(gimg_path)
                    gimg = cv2.resize(gimg, (width, height))
                    gimg = cv2.copyMakeBorder(
                        gimg, BW, BW, BW, BW, cv2.BORDER_CONSTANT, value=border_color
                    )
                    gimg = cv2.resize(gimg, (width, height))
                    start = (
                        rank_idx * width + rank_idx * GRID_SPACING + QUERY_EXTRA_SPACING
                    )
                    end = (
                        (rank_idx + 1) * width
                        + rank_idx * GRID_SPACING
                        + QUERY_EXTRA_SPACING
                    )
                    grid_img[:, start:end, :] = gimg
                else:
                    _cp_img_to(
                        gimg_path,
                        qdir,
                        rank=rank_idx,
                        prefix="gallery",
                        matched=matched,
                    )

                rank_idx += 1
                if rank_idx > topk:
                    break

        if data_type == "image":
            imname = os.path.basename(os.path.splitext(qimg_path_name)[0])
            cv2.imwrite(os.path.join(save_dir, imname + ".png"), grid_img)

        if (q_idx + 1) % 100 == 0:
            print("- done {}/{}".format(q_idx + 1, num_q))

        if q_idx >= int(cfg.TEST.VISUALIZE_MAX_NUMBER):
            break

    print('Done. Images have been saved to "{}" ...'.format(save_dir))
