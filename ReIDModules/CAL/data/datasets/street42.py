import glob
import logging
import os
import os.path as osp
from pathlib import Path

from ProcessData.process_data_constants import TRACKLETS, GALLERY


class Street42(object):
    dataset_dir = 'street42'

    def __init__(self, root='data', **kwargs):
        self.dataset_dir = root
        self.query_dir_val = osp.join(self.dataset_dir, 'val', TRACKLETS)
        self.query_dir_test = osp.join(self.dataset_dir, 'test', TRACKLETS)
        self.gallery_dir = osp.join(self.dataset_dir, GALLERY)
        self.enriched_dir = None
        self._check_before_run()

        query_test_tracklets, test_num_pids, test_num_tracklets, test_num_imgs, test_num_videos = self._process_tracklets(
            self.query_dir_test)
        gallery_imgs, gallery_num_pids = self._create_gallery_paths()
        gallery_num_imgs = len(gallery_imgs)

        logger = logging.getLogger('reid.dataset')
        logger.info("=> 42Street loaded")
        logger.info("Dataset statistics:")
        logger.info("  -----------------------------------------------------")
        logger.info("  subset   | # ids | # images | # tracklets | # vids |")
        logger.info("  -----------------------------------------------------")
        logger.info("  test     | {:5d} | {:8d} | {:5d} | {:5d} |".format(test_num_pids, test_num_imgs, test_num_tracklets, test_num_videos))
        logger.info("  gallery  | {:5d} | {:8d} |".format(gallery_num_pids, gallery_num_imgs))

        self.query = query_test_tracklets
        self.gallery = gallery_imgs

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.query_dir_test):
            raise RuntimeError("'{}' is not available".format(self.query_dir_test))
        if not osp.exists(self.query_dir_val):
            raise RuntimeError("'{}' is not available".format(self.query_dir_val))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_tracklets(self, path):
        total_pids = set()
        total_tracklets = 0
        total_imgs = 0

        videos = os.listdir(path)
        tracklets = []
        for vid in videos:
            video_tracks = os.listdir(os.path.join(path, vid))
            total_tracklets += len(video_tracks)
            for tracklet_path in video_tracks:
                pid = int(tracklet_path.split('_')[0])
                total_pids.add(pid)
                img_paths = glob.glob(osp.join(path, vid, tracklet_path, '*.png'))
                img_paths.sort()
                total_imgs += len(img_paths)
                tracklets.append((img_paths, pid))

        total_videos = len(videos)
        return tracklets, len(total_pids), total_tracklets, total_imgs, total_videos

    def _create_gallery_paths(self) -> []:
        imgs_paths = []
        total_pids = set()
        print(f"Loading gallery for dataset..")
        for img in glob.glob(self.gallery_dir + "/*"):
            suffix = img[-3:]
            if suffix != 'jpg' and suffix != 'png':
                continue
            file_name = Path(img).name
            if os.path.isfile(img):
                pid = int(file_name.split('_')[0])
                total_pids.add(pid)
                imgs_paths.append((img, pid))
        print(f'Done. {len(imgs_paths)} loaded.')
        return imgs_paths, len(total_pids)
