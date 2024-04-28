import torch
from insightface.app import FaceAnalysis
from ProcessData.process_data_constants import *
from ProcessData.process_dataset import ProcessDataset
import cv2
import numpy as np
import tqdm
FACE_EMBEDDING_SIZE = 512

class InsightFace:
    def __init__(self, gallery_imgs_paths: list, query_imgs_paths: list, dataset_processor: ProcessDataset, device: str,
                 detection_threshold: float = 0.0, similarity_threshold: float = 0.0, CC: bool = False, rank_diff: float = 0.0):
        self.gallery_imgs_paths = gallery_imgs_paths
        self.query_imgs_paths = query_imgs_paths
        self.dataset_processor = dataset_processor
        self.device = device
        self.detection_threshold = detection_threshold
        self.similarity_threshold = similarity_threshold
        self.rank_diff = rank_diff
        self.CC = CC
        self.gallery_features = []
        self.query_features = []
        self.g_camids = None
        self.q_camids = None
        self.face_recognition = None
        self.face_detector = None
        self.similarities = None
        self.predicted_labels = None
        self.gallery_references = None

    def init(self):
        self.g_camids = self.dataset_processor.extract_camids(self.gallery_imgs_paths)
        self.q_camids = self.dataset_processor.extract_camids(self.query_imgs_paths)
        self.q_pids = np.array(self.dataset_processor.convert_imgs_path_to_labels(self.query_imgs_paths))
        self.g_pids = np.array(self.dataset_processor.convert_imgs_path_to_labels(self.gallery_imgs_paths))
        self._init_face_model()

    def _init_face_model(self):
        """
        To select a specific device use:
        providers = [('CUDAExecutionProvider', {'device_id': 0,})]
        """
        if 'cuda' in self.device:
            providers = [('CUDAExecutionProvider', {'device_id': int(self.device.split(':')[1])})]
            app = FaceAnalysis(providers=providers)
            app.prepare(ctx_id=int(self.device.split(':')[1]))
        else:
            providers = ['CPUExecutionProvider']
            app = FaceAnalysis(providers=providers)
            app.prepare(ctx_id=-1)
        self.face_recognition = app.models['recognition']
        self.face_detector = app

    def compute_feature_vectors(self):
        self._detect_faces_and_compute_features_factory('gallery')
        self._detect_faces_and_compute_features_factory('query')

    def _detect_faces_and_compute_features_factory(self, data_type: str):
        """
        Iterate over the images in the given data type, and for each image detect faces and compute a feature vector.
        :param data_type: the data for which to detect faces and compute feature vectors.
        """
        print(f'Detecting faces and computing feature vectors for {data_type}')
        if data_type == 'gallery':
            self._detect_faces_and_compute_feature_vectors(self.gallery_imgs_paths, self.gallery_features)

        elif data_type == 'query':
            self._detect_faces_and_compute_feature_vectors(self.query_imgs_paths, self.query_features)
        else:
            raise Exception(f'Unsupported data type {data_type}. Possible types are ["gallery", "query"]')

    def _detect_faces_and_compute_feature_vectors(self, img_paths, feature_vectors):
        for img_path in tqdm.tqdm(img_paths, total=len(img_paths)):
            # Detect the face in the image
            face_img, face_obj = self._detect_face_from_crop(img_path)

            feature_vector = np.nan
            # If a face was detected, create a feature vector for it
            if face_img is not None:
                feature_vector = face_obj[0]['embedding']
                feature_vector = feature_vector[None, :]
            feature_vectors.append(feature_vector)

    def _detect_face_from_crop(self, crop_img):
        """
        Given a path to an image, detect the face in the image and return it. Returns None if no face was detected.
        :param crop_img: either the crop image or the path to the image in which a face should be detected.
        """
        if isinstance(crop_img, str):
            crop_img = cv2.imread(crop_img)
        face_img = None
        if crop_img is not None:
            detection_res = self.face_detector.get(crop_img)
            if len(detection_res) > 0 and detection_res[0] and detection_res[0]['det_score'] >= self.detection_threshold:
                face_bbox = detection_res[0]['bbox']
                X = np.max([int(face_bbox[0]), 0])
                Y = np.max([int(face_bbox[1]), 0])
                W = np.min([int(face_bbox[2]), crop_img.shape[1]])
                H = np.min([int(face_bbox[3]), crop_img.shape[0]])
                face_img = crop_img[Y:H, X:W]
        else:
            face_img, detection_res = None, None
            print(f'No image found for {crop_img}.')
        return face_img, detection_res

    def compute_similarities(self, gallery_enrichment):
        """
        Iterates over all query feature vectors and computes the similarity to the gallery feature vectors.
        """
        print("Computing similarities for face similarity matrix")
        num_query = len(self.query_features)
        num_gallary = len(self.gallery_features)
        if gallery_enrichment:
            self.similarities = np.ones((num_query, 2)) * -1
        else:
            self.similarities = np.ones((num_query, num_gallary)) * -1
        device = torch.device(self.device)
        gallery_indices = [i for i, v in enumerate(self.gallery_features) if np.isnan(v).all() == False]
        gallery_features_np = np.zeros((num_gallary, self.face_recognition.output_shape[1]))
        for i in gallery_indices:
            gallery_features_np[i, :] = self.gallery_features[i]
        gallery_tensors = torch.from_numpy(gallery_features_np).float().to(device)
        for i, q_feat in tqdm.tqdm(enumerate(self.query_features), total=num_query):
            if q_feat is np.nan:  # no query feature exists since no face was detected in the image
                continue

            # The similarity of the current query to all gallery samples, shape: (1,len(gallery_indices))
            sims = torch.cosine_similarity(torch.from_numpy(q_feat).float().to(device), gallery_tensors, dim=1).detach().cpu().numpy()

            # set to -inf all gallery scores where the camid is the same as the camid of the query
            if self.g_camids is not None and self.q_camids is not None and len(self.g_camids) > 0 and len(self.q_camids) > 0:
                same_camid_indices = np.where(self.g_camids == self.q_camids[i])[0]
                sims[same_camid_indices] = -np.inf
            if gallery_enrichment:
                if self.CC:
                    same_clothes_indices = np.where(np.array(self.dataset_processor.clothes_ids_gallery) ==
                                                    self.dataset_processor.clothes_ids_query[i])[0]
                    convert_to_sims_indices = np.where(np.in1d(np.array(gallery_indices), same_clothes_indices))[0]
                    sims[convert_to_sims_indices] = -np.inf
                sim_i, arg_sim_i = [np.max(sims), np.argmax(sims)]
                if self.rank_diff > 0:
                    from Scripts.inference import get_score_all_ids_full_gallery
                    gpid_scores = sorted(get_score_all_ids_full_gallery(np.expand_dims(sims, axis=0), self.g_pids).values(), reverse=True)
                    calc_ranking_diff = abs(gpid_scores[0] - gpid_scores[1])
                    if calc_ranking_diff < self.rank_diff: # difference between rank-1 and rank-2 not high enough, do not enrich
                        sim_i, arg_sim_i = [-np.inf, np.argmax(sims)]
                self.similarities[i, :] = [sim_i, arg_sim_i]
            else:
                self.similarities[i,:] = sims

    def predict_labels(self):
        predicted_labels = self.g_pids[self.similarities[:, 1].astype(int)]
        gallery_references = np.array(self.gallery_imgs_paths)[self.similarities[:, 1].astype(int)]
        max_scores = np.array(self.similarities[:, 0])
        # set all queries for which no face was detected or that the similarity score is below a given threshold to -1
        th_indices = np.where(max_scores < float(self.similarity_threshold))[0]
        predicted_labels[th_indices] = -1
        gallery_references[th_indices] = ''
        self.predicted_labels = [f"{int(x):06d}" for x in predicted_labels]
        self.gallery_references = gallery_references

        # For the CCVID dataset, convert the per image labels to the majority vote per sequence
        if self.dataset_processor.dataset == CCVID:
            self.predict_labels_with_majority_vote()

    def predict_labels_with_majority_vote(self):
        # First, iterate once on all query images to make majority vote
        query_to_gallery_mapping = {}
        prev_seq = ''
        cur_seq_votes = {}
        for i, img_path in enumerate(self.query_imgs_paths):
            cur_seq = self.dataset_processor.convert_img_path_to_sequence(img_path)
            predicted_label = self.predicted_labels[i]
            # count all votes of the same sequence in a dictionary
            if prev_seq == cur_seq:
                cur_seq_votes[predicted_label] = cur_seq_votes.get(predicted_label, 0) + 1

            # when done with a sequence, take the majority vote of its elements
            else:
                if not prev_seq:  # first image
                    cur_seq_votes[predicted_label] = 1
                    prev_seq = cur_seq
                    continue
                if UNDETECTED_FACE in cur_seq_votes:
                    # remove this key to avoid cases where this is the maximum in the majority vote
                    cur_seq_votes.pop(UNDETECTED_FACE)

                # no faces were identified in the sequence, hence all votes are UNDETECTED_FACE and cur_seq_votes is empty
                if len(cur_seq_votes) == 0:
                    query_to_gallery_mapping[prev_seq] = UNDETECTED_FACE
                else:
                    query_to_gallery_mapping[prev_seq] = max(cur_seq_votes, key=cur_seq_votes.get)
                cur_seq_votes = {}
                cur_seq_votes[predicted_label] = 1
            prev_seq = cur_seq
        if UNDETECTED_FACE in cur_seq_votes:
            cur_seq_votes.pop(UNDETECTED_FACE)
        if len(cur_seq_votes) == 0:
            query_to_gallery_mapping[prev_seq] = UNDETECTED_FACE
        else:
            query_to_gallery_mapping[prev_seq] = max(cur_seq_votes, key=cur_seq_votes.get)
        query_to_gallery_mapping[prev_seq] = max(cur_seq_votes, key=cur_seq_votes.get)  # add last sequence

        # Display accuracy computed per sequence and not per image:
        correct_labels = 0
        incorrect_labels = 0
        for seq, pred in query_to_gallery_mapping.items():
            gt_label = f'{int(seq.split("_")[1]):06d}'
            if gt_label == pred:
                correct_labels += 1
            else:
                incorrect_labels += 1
        print(f'Per sequences accuracy:\n'
              f'Correct: {correct_labels}\n'
              f'Incorrect: {incorrect_labels}\n'
              f'Accuracy: {correct_labels / len(query_to_gallery_mapping)}\n')

        # Next, iterate again over all query images and set the label according to the majority vote of the sequence
        for i, img_path in enumerate(self.query_imgs_paths):
            img_seq = self.dataset_processor.convert_img_path_to_sequence(img_path)
            if query_to_gallery_mapping.get(img_seq):
                self.predicted_labels[i] = query_to_gallery_mapping.get(img_seq)
            else:  # no face image was found in the sequence, set label as -1
                self.predicted_labels[i] = UNDETECTED_FACE