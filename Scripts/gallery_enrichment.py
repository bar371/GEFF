import sys

sys.path.append('.')
from ProcessData.utils import prepare_dataset
from FaceModules.insightface import InsightFace
from GalleryEnrichment.gallery_builder_CCVID import GalleryBuilderCCVID
from GalleryEnrichment.gallery_builder_PRCC import GalleryBuilderPRCC
from GalleryEnrichment.gallery_builder_LTCC import GalleryBuilderLTCC
from GalleryEnrichment.gallery_builder_LaST import GalleryBuilderLaST
from GalleryEnrichment.gallery_builder_VCClothes import GalleryBuilderVCClothes
from ProcessData.process_data_constants import *
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument('dataset',
                        help='On which dataset to run the face model?',
                        choices=DATASETS)
    parser.add_argument('--dataset_path',
                        help='Full path to the dataset.')
    parser.add_argument('--detection_threshold',
                        help='The threshold that should be used for face detection in the gallery images.',
                        default=0.0)
    parser.add_argument('--similarity_threshold',
                        help='The threshold that should be used for similarity between query and gallery images.',
                        default=0.0)
    parser.add_argument('--CC',
                        help='Ignore images that have same clothes id between query and gallery samples.',
                        action='store_true')
    parser.add_argument('--device', help='Cuda device to use. Should be in the format of `cuda:X`.',
                        default='cuda:0')
    return parser.parse_args()


def build_gallery(args, dataset_processor, face_model, query_paths, gallery_references):
    if args.dataset == PRCC:
        gallery_builder = GalleryBuilderPRCC(dataset_processor.data_base_path, query_paths,
                                             face_model.predicted_labels,
                                             f'{os.path.join("enriched_gallery", "A")}',
                                             gallery_references)
    elif args.dataset == CCVID:
        gallery_builder = GalleryBuilderCCVID(dataset_processor.data_base_path, query_paths,
                                              face_model.predicted_labels,
                                              f'enriched_gallery')
    elif args.dataset == LTCC:
        gallery_builder = GalleryBuilderLTCC(dataset_processor.data_base_path, query_paths,
                                             face_model.predicted_labels,
                                             f'enriched_gallery',
                                             gallery_references)
    elif args.dataset == LAST:
        gallery_builder = GalleryBuilderLaST(dataset_processor.data_base_path, query_paths,
                                             face_model.predicted_labels,
                                             f'enriched_gallery', gallery_references)

    elif args.dataset == VCCLOTHES:
        gallery_builder = GalleryBuilderVCClothes(dataset_processor.data_base_path, query_paths,
                                                  face_model.predicted_labels,
                                                  f'enriched_gallery', gallery_references)
    else:
        raise Exception(f'Supported dataset types are: {DATASETS}.')

    gallery_builder.create_query_based_gallery()


def main(args):
    # Prepare the dataset that should be evaluated:
    dataset_processor, gallery_paths, query_paths = prepare_dataset(args)
    if not gallery_paths or not query_paths:
        raise Exception(f'No images were found for the gallery or query. Please check the dataset path and type.')

    # Initialize the face model used for gallery enrichment:
    face_model = InsightFace(gallery_paths, query_paths, dataset_processor, device=args.device,
                             detection_threshold=float(args.detection_threshold),
                             similarity_threshold=args.similarity_threshold, CC=args.CC)
    face_model.init()

    # Run the model (detect faces in query and gallery and compute similarities)
    face_model.compute_feature_vectors()
    face_model.compute_similarities(gallery_enrichment=True)
    face_model.predict_labels()

    # Build the gallery from query with the predicted labels:
    build_gallery(args, dataset_processor, face_model, query_paths, gallery_references=face_model.gallery_references)


if __name__ == '__main__':
    args = get_args()
    main(args)
