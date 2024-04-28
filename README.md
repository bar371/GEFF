# ReFace: Improving Clothes-Changing Re-Identification With Face Features

Official implementation of the paper [*ReFace: Improving Clothes-Changing Re-Identification With Face Features*](https://arxiv.org/pdf/2211.13807.pdf).
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reface-improving-clothes-changing-re/person-re-identification-on-ltcc)](https://paperswithcode.com/sota/person-re-identification-on-ltcc?p=reface-improving-clothes-changing-re)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reface-improving-clothes-changing-re/person-re-identification-on-prcc)](https://paperswithcode.com/sota/person-re-identification-on-prcc?p=reface-improving-clothes-changing-re)

## Quick start
To evaluate the performance of our model, we provide a [colab notebook](https://colab.research.google.com/drive/1fd91h6WhSSwuvPcUoVjNmPENBN1i5vqc?usp=sharing).
In this notebook, we first create an enriched gallery as described in the paper and then run the inference of our model using the enriched gallery.

### Usage Example - LTCC
#### Gallery Enrichment
`
ltcc --dataset_path <path-to-dataset> --detection_threshold 0.8  --similarity_threshold 0.5 --device <device> --CC
`

#### Inference
`
ltcc AIM --reid_config <path-to-reid-config-file> --dataset_path <path-to-dataset> --detection_threshold 0.7 --similarity_threshold 0.5 --alpha 0.75 --reid_checkpoint <path-to-checkpoints> --device <device>
`

#### Notes
- To download the datasets see the original pages of each dataset (listed [below](#datasets)).
- The `reid_config` files of the supported models can be found under `ReIDModules\<reid-model>\configs`.
- Checkpoints of the different models can be downloaded from [here](#trained-model-weights).

## Datasets

### Existing Benchmarks
In this paper we compare the results of our model on the LTCC, PRCC, and LaST datasets.
The different datasets can be downloaded through the official pages of these datasets:
* [CCVID](https://github.com/guxinqian/Simple-CCReID)
* [LTCC](https://naiq.github.io/LTCC_Perosn_ReID.html)
* [PRCC](https://www.isee-ai.cn/~yangqize/clothing.html)
* [LaST](https://github.com/shuxjweb/last)
* [VC-Clothes](https://wanfb.github.io/dataset.html#) 

### The 42street Dataset
The 42street dataset can be downloaded from the following link:
* [dataset]()
* [extra-data-1]()
* [extra-data-2]()

#### Dataset Structure
- *gallery*: folder with annotated crops - 16,668 images of 13 identities + 1 category for unidentified persons.
- *test*:
  - *vids*: raw videos that were used for testing - 10 videos of ~17 seconds each. 
  - *tracklets*: annotated tracklets from the test videos - 26,427 images in 239 tracklets.
- *extra-data*: a folder with unannotated crops taken from the same part in the play as the test videos (downloaded separately).

To use the extra data, download both folders above and extract them to the `extra-data` folder.

## Trained model weights
Our model relies on pre-trained face and ReID models and does not require any further training.
See [this](https://drive.google.com/drive/folders/1qm1D38WzH2Rqv8NKteulTB3bU4W3nBFh) folder for trained weights of the ReID model, trained by us on the original LTCC, PRCC, LaST and CCVID datasets (the checkpoints are automatically downloaded when running the [colab notebook](#quick-start)).

## Results
Below we provide the results achieved by our model on the clothes-changing settings in the different datasets. 

| Dataset | PRCC | LTCC | LaST | VC-Clothes | CCVID | 42Street | 42Street (w. extra-data) |
|-------|------|------|------|------------|-------|----------|--------------------------|
| Top-1 | 81.9 | 76.3 | 78.0 | 94.9       | 89.2  | 75.0     | 92.2                     |
| mAP   | 58.8 | 42.3 | 37.2 | 88.9       | NaN   | NaN      | NaN                      |


## Acknowledgments
In our work we use [Simple-CCReID](https://github.com/guxinqian/Simple-CCReID) as the ReID module and [Insightface](https://github.com/deepinsight/insightface) as the face module.
We thank them for their great works.

## Citation
```
@article{arkushin2022reface,
  title={ReFace: Improving Clothes-Changing Re-Identification With Face Features},
  author={Arkushin, Daniel and Cohen, Bar and Peleg, Shmuel and Fried, Ohad},
  journal={arXiv preprint arXiv:2211.13807},
  year={2022}
```