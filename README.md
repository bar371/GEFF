# ReFace: Improving Clothes-Changing Re-Identification With Face Features

Official implementation of the paper [*ReFace: Improving Clothes-Changing Re-Identification With Face Features*](https://arxiv.org/pdf/2211.13807.pdf).
	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reface-improving-clothes-changing-re/person-re-identification-on-ltcc)](https://paperswithcode.com/sota/person-re-identification-on-ltcc?p=reface-improving-clothes-changing-re)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/reface-improving-clothes-changing-re/person-re-identification-on-prcc)](https://paperswithcode.com/sota/person-re-identification-on-prcc?p=reface-improving-clothes-changing-re)

## Quick start
To evaluate the performance of our model, we provide a [colab notebook](https://colab.research.google.com/drive/1fd91h6WhSSwuvPcUoVjNmPENBN1i5vqc?usp=sharing).
In this notebook, we first create an enriched gallery as described in the paper and then run the inference of our model using the enriched gallery.

## Datasets
In this paper we compare the results of our model on the LTCC, PRCC, and LaST datasets.
The different datasets can be downloaded through the official pages of these datasets:
* [CCVID](https://github.com/guxinqian/Simple-CCReID)
* [LTCC](https://naiq.github.io/LTCC_Perosn_ReID.html)
* [PRCC](https://www.isee-ai.cn/~yangqize/clothing.html)
* [LaST](https://github.com/shuxjweb/last)

### Custom Dataset
Inference on a custom dataset including person tracking, will be released soon, together with the ***42Street*** dataset presented in the paper. 

## Trained model weights
Our model relies on pre-trained face and ReID models and does not require any further training.
See [this](https://drive.google.com/drive/folders/1qm1D38WzH2Rqv8NKteulTB3bU4W3nBFh) folder for trained weights of the ReID model, trained by us on the original LTCC, PRCC, LaST and CCVID datasets (the checkpoints are automatically downloaded when running the [colab notebook](#quick-start)).

## Results
Below we provide the results achieved by our model on the clothes-changing settings in the different datasets. 

| Dataset | PRCC | LTCC | LaST | CCVID |
|---------|------|------|------|-------|
| Top-1   | 83.7 | 74.8 | 75.8 | 89.2  |
| mAP     | 66.7 | 48.4 | 29.6 | NaN   |


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