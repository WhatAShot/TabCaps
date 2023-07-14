# TabCaps
This is the pytorch code for [TabCaps: A Capsule Neural Network for Tabular Data Classification with BoW Routing](https://openreview.net/pdf?id=OgbtSLESnI), an ICLR2023 publication that introduces an effective and robust capsule neural network designed specifically for tabular data. It is an excellent capsule neural network performing well on biased datasets or with limited training data scales.

## Brief Introduction
Records in a table are represented by a collection of heterogeneous scalar features. Previous work often made predictions for records in a paradigm that processed each feature as an operating unit, which requires to well cope with the heterogeneity. In this paper, we propose to encapsulate all feature values of a record into vectorial features and process them collectively rather than have to deal with individual ones, which directly captures the representations at the data level and benefits robust performances. Specifically, we adopt the concept of "capsules" to organize features into vectorial features, and devise a novel capsule neural network called "TabCaps" to process the vectorial features for classification. In TabCaps, a record is encoded into several vectorial features by some optimizable multivariate Gaussian kernels in the primary capsule layer, where each vectorial feature represents a specific "profile" of the input record and is transformed into senior capsule layer under the guidance of a new straightforward routing algorithm. The design of routing algorithm is motivated by the Bag-of-Words (BoW) model, which performs capsule feature grouping straightforwardly and efficiently, in lieu of the computationally complex clustering of previous routing algorithms. Comprehensive experiments show that TabCaps achieves competitive and robust performances in tabular data classification tasks. Please read the paper for more details.
## Pipeline Illustration of Tabcaps
![Tabcaps processing an tabular sample](https://github.com/WhatAShot/TabCaps/blob/main/tabcaps.jpg)

## Citations
```
@inproceedings{tabcaps, 
   title={TabCaps: A Capsule Neural Network for Tabular Data Classification with BoW Routing}, 
   author={Chen, Jintai and Liao, Kuanlun and Fang, Yanwen and Chen, Danny Z and Wu, Jian}, 
   booktitle={ICLR}, 
   year={2023}
 }
 ```
