# Skeletons in the Cloud: Motion Retargeting for Anonymisation of Smart Home Data

## Abstract

Skeleton-based activity recognition (SAR) is popular due to its spatially rich yet compact representation of human motion. Recognising Activities of Daily Living (ADLs) is critical for in-home health monitoring of older adults, for which skeleton-based data offer a non-invasive solution. Despite perceived privacy, linkage attacks can recover individuals’ identities from skeletons. Motion retargeting has been proposed to reduce this risk while maintaining motion utility. We present a novel dataset for privacy-preserving SAR. The dataset was collected from 16 subjects and includes 20 activity classes such as eating and bathing. Participants were male (n=12) and female (n=4) adults, including older adults (n=5). This dataset will be used to evaluate adversary-guided motion retargeting on real-world data for in-home monitoring of older adults. This approach could be used in IoT-based systems, offering a privacy-preserving solution for ageing in place.

## Project Description

This is the code for the guided research project "Skeletons in the Cloud: Motion Retargeting for Anonymisation of Smart Home Data", which was conducted from October 2024 to March 2025 as part of a student exchange at the Ubiquitous Computing Systems Laboratory, Nara Institute of Science and Technology.

The results of this project will be presented at the CHI 2025 workshop on "Technology-Mediated Caregiving for Aging in Place" taking place in Yokohama, Japan on 27th April 2025. 

## 0. Accessing the dataset

The dataset can be provided upon request.

## 1. Available submodules

There are various submodules responsible for different aspects of the system:

1. Data collection: these are provided by [`sitc-ak-driver`](https://github.com/marinaaoki/sitc-ak-driver/blob/55ca462b28664ab7ce11d9a79a26c6893472f0ab) and [`sitc-studio`](https://github.com/marinaaoki/sitc-studio/blob/a5a1a78c79790080e4de5451c546b87a7bc788b8).
2. Data pre-processing: these are handled by the scripts in [`src/preprocessing`](src/preprocessing/).
3. Data visualisation: these are handled by the scripts in [`src/visualisation`](src/visualisation).
4. Human pose estimation: handled by [`simple-HRNet`](https://github.com/stefanopini/simple-HRNet/tree/dcfdb8ee0415b615d68f578b3eb172c73e5dda74) submodule.
5. SAR and SRID training: handled by [`SGN`](https://github.com/microsoft/SGN/tree/42c5784422db2823c6e826d60da4cde8b718f2c6) submodule.
6. Data conversions: these are handled by the scripts in [`src/conversions`](src/conversions). Note that the BODY15 format is similar to BODY25 but with the last ten joints simply omitted.
7. Motion retargeting: handled by [`2D-Motion-Retargeting`](https://github.com/ChrisWu1997/2D-Motion-Retargeting/tree/bdb4e78ae6e586fbc3d1145b82170645c4bcde60) submodule.
