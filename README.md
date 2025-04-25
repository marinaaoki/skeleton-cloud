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

1. Data collection: these are provided by `sitc-ak-driver` and `sitc-studio`.
2. Data pre-processing: these are handled by the scripts in `src/processing`.
3. Data visualisation: these are handled by the scripts in `src/visualisation`.
4. SAR and SRID training: handled by `SGN` submodule.
5. Data conversions: these are handled by the scripts in `src/conversions`. Note that the BODY15 format is similar to BODY25 but with the last ten joints simply omitted.
6. Motion retargeting: handled by `2D-Motion-Retargeting` submodule.