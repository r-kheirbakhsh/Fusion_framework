# Modality Fusion of MRI and Clinical Data for Glioma Grading
This repository contains the official implementation of our multimodal fusion framework for glioma grade classification. Gliomas are among the most common and aggressive brain tumors, and accurate grading is essential for treatment planning and prognosis. While MRI provides rich structural and functional information, clinical data (such as age and sex) offer complementary context that can improve diagnostic performance. Despite this potential, systematic evaluations of how to effectively integrate clinical data with MRI across different fusion strategies remain limited. To address this gap, our framework explores and compares early, intermediate, and late fusion methods, and introduces a novel Modality Weighting Block (MWB) that adaptively learns the contribution of each modality. Beyond improving predictive performance, the framework also provides interpretable insights into how MRI modalities and clinical data jointly shape model decisions, intending to support more transparent and clinically useful AI systems.
## Fusion strategies included in this framework
The fusion architectures in our framework are illustrated in the following figure. They are referred to as Early Raw Features (ERF), Early Learned Features (ELF), Intermediate Separate Features (ISF), Intermediate Merged Features (IMF), Late Probability Averaging (LPA), and Late Majority Voting (LMV). 


<img width="1100" height="652" alt="fusion_strategies_github" src="https://github.com/user-attachments/assets/2cb00303-520b-4383-94dd-8f35da3be278" />


## Dataset used
This project is based on the the University of California, San Francisco Preoperative Diffuse Glioma MRI (UCSF-PDGM) dataset [[1]](#1), which includes 501 adult patients with WHO grade 2-4 diffuse gliomas (43, 56, and 396 patients, respectively, for grades 2, 3, and 4) who underwent preoperative MRI and resection between 2015 and 2021. The dataset provides standardized 3T MRI scans across multiple sequences, including T1 (pre- and post-contrast), T2, and FLAIR, along with advanced modalities such as DWI, SWI, and perfusion imaging. It also contains clinical and molecular data (age, sex, tumour grade, IDH mutation, MGMT methylation) and expert tumour segmentations.

The UCSF-PDGM dataset is available at https://www.cancerimagingarchive.net/collection/ucsf-pdgm/. Access requires permission from the data providers, and researchers must request approval directly from the UCSF repository.

## Datset preparation
To construct the input data for our framework, we leveraged the provided brain masks and tumor segmentation masks. Once you have access to the UCSF glioma dataset, the following preprocessing steps are required to prepare the data for training with this framework:

1. MRI Slice Extraction
    * Use the provided brain masks to identify slices containing brain. 
    * Extract the corresponding 2D slices for each MRI modality (T1, T1c, T2, FLAIR) from the *_bias.nii.gz 3D files. In our experiments we uses Axia plane, if you want to choose other planes, change the struchture/code accordingly.
    * Store the slices in .npz format, one file for each modality according to the dataset structure (refer to "4. Dataset Structure" for more details).
    * Use the provided tumor segmentation masks to identify slices containing tumor regions.
    * Save the indexis to the slices with tumour in a CSV file (refer to "4. Dataset Structure" for more details).

2. MRI Preprocessing
    * Since the UCSF dataset already provides preprocessed MRI scans (in *_bias.nii.gz files), no additional preprocessing (e.g., skull-stripping, registration, bias-field correction) is required.
    * Modality-wise intensity scaling to normalize the values across MRI modalities will be done in the dataloaders of the framework.

3. Clinical Data Preprocessing
    * Encode sex as binary (0 = female, 1 = male).
    * Min–max scaling to normalize age values between 0 and 1 will be done by the framework.

4. Dataset Structure
    * Organize the 2D MRi slices so that each patient has a unique ID, with MRI slices stored in a consistent folder as shown bellow:

    ```
        UCSF-PDGM-SLICED/
        │
        ├── UCSF-PDGM-0004/
        │   ├──Axial/
        │       ├──T1_bias/
        │           ├──slice_22.npz
        │           ├──slice_23.npz
        │           ├── ...
        │       ├──T1c_bias/
        │           ├──slice_22.npz
        │           ├──slice_23.npz
        │           ├── ...
        │       ├──T2_bias/
        │           ├──slice_22.npz
        │           ├──slice_23.npz
        │           ├── ...
        │       ├──FLAIR_bias/
        │           ├──slice_22.npz
        │           ├──slice_23.npz
        │           ├── ...
        │
        ├── UCSF-PDGM-0005/
        │   ├──Axial/
        │       ├──T1_bias/
        │           ├──slice_10.npz
        │           ├──slice_11.npz
        │           ├── ...
        │       ├──T1c_bias/
        │           ├──slice_10.npz
        │           ├──slice_11.npz
        │           ├── ...
        │       ├──T2_bias/
        │           ├──slice_10.npz
        │           ├──slice_11.npz
        │           ├── ...
        │       ├──FLAIR_bias/
        │           ├──slice_10.npz
        │           ├──slice_11.npz
        │           ├── ...        
        │
        └── ... (and so on for each patient)
    ```
    * Organize the CSV file containing the metadata of the dataset for the framework with the following columns:

    ```
    dataset_metadata.csv:
    
    ID,slice_name,sex,age,WHO_grade
    0004,slice_81,1,66,4
    0004,slice_82,1,66,4
    ...
    0004,slice_126,1,66,4
    0005,slice_59,0,80,4
    ...
    0005,slice_116,0,80,4
    0007,slice_46,1,70,4
    ...
    0007,slice_124,1,70,4
    ... (and so on for each patient)
    ```    

## Table of contents
## How to run the code

## References
<a id="1">[1]</a> 
Calabrese, E., Villanueva-Meyer, J.E., Rudie, J.D., Rauschecker, A.M., Baid, U., Bakas, S., Cha, S., Mongan, J.T. and Hess, C.P., 2022. The University of California San Francisco preoperative diffuse glioma MRI dataset. Radiology: Artificial Intelligence, 4(6), p.e220058.





