# Modality Fusion of MRI and Clinical Data for Glioma Grading
This repository contains the official implementation of our multimodal fusion framework for glioma grade classification. Gliomas are among the most common and aggressive brain tumors, and accurate grading is essential for treatment planning and prognosis. While MRI provides rich structural and functional information, clinical data (such as age and sex) offer complementary context that can improve diagnostic performance. Despite this potential, systematic evaluations of how to effectively integrate clinical data with MRI across different fusion strategies remain limited. To address this gap, our framework explores and compares early, intermediate, and late fusion methods, and introduces a novel Modality Weighting Block (MWB) that adaptively learns the contribution of each modality. Beyond improving predictive performance, the framework also provides interpretable insights into how MRI modalities and clinical data jointly shape model decisions, intending to support more transparent and clinically useful AI systems.
## Fusion strategies included in this framework
The fusion architectures in our framework are illustrated in the following figure. They are referred to as Early Raw Features (ERF), Early Learned Features (ELF), Intermediate Separate Features (ISF), Intermediate Merged Features (IMF), Late Probability Averaging (LPA), and Late Majority Voting (LMV). 


<img width="1100" height="652" alt="fusion_strategies_github" src="https://github.com/user-attachments/assets/2cb00303-520b-4383-94dd-8f35da3be278" />


## Dataset used
This project is based on the UCSF glioma dataset, which provides preprocessed MRI scans and associated clinical information. We use four MRI modalities — T1, T1c, T2, and FLAIR — along with two clinical variables: sex and age. To construct the input data for our framework, we leveraged the provided tumor segmentation masks to extract 2D slices from each MRI sequence that contain tumor regions. Since the MRI scans are already preprocessed in the UCSF dataset, no additional preprocessing was applied beyond modality-wise intensity scaling. For the clinical features, we applied min–max scaling to normalize age values, while sex was already encoded as binary (0/1). This prepared dataset serves as the input to our multimodal fusion framework.

Note: The UCSF glioma dataset is not publicly available. Access requires permission from the data providers, and researchers must request approval directly from the UCSF repository.

## Datset preparation
Once you have access to the UCSF glioma dataset, the following preprocessing steps are required to prepare the data for training with this framework:

1. MRI Slice Extraction

* Use the provided tumor segmentation masks to identify slices containing tumor regions.

* Extract the corresponding slices from each MRI modality (T1, T1c, T2, FLAIR).

* Store the slices in .npz format, where each file contains four channels corresponding to the four MRI modalities.

2. MRI Preprocessing

* Since the UCSF dataset already provides preprocessed MRI scans, no additional preprocessing (e.g., skull-stripping, registration, bias-field correction) is required.

* Apply modality-wise intensity scaling to normalize the values across MRI modalities.

3. Clinical Data Preprocessing

* Encode sex as binary (0 = female, 1 = male).

* Apply min–max scaling to normalize age values between 0 and 1.

4. Dataset Structure

* Organize the processed data so that each patient has a unique ID, with MRI slices stored in a consistent folder structure and clinical features stored in a tabular format (e.g., CSV).

* The codebase includes dataset classes and loaders to handle this structure automatically.

## Table of contents
## How to run the code






