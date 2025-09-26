# Modality Fusion of MRI and Clinical Data for Glioma Grading
This repository contains the official implementation of our multimodal fusion framework for glioma grade classification. Gliomas are among the most common and aggressive brain tumors, and accurate grading is essential for treatment planning and prognosis. While MRI provides rich structural and functional information, clinical data (such as age and sex) offer complementary context that can improve diagnostic performance. Despite this potential, systematic evaluations of how to effectively integrate clinical data with MRI across different fusion strategies remain limited. To address this gap, our framework explores and compares early, intermediate, and late fusion methods, and introduces a novel Modality Weighting Block (MWB) that adaptively learns the contribution of each modality. Beyond improving predictive performance, the framework also provides interpretable insights into how MRI modalities and clinical data jointly shape model decisions, intending to support more transparent and clinically useful AI systems.
## Fusion strategies included in this framework
The fusion architectures in our framework are illustrated in the following figure. They are referred to as Early Raw Features (ERF), Early Learned Features (ELF), Intermediate Separate Features (ISF), Intermediate Merged Features (IMF), Late Probability Averaging (LPA), and Late Majority Voting (LMV). 


<img width="1100" height="652" alt="fusion_strategies_github" src="https://github.com/user-attachments/assets/2cb00303-520b-4383-94dd-8f35da3be278" />


## Dataset used
## Table of contents
## How to run the code






