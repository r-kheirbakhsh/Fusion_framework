import argparse


def parse_args():
    args = argparse.ArgumentParser(description='Modality fusion framework')
    
    args.add_argument('--project_name', default=None, type=str)
    args.add_argument('--fusion_method', default='early_concat_fusion', type=str) # it consists of ['early_1_fusion', 'early_2_fusion', 'intermediate_1_fusion', 'intermediate_2_fusion', 'late_fusion']
    args.add_argument('--dataset_csv_path', default=None, type=str)   # the path to the csv file containing the metadata of each slice (columns: ID,slice_name,sex,age,WHO_grade,final_diagnosis,MGMT_status,1p/19q,IDH)
    args.add_argument('--train_csv_path', default=None, type=str)
    args.add_argument('--val_csv_path', default=None, type=str)
    args.add_argument('--test_csv_path', default=None, type=str)
    args.add_argument('--dataset_image_path', default=None, type=str)  # the path to the main folder contaning the sliced MRI scans 
    args.add_argument('--label', default=4, type=int)    # LABEL2 on metadata after slice for new dataloader (WHO_grade->4 ,final_diagnosis->5 ,MGMT_status->6 ,1p/19q->7 ,IDH->8)
    args.add_argument('--num_class', default=2, type=int)
    args.add_argument('--axis', default=2, choices=[0,1,2], type=int)    # axis of the slices -> 0: Sagittal, 1: Coronal, 2: Axial
    args.add_argument('--modalities', nargs='+', type=str, help="List of modalities for late fuaion") # e.g. ['T1_bias', 'T2_bias', 'Clinical']
    args.add_argument('--scale_clinical_modality', default='Minmax', type=str) # e.g. Minmxa, Normalize   
    args.add_argument('--mri_model', default="DenseNet121", type=str) # the architecture of NN for MRI encoder
    args.add_argument('--cl_model', default="AutoInt", type=str) # the architecture of NN for the Clinical encoder
    args.add_argument('--fused_model', default="MLP", type=str) # the model for fusion of the modalities: 
    args.add_argument('--pretrained', default=0, type=int)   # pretrained mri_model: 1, non-pretrained mri_model: 0 
    args.add_argument('--lr_mri', default=1e-6, type=float)   # learning rate for mri_model
    args.add_argument('--lr_cl', default=1e-6, type=float)   # learning rate for cl_model
    args.add_argument('--lr_fused', default=1e-6, type=float)   # learning rate for fused_model
    args.add_argument('--lmbda', default=1e-4, type=float)
    args.add_argument('--batch_size_mri', default=16, type=int)   # number of batch size for mri_model
    args.add_argument('--batch_size_cl', default=16, type=int)   # number of batch size for cl_model
    args.add_argument('--batch_size_fused', default=16, type=int)   # number of batch size for fused_model
    args.add_argument('--n_epochs_mri', default=40, type=int)    # number of epochs for mri_model
    args.add_argument('--n_epochs_cl', default=40, type=int)    # number of epochs for cl_model
    args.add_argument('--n_epochs_fused', default=40, type=int)    # number of epochs for fused_model
    args.add_argument('--n_gpu', default=1, type=int)
    args.add_argument('--seed', default=7, type=int)     # the seed to split the data into train, val, and test, and also to set the seed for reproducability of the training process
    args.add_argument('--num_folds', default=3, type=int)  # the number of folds for cross-validation
    args.add_argument('--fold', default=100, type=int)  # the fold number for cross-validation, starting from 0 to num_folds-1

    return args.parse_args()