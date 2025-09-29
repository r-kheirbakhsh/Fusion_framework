#!/bin/bash


CODEBASE="/your/path/to/Fusion_framework"
PROJECT_NAME='Glioma_Classification'
FUSION_STRATEGY='L'     # or 'ELF' or 'ERF' or 'ISF' or 'IMF'
DATASET_CSV_PATH='/your/path/to/dataset_metadata.csv'
TRAIN_CSV_PATH='-'  # if you do not want to have cross-validation, you can provide the path to the csv file containing the metadata of each slice for training and change the code accordingly
VAL_CSV_PATH='-'    # if you do not want to have cross-validation, you can provide the path to the csv file containing the metadata of each slice for validation and change the code accordingly
TEST_CSV_PATH='-'   # if you do not want to have cross-validation, you can provide the path to the csv file containing the metadata of each slice for testing and change the code accordingly
DATASET_IMAGE_PATH='/your/path/to/UCSF-PDGM-SLICED'
LABEL=4         # the column containing labels for WHO_grade on dataset_metadata.csv file
NUM_CLASS=2     # binary classification
AXIS=2          #axis along which the slices are to be taken -> 0: Sagittal, 1: Coronal, 2: Axial
MODALITIES=("Clinical" "T1_bias" "T1c_bias" "T2_bias" "FLAIR_bias")
SCALE_CLINICAL_MODALITY='Minmax'    # Or 'Quantile-normal' or 'Non_scaled'
MRI_MODEL='denseNet121'     # for L, ELF, ISF, IMF, and ISF with MWB
CL_MODEL='AutoInt'          # for L, ELF, ISF, IMF, and ISF with MWB
FUSED_MODEL='majority_voting'       # or 'probability_averaging' for L, 'MLP' for ERF, 'Inter_1_concat' for ELF (the encoder layers will be freezed), 'Inter_1_concat' for ISF, 'Inter_2_concat' for IMF, and 'Inter_1_concat_attn' for ISF with MWB
PRETRAINED=1
LR_MRI=5e-5
LR_CL=5e-3
LR_FUSED=0.0
LMBDA=1e-6
BATCH_SIZE_MRI=16
BATCH_SIZE_CL=16
BATCH_SIZE_FUSED=0
N_EPOCHS_MRI=20
N_EPOCHS_CL=15
N_EPOCHS_FUSED=0
N_GPU=1
SEED=5
NUM_FOLDS=5
FOLD=100        # for the start, use 100, then it will be changed to 0, 1, 2, ..., NUM_FOLDS-1


LOGFILE="log_${SEED}.log"

python ${CODEBASE}/train_test.py --project_name $PROJECT_NAME \
                    --fusion_method $FUSION_METHOD \
                    --dataset_csv_path $DATASET_CSV_PATH\
                    --train_csv_path $TRAIN_CSV_PATH \
                    --val_csv_path $VAL_CSV_PATH \
                    --test_csv_path $TEST_CSV_PATH \
                    --dataset_image_path $DATASET_IMAGE_PATH\
                    --label $LABEL \
                    --num_class $NUM_CLASS \
                    --axis $AXIS \
                    --modalities "${MODALITIES[@]}" \
                    --scale_clinical_modality $SCALE_CLINICAL_MODALITY \
                    --mri_model $MRI_MODEL \
                    --cl_model $CL_MODEL \
                    --fused_model $FUSED_MODEL \
                    --pretrained $PRETRAINED \
                    --lr_mri $LR_MRI \
                    --lr_cl $LR_CL \
                    --lr_fused $LR_FUSED \
                    --lmbda $LMBDA \
                    --batch_size_mri $BATCH_SIZE_MRI \
                    --batch_size_cl $BATCH_SIZE_CL \
                    --batch_size_fused $BATCH_SIZE_FUSED \
                    --n_epochs_mri $N_EPOCHS_MRI \
                    --n_epochs_cl $N_EPOCHS_CL \
                    --n_epochs_fused $N_EPOCHS_FUSED \
                    --n_gpu $N_GPU \
                    --seed $SEED \
                    --num_folds $NUM_FOLDS \
                    --fold $FOLD > $LOGFILE 2>&1