
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from dataloader.transfromation import CustomCompose_not_tensor, scale_mri_image



def _transform (mri_modalities_dict, do):
    ''' Do augmentation using CustomCompose
    Args:
        mri_modalities_dic (_type:dictionary_): a dictionary of the slices of MRI 
        do (_type:int_): 1 means do augmentation, 0 means do not augmentation

    Returns:
        mri_modalities_dict (_type:dictionary_): a dictionary of the transformed slices of MRI
    '''

    transform = CustomCompose_not_tensor(
                vert_flip_th=0.5,
                hor_flip_th=0.5,
                rotation_degree=20
        )

    # Scale the images in the dictionary
    for item in mri_modalities_dict:
        image = mri_modalities_dict[item]
             
        # Scale the MRI image to [0, 1] range
        mri_modalities_dict[item] = scale_mri_image(image, item)  


    if (do == 1):   # transform should be performed (for train)
        return transform(mri_modalities_dict)

    else:   # no transform (for validation and test)
        return mri_modalities_dict
        
        

class UCSFslice_early_2_fusion(Dataset):
    '''UCSFslice_early_2_fusion for ERF fusion strategy
    Args:
        Dataset: Parent torch dataset class
    '''
    def __init__(self, metadata_df, config, do_transform)-> None:
        ''' Sets the class variables

        Args:
            metadata_df (_type:panda dataframe_): it contains the metadata of the dataset with columns: ID,slice_name,sex,age,WHO_grade,final_diagnosis,MGMT_status,1p/19q,IDH
            config (_type:config_): it contains the configeration of the problem, the ones used in this class:
                config.image_path (_type:str_): the path to the main folder of images, like /mnt/storage/reyhaneh/data/UCSF/UCSF-PDGM-SLICED            
                config.axis (_type:int_): axis along which the slices are in the dataset -> 0: Sagittal, 1: Coronal, 2: Axial  
                config.data_label (_type:int_): the label for classification (WHO_grade->4 ,final_diagnosis->5 ,MGMT_status->6 ,1p/19q->7 ,IDH->8)
                config.num_class (_type:int_): the number of classes we wanted for classification 
            do_transform (_type:int_): 1 means do augmentation (for train), 0 means do not augmentation (for validation and test)
        '''

        self.metadata_df = metadata_df
        self.config = config
        self.transformation = do_transform

 
    def __len__(self)-> int:
        '''Gets the length of the dataset

        Returns:
            int: total number of data points
        '''
        return len(self.metadata_df)


    def __getitem__(self, idx)-> tuple[torch.Tensor, torch.Tensor]: 
        '''_summary_

        Args:
            idx (_type:int_): the index of a slice

        Returns:
            fused_features_tensor (_type:dic_): a vector of fused features (of MRI slices and clinical data) for an instance in tensor format
            label_tensor (-type:tensor-): the label of that instance
        '''

        label = self.metadata_df.iloc[idx, self.config.label] 
        # No one-hot encoding, because CrossEntropyLoss (for multiclass classification) accepts labels as 0, 1, 2, ...
        
        # NOTE: This part of code is specific to have column 'WHO_grade' as label, not other labels
        if self.config.num_class == 3:
            match label:
                case 4:
                    label= 2
                case 3:
                    label= 1
                case 2:
                    label= 0
        else:
            match label:
                case 4:
                    label= 0
                case 3:
                    label= 1
                case 2:
                    label= 1

        label_tensor= torch.tensor(label, dtype=torch.float32) # the type should be Tensor

        data_modalities_feature_list = []   # list of fused modalities 
        mri_modalities_dict = {}   # dictionary of the MRI modalities in numpy array format

        for modality in self.config.modalities:
            
            if modality == 'Clinical':                
                # Fetch the Clinical data and MinMax normalizing the age
                clinical_np = np.array([self.metadata_df.iloc[idx, 2], self.metadata_df.iloc[idx, 3]], dtype=np.float32) 
                
                # clinical_tensor = torch.tensor(clinical_np, dtype=torch.float32)
                data_modalities_feature_list.append(clinical_np)
            
            else:
                axis_dic = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
                # Create the path of the slice of the MRI modality and load the corresponding slice 
                img_path = os.path.join(self.config.dataset_image_path, 
                                        f'UCSF-PDGM-{self.metadata_df.iloc[idx, 0]}', # ID
                                        axis_dic[self.config.axis],
                                        modality,
                                        f'{self.metadata_df.iloc[idx, 1]}.npz') # slice_name
                        
                # Load the image
                A = np.load(img_path)
                img = A[A.files[0]]  # the file is in .npz format

                mri_modalities_dict[modality] = img


        # Do transformation on images
        mri_modalities_dic_transformed = _transform(mri_modalities_dict, self.transformation)

        # Convert the transformed MRI modalities to numpy array and append to the list
        for modality in mri_modalities_dic_transformed.keys():
            data_modalities_feature_list.append(mri_modalities_dic_transformed[modality].flatten())  # flatten the image and convert to numpy array

        data_modalities_fused_features_nparray = np.concatenate(data_modalities_feature_list, axis=0) # concatenate the features of all modalities, shape is (num_features,)
        data_modalities_fused_features_tensor = torch.tensor(data_modalities_fused_features_nparray, dtype=torch.float32).unsqueeze(0)  # convert to tensor and add batch dimension, shape is (batch_size, num_features)


        # return dictionary of tensors of modalities in the config.modalities list and label_tensor
        return data_modalities_fused_features_tensor, label_tensor