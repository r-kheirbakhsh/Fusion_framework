
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import cv2
from torchvision.transforms import v2

from dataloader.transfromation import CustomCompose, scale_mri_image

 
 
def _transform(mri_multichannel, pretrained, do, list_of_mri_modalities):
    ''' Do augmentation using CustomCompose
    Args:
        mri_multichannel (_type:dictionary_): a multichannel image of slices of MRI 
        pretrained (_type:int_): 1 means the model is pretrained, 0 meant it is not pretrained
        do (_type:int_): 1 means do augmentation, 0 means do not augmentation

    Returns:
        tensor: the transformed image in the tensor format
    '''

    if pretrained == 1:
        transform_2= v2.Compose([
            v2.ToImage(),
            v2.Resize((224, 224)),  # Resize to 224x224
            v2.ToDtype(torch.float32, scale=False)     
        ])

        transform_1= v2.Compose([
            v2.ToImage(),
            v2.Resize((224, 224)),  # Resize to 224x224
            v2.RandomHorizontalFlip(p=0.5),          # Random horizontal flip
            v2.RandomVerticalFlip(p=0.5),            # Random vertical flip
            v2.RandomRotation(degrees=20),           # Random rotation
            v2.ToDtype(torch.float32, scale=False)     
        ])

    else:
        transform_2= v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=False)     
        ])

        transform_1= v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),          # Random horizontal flip
            v2.RandomVerticalFlip(p=0.5),            # Random vertical flip
            v2.RandomRotation(degrees=20),           # Random rotation
            v2.ToDtype(torch.float32, scale=False)     
        ])

    # Scale each channel, and replace the scaled one in the image
    for channel in range(0, mri_multichannel.shape[2]):

        # min_value = np.min(mri_multichannel[:,:,channel])
        # max_value = np.max(mri_multichannel[:,:,channel])
        # range_val = max_value - min_value
        # # Scale the array
        # if range_val > 0:
        #     mri_multichannel[:,:,channel] = (mri_multichannel[:,:,channel] - min_value) / range_val
        # else:
        #     mri_multichannel[:,:,channel] = 0.0

        # scale the MRI image to [0, 1] range
        mri_multichannel[:,:,channel] = scale_mri_image(mri_multichannel[:,:,channel], list_of_mri_modalities[channel])


    if (do == 1):   # transform_1 should be performed (for train)
        return transform_1(mri_multichannel)

    else:   # transform_2 should be performed (for validation and test)
        return transform_2(mri_multichannel)


class UCSFslice_intermediate_2_fusion(Dataset):
    '''UCSFslice_intermediate_2_fusion
    Args:
        Dataset: Parent torch dataset class
    '''
    def __init__(self, metadata_df, config, do_transform) -> None:
        ''' Sets the class variables

        Args:
            metadata_df (_type:panda dataframe_): it contains the metadata of the dataset with columns: ID,slice_name,sex,age,WHO_grade,final_diagnosis,MGMT_status,1p/19q,IDH
            config (_type:config_): it contains the configeration of the problem, the ones used in this class:
                config.image_path (_type:str_): the path to the main folder of images, like /mnt/storage/reyhaneh/data/UCSF/UCSF-PDGM-SLICED            
                config.axis (_type:int_): axis along which the slices are in the dataset -> 0: Sagittal, 1: Coronal, 2: Axial  
                config.pretrained (_type:int_): 1 means the model is pretrained, 0 meant it is not pretrained
                config.data_label (_type:int_): the label for classification (WHO_grade->4 ,final_diagnosis->5 ,MGMT_status->6 ,1p/19q->7 ,IDH->8)
                config.num_class (_type:int_): the number of classes we wanted for classification 
            do_transform (_type:int_): 1 means do augmentation (for train), 0 means do not augmentation (for validation and test)
        '''

        self.metadata_df = metadata_df
        self.config = config
        self.transformation = do_transform


    def __len__(self) -> int:
        '''Gets the length of the dataset

        Returns:
            int: total number of data points
        '''
        return len(self.metadata_df)
    

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]: 
        '''_summary_

        Args:
            idx (_type:int_): the index of a slice

        Returns:
            data_modalities_dic_tensor (_type:dic_): a dictionary of data modalities with at most two tensors: 1. a multichannel MR image, and 2. clinical data
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

        label_tensor= torch.tensor(label, dtype=torch.long) # the type should be Tensor

        data_modalities_dict_tensor = {}   # dictionary of at most two item (Clinical and multichannel MRI) in tensor format

        if 'Clinical' in self.config.modalities:
            # Fetch the Clinical data and MinMax normalizing the age
            clinical_np = np.array([self.metadata_df.iloc[idx, 2], self.metadata_df.iloc[idx, 3]], dtype=np.float32) 
            
            clinical_tensor = torch.tensor(clinical_np, dtype=torch.float32)
            data_modalities_dict_tensor['Clinical'] = clinical_tensor


        if 'T1_bias' in self.config.modalities or 'T1c_bias' in self.config.modalities or 'T2_bias' in self.config.modalities or 'FLAIR_bias' in self.config.modalities:
            axis_dic = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
            slices_of_modalities = [] # a list of slices of different modalities
            list_of_mri_modalities = []  # a list of modalities in the config.modalities

            # Create the path of the slices in differen modalities, load the corresponding slice and then append that slice to the list of slices
            for modality in self.config.modalities:
                
                if modality in ['T1_bias', 'T1c_bias', 'T2_bias', 'FLAIR_bias']:
                    img_path = os.path.join(self.config.dataset_image_path, 
                                        f'UCSF-PDGM-{self.metadata_df.iloc[idx, 0]}', # ID
                                        axis_dic[self.config.axis],
                                        modality,
                                        f'{self.metadata_df.iloc[idx, 1]}.npz') # slice_name
                    
                    # Load the image
                    A = np.load(img_path)
                    img = A[A.files[0]]
                    slices_of_modalities.append(img)
                    list_of_mri_modalities.append(modality)

            # Stack the arrays along the third axis (channel axis)
            fused_image = np.stack(slices_of_modalities, axis=-1)  # Shape: (240, 240, len(modalities))

            image_tensor = _transform(fused_image, self.config.pretrained, self.transformation, list_of_mri_modalities)

            data_modalities_dict_tensor['MRI'] = image_tensor


        # return dictionary of tensors of modalities in the config.modalities list and labe_tensor
        return data_modalities_dict_tensor, label_tensor