import os
from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
from torchvision.transforms import v2

from dataloader.transfromation import scale_mri_image



def _transform (image, pretrained, do, modality):
    ''' Do augmentation using torchvision.transform
    Args:
        image (_type:numpy array_): a slice of MRI in numpy array format
        model (_type:str_): the name of the class of the model we want to use
        pretrained (_type:int_): 1 means the model is pretrained, 0 meant it is not pretrained
        do (_type:int_): 1 means do augmentation, 0 means do not augmentation

    Returns:
        tensor: the transformed image in the tensor format
    '''

    transform_1= v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)     
    ])

    transform_2= v2.Compose([
            v2.ToImage(),
            v2.RandomHorizontalFlip(p=0.5),          # Random horizontal flip
            v2.RandomVerticalFlip(p=0.5),            # Random vertical flip
            v2.RandomRotation(degrees=20),           # Random rotation
            v2.ToDtype(torch.float32, scale=True)     
    ])

    # # find the max and min value of the array (image) for scaling
    # min_value = np.min(image)
    # max_value = np.max(image)
    # # Scale the array (image)
    # image = (image - min_value) / (max_value - min_value)

    # Scale the MRI image to [0, 1] range
    image = scale_mri_image(image, modality)

    if (do == 1):   # transform_2 should be performed (for train)
        if pretrained == 1: 
            resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)  # change the size of image; the pretrained model has been train on ImageNet (images of size 224*224)
            return transform_2(resized_image)
        else:
            return transform_2(image)

    else:   # transform_1 should be performed (for validation and test)
        if pretrained == 1:
            resized_image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA) # change the size of image; the pretrained model has been train on ImageNet (images of size 224*224)
            return transform_1(resized_image)
        else:          
            return transform_1(image)
        
        

class UCSFslice_late_fusion(Dataset):
    '''UCSFslice_late_fusion
    Args:
        Dataset: Parent torch dataset class
    '''
    def __init__(self, metadata_df, config, do_transform, modality) -> None:
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
            modality (_type:str_): modality of dataset; the choices are T1_bias, T1c_bias, T2_bias, FLAIR_bias, and Clinical 
        '''

        self.metadata_df = metadata_df
        self.config = config
        self.modality = modality
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
            image_tensor (_type:tensor_): an instance of data (a slice of MRI) in the modalitity of the dataset in tensor format
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

        #label_tensor= torch.tensor(label, dtype=torch.long) # the type should be Tensor
        label_tensor = torch.tensor(label, dtype=torch.float32)


        if self.modality == 'Clinical':
            
            # Fetch the Clinical data and MinMax normalizing the age
            tabular_np = np.array([self.metadata_df.iloc[idx, 2], self.metadata_df.iloc[idx, 3]], dtype=np.float32) # it works both for the slice data set, and ptient data set (as I do .groupby('ID').first().reset_index())
            
            tabular_tensor = torch.tensor(tabular_np, dtype=torch.float32)

            # return tabular_tensor, labe_tensor
            return tabular_tensor, label_tensor
        
        else:
            
            axis_dic = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
            # Create the path of the slice of the MRI modality and load the corresponding slice 
            img_path = os.path.join(self.config.dataset_image_path, 
                                    f'UCSF-PDGM-{self.metadata_df.iloc[idx, 0]}', # ID
                                    axis_dic[self.config.axis],
                                    self.modality,
                                    f'{self.metadata_df.iloc[idx, 1]}.npz') # slice_name
                    
            # Load the image
            A = np.load(img_path)
            img = A[A.files[0]]  # the file is in .npz format

            image_tensor = _transform(img, self.config.pretrained, self.transformation, self.modality)

            # return image_tensor, labe_tensor
            return image_tensor, label_tensor