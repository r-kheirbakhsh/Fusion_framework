import numpy as np
import cv2
import torch
from sklearn.preprocessing import StandardScaler

class CustomCompose(object):
    def __init__(self, resize=None, vert_flip_th=0.5, hor_flip_th=0.5, rotation_degree=20):
        self.resize = resize
        self.vert_flip_th = vert_flip_th
        self.hor_flip_th = hor_flip_th
        self.rotation_degree = rotation_degree
    
    def __call__(self, modalities_dict):
        if self.resize is not None:
            for key in modalities_dict.keys():
                modalities_dict[key] = cv2.resize(modalities_dict[key], self.resize, interpolation=cv2.INTER_AREA)

        # Random vertical flip
        if np.random.rand() < self.vert_flip_th:
            for key in modalities_dict.keys():
                modalities_dict[key] = cv2.flip(modalities_dict[key], 0)

        # Random horizontal flip
        if np.random.rand() < self.hor_flip_th:
            for key in modalities_dict.keys():
                modalities_dict[key] = cv2.flip(modalities_dict[key], 1)

        if self.rotation_degree > 0:        
            # Random rotation                                                       
            rotation_angle = np.random.uniform(-self.rotation_degree, self.rotation_degree) 
            for key in modalities_dict.keys():
                h, w = modalities_dict[key].shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                modalities_dict[key] = cv2.warpAffine(modalities_dict[key], M, (w, h))  

        # NOTE: If you use torchvision.transforms, note that ToTensor() converts a PIL.Image or numpy.ndarray of shape (H, W) into a tensor of shape [1, H, W] — but only if the input is correctly formatted.
        # Alternatively, if you’re starting with a NumPy array and converting it manually:
        # image = torch.tensor(image, dtype=torch.float32)  # [H, W]
        # image = image.unsqueeze(0)  # [1, H, W]

        # Convert to tensor
        for key in modalities_dict.keys():
            modalities_dict[key] = torch.tensor(modalities_dict[key], dtype=torch.float32).unsqueeze(0)

        return modalities_dict   


class CustomCompose_not_tensor(object):
    def __init__(self, resize=None, vert_flip_th=0.5, hor_flip_th=0.5, rotation_degree=20):
        self.resize = resize
        self.vert_flip_th = vert_flip_th
        self.hor_flip_th = hor_flip_th
        self.rotation_degree = rotation_degree
    
    def __call__(self, modalities_dict):
        if self.resize is not None:
            for key in modalities_dict.keys():
                modalities_dict[key] = cv2.resize(modalities_dict[key], self.resize, interpolation=cv2.INTER_AREA)

        # Random vertical flip
        if np.random.rand() < self.vert_flip_th:
            for key in modalities_dict.keys():
                modalities_dict[key] = cv2.flip(modalities_dict[key], 0)

        # Random horizontal flip
        if np.random.rand() < self.hor_flip_th:
            for key in modalities_dict.keys():
                modalities_dict[key] = cv2.flip(modalities_dict[key], 1)

        if self.rotation_degree > 0:        
            # Random rotation                                                       
            rotation_angle = np.random.uniform(-self.rotation_degree, self.rotation_degree) 
            for key in modalities_dict.keys():
                h, w = modalities_dict[key].shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                modalities_dict[key] = cv2.warpAffine(modalities_dict[key], M, (w, h))  

        return modalities_dict  


def scale_mri_image(image, modality):
    ''' This function takes a slice of MRI image and scales it to [0, 1] range

    Args:
        image (_type:numpy array_): a slice of MRI in numpy array format
        modality (_type:str_): the modality of the MRI image (T1_bias, T1c_bias, T2_bias, FLAIR_bias)

    Returns:
        image (_type:numpy array_): the scaled slice of MRI in numpy array format

    '''

    # Set the min and max values for global scaling based on the modality
    match modality:
        case 'T1_bias':
            min_value = 0   
            max_value = 8690 # 8689.239481016994
        case 'T1c_bias':
            min_value = 0
            max_value = 17118 # 17117.279822677374
        case 'T2_bias':
            min_value = 0
            max_value = 5347 # 5346.053190082312
        case 'FLAIR_bias':
            min_value = 0
            max_value = 6355 # 6354.434775821865
 

    # scalar = StandardScaler()
    # image = scalar.fit_transform(image)

    # # find the max and min value of the array (image) for scaling
    # min_value = np.min(image)
    # max_value = np.max(image)
    # # Scale the array (image)
    # image = (image - min_value) / (max_value - min_value)

    return image 