

import sys
import numpy as np
import pandas as pd
import json
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, matthews_corrcoef, log_loss
from sklearn.preprocessing import QuantileTransformer

from dataloader import UCSFslice_late_fusion, UCSFslice_intermediate_1_fusion, UCSFslice_intermediate_2_fusion, UCSFslice_early_2_fusion



def prepare_device(n_gpu_use):
    '''setup GPU device if available. get gpu device indices which are used for DataParallel
    Args:
        n_gpu_use (__type:int__): indicates the number of gpus configured to use
    '''

    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    
    return device, list_ids


def move_to_device(batch, device):
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    return batch.to(device)


def scale_clinical_data(config, train_df, val_df, test_df)->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ''' This function takes a config, and the splits of the dataset (slice level) and scale the Clinical data

    Args:
        config (_type:Config_): the configeration of the problem, the ones used in this class:
            config.scale_clinical_modality (_type:str_): the method to scale the Clinical data ('Minmax', 'Normalize', ...)

    Returns:
        train_scaled_df (_type:pandas Dataframe_): the scaled train dataset (slice level) 
        val_scaled_df (_type:pandas Dataframe_): the scaled validation dataset (slice level)
        test_scaled_df (_type:pandas Dataframe_): the scaled test dataset (slice level)

    ''' 
    column_to_scale = 'age'
    if config.scale_clinical_modality == 'Minmax':
        train_df.loc[:,column_to_scale] = train_df.loc[:,column_to_scale]/100
        val_df.loc[:,column_to_scale] = val_df.loc[:,column_to_scale]/100
        test_df.loc[:,column_to_scale] = test_df.loc[:,column_to_scale]/100
    
    elif config.scale_clinical_modality == 'Quantile-normal':
        # Initialize the transformer
        qt = QuantileTransformer(output_distribution='normal', random_state=config.seed)

        # Fit on the training age column only
        qt.fit(train_df[[column_to_scale]])

        # Transform Age column for all splits
        train_df[column_to_scale] = qt.transform(train_df[[column_to_scale]])
        val_df[column_to_scale] = qt.transform(val_df[[column_to_scale]])
        test_df[column_to_scale] = qt.transform(test_df[[column_to_scale]])
    
    else:
        pass

    return train_df, val_df, test_df

       

def prepare_dataset_split(config, dataset_patient_df, train_index, test_index)->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ''' This function takes a config, and the splits of the dataset (patient level) and prepare the train, val, and test datasets (slice level)

    Args:
        config (_type:Config_): the configeration of the problem, the ones used in this class:
            config.seed (_type:int_): for reproducability
        dataset_patient_df (_type:pandas Dataframe_): the patient level dataset
        train_index (_type:list_): the index of the train dataset
        temp_index (_type:list_): the index of the temp dataset

    Returns:
        train_slice_df (_type:pandas Dataframe_): the train dataset (slice level) 
        val_slice_df (_type:pandas Dataframe_): the validation dataset (slice level)
        test_slice_df (_type:pandas Dataframe_): the test dataset (slice level)

    '''
    # Load dataset
    dtype_dict = {'ID': str, 'slice_name': str, 'sex': int, 'age': int, 'WHO_grade': int}
    dataset_slice_df = pd.read_csv(config.dataset_csv_path, dtype=dtype_dict)
    
    # Split to train, vel, and test datasets
    test_patient_df = dataset_patient_df.iloc[test_index]
    train_patient_df, val_patient_df = train_test_split(dataset_patient_df.iloc[train_index], train_size=0.875, random_state=config.seed, shuffle=True, stratify=dataset_patient_df['WHO_grade'].iloc[train_index])
    
    # Produce the slice level splits
    train_slice_df = dataset_slice_df[dataset_slice_df['ID'].isin(train_patient_df['ID'])]
    val_slice_df = dataset_slice_df[dataset_slice_df['ID'].isin(val_patient_df['ID'])]
    test_slice_df = dataset_slice_df[dataset_slice_df['ID'].isin(test_patient_df['ID'])]
   
    # ################################### for train/test
    # train_patient_df = dataset_patient_df.iloc[train_index]
    # test_patient_df = dataset_patient_df.iloc[test_index]

    # train_slice_df = dataset_slice_df[dataset_slice_df['ID'].isin(train_patient_df['ID'])]
    # val_slice_df = dataset_slice_df[dataset_slice_df['ID'].isin(test_patient_df['ID'])]
    # test_slice_df = dataset_slice_df[dataset_slice_df['ID'].isin(test_patient_df['ID'])]
    # #########################################

    # Scale Clinical data
    train_slice_scaled_df, val_slice_scaled_df, test_slice_scaled_df = scale_clinical_data(config, train_slice_df, val_slice_df, test_slice_df)

    # Save the statistics of data sets into a text file
    save_dataset_splited_statistics(config, train_slice_scaled_df, val_slice_scaled_df, test_slice_scaled_df)

    return train_slice_scaled_df, val_slice_scaled_df, test_slice_scaled_df      



def dataset_slice_split(config)->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ''' This function takes a config, and split the dataset (slice level) specified in the config into train, val, and test, saves them, and then returns their data frames

    Args:
        config (_type:Config_): the configeration of the problem, the ones used in this class:
            config.dataset_csv_path (_type:int_): the path to the csv file containing the slice level data
            config.seed (_type:int_): for reproducability

    Returns:
        train_df (_type:pandas Dataframe_): the train dataset (slice level) 
        val_df (_type:pandas Dataframe_): the validation dataset (slice level)
        test_df (_type:pandas Dataframe_): the test dataset (slice level)

    '''

    dtype_dict = {'ID': str, 'slice_name': str, 'sex': int, 'age': int, 'WHO_grade': int}
    dataset_slice_df = pd.read_csv(config.dataset_csv_path, dtype=dtype_dict)
    
    # using the config.seed produce train, val, and test datasets
    # Group by slice level data to have patient level dataset
    dataset_temp_df = dataset_slice_df.groupby('ID')
    dataset_patient_df = dataset_temp_df.first().reset_index()

    # Split to train, vel, and test datasets
    train_ratio = 0.7
    train_patient_df, temp_df= train_test_split(dataset_patient_df, train_size=train_ratio, random_state=config.seed, shuffle=True, stratify=dataset_patient_df['WHO_grade'])
    val_patient_df, test_patient_df= train_test_split(temp_df, train_size=0.33, random_state=config.seed, shuffle=True, stratify=temp_df['WHO_grade'])
    
    # Produce the slice level splits
    train_index = train_patient_df['ID']
    val_index = val_patient_df['ID']
    test_index = test_patient_df['ID']

    train_slice_df = dataset_slice_df[dataset_slice_df['ID'].isin(train_index)]
    val_slice_df = dataset_slice_df[dataset_slice_df['ID'].isin(val_index)]
    test_slice_df = dataset_slice_df[dataset_slice_df['ID'].isin(test_index)]

    # Scale Clinical data
    train_slice_scaled_df, val_slice_scaled_df, test_slice_scaled_df = scale_clinical_data(config, train_slice_df, val_slice_df, test_slice_df)    
    
    # Save the statistics of data sets into a text file
    save_dataset_splited_statistics(config, train_slice_scaled_df, val_slice_scaled_df, test_slice_scaled_df)

    # # Save the data frames to csv files
    # train_slice_scaled_df.to_csv(f'train_dataset_{config.seed}.csv', index=False)
    # val_slice_scaled_df.to_csv(f'val_dataset_{config.seed}.csv', index=False)
    # test_slice_scaled_df.to_csv(f'test_dataset_{config.seed}.csv', index=False)
    
    return train_slice_scaled_df, val_slice_scaled_df, test_slice_scaled_df 



def dataset_slice_get(config)->tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ''' This function takes a config, and split the dataset (slice level) specified in the config into train, val, and test, saves them, and then returns their data frames

    Args:
        config (_type:Config_): the configeration of the problem, the ones used in this class:
            config.train_csv_path (_type:int_): the path to the csv file containing the slice level data
            config.val_csv_path (_type:int_): the path to the csv file containing the slice level data
            config.test_csv_path (_type:int_): the path to the csv file containing the slice level data

    Returns:
        tuple: A tuple containing the train, validation, and test data frames

    '''

    dtype_dict = {'ID': str, 'slice_name': str, 'sex': int, 'age': int, 'WHO_grade': int}
    train_slice_df = pd.read_csv(config.train_csv_path, dtype=dtype_dict)
    val_slice_df = pd.read_csv(config.val_csv_path, dtype=dtype_dict)
    test_slice_df = pd.read_csv(config.test_csv_path, dtype=dtype_dict)

    # Scale Clinical data
    train_slice_scaled_df, val_slice_scaled_df, test_slice_scaled_df = scale_clinical_data(config, train_slice_df, val_slice_df, test_slice_df)

    return train_slice_scaled_df, val_slice_scaled_df, test_slice_scaled_df



def save_dataset_splited_statistics(config, train_df, val_df, test_df)->None:
    ''' This function takes the train, val, and test data frames (slice level) and saves the statistics of the dataset into a text file   
    Args:
        config (_type:Config_): the configeration of the problem, the ones used in this class:
            config.axis (_type:int_): the axis of the slices (0: Sagittal, 1: Coronal, 2: Axial)
            config.seed (_type:int_): for reproducability
            config.fold (_type:int_): the fold number for cross-validation
        train_df (_type:pandas Dataframe_): the train dataset (slice level) 
        val_df (_type:pandas Dataframe_): the validation dataset (slice level)
        test_df (_type:pandas Dataframe_): the test dataset (slice level)   
    
    Returns:
        None: the function saves the statistics of the dataset into a text file

    '''

    # Convert the slice level dataframes to patient level dataframes
    train_patient_df = slice_to_patient_dataset(train_df)
    val_patient_df = slice_to_patient_dataset(val_df)
    test_patient_df = slice_to_patient_dataset(test_df)

    # Get the number of patients in each data set
    train_patient_size = len(train_patient_df)
    val_patient_size = len(val_patient_df)
    test_patient_size = len(test_patient_df)
    total_patient = train_patient_size + test_patient_size + val_patient_size # for the moment that I have test==val

    # Get the number of data samples in each data set
    train_size = len(train_df)
    val_size = len(val_df)
    test_size = len(test_df)
    total = train_size + test_size + val_size  # for the moment that I have test==val
    

    # Save the original stdout so you can restore it later
    original_stdout = sys.stdout 

    #axis_dic = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
    # Redirect output to a file
    with open(f'dataset_{config.seed}_{config.fold}_statistics.txt', 'w') as f:
        sys.stdout = f  # Redirects all print statements to f

        # View the counts
        print(f'\n\nStatistics of Axial_43_56_396_{config.seed}_{config.fold} dataset')
        print('\nNumber of datapoints:')
        print(f'\t\tGrade 2:\t43')
        print(f'\t\tGrade 3:\t56')
        print(f'\t\tGrade 4:\t396\n')

        print(f'train patients count:\t\t{train_patient_size}\t({(100 * train_patient_size/total_patient):.2f}%)')
        print(f'validation patients count:\t{val_patient_size}\t({(100 * val_patient_size/total_patient):.2f}%)')
        print(f'test patients count:\t\t{test_patient_size}\t({(100 * test_patient_size/total_patient):.2f}%)')
        print('================================================')
        print(f'TOTAL:\t\t\t\t{total_patient}\t({(100 * total_patient/total_patient):.2f}%)')

        print('================================================')
        print(f'train samples count:\t\t{train_size}\t({(100 * train_size/total):.2f}%)')
        print(f'validation samples count:\t{val_size}\t({(100 * val_size/total):.2f}%)')
        print(f'test samples count:\t\t{test_size}\t({(100 * test_size/total):.2f}%)')
        print('================================================')
        print(f'TOTAL:\t\t\t\t{total}\t({(100 * total/total):.2f}%)')

        print('================================================')
        print('Number of slices of each classes in the train, validation, and test dataset:\n')
        
        labels_dic_2= {4: "WHO_grade", 5: "final_diagnosis", 6: "MGMT_status", 7: "1p/19q", 8: "IDH"}
        # Statistics of the classes in each of the train, val, and test dataset
        for label, group in train_df.groupby(labels_dic_2[4]):
            print(f'Number of label {label} in train dataset:\t{len(group)}\t({(100 * len(group)/train_size):.2f}%)')
    
        print('----------------------------------------------------------')
        for label, group in val_df.groupby(labels_dic_2[4]):
            print(f'Number of label {label} in val dataset:\t{len(group)}\t({(100 * len(group)/val_size):.2f}%)')

        print('----------------------------------------------------------')
        for label, group in test_df.groupby(labels_dic_2[4]):
            print(f'Number of label {label} in test dataset:\t{len(group)}\t({(100 * len(group)/test_size):.2f}%)')  
    
    # Restore stdout to its original value
    sys.stdout = original_stdout

    

def slice_to_patient_dataset(dataset_df_slice):
    ''' This function takes a data set in pandas dataframe fromat at slice level, and retuens that data frame 
        at patient level 

    Args:
        dataset_df_slice (_type:pd.dataframe_): the dataframe contaning data at slice level

    Returns:
        dataset_df_patient (_type:pd.dataframe_): the dataframe contaning data at patient with scaled 'age' data

    '''

    dataset_df_temp = dataset_df_slice.groupby('ID')
    dataset_df_patient = dataset_df_temp.first().reset_index()

    return dataset_df_patient



def prepare_classic_classifier_dataset(dataset)->tuple[np.ndarray, np.ndarray]:

    # data_np = np.array([instance.item() for instance in dataloader][0])
    # labels_np = np.array([instance.item() for instance in dataloader][1])

    # Create lists to store the data
    tabular_data = []
    labels = []

    # Iterate through the dataset
    for i in range(len(dataset)):
        clinical_tensor, label_tensor = dataset[i]  # Extract data
        tabular_data.append(clinical_tensor.numpy())  # Convert tensor to numpy
        labels.append(label_tensor.numpy())

    # Convert lists to NumPy arrays
    tabular_data_np = np.stack(tabular_data)  # Shape: (num_samples, num_features)
    labels_np = np.array(labels)  # Shape: (num_samples,)

    return tabular_data_np, labels_np



def get_dataloader_late(metadata_df, config, train_flag, modality, batch_size) -> tuple[UCSFslice_late_fusion,DataLoader]:
    ''' This function takes a data set in pandas dataframe fromat and creats the relevant dataset class and dataloader 

    Args:
        metadata_df (_type:dataframe_): the dataframe contaning data 
        config (_type:Config_): it contains the configeration of the problem, the ones used in this class:
            config.seed (_type:int_): seed for reproducibility
        modality (_type:str_): modality of data we wan for the dataset/dataloader    
        train_flag (_type:int_): 1 means train dataset and datloader, 0 means val/test dataset and dataloader 
        batch_size (_type:int_): batch size for the dataloader

    Returns:
        dataset (_type:class_): an instance of UCSF_late_fusion class contaning dataset 
        dataloader (_type:class_): an instance of Dataloader for the above dataset

    '''

    # Use a generator with the same seed to make DataLoader shuffling reproducible:
    g = torch.Generator()
    g.manual_seed(config.seed)

    dataset = UCSFslice_late_fusion(
            metadata_df = metadata_df,
            config = config,
            do_transform = train_flag,
            modality = modality,
        )

    # During training, it is often beneficial to use drop_last to maintain a consistent batch size. This can help in stabilizing the gradient updates and improving convergence rates.
    if train_flag == 1:
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True, 
            num_workers = 2,
            drop_last = True,
            pin_memory=True,
            generator = g 
        )
    # during evaluation or validation, you might want to keep all data points, including the last incomplete batch. Therefore, it is common to set drop_last to False in these scenarios.
    else:
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size, 
            shuffle = False,             
            num_workers = 2,
            drop_last = False,
            pin_memory=True
        ) 

    return dataset, dataloader



def get_labels_from_df(metadata_df, num_class, label_col):
    ''' This function takes a data set in pandas dataframe fromat and returns the labels of the data
    
    Args:
        metadata_df (_type:dataframe_): the dataframe contaning data 
        num_class (_type:int_): the number of class we have 2 or >2 
        label_col (_type:int_): the column number of the labels in the dataframe
    
    Returns:    
        labels (_type:list_): the labels of the data in the dataframe
    '''
    
    labels = []

    for idx in range(len(metadata_df)):
        label = metadata_df.iloc[idx, label_col]
        if num_class == 3:
            label = {4: 2, 3: 1, 2: 0}[label]
        else:
            label = 0 if label == 4 else 1
        labels.append(label)

    return labels



def get_dataloader_late_sampler(metadata_df, config, modality, train_flag, batch_size) -> tuple[UCSFslice_late_fusion,DataLoader]:
    ''' This function takes a data set in pandas dataframe fromat and creates the relevant dataset class and dataloader with sampler
    
    Args:
        metadata_df (_type:dataframe_): the dataframe contaning data 
        config (_type:Config_): it contains the configeration of the problem
        modality (_type:str_): modality of data we wan for the dataset/dataloader
        train_flag (_type:int_): 1 means train dataset and datloader, 0 means val/test dataset and dataloader
        batch_size (_type:int_): batch size for the dataloader       
    
    Returns:
        dataset (_type:class_): an instance of UCSF_late_fusion class contaning dataset 
        dataloader (_type:class_): an instance of Dataloader for the above dataset
    ''' 

    # Get list of labels from the metadata DataFrame and replace these with your actual values
    labels = get_labels_from_df(metadata_df, config.num_class, config.label)

    # Count the frequency of each class
    label_counts = Counter(labels)

    # Compute weight for each sample: inverse of its class frequency
    weights = [1.0 / label_counts[label] for label in labels]

    sampler = WeightedRandomSampler(
    weights = weights,
        num_samples = len(weights),  # or e.g., 1000 * batch_size for debugging
        replacement = True
    )

    dataset = UCSFslice_late_fusion(
            metadata_df = metadata_df,
            config = config,
            do_transform = train_flag,
            modality = modality,
        )

    if train_flag == 1:
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            sampler = sampler,
            num_workers = 2,
            drop_last = True,
            pin_memory=True
        )
        
    else:       
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            sampler = None,
            num_workers = 2,
            drop_last = False,
            pin_memory=True
        )

    return dataset, dataloader



def get_dataloader_intermediate_1(metadata_df, config, train_flag, batch_size) -> tuple[UCSFslice_intermediate_1_fusion,DataLoader]:
    ''' This function takes a data set in pandas dataframe fromat and creats the relevant dataset class and dataloader 

    Args:
         metadata_df (_type:dataframe_): the dataframe contaning data 
         config (_type:Config_): it contains the configeration of the problem, the ones used in this class:
             config.batch_size_nn (_type:int_): batch size for nn_model for MRI modalities
         train_flag (_type:int_): 1 means train dataset and datloader, 0 means val/test dataset and dataloader

    Returns:
         dataset (_type:class_): an instance of UCSF_intermediate_1_fusion class contaning dataset 
         dataloader (_type:class_): an instance of Dataloader for the above dataset

    '''

    # Use a generator with the same seed to make DataLoader shuffling reproducible:
    g = torch.Generator()
    g.manual_seed(config.seed)

    dataset = UCSFslice_intermediate_1_fusion(
             metadata_df = metadata_df,
             config = config,
             do_transform = train_flag,
         )


    if train_flag == 1:
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True, 
            num_workers = 2,
            drop_last = True,
            generator = g 
        )

    else:
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size, 
            shuffle = False,             
            num_workers = 2,
            drop_last = False,
        )
  
    return dataset, dataloader



def get_dataloader_intermediate_1_sampler(metadata_df, config, train_flag, batch_size) -> tuple[UCSFslice_intermediate_1_fusion, DataLoader]:
    ''' This function takes a data set in pandas dataframe fromat and creates the relevant dataset class and dataloader with sampler
    
    Args:
        metadata_df (_type:dataframe_): the dataframe contaning data 
        config (_type:Config_): it contains the configeration of the problem
        train_flag (_type:int_): 1 means train dataset and datloader, 0 means val/test dataset and dataloader
        batch_size (_type:int_): batch size for the dataloader       
    
    Returns:
        dataset (_type:class_): an instance of UCSF_intermediate_1_fusion class contaning dataset 
        dataloader (_type:class_): an instance of Dataloader for the above dataset
    ''' 

    # Get list of labels from the metadata DataFrame and replace these with your actual values
    labels = get_labels_from_df(metadata_df, config.num_class, config.label)

    # Count the frequency of each class
    label_counts = Counter(labels)

    # Compute weight for each sample: inverse of its class frequency
    weights = [1.0 / label_counts[label] for label in labels]

    sampler = WeightedRandomSampler(
    weights = weights,
        num_samples = len(weights),  # or e.g., 1000 * batch_size for debugging
        replacement = True
    )

    dataset = UCSFslice_intermediate_1_fusion(
            metadata_df = metadata_df,
            config = config,
            do_transform = train_flag,
        )


    if train_flag == 1:
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            sampler = sampler,
            num_workers = 2,
            drop_last = True
        )
        
    else:       
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            sampler = None,
            num_workers = 2,
            drop_last = False
        )

    return dataset, dataloader



def get_dataloader_intermediate_2(metadata_df, config, train_flag, batch_size) -> tuple[UCSFslice_intermediate_2_fusion,DataLoader]:
    ''' This function takes a data set in pandas dataframe fromat and creats the relevant dataset class and dataloader 

    Args:
        metadata_df (_type:dataframe_): the dataframe contaning data 
        config (_type:Config_): it contains the configeration of the problem, the ones used in this class:
        train_flag (_type:int_): 1 means train dataset and datloader, 0 means val/test dataset and dataloader
        batch_size (_type:int_): batch size for the dataloader 

    Returns:
        dataset (_type:class_): an instance of UCSF_intermediate_2_fusion class contaning dataset 
        dataloader (_type:class_): an instance of Dataloader for the above dataset

    '''

    # Use a generator with the same seed to make DataLoader shuffling reproducible:
    g = torch.Generator()
    g.manual_seed(config.seed)

    dataset = UCSFslice_intermediate_2_fusion(
             metadata_df = metadata_df,
             config = config,
             do_transform = train_flag,
         )


    if train_flag == 1:
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True, 
            num_workers = 2,
            drop_last = True,
            generator = g 
        )

    else:
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size, 
            shuffle = False,             
            num_workers = 2,
            drop_last = False,
        )
  
    return dataset, dataloader



def get_dataloader_intermediate_2_sampler(metadata_df, config, train_flag, batch_size) -> tuple[UCSFslice_intermediate_2_fusion, DataLoader]:
    ''' This function takes a data set in pandas dataframe fromat and creates the relevant dataset class and dataloader with sampler
    
    Args:
        metadata_df (_type:dataframe_): the dataframe contaning data 
        config (_type:Config_): it contains the configeration of the problem
        train_flag (_type:int_): 1 means train dataset and datloader, 0 means val/test dataset and dataloader
        batch_size (_type:int_): batch size for the dataloader       
    
    Returns:
        dataset (_type:class_): an instance of UCSF_intermediate_2_fusion class contaning dataset 
        dataloader (_type:class_): an instance of Dataloader for the above dataset
    ''' 
    g = torch.Generator()
    g.manual_seed(config.seed)

    # Get list of labels from the metadata DataFrame and replace these with your actual values
    labels = get_labels_from_df(metadata_df, config.num_class, config.label)

    # Count the frequency of each class
    label_counts = Counter(labels)

    # Compute weight for each sample: inverse of its class frequency
    weights = [1.0 / label_counts[label] for label in labels]

    sampler = WeightedRandomSampler(
    weights = weights,
        num_samples = len(weights),  # or e.g., 1000 * batch_size for debugging
        replacement = True
    )

    dataset = UCSFslice_intermediate_2_fusion(
            metadata_df = metadata_df,
            config = config,
            do_transform = train_flag,
        )


    if train_flag == 1:
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            sampler = sampler,
            num_workers = 2,
            drop_last = True
        )
        
    else:       
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            sampler = None,
            num_workers = 2,
            drop_last = False
        )

    return dataset, dataloader



def get_dataloader_early_2(metadata_df, config, train_flag, batch_size) -> tuple[UCSFslice_early_2_fusion, DataLoader]:
    ''' This function takes a data set in pandas dataframe fromat and creats the relevant dataset class and dataloader 

    Args:
        metadata_df (_type:dataframe_): the dataframe contaning data 
        config (_type:Config_): it contains the configeration of the problem, the ones used in this class:
        train_flag (_type:int_): 1 means train dataset and datloader, 0 means val/test dataset and dataloader
        batch_size (_type:int_): batch size for the dataloader 

    Returns:
        dataset (_type:class_): an instance of UCSF_early_2_fusion class contaning dataset 
        dataloader (_type:class_): an instance of Dataloader for the above dataset

    '''

    # Use a generator with the same seed to make DataLoader shuffling reproducible:
    g = torch.Generator()
    g.manual_seed(config.seed)

    dataset = UCSFslice_early_2_fusion(
             metadata_df = metadata_df,
             config = config,
             do_transform = train_flag,
         )


    if train_flag == 1:
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = True, 
            num_workers = 2,
            drop_last = True,
            generator = g 
        )
 
    else:
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size, 
            shuffle = False,             
            num_workers = 2,
            drop_last = False,
        )
  
    return dataset, dataloader



def get_dataloader_early_2_sampler(metadata_df, config, train_flag, batch_size) -> tuple[UCSFslice_early_2_fusion, DataLoader]:
    ''' This function takes a data set in pandas dataframe fromat and creats the relevant dataset class and dataloader 

    Args:
        metadata_df (_type:dataframe_): the dataframe contaning data 
        config (_type:Config_): it contains the configeration of the problem, the ones used in this class:
        train_flag (_type:int_): 1 means train dataset and datloader, 0 means val/test dataset and dataloader
        batch_size (_type:int_): batch size for the dataloader 

    Returns:
        dataset (_type:class_): an instance of UCSF_early_2_fusion class contaning dataset 
        dataloader (_type:class_): an instance of Dataloader for the above dataset

    '''

    # Use a generator with the same seed to make DataLoader shuffling reproducible:
    g = torch.Generator()
    g.manual_seed(config.seed)

    # Get list of labels from the metadata DataFrame and replace these with your actual values
    labels = get_labels_from_df(metadata_df, config.num_class, config.label)

    # Count the frequency of each class
    label_counts = Counter(labels)

    # Compute weight for each sample: inverse of its class frequency
    weights = [1.0 / label_counts[label] for label in labels]

    sampler = WeightedRandomSampler(
    weights = weights,
        num_samples = len(weights),  # or e.g., 1000 * batch_size for debugging
        replacement = True
    )

    dataset = UCSFslice_early_2_fusion(
             metadata_df = metadata_df,
             config = config,
             do_transform = train_flag,
         )

    if train_flag == 1:
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            sampler = sampler,
            num_workers = 2,
            drop_last = True
        )
        
    else:       
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            sampler = None,
            num_workers = 2,
            drop_last = False
        )

    return dataset, dataloader



def calculate_save_metrics_late(config, modality, y_labels, y_predicted, multi, training_time_spent, test_loss=None) -> tuple[float, float, float, float, float]:   
    ''' This function takes two numpy array, one the labels and the other predicted value, and then calculates 
    the performance metrics and saves the reports on three files of .tex, .json, and .csv

    Args:
        config (_type:Config_): the configeration of the problem, the ones used in this class:
            config.mri_model (_type:str_): the name of model for MRI modality
            config.cl_model (_type:str_): the name of model for Clinical modality
            config.seed (_type:str_): for reproducability
            config.num_class (_type:int_): the number of class we have 2 or >2 
        modality(_type:str_): the modality(ies) used in the model
        y_labels (_type:np.array_): the true labels
        y_predicted (_type:np.array_): the predicted values for y_labels
        multi (_type:int_): 0 means unimodal, 1 means multimodal model
        training_time_spent (_type:float_): the time spent for training the model
        test_loss (_type:float_): -1 the call is for fused model, o.w. the test loss of the model

    Returns:
        None: the function saves the metrics in three files, .tex, .json, and .csv
        or
        test_accuracy, MCC (_type:float, float_): the test accuracy and MCC of the model if it is a multimodal model
        f1_w (_type:float_): The weighted average of f1_score of the model on the test set
        recall_w (_type:float_): The weighted average of recall of the model on the test set
        precision_w (_type:float_): The weighted average of precision of the model on the test set

    '''

    if multi == 0: # it is unimodal model
        if modality == 'Clinical':
            model_type = config.cl_model
            if config.cl_model == 'MLP' or config.cl_model == 'AutoInt':
                batch_size = config.batch_size_cl
                n_epochs = config.n_epochs_cl
                lr_rate = config.lr_cl
            else:  # for XGBoost model              
                batch_size = 0
                n_epochs = 0
                lr_rate = 0 
        else:
            model_type = config.mri_model
            batch_size = config.batch_size_mri
            n_epochs = config.n_epochs_mri
            lr_rate = config.lr_mri
    
    else: # it is multimodal model
        batch_size = 0
        n_epochs = 0
        lr_rate = 0
        if 'Clinical' in config.modalities:
            model_type =  f'{config.mri_model}_{config.cl_model}_{config.fused_model}' 
        else:
            model_type = f'{config.mri_model}_{config.fused_model}'

    # Compute classification metrics
    conf_matrix = confusion_matrix(y_labels, y_predicted).tolist()  # Convert to list for JSON serialization
    match config.num_class:
        case 3: 
            report = classification_report(y_labels, y_predicted, target_names=['Grade 2', 'Grade 3', 'Grade 4'], output_dict=True)
        case 2: 
            report = classification_report(y_labels, y_predicted, target_names=['Grade 4', 'Grade 2&3'], output_dict=True)

    # comput metrics
    MCC = matthews_corrcoef(y_labels, y_predicted)
    test_accuracy = accuracy_score(y_labels, y_predicted) 
    f1_w = report['weighted avg']['f1-score']
    recall_w = report['weighted avg']['recall']
    precision_w = report['weighted avg']['precision']

    axis_dic = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
    # Save metrics to a text file
    with open(f'{modality}_fold_{config.fold}_metrics.txt', 'w') as f:
        f.write(f'Dataset Spec: {axis_dic[config.axis]}_43_56_396_seed_{config.seed}_{config.fold}\n')
        f.write(f'Modality: {modality}\n')
        f.write(f'Fusion Method: {config.fusion_method}\n')
        f.write(f'Model: {model_type}\n')
        f.write(f'Batch size: {batch_size}\n')
        f.write(f'Number of epochs: {n_epochs}\n')
        f.write(f'Time spent for training: {training_time_spent:.2f} minutes\n\n')
        f.write(f'Test Loss: {test_loss:.4f}\n')
        f.write(f'Test Accuracy: {test_accuracy:.4f}\n')
        f.write(f'Test MCC: {MCC:.4f}\n\n')
        f.write('Confusion Matrix:\n')
        f.write(f'{conf_matrix}\n\n')
        f.write('Classification Report:\n')
        f.write(classification_report(y_labels, y_predicted, target_names=['Grade 4', 'Grade 2&3'], output_dict=False))

    # Save metrics and configuration to a JSON file
    results = {
        "config": {
            "dataset": f'{axis_dic[config.axis]}_43_56_396_seed_{config.seed}_{config.fold}',
            "fusion_method": f'{config.fusion_method}',
            "modality": modality,
            "model": model_type,
            "batch_size": batch_size,
            "num_epochs": n_epochs, 
            "training_time": training_time_spent,       
            "learning_rate": lr_rate, 
            "clinical_scaling": config.scale_clinical_modality
        },
        "metrics": {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy, 
            "test_MCC": MCC,
            "confusion_matrix": conf_matrix,
            "grade 2&3 precision": report['Grade 2&3']['precision'],
            "grade 2&3 recall": report['Grade 2&3']['recall'],
            "grade 2&3 f1-score": report['Grade 2&3']['f1-score'],
            "grade 2&3 support": report['Grade 2&3']['support'],
            "grade 4 precision": report['Grade 4']['precision'],
            "grade 4 recall": report['Grade 4']['recall'],
            "grade 4 f1-score": report['Grade 4']['f1-score'],
            "grade 4 support": report['Grade 4']['support'],
            "precision-macro avg": report['macro avg']['precision'],
            "recall-macro avg": report['macro avg']['recall'],
            "f1-score-macro avg": report['macro avg']['f1-score'],
            "support-macro avg": report['macro avg']['support'],
            "precision-weighted avg": report['weighted avg']['precision'],
            "recall-weighted avg": report['weighted avg']['recall'],
            "f1-score-weighted avg": report['weighted avg']['f1-score'],
            "support-weighted avg": report['weighted avg']['support']    
        }
    }

    with open(f'{modality}_{config.fold}_metrics.json', 'w') as json_file:
        json.dump(results, json_file, indent=4)

    # Save the configs and metrics to a csv file
    # Flatten nested structures and convert to a DataFrame
    # Combine 'config' and 'metrics' into a single dictionary for CSV export
    combined_data = {**results['config'], **results['metrics']}

    # Convert the combined dictionary into a DataFrame
    df = pd.DataFrame([combined_data])  # Create a DataFrame with one row

    csv_file_path = '/mnt/storage/reyhaneh/experiments/gl_classification/Modality_fusion_framework_experiments/AICS25/late_results.csv' 
    # Load the existing CSV file
    try:
        csv_df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # Create a new DataFrame if the CSV does not exist
        csv_df = pd.DataFrame()

    # Append the new JSON data
    updated_csv_df = pd.concat([csv_df, df], ignore_index=True)

    # Save the updated CSV
    updated_csv_df.to_csv(csv_file_path, index=False)

    if multi == 0: # it is unimodal model
        return None
    else: # it is multimodal model
        return test_accuracy, MCC, f1_w, recall_w, precision_w  # return the accuracy and MCC for the fused model to be used in the loop over folds



def calculate_save_metrics_intermediate_1(config, modality, y_labels, y_predicted, training_time_spent, test_loss=None, all_weights=None)-> tuple[float, float, float, float, float]:   
    ''' This function takes two numpy array, one the labels and the other predicted value, and then calculates 
    the performance metrics and saves the reports on three files of .tex, .json, and .csv

    Args:
        config (_type:Config_): the configeration of the problem, the ones used in this class:
            config.fused_model (_type:str_): the name of intermediate fusion model
            config.mri_model (_type:str_): the name of model for MRI modality
            config.cl_model (_type:str_): the name of model for Clinical modality
            config.fusion_method (_type:str_): the fusion method used in the model
            config.axis (_type:int_): the axis of the data, 0 for Sagittal, 1 for Coronal, and 2 for Axial
            config.batch_size_fused (_type:int_): batch size for the fused model
            config.n_epochs_fused (_type:int_): number of epochs for the fused model
            config.lr_fused (_type:str_): learning rate for the fused model
            config.scale_clinical_modality (_type:bool_): whether to scale the clinical modality or not
            config.seed (_type:str_): for reproducability
            config.num_class (_type:int_): the number of class we have 2 or >2 
        modality(_type:str_): the modality(ies) used in the model
        y_labels (_type:np.ndarray_): the labels
        y_predicted (_type:np.ndarray_): the predicted values for y_labels
        training_time_spent (_type:float_): the time spent for training the model
        test_loss (_type:float_): the test loss of the intermediate fusion model 

    Returns:
        test_accuracy (_type:float_): the accuracy of the model
        MCC (_type:float_): the Matthews correlation coefficient of the model
        f1_w (_type:float_): The weighted average of f1_score of the model on the test set
        recall_w (_type:float_): The weighted average of recall of the model on the test set
        precision_w (_type:float_): The weighted average of precision of the model on the test set

    '''

    # Compute classification metrics
    conf_matrix = confusion_matrix(y_labels, y_predicted).tolist()  # Convert to list for JSON serialization
    match config.num_class:
        case 3: 
            report = classification_report(y_labels, y_predicted, target_names=['Grade 2', 'Grade 3', 'Grade 4'], output_dict=True)
        case 2: 
            report = classification_report(y_labels, y_predicted, target_names=['Grade 4', 'Grade 2&3'], output_dict=True)

    # comput MCC
    MCC = matthews_corrcoef(y_labels, y_predicted)
    test_accuracy = accuracy_score(y_labels, y_predicted)
    f1_w = report['weighted avg']['f1-score']
    recall_w = report['weighted avg']['recall']
    precision_w = report['weighted avg']['precision']

    # Calculate avg attention weights
    if all_weights is not None:
        modality_cont_avg, modality_cont_label_0_avg, modality_cont_label_1_avg = calculate_avg_attn_weights(y_labels, y_predicted, all_weights)
    else:
        modality_cont_avg = None
        modality_cont_label_0_avg = None
        modality_cont_label_1_avg = None

    axis_dic = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
    # Save metrics to a text file
    with open(f'fold_{config.fold}_metrics.txt', 'w') as f:
        f.write(f'Dataset Spec: {axis_dic[config.axis]}_43_56_396_seed_{config.seed}_{config.fold}\n')
        f.write(f'Fusion method: {config.fusion_method}\n')
        f.write(f'Modality: {modality}\n')
        f.write(f'Model: {config.fused_model}\n')
        f.write(f'MRI backbone: {config.mri_model}\n')
        f.write(f'Clinical backbone: {config.cl_model}\n')
        f.write(f'Batch size: {config.batch_size_fused}\n')
        f.write(f'Number of epochs: {config.n_epochs_fused}\n')
        f.write(f'Time spent for training: {training_time_spent:.2f} minutes\n\n')
        f.write(f'Test Loss: {test_loss:.4f}\n')
        f.write(f'Test Accuracy: {test_accuracy:.4f}\n')
        f.write(f'Test MCC: {MCC:.4f}\n\n')
        f.write(f'Modality Continuous Avg: {modality_cont_avg}\n')
        f.write(f'Modality Continuous Label 0 Avg: {modality_cont_label_0_avg}\n')
        f.write(f'Modality Continuous Label 1 Avg: {modality_cont_label_1_avg}\n')
        f.write('Confusion Matrix:\n')
        f.write(f'{conf_matrix}\n\n')
        f.write('Classification Report:\n')
        f.write(classification_report(y_labels, y_predicted, target_names=['Grade 4', 'Grade 2&3'], output_dict=False))

    # Save metrics and configuration to a JSON file
    results = {
        "config": {
            "dataset": f'{axis_dic[config.axis]}_43_56_396_seed_{config.seed}_{config.fold}',
            "fusion_method": config.fusion_method,
            "modality": modality,
            "model": config.fused_model,
            "mri_backbone": config.mri_model,
            "clinical_backbone": config.cl_model,
            "batch_size": config.batch_size_fused,
            "num_epochs": config.n_epochs_fused, 
            "training_time": training_time_spent,      
            "learning_rate": config.lr_fused, 
            "clinical_scaling": config.scale_clinical_modality
        },
        "metrics": {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy, 
            "test_MCC": MCC,
            #"modality_continuous_avg": modality_cont_avg,
            "modality_continuous_label_4_avg": modality_cont_label_0_avg,
            "modality_continuous_label_2&3_avg": modality_cont_label_1_avg,
            "confusion_matrix": conf_matrix,
            "grade 2&3 precision": report['Grade 2&3']['precision'],
            "grade 2&3 recall": report['Grade 2&3']['recall'],
            "grade 2&3 f1-score": report['Grade 2&3']['f1-score'],
            "grade 2&3 support": report['Grade 2&3']['support'],
            "grade 4 precision": report['Grade 4']['precision'],
            "grade 4 recall": report['Grade 4']['recall'],
            "grade 4 f1-score": report['Grade 4']['f1-score'],
            "grade 4 support": report['Grade 4']['support'],
            "precision-macro avg": report['macro avg']['precision'],
            "recall-macro avg": report['macro avg']['recall'],
            "f1-score-macro avg": report['macro avg']['f1-score'],
            "support-macro avg": report['macro avg']['support'],
            "precision-weighted avg": report['weighted avg']['precision'],
            "recall-weighted avg": report['weighted avg']['recall'],
            "f1-score-weighted avg": report['weighted avg']['f1-score'],
            "support-weighted avg": report['weighted avg']['support']    
        }
    }

    with open(f'fold_{config.fold}_metrics.json', 'w') as json_file: 
        json.dump(results, json_file, indent=4)

    # Save the configs and metrics to a csv file
    # Flatten nested structures and convert to a DataFrame
    # Combine 'config' and 'metrics' into a single dictionary for CSV export
    combined_data = {**results['config'], **results['metrics']}

    # Convert the combined dictionary into a DataFrame
    df = pd.DataFrame([combined_data])  # Create a DataFrame with one row

    csv_file_path = '/mnt/storage/reyhaneh/experiments/gl_classification/Modality_fusion_framework_experiments/AICS25/intermediate_1_results.csv'
    try:
        csv_df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # Create a new DataFrame if the CSV does not exist
        csv_df = pd.DataFrame()

    # Append the new JSON data
    updated_csv_df = pd.concat([csv_df, df], ignore_index=True)

    # Save the updated CSV
    updated_csv_df.to_csv(csv_file_path, index=False)

    return test_accuracy, MCC, f1_w, recall_w, precision_w  # return the accuracy and MCC for the fused model to be used in the loop over folds



def Inter_2_calculate_avg_attn_weights(y_labels, y_predicted, all_weights)-> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ''' This function takes a list of attention weights arrays and calculates the average attention weights across all arrays

    Args:
        y_labels (_type:np.ndarray_): The true labels
        y_predicted (_type:np.ndarray_): The predicted labels
        attn_weights (_type:np.ndarray_): A NumPy array containing attention weights

    Returns:
        modality_cont_avg (_type:np.ndarray_): A NumPy array containing the average attention weights
        modality_cont_label_0_avg (_type:np.ndarray_): A NumPy array containing the average attention weights for label 0 (grade 4)
        modality_cont_label_1_avg (_type:np.ndarray_): A NumPy array containing the average attention weights for label 1 (grade 2&3)
        modality_cont_label_0_correct_avg (_type:np.ndarray_): A NumPy array containing the average attention weights for label 0 (grade 4) for correctly predicted samples
        modality_cont_label_1_correct_avg (_type:np.ndarray_): A NumPy array containing the average attention weights for label 1 (grade 2&3) for correctly predicted samples
        modality_cont_correct_avg (_type:np.ndarray_): A NumPy array containing the average attention weights for correctly predicted samples across all labels

    '''
    # Calculate the mean attention weights
    modality_cont_avg = all_weights.mean(axis=0).squeeze()

    # Reshape attention weights to (number of instances in test, number of modalities)
    attention_weights = attention_weights.reshape(attention_weights.shape[0], attention_weights.shape[1])
    
    df = pd.DataFrame({
    'True_Label': y_labels,
    'Predicted': y_predicted,
    'Attention_Weight_MRI': attention_weights[:, 0],
    'Attention_Weight_Clinical': attention_weights[:, 1],
    })

    # Calculate mean attention weights for each label
    MRI_contribution_label_0_avg = df[df['True_Label'] == 0]['Attention_Weight_MRI'].mean()
    Clinical_contribution_label_0_avg = df[df['True_Label'] == 0]['Attention_Weight_Clinical'].mean()
    modality_cont_label_0_avg = np.array([MRI_contribution_label_0_avg, Clinical_contribution_label_0_avg])

    MRI_contribution_label_1_avg = df[df['True_Label'] == 1]['Attention_Weight_MRI'].mean()
    Clinical_contribution_label_1_avg = df[df['True_Label'] == 1]['Attention_Weight_Clinical'].mean()
    modality_cont_label_1_avg = np.array([MRI_contribution_label_1_avg, Clinical_contribution_label_1_avg])

    # Calculate mean attention weights for the correctly predicted label 0 (grade 4)
    MRI_contribution_label_0_correct_avg = df[df['True_Label'] == 0 & df['Predicted'] == 0]['Attention_Weight_MRI'].mean()
    Clinical_contribution_label_0_correct_avg = df[df['True_Label'] == 0 & df['Predicted'] == 0]['Attention_Weight_Clinical'].mean()
    modality_cont_label_0_correct_avg = np.array([MRI_contribution_label_0_correct_avg, Clinical_contribution_label_0_correct_avg])

    # Calculate mean attention weights for the correctly predicted label 1 (grade 2&3)
    MRI_contribution_label_1_correct_avg = df[df['True_Label'] == 1 & df['Predicted'] == 1]['Attention_Weight_MRI'].mean()
    Clinical_contribution_label_1_correct_avg = df[df['True_Label'] == 1 & df['Predicted'] == 1]['Attention_Weight_Clinical'].mean()
    modality_cont_label_1_correct_avg = np.array([MRI_contribution_label_1_correct_avg, Clinical_contribution_label_1_correct_avg])

    # Calculate mean attention weights for the correctly predicted samples
    MRI_contribution_correct_avg = df[(df['True_Label'] == 1 & df['Predicted'] == 1) | (df['True_Label'] == 0 & df['Predicted'] == 0)]['Attention_Weight_MRI'].mean()
    Clinical_contribution_correct_avg = df[(df['True_Label'] == 1 & df['Predicted'] == 1) | (df['True_Label'] == 0 & df['Predicted'] == 0)]['Attention_Weight_Clinical'].mean()
    modality_cont_correct_avg = np.array([MRI_contribution_correct_avg, Clinical_contribution_correct_avg])


    return modality_cont_avg, modality_cont_label_0_avg, modality_cont_label_1_avg, modality_cont_label_0_correct_avg, modality_cont_label_1_correct_avg, modality_cont_correct_avg


def calculate_save_metrics_intermediate_2(config, modality, y_labels, y_predicted, training_time_spent, test_loss=None, all_weights=None)-> tuple[float, float, float, float, float]:   
    ''' This function takes two numpy array, one the labels and the other predicted value, and then calculates 
    the performance metrics and saves the reports on three files of .tex, .json, and .csv

    Args:
        config (_type:Config_): the configeration of the problem, the ones used in this class:
            config.fused_model (_type:str_): the name of intermediate fusion model
            config.mri_model (_type:str_): the name of model for MRI modality
            config.cl_model (_type:str_): the name of model for Clinical modality
            config.fusion_method (_type:str_): the fusion method used in the model
            config.axis (_type:int_): the axis of the data, 0 for Sagittal, 1 for Coronal, and 2 for Axial
            config.batch_size_fused (_type:int_): batch size for the fused model
            config.n_epochs_fused (_type:int_): number of epochs for the fused model
            config.lr_fused (_type:str_): learning rate for the fused model
            config.scale_clinical_modality (_type:bool_): whether to scale the clinical modality or not
            config.seed (_type:str_): for reproducability
            config.num_class (_type:int_): the number of class we have 2 or >2 
        modality(_type:str_): the modality(ies) used in the model
        y_labels (_type:np.array_): the labels
        y_predicted (_type:np.array_): the predicted values for y_labels
        training_time_spent (_type:float_): the time spent for training the model
        test_loss (_type:float_): the test loss of the intermediate fusion model 
        all_weights (_type:numpy.ndarray_): the attention weights of the model on the test dataset (only for attention-based models)

    Returns:
        test_accuracy (_type:float_): the accuracy of the model
        MCC (_type:float_): the Matthews correlation coefficient of the model
        f1_w (_type:float_): The weighted average of f1_score of the model on the test set
        recall_w (_type:float_): The weighted average of recall of the model on the test set
        precision_w (_type:float_): The weighted average of precision of the model on the test set

    '''

    # Compute classification metrics
    conf_matrix = confusion_matrix(y_labels, y_predicted).tolist()  # Convert to list for JSON serialization
    match config.num_class:
        case 3: 
            report = classification_report(y_labels, y_predicted, target_names=['Grade 2', 'Grade 3', 'Grade 4'], output_dict=True)
        case 2: 
            report = classification_report(y_labels, y_predicted, target_names=['Grade 4', 'Grade 2&3'], output_dict=True)

    # comput metrics
    MCC = matthews_corrcoef(y_labels, y_predicted)
    test_accuracy = accuracy_score(y_labels, y_predicted)
    f1_w = report['weighted avg']['f1-score']
    recall_w = report['weighted avg']['recall']
    precision_w = report['weighted avg']['precision']

    # Calculate avg attention weights
    if all_weights is not None:
        modality_cont_avg, modality_cont_label_0_avg, modality_cont_label_1_avg, \
            modality_cont_label_0_correct_avg, modality_cont_label_1_correct_avg, \
            modality_cont_correct_avg = Inter_2_calculate_avg_attn_weights(y_labels, y_predicted, all_weights)
    else:
        modality_cont_avg = None
        modality_cont_label_0_avg = None
        modality_cont_label_1_avg = None
        modality_cont_label_0_correct_avg = None
        modality_cont_label_1_correct_avg = None
        modality_cont_correct_avg = None

    axis_dic = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
    # Save metrics to a text file
    with open(f'fold_{config.fold}_metrics.txt', 'w') as f:
        f.write(f'Dataset Spec: {axis_dic[config.axis]}_43_56_396_seed_{config.seed}_{config.fold}\n')
        f.write(f'Fusion method: {config.fusion_method}\n')
        f.write(f'Modality: {modality}\n')
        f.write(f'Model: {config.fused_model}\n')
        f.write(f'MRI backbone: {config.mri_model}\n')
        f.write(f'Clinical backbone: {config.cl_model}\n')
        f.write(f'Batch size: {config.batch_size_fused}\n')
        f.write(f'Number of epochs: {config.n_epochs_fused}\n')
        f.write(f'Time spent for training: {training_time_spent:.2f} minutes\n\n')
        f.write(f'Test Loss: {test_loss:.4f}\n')
        f.write(f'Test Accuracy: {test_accuracy:.4f}\n')
        f.write(f'Test MCC: {MCC:.4f}\n\n')
        f.write(f'Modality Contribution Avg: {modality_cont_avg}\n')
        f.write(f'Modality Contribution Label 0 Avg: {modality_cont_label_0_avg}\n')
        f.write(f'Modality Contribution Label 1 Avg: {modality_cont_label_1_avg}\n')
        f.write(f'Modality Contribution Label 0 Correct Avg: {modality_cont_label_0_correct_avg}\n')
        f.write(f'Modality Contribution Label 1 Correct Avg: {modality_cont_label_1_correct_avg}\n')
        f.write(f'Modality Contribution Correct Avg: {modality_cont_correct_avg}\n\n')
        f.write('Confusion Matrix:\n')
        f.write(f'{conf_matrix}\n\n')
        f.write('Classification Report:\n')
        f.write(classification_report(y_labels, y_predicted, target_names=['Grade 4', 'Grade 2&3'], output_dict=False))

    # Save metrics and configuration to a JSON file
    results = {
        "config": {
            "dataset": f'{axis_dic[config.axis]}_43_56_396_seed_{config.seed}_{config.fold}',
            "fusion_method": config.fusion_method,
            "modality": modality,
            "model": config.fused_model,
            "mri_backbone": config.mri_model,
            "clinical_backbone": config.cl_model,
            "batch_size": config.batch_size_fused,
            "num_epochs": config.n_epochs_fused, 
            "training_time": training_time_spent,      
            "learning_rate": config.lr_fused, 
            "clinical_scaling": config.scale_clinical_modality
        },
        "metrics": {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy, 
            "test_MCC": MCC,
            #"modality_contribution_avg": modality_cont_avg,
            #"modality_contribution_label_4_avg": modality_cont_label_0_avg,
            #"modality_contribution_label_2&3_avg": modality_cont_label_1_avg,
            "confusion_matrix": conf_matrix,
            "grade 2&3 precision": report['Grade 2&3']['precision'],
            "grade 2&3 recall": report['Grade 2&3']['recall'],
            "grade 2&3 f1-score": report['Grade 2&3']['f1-score'],
            "grade 2&3 support": report['Grade 2&3']['support'],
            "grade 4 precision": report['Grade 4']['precision'],
            "grade 4 recall": report['Grade 4']['recall'],
            "grade 4 f1-score": report['Grade 4']['f1-score'],
            "grade 4 support": report['Grade 4']['support'],
            "precision-macro avg": report['macro avg']['precision'],
            "recall-macro avg": report['macro avg']['recall'],
            "f1-score-macro avg": report['macro avg']['f1-score'],
            "support-macro avg": report['macro avg']['support'],
            "precision-weighted avg": report['weighted avg']['precision'],
            "recall-weighted avg": report['weighted avg']['recall'],
            "f1-score-weighted avg": report['weighted avg']['f1-score'],
            "support-weighted avg": report['weighted avg']['support']    
        }
    }

    with open(f'fold_{config.fold}_metrics.json', 'w') as json_file: 
        json.dump(results, json_file, indent=4)

    # Save the configs and metrics to a csv file
    # Flatten nested structures and convert to a DataFrame
    # Combine 'config' and 'metrics' into a single dictionary for CSV export
    combined_data = {**results['config'], **results['metrics']}

    # Convert the combined dictionary into a DataFrame
    df = pd.DataFrame([combined_data])  # Create a DataFrame with one row

    csv_file_path = '/mnt/storage/reyhaneh/experiments/gl_classification/Modality_fusion_framework_experiments/AICS25/intermediate_2_results.csv'
    try:
        csv_df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # Create a new DataFrame if the CSV does not exist
        csv_df = pd.DataFrame()

    # Append the new JSON data
    updated_csv_df = pd.concat([csv_df, df], ignore_index=True)

    # Save the updated CSV
    updated_csv_df.to_csv(csv_file_path, index=False)

    return test_accuracy, MCC, f1_w, recall_w, precision_w  # return the metrics for the fused model to be used in the loop over folds


def calculate_save_metrics_early_1(config, modality, y_labels, y_predicted, training_time_spent, uni, test_loss=None)-> tuple[float, float, float, float, float]:   
    ''' This function takes two numpy array, one the labels and the other predicted value, and then calculates 
    the performance metrics and saves the reports on three files of .tex, .json, and .csv

    Args:
        config (_type:Config_): the configeration of the problem, the ones used in this class:
            config.mri_model (_type:str_): the name of the model for MRI modality
            config.cl_model (_type:str_): the name of the model for Clinical modality
            config.fused_model (_type:str_): the name of early fusion model
            config.n_epochs_mri (_type:int_): number of epochs for MRI modality
            config.n_epochs_cl (_type:int_): number of epochs for Clinical modality
            config.n_epochs_fused (_type:int_): number of epochs for early fusion model
            config.batch_size_mri (_type:int_): batch size for MRI modality
            config.batch_size_cl (_type:int_): batch size for Clinical modality
            config.batch_size_fused (_type:int_): batch size for early fusion model
            config.lr_mri (_type:float_): learning rate for MRI modality
            config.lr_cl (_type:float_): learning rate for Clinical modality
            config.lr_fused (_type:float_): learning rate for early fusion model 
            config.seed (_type:str_): for reproducability   
            config.num_class (_type:int_): the number of class we have 2 or >2 
        modality(_type:str_): the modality(ies) used in the model
        y_labels (_type:np.array_): the labels
        y_predicted (_type:np.array_): the predicted values for y_labels
        training_time_dict (_type:dict_): the time spent for training the model for each modality
        uni (_type:int_): 1 means unimodal, 0 means multimodal model
        test_loss (_type:float_): the test loss of the intermediate fusion model 
        
    Returns:
        test_accuracy (_type:float_): the accuracy of the model
        MCC (_type:float_): the Matthews correlation coefficient of the model
        f1_w (_type:float_): The weighted average of f1_score of the model on the test set
        recall_w (_type:float_): The weighted average of recall of the model on the test set
        precision_w (_type:float_): The weighted average of precision of the model on the test set

    '''
    if uni == 1: # it is the feature extraction phase
        if modality == 'Clinical':
            model_type = config.cl_model
            n_epochs = config.n_epochs_cl
            n_batches = config.batch_size_cl
            lr_rate = config.lr_cl
        else:
            model_type = config.mri_model
            n_epochs = config.n_epochs_mri
            n_batches = config.batch_size_mri
            lr_rate = config.lr_mri
    else: # it is the main training phase for early fusion model
        model_type = config.fused_model
        n_epochs = config.n_epochs_fused
        n_batches = config.batch_size_fused
        lr_rate = config.lr_fused


    # Compute classification metrics
    conf_matrix = confusion_matrix(y_labels, y_predicted).tolist()  # Convert to list for JSON serialization
    match config.num_class:
        case 3: 
            report = classification_report(y_labels, y_predicted, target_names=['Grade 2', 'Grade 3', 'Grade 4'], output_dict=True)
        case 2: 
            report = classification_report(y_labels, y_predicted, target_names=['Grade 4', 'Grade 2&3'], output_dict=True)

    # comput metrics
    MCC = matthews_corrcoef(y_labels, y_predicted)
    test_accuracy = accuracy_score(y_labels, y_predicted)
    f1_w = report['weighted avg']['f1-score']
    recall_w = report['weighted avg']['recall']
    precision_w = report['weighted avg']['precision']

    axis_dic = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
    # Save metrics to a text file
    with open(f'{modality}_{config.fold}_metrics.txt', 'w') as f:
        f.write(f'Dataset Spec: {axis_dic[config.axis]}_43_56_396_seed_{config.seed}_{config.fold}\n')
        f.write(f'Fusion Method: {config.fusion_method}\n')
        f.write(f'Modality: {modality}\n')
        f.write(f'Model: {model_type}\n')
        f.write(f'Batch size: {n_batches}\n')
        f.write(f'Number of epochs: {n_epochs}\n')
        f.write(f'Time spent for training: {training_time_spent:.2f} minutes\n\n')
        f.write(f'Test Loss: {test_loss:.4f}\n')
        f.write(f'Test Accuracy: {test_accuracy:.4f}\n')
        f.write(f'Test MCC: {MCC:.4f}\n\n')
        f.write('Confusion Matrix:\n')
        f.write(f'{conf_matrix}\n\n')
        f.write('Classification Report:\n')
        f.write(classification_report(y_labels, y_predicted, target_names=['Grade 4', 'Grade 2&3'], output_dict=False))


    # Save metrics and configuration to a JSON file
    results = {
        "config": {
            "dataset": f'{axis_dic[config.axis]}_43_56_396_seed_{config.seed}_{config.fold}',
            "fusion_method": config.fusion_method,
            "modality": modality,
            "model": model_type,
            "batch_size": n_batches,
            "num_epochs": n_epochs,
            "training_time": training_time_spent,
            "learning_rate": lr_rate,  
            "clinical_scaling": config.scale_clinical_modality
        },
        "metrics": {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,  
            "test_MCC": MCC,
            "confusion_matrix": conf_matrix,
            "grade 2&3 precision": report['Grade 2&3']['precision'],
            "grade 2&3 recall": report['Grade 2&3']['recall'],
            "grade 2&3 f1-score": report['Grade 2&3']['f1-score'],
            "grade 2&3 support": report['Grade 2&3']['support'],
            "grade 4 precision": report['Grade 4']['precision'],
            "grade 4 recall": report['Grade 4']['recall'],
            "grade 4 f1-score": report['Grade 4']['f1-score'],
            "grade 4 support": report['Grade 4']['support'],
            "precision-macro avg": report['macro avg']['precision'],
            "recall-macro avg": report['macro avg']['recall'],
            "f1-score-macro avg": report['macro avg']['f1-score'],
            "support-macro avg": report['macro avg']['support'],
            "precision-weighted avg": report['weighted avg']['precision'],
            "recall-weighted avg": report['weighted avg']['recall'],
            "f1-score-weighted avg": report['weighted avg']['f1-score'],
            "support-weighted avg": report['weighted avg']['support']    
        }
    }

    with open(f'{modality}_{config.fold}_metrics.json', 'w') as json_file: 
        json.dump(results, json_file, indent=4)

    # Save the configs and metrics to a csv file
    # Flatten nested structures and convert to a DataFrame
    # Combine 'config' and 'metrics' into a single dictionary for CSV export
    combined_data = {**results['config'], **results['metrics']}

    # Convert the combined dictionary into a DataFrame
    df = pd.DataFrame([combined_data])  # Create a DataFrame with one row

    csv_file_path = '/mnt/storage/reyhaneh/experiments/gl_classification/Modality_fusion_framework_experiments/AICS25/early_1_results.csv'
    # Load the existing CSV file
    try:
        csv_df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # Create a new DataFrame if the CSV does not exist
        csv_df = pd.DataFrame()

    # Append the new JSON data
    updated_csv_df = pd.concat([csv_df, df], ignore_index=True)

    # Save the updated CSV
    updated_csv_df.to_csv(csv_file_path, index=False)

    return test_accuracy, MCC, f1_w, recall_w, precision_w  # return the metrics for the fused model to be used in the loop over folds



def calculate_save_metrics_early_2(config, modality, y_labels, y_predicted, training_time_spent, test_loss=None) -> tuple[float, float, float, float, float]:   
    ''' This function takes two numpy array, one the labels and the other predicted value, and then calculates 
    the performance metrics and saves the reports on three files of .tex, .json, and .csv

    Args:
        config (_type:Config_): the configeration of the problem, the ones used in this class:
            config.seed (_type:str_): for reproducability
            config.num_class (_type:int_): the number of class we have 2 or >2 
        modality(_type:str_): the modality(ies) used in the model
        y_labels (_type:np.array_): the labels
        y_predicted (_type:np.array_): the predicted values for y_labels
        training_time_dict (_type:dict_): the time spent for training the model for each modality
        uni (_type:int_): 1 means unimodal, 0 means multimodal model
        test_loss (_type:float_): the test loss of the intermediate fusion model 
        
    Returns:
        test_accuracy (_type:float_): The accuracy of the model on the test set
        MCC (_type:float_): The Matthews correlation coefficient of the model on the test set
        f1_w (_type:float_): The weighted average of f1_score of the model on the test set
        recall_w (_type:float_): The weighted average of recall of the model on the test set
        precision_w (_type:float_): The weighted average of precision of the model on the test set

    '''

    # Compute classification metrics
    conf_matrix = confusion_matrix(y_labels, y_predicted).tolist()  # Convert to list for JSON serialization
    match config.num_class:
        case 3: 
            report = classification_report(y_labels, y_predicted, target_names=['Grade 2', 'Grade 3', 'Grade 4'], output_dict=True)
        case 2: 
            report = classification_report(y_labels, y_predicted, target_names=['Grade 4', 'Grade 2&3'], output_dict=True)

    # comput metrics
    MCC = matthews_corrcoef(y_labels, y_predicted)
    test_accuracy = accuracy_score(y_labels, y_predicted)
    f1_w = report['weighted avg']['f1-score']
    recall_w = report['weighted avg']['recall']
    precision_w = report['weighted avg']['precision']

    axis_dic = {0: "Sagittal", 1: "Coronal", 2: "Axial"}
    # Save metrics to a text file
    with open(f'fold_{config.fold}_metrics.txt', 'w') as f:
        f.write(f'Dataset Spec: {axis_dic[config.axis]}_43_56_396_seed_{config.seed}_{config.fold}\n')
        f.write(f'Fusion Method: {config.fusion_method}\n')
        f.write(f'Modality: {modality}\n')
        f.write(f'Model: {config.fused_model}\n')
        f.write(f'Batch size: {config.batch_size_fused}\n')
        f.write(f'Number of epochs: {config.n_epochs_fused}\n')
        f.write(f'Time spent for training: {training_time_spent:.2f} minutes\n\n')
        f.write(f'Test Loss: {test_loss:.4f}\n')
        f.write(f'Test Accuracy: {test_accuracy:.4f}\n')
        f.write(f'Test MCC: {MCC:.4f}\n\n')
        f.write('Confusion Matrix:\n')
        f.write(f'{conf_matrix}\n\n')
        f.write('Classification Report:\n')
        f.write(classification_report(y_labels, y_predicted, target_names=['Grade 4', 'Grade 2&3'], output_dict=False))


    # Save metrics and configuration to a JSON file
    results = {
        "config": {
            "dataset": f'{axis_dic[config.axis]}_43_56_396_seed_{config.seed}_{config.fold}',
            "fusion_method": config.fusion_method,
            "modality": modality,
            "model": config.fused_model,
            "batch_size": config.batch_size_fused,
            "num_epochs": config.n_epochs_fused,
            "training_time": training_time_spent,
            "learning_rate": config.lr_fused,  
            "clinical_scaling": config.scale_clinical_modality
        },
        "metrics": {
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,  
            "test_MCC": MCC,
            "confusion_matrix": conf_matrix,
            "grade 2&3 precision": report['Grade 2&3']['precision'],
            "grade 2&3 recall": report['Grade 2&3']['recall'],
            "grade 2&3 f1-score": report['Grade 2&3']['f1-score'],
            "grade 2&3 support": report['Grade 2&3']['support'],
            "grade 4 precision": report['Grade 4']['precision'],
            "grade 4 recall": report['Grade 4']['recall'],
            "grade 4 f1-score": report['Grade 4']['f1-score'],
            "grade 4 support": report['Grade 4']['support'],
            "precision-macro avg": report['macro avg']['precision'],
            "recall-macro avg": report['macro avg']['recall'],
            "f1-score-macro avg": report['macro avg']['f1-score'],
            "support-macro avg": report['macro avg']['support'],
            "precision-weighted avg": report['weighted avg']['precision'],
            "recall-weighted avg": report['weighted avg']['recall'],
            "f1-score-weighted avg": report['weighted avg']['f1-score'],
            "support-weighted avg": report['weighted avg']['support']    
        }
    }

    with open(f'fold_{config.fold}_metrics.json', 'w') as json_file: 
        json.dump(results, json_file, indent=4)

    # Save the configs and metrics to a csv file
    # Flatten nested structures and convert to a DataFrame
    # Combine 'config' and 'metrics' into a single dictionary for CSV export
    combined_data = {**results['config'], **results['metrics']}

    # Convert the combined dictionary into a DataFrame
    df = pd.DataFrame([combined_data])  # Create a DataFrame with one row

    csv_file_path = '/mnt/storage/reyhaneh/experiments/gl_classification/Modality_fusion_framework_experiments/AICS25/early_2_results.csv'
    # Load the existing CSV file
    try:
        csv_df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        # Create a new DataFrame if the CSV does not exist
        csv_df = pd.DataFrame()

    # Append the new JSON data
    updated_csv_df = pd.concat([csv_df, df], ignore_index=True)

    # Save the updated CSV
    updated_csv_df.to_csv(csv_file_path, index=False)

    return test_accuracy, MCC, f1_w, recall_w, precision_w  # return the metrics for the fused model to be used in the loop over folds