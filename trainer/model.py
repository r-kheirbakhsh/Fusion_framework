import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import log_loss

from utils import slice_to_patient_dataset, prepare_device, move_to_device, prepare_classic_classifier_dataset, \
                    get_dataloader_late_sampler, get_dataloader_late, \
                    get_dataloader_intermediate_1_sampler, get_dataloader_intermediate_1, \
                    get_dataloader_early_2_sampler, get_dataloader_early_2, \
                    get_dataloader_intermediate_2_sampler, get_dataloader_intermediate_2, \
                    calculate_save_metrics_early_1, calculate_save_metrics_early_2, \
                    calculate_save_metrics_intermediate_1, calculate_save_metrics_intermediate_2, \
                    calculate_save_metrics_late

from fusion import late_fusion_function                   
from trainer import nn_Trainer_early, nn_Trainer_intermediate, nn_Trainer_late, cl_Trainer
from model import model_selection_early, model_selection_intermediate, model_selection_late




class Model:
    def __init__(self, config):
        self.config = config
        # Initialize model parameters here

    def set_seed(self):
        '''This function sets random seeds for reproducibility
        
        '''
        # Set seeds for PyTorch to ensure consistency across runs
        torch.manual_seed(self.config.seed)

        # Using a GPU, make operations deterministic by setting:
        torch.cuda.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def train(self, train_slice_df, val_slice_df, test_slice_df)-> None:
        '''This function handles the training process of the model based on the fusion method specified in the config.
        
        Args:
            train_slice_df (_type:Pandas DataFrame_): containing the training data in slice level
            val_slice_df (_type:PandasDataFrame_): containing the validation data in slice level
            test_slice_df (_type:Pandas DataFrame_): containing the test data in slice level

        Returns:
            None    
        
        '''
        # Set the random seed for reproducibility
        self.set_seed()

        match self.config.fusion_method:
            case 'early_1_fusion':
                self._train_early_1(train_slice_df, val_slice_df, test_slice_df)

            case 'early_2_fusion':
                self._train_early_2(train_slice_df, val_slice_df)

            case 'intermediate_1_fusion':
                self._train_intermediate_1(train_slice_df, val_slice_df)

            case 'intermediate_2_fusion':
                self._train_intermediate_2(train_slice_df, val_slice_df)

            case 'late_fusion':
                self._train_late(train_slice_df, val_slice_df)

            case '-':
                raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")
        return
            
    
    def evaluate(self, test_slice_df, training_time_spent)-> tuple:
        '''This function handles the evaluation process of the model on the test dataset.
        
        Args:
            test_slice_df (_type:Pandas DataFrame_): containing the test data in slice level

        Returns:
            None

        '''
        # Set the random seed for reproducibility
        self.set_seed()

        if len(self.config.modalities) > 1:
            modality = '+'.join(self.config.modalities)
        else:
            modality = self.config.modalities[0] 

        match self.config.fusion_method:
            case 'early_1_fusion':
                # acc, mcc, f1_w, recall_w, precision_w = self._test_early_1(test_slice_df, modality, 0, training_time_spent)
                return self._test_early_1(test_slice_df, modality, 0, training_time_spent)

            case 'early_2_fusion':
                return self._test_early_2(test_slice_df, modality, training_time_spent)

            case 'intermediate_1_fusion':
                return self._test_intermediate_1(test_slice_df, modality, training_time_spent)

            case 'intermediate_2_fusion':
                return self._test_intermediate_2(test_slice_df, modality, training_time_spent)

            case 'late_fusion':
                return self._test_late(test_slice_df, training_time_spent)

            case '-':
                raise ValueError(f"Unknown fusion method: {self.config.fusion_method}")
                return
            
    
    def _test_loop(self, model, test_dataloader)-> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        '''This function is the testing loop for the early_2, intermediate_1 fusion method.
        
        Args:
            model (_type:torch.nn.Module_): the model to be tested
            test_dataloader (_type:torch.utils.data.DataLoader_): the dataloader for the test dataset

        Returns:
            y_labels (_type:numpy.ndarray_): the true labels of the test dataset
            y_outputs (_type:numpy.ndarray_): the outputs of the model on the test dataset
            y_predicted (_type:numpy.ndarray_): the predicted labels of the model on the test dataset
            test_loss (_type:float_): the average loss of the model on the test dataset
            all_weights (_type:numpy.ndarray_): the attention weights of the model on the test dataset (only for attention-based models)

        '''
        # Set the random seed for reproducibility
        self.set_seed()

        # Define device
        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(self.config.n_gpu)
        model = model.to(device)
 
        # Define loss function
        match self.config.num_class:
            case 2: criterion= nn.BCEWithLogitsLoss()  # For binary classification
            case 3: criterion= nn.CrossEntropyLoss()  # For multi-class classification

        # Set the model to evaluation mode (for inference)
        model.eval()  
        test_loss = 0.0
        correct = 0
        y_labels = []
        y_outputs = []  
        y_predicted = []
        all_weights = []

        with torch.no_grad():
            for inputs, labels in test_dataloader:

                inputs = move_to_device(inputs, device)
                labels = move_to_device(labels, device)
                
                if self.config.fused_model in ['Inter_2_concat_attn', 'Inter_1_concat_attn', 'Inter_1_gated_attn']:
                    outputs, attn_weights = model(inputs)
                    attn_weights = torch.squeeze(attn_weights, 0)
                    all_weights.append(attn_weights.detach().cpu()) 

                else:
                    outputs = model(inputs)

                if self.config.num_class == 2:
                    labels = labels.float()
                    outputs = torch.squeeze(outputs, 0)  # Remove one extra dimension, this works for batch_size=1
                    predicted = torch.round(torch.sigmoid(outputs))  # for Binary classification
                else:
                    _, predicted = torch.max(outputs, 1)  # for Multi-class classification 
                
                try:
                    loss = criterion(outputs, labels)
                except ValueError:
                    print('outputs: ', outputs, 'labels: ', labels)
                    raise ValueError('ERROR')
            
                test_loss += loss.item() 

                # Collect true and predicted labels for metrics
                y_labels.extend(labels.cpu().numpy())
                y_outputs.extend(outputs.cpu().numpy())
                y_predicted.extend(predicted.cpu().numpy())
                                
                correct += (predicted == labels).sum().item()

        test_loss /= len(test_dataloader)

        # Convert lists to numpy arrays
        y_labels = np.array(y_labels)
        y_outputs = np.array(y_outputs)
        y_predicted = np.array(y_predicted)
        print(f"True labels shape: {y_labels.shape}, True labels type: {type(y_labels)}")
        print(f"Predicted shape: {y_outputs.shape}, Predicted type: {type(y_outputs)}")

        if self.config.fused_model in ['Inter_2_concat_attn', 'Inter_1_concat_attn', 'Inter_1_gated_attn']:
            all_weights = np.array(all_weights)
            print(f"Attention weights shape: {all_weights.shape}", f"Attention weights type: {type(all_weights)}")
            mean_weights = all_weights.mean(axis=0).squeeze()
            print("Average modality importance:", mean_weights)
            return y_labels, y_outputs, y_predicted, test_loss, all_weights
        
        else:
            return y_labels, y_outputs, y_predicted, test_loss 


################################################ Code for Early_1 Fusion ################################################


    def _train_early_1(self, train_slice_df, val_slice_df, test_slice_df)-> None:
        '''This function handles the training process for the early_1 fusion method.

        Args:
            train_slice_df (_type:Pandas DataFrame_): containing the training data in slice level
            val_slice_df (_type:Pandas DataFrame_): containing the validation data in slice level

        Returns:
            None

        '''
        # training time dictionary for feature extraction phase
        training_time_dict = {}    
        # feature extraction phase
        print('=========================================================')
        print(f'Training models for feature extraction on fold {self.config.fold} is in progress...')

        # Train seperat models for each modality in the list of modalities (config.modalities) to use for feature extraction
        for modality in self.config.modalities: 
            start_time = time.time()
            print('------------------------------------------------------------------')
            print(f'Training the model for {modality} modality on fold {self.config.fold} started:')

            train_df = train_slice_df
            val_df = val_slice_df
            if modality == 'Clinical':
                train_df = slice_to_patient_dataset(train_slice_df) # Change dataset from slice level to patient level
                val_df = slice_to_patient_dataset(val_slice_df)

            self._train_model_early_1(train_df, val_df, modality, 1)

            print(f'Training the model for {modality} modality on fold {self.config.fold} completed.')
            train_time = (time.time() - start_time) / 60
            training_time_dict[modality] = train_time
            print(f'Training time for {modality} model on fold {self.config.fold} is: {training_time_dict[modality]:.2f} minutes')

        print('=========================================================')
        print(f'Training models for feature extraction on fold {self.config.fold} completed.')
        # calculate the total training time for all the models
        total_training_time = sum(training_time_dict.values())
        print(f'Total training time for feature extraction models on fold {self.config.fold}: {total_training_time:.2f} minutes')

        # Test models for feature extraction
        print('=========================================================')
        print(f'Evaluation of models for feature extraction on fold {self.config.fold} is in progress...')
        for modality in self.config.modalities:
            self._test_early_1(test_slice_df, modality, 1, training_time_dict[modality])
            print(f'Evaluation of the model for feature extraction of {modality} modality is finished and the results recorded in .text, .json, and on the CSV file')

        # Training phase for early fused features
        print('=========================================================')
        print('Training the model with Early_1 fusion method is in progress...')
        start_time_2 = time.time()
        # If there are multiple modalities, concatenate them for training
        if len(self.config.modalities) > 1:
            modality = '+'.join(self.config.modalities)
        else:
            modality = self.config.modalities[0]

        self._train_model_early_1(train_slice_df, val_slice_df, modality, 0)

        print('=========================================================')
        print('Training the model for Early_1 fusion method completed.')
        train_time_2 = (time.time() - start_time_2) / 60
        training_time_dict[modality] = train_time_2
        print(f'Training time for the Early_1 fused features model: {training_time_dict[modality]:.2f} minutes')
        return


    def _train_model_early_1(self, train_metadata_df, val_metadata_df, modality, uni)-> None:
        '''This function is the internal training function for the early_1 fusion method
        
        Args:
            train_metadata_df (_type:Pandas DataFrame_): containing the training data in patient level
            val_metadata_df (_type:Pandas DataFrame_): containing the validation data in patient level
            modality (_type:str_): the modality to train the model on
            uni (_type:int_): indicates whether the model is being trained for feature extraction (1) or for the main training phase (0)

        Returns:
            None
        '''

        # Set the random seed for reproducibility
        self.set_seed()

        if  uni == 1: # for feature extraction in Early_1 fusion
            if modality == 'Clinical':
                model_type = self.config.cl_model
                learning_rate = self.config.lr_cl
                batch_size = self.config.batch_size_cl
            else:
                model_type = self.config.mri_model
                learning_rate = self.config.lr_mri
                batch_size = self.config.batch_size_mri

            # create dataloaders with sampler
            train_dataset, train_dataloader = get_dataloader_late_sampler(metadata_df=train_metadata_df, config=self.config, train_flag=1, modality=modality, batch_size=batch_size)
            val_dataset, val_dataloader = get_dataloader_late(metadata_df=val_metadata_df, config=self.config, train_flag=0, modality=modality, batch_size=batch_size)

        else: # for the main training phase of Early_1 fusion
            model_type = self.config.fused_model
            learning_rate = self.config.lr_fused
            self.config.lmbda = 1e-4 # regularization parameter for Early_1 fusion hard-coded to 1e-4

            # create dataloaders with sampler
            train_dataset, train_dataloader = get_dataloader_intermediate_1_sampler(metadata_df=train_metadata_df, config=self.config, train_flag=1, batch_size=self.config.batch_size_fused)
            val_dataset, val_dataloader = get_dataloader_intermediate_1(metadata_df=val_metadata_df, config=self.config, train_flag=0, batch_size=self.config.batch_size_fused)

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(self.config.n_gpu)

        # select the backbone model to be trained
        model = model_selection_early(self.config, model_type) 

        # upload the weights of the unimodal feature extrator models    
        if model_type == 'Inter_1_concat':
            for modality_1 in self.config.modalities:

                if modality_1 == 'Clinical':
                    saved_model_path = f'best_model_{modality_1}_{self.config.fold}.pth'
                    clinical_dict = torch.load(saved_model_path , weights_only = True)
                    # Remove classifier weights (keys that start with 'classifier')
                    clinical_dict = {k: v for k, v in clinical_dict.items() if not k.startswith("classifier")}
                    
                    model.clinical_encoder.load_state_dict(clinical_dict, strict=False)
                    # freeze the weights of the Clinical feature extractor
                    for param in model.clinical_encoder.parameters():
                        param.requires_grad = False
                else:
                    saved_model_path = f'best_model_{modality_1}_{self.config.fold}.pth'
                    mri_dict = torch.load(saved_model_path , weights_only = True)
                    # Remove classifier weights (keys that start with 'classifier')
                    mri_dict = {k: v for k, v in mri_dict.items() if not k.startswith("classifier")}

                    model.mri_encoders[modality_1].load_state_dict(mri_dict, strict=False)
                    # freeze the weights of the MRI feature extractor
                    for param in model.mri_encoders[modality_1].parameters():
                            param.requires_grad = False

            # Define optimizer
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=self.config.lmbda)

        else:
            # Define optimizer
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=self.config.lmbda)
        
        model = model.to(device)

        # Define loss function
        match self.config.num_class:
            case 2: criterion = nn.BCEWithLogitsLoss()  # OR nn.BCELoss()   For binary classification
            case 3: criterion = nn.CrossEntropyLoss()  # For multi-class classification               

        nn_trainer = nn_Trainer_early(model, modality, criterion, optimizer,
                            config = self.config,
                            device = device,
                            train_dataloader = train_dataloader,
                            val_dataloader = val_dataloader,
                            uni=uni,
                            )

        nn_trainer.nn_train()
        return
            

    def _test_early_1(self, test_slice_df, modality, uni, training_time_spent)-> tuple[float, float, float, float, float]:
        '''This function is the testing function for the early_1 fusion method
        
        Args:
            test_slice_df (_type:Pandas DataFrame_): containing the test data in slice level
            modality (_type:str_): the modality to test the model on
            uni (_type:int_): indicates whether the model is being tested for feature extraction (1) or for the main training phase (0)
            training_time_spent (_type:dict_): the dictionary that contains the training time for each modality
        Returns:
            None
        '''
        # Set the random seed for reproducibility
        self.set_seed()

        if  uni == 1:
            if modality == 'Clinical':
                model_type = self.config.cl_model
            else:
                model_type = self.config.mri_model
            # create dataloaders
            test_dataset, test_dataloader = get_dataloader_late(metadata_df=test_slice_df, config=self.config, train_flag=0, modality=modality, batch_size=1) # batch_size=1 for testing test batch size of 1

        else:
            model_type = self.config.fused_model
            # create dataloaders
            test_dataset, test_dataloader = get_dataloader_intermediate_1(metadata_df=test_slice_df, config=self.config, train_flag=0, batch_size=1) # batch_size=1 for testing test batch size of 1

        # select the backbone model to be trained
        model = model_selection_early(self.config, model_type)

        # Load the saved state_dict
        try:
            model.load_state_dict(torch.load(f'best_model_{modality}_{self.config.fold}.pth', weights_only = True))
        
        except FileNotFoundError:
            print("Model checkpoint not found. Ensure the path to 'best_model.pth' is correct.")
            return
        
        y_labels, y_outputs, y_predicted, test_loss = self._test_loop(model, test_dataloader)

        # calculate the metrics for the model and save the results in three file of .text, .json, and .csv
        acc, mcc, f1_w, recall_w, precision_w = calculate_save_metrics_early_1(self.config, modality, y_labels, y_predicted, training_time_spent, uni, test_loss)

        if uni == 1:
            return None
        else:
            return acc, mcc, f1_w, recall_w, precision_w       
        
            
########################################## End of code for Early_1 Fusion ################################################

############################################## Code for Early_2 Fusion ###################################################            


    def _train_early_2(self, train_slice_df, val_slice_df)-> None:
        '''This function handles the training process for the early_2 fusion method.

        Args:
            train_slice_df (_type:Pandas DataFrame_): containing the training data in slice level
            val_slice_df (_type:Pandas DataFrame_): containing the validation data in slice level

        Returns:
            None

        '''

        # Set the random seed for reproducibility
        self.set_seed()
        
        # create dataloader with sampler
        train_dataset, train_dataloader = get_dataloader_early_2_sampler(metadata_df=train_slice_df, config=self.config, train_flag=1, batch_size=self.config.batch_size_fused)
        val_dataset, val_dataloader = get_dataloader_early_2(metadata_df=val_slice_df, config=self.config, train_flag=0, batch_size=self.config.batch_size_fused)

        # select the backbone model to be trained
        model = model_selection_early(self.config, self.config.fused_model)

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(self.config.n_gpu)
        model = model.to(device)

        # Define loss function
        match self.config.num_class:
            case 2: criterion = nn.BCEWithLogitsLoss()  # For binary classification
            case 3: criterion = nn.CrossEntropyLoss()  # For multi-class classification

        # Define optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.config.lr_fused, weight_decay=self.config.lmbda)

        if len(self.config.modalities) > 1:
            modality = '+'.join(self.config.modalities)
        else:
            modality = self.config.modalities[0]

        # nn_Trainer_intermediate is useful for training the early_2 fusion model
        nn_trainer = nn_Trainer_intermediate(model, modality, criterion, optimizer,
                            config = self.config,
                            device = device,
                            train_dataloader = train_dataloader,
                            val_dataloader = val_dataloader,
                            )

        # Train the model
        nn_trainer.nn_train()
        return


    def _test_early_2(self, test_slice_df, modality, training_time_spent)-> tuple[float, float, float, float, float]:
        '''This function is the testing function for the early_2 fusion method
        
        Args:
            test_slice_df (_type:Pandas DataFrame_): containing the test data in slice level
            training_time_spent (_type:dict_): the time spent on the training 

        Returns:
            acc (_type:float_): the accuracy of the model on the test dataset
            mcc (_type:float_): the Matthews correlation coefficient of the model on the test dataset
            f1_w (_type:float_): The weighted average of f1_score of the model on the test set
            recall_w (_type:float_): The weighted average of recall of the model on the test set
            precision_w (_type:float_): The weighted average of precision of the model on the test set

        '''
        # Set the random seed for reproducibility
        self.set_seed()

        # create dataloaders
        test_dataset, test_dataloader = get_dataloader_early_2(metadata_df=test_slice_df, config=self.config, train_flag=0, batch_size=1) # batch_size=1 for testing test batch size of 1
            
         # select the backbone model to be trained
        model = model_selection_early(self.config, self.config.fused_model)

        # Load the saved state_dict
        try:
            model.load_state_dict(torch.load(f'best_model_{self.config.fold}.pth' , weights_only = True))
        
        except FileNotFoundError:
            print("Model checkpoint not found. Ensure the path to 'best_model.pth' is correct.")
            return

        y_labels, y_outputs, y_predicted, test_loss = self._test_loop(model, test_dataloader)

        # calculate the metrics for the model and save the results in three file of .text, .json, and .csv and return
        return calculate_save_metrics_early_2(self.config, modality, y_labels, y_predicted, training_time_spent, test_loss)
    
     
################################################### End of code for Early_2 Fusion ################################################ 

################################################### Code for Intermediate_1 Fusion ################################################

    def _train_intermediate_1(self, train_slice_df, val_slice_df)-> None:
        '''This function handles the training process for the intermediate_1 fusion method.

        Args:
            train_slice_df (_type:Pandas DataFrame_): containing the training data in slice level
            val_slice_df (_type:Pandas DataFrame_): containing the validation data in slice level

        Returns:
            None

        '''                

        # create dataloader with sampler
        train_dataset, train_dataloader = get_dataloader_intermediate_1_sampler(metadata_df=train_slice_df, config=self.config, train_flag=1, batch_size=self.config.batch_size_fused)
        val_dataset, val_dataloader = get_dataloader_intermediate_1(metadata_df=val_slice_df, config=self.config, train_flag=0, batch_size=self.config.batch_size_fused)    


        # select the backbone model to be trained
        model = model_selection_intermediate(self.config)    

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(self.config.n_gpu)
        model = model.to(device)

        # Define loss function
        match self.config.num_class:
            case 2: criterion = nn.BCEWithLogitsLoss()  # OR nn.BCELoss()   For binary classification
            case 3: criterion = nn.CrossEntropyLoss()  # For multi-class classification
                    
    
        # Define optimizer with different learning rates
        if 'Clinical' in self.config.modalities:
            optimizer = optim.AdamW([
                {'params': model.mri_encoders.parameters(), 'lr': self.config.lr_mri, 'weight_decay': self.config.lmbda},
                {'params': model.clinical_encoder.parameters(), 'lr': self.config.lr_cl, 'weight_decay': self.config.lmbda},
                {'params': model.modality_attention.parameters(), 'lr': self.config.lr_fused, 'weight_decay': self.config.lmbda}, 
                # {'params': model.mri_attention_shared.parameters(), 'lr': self.config.lr_fused, 'weight_decay': self.config.lmbda},
                {'params': model.classifier.parameters(), 'lr': self.config.lr_fused, 'weight_decay': self.config.lmbda}
                ])
        else:
            optimizer = optim.AdamW([
                {'params': model.mri_encoders.parameters(), 'lr': self.config.lr_mri, 'weight_decay': self.config.lmbda},
                {'params': model.modality_attention.parameters(), 'lr': self.config.lr_fused, 'weight_decay': self.config.lmbda}, 
                {'params': model.classifier.parameters(), 'lr': self.config.lr_fused, 'weight_decay': self.config.lmbda}
            ])

        if len(self.config.modalities) > 1:
            modality = '+'.join(self.config.modalities)
        else:
            modality = self.config.modalities[0]

        # Define the trainer
        nn_trainer = nn_Trainer_intermediate(model, modality, criterion, optimizer,
                            config = self.config,
                            device = device,
                            train_dataloader = train_dataloader,
                            val_dataloader = val_dataloader,
                            )

        # Train the model
        nn_trainer.nn_train()
        return


    def _test_intermediate_1(self, test_slice_df, modality, training_time_spent)-> tuple[float, float, float, float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''This function is the testing function for the intermediate_1 fusion method
        
        Args:
            test_slice_df (_type:Pandas DataFrame_): containing the test data in slice level
            training_time_spent (_type:dict_): the time spent on the training 

        Returns:
            acc (_type:float_): the accuracy of the model on the test dataset
            mcc (_type:float_): the Matthews correlation coefficient of the model on the test dataset
            f1_w (_type:float_): The weighted average of f1_score of the model on the test set
            recall_w (_type:float_): The weighted average of recall of the model on the test set
            precision_w (_type:float_): The weighted average of precision of the model on the test set
            modality_cont_avg (_type:np.ndarray_): The average contribution of the modality to the final prediction
            modality_cont_label_0_avg (_type:np.ndarray_): The average contribution of the modality to the final prediction for label 0
            modality_cont_label_1_avg (_type:np.ndarray_): The average contribution of the modality to the final prediction for label 1
            modality_cont_label_0_correct_avg (_type:np.ndarray_): The average contribution of the modality to the final prediction for label 0 when the prediction is correct
            modality_cont_label_1_correct_avg (_type:np.ndarray_): The average contribution of the modality to the final prediction for label 1 when the prediction is correct
            modality_cont_correct_avg (_type:np.ndarray_): The average contribution of the modality to the final prediction when the prediction is correct

        '''
        # Set the random seed for reproducibility
        self.set_seed()

        # create dataloaders
        test_dataset, test_dataloader = get_dataloader_intermediate_1(metadata_df=test_slice_df, config=self.config, train_flag=0, batch_size=1) # batch_size=1 for testing test batch size of 1

        # select the backbone model
        model = model_selection_intermediate(self.config) 

        # Load the saved state_dict
        try:
            model.load_state_dict(torch.load(f'best_model_{self.config.fold}.pth' , weights_only = True))
            
        except FileNotFoundError:
            print("Model checkpoint not found. Ensure the path to 'best_model.pth' is correct.")
            return

        if self.config.fused_model in ['Inter_1_concat_attn', 'Inter_1_gated_attn']:
            y_labels, y_outputs, y_predicted, test_loss, all_weights = self._test_loop(model, test_dataloader)
        else:
            y_labels, y_outputs, y_predicted, test_loss = self._test_loop(model, test_dataloader)

        return calculate_save_metrics_intermediate_1(self.config, modality, y_labels, y_predicted, training_time_spent, test_loss, all_weights)


############################################## End of code for Intermediate_1 Fusion #####################################

############################################### Code for Intermediate_2 Fusion ###########################################
    

    def _train_intermediate_2(self, train_slice_df, val_slice_df)-> None:
        '''This function handles the training process for the intermediate_1 fusion method.

        Args:
            train_slice_df (_type:Pandas DataFrame_): containing the training data in slice level
            val_slice_df (_type:Pandas DataFrame_): containing the validation data in slice level

        Returns:
            None

        '''
        # Set the random seed for reproducibility
        self.set_seed()

        # # create dataloader with sampler
        train_dataset, train_dataloader = get_dataloader_intermediate_2_sampler(metadata_df=train_slice_df, config=self.config, train_flag=1, batch_size=self.config.batch_size_fused)
        val_dataset, val_dataloader = get_dataloader_intermediate_2(metadata_df=val_slice_df, config=self.config, train_flag=0, batch_size=self.config.batch_size_fused)

        # select the backbone model to be trained
        model = model_selection_intermediate(self.config)

        # prepare for (multi-device) GPU training
        device, device_ids = prepare_device(self.config.n_gpu)
        model = model.to(device)

        # Define loss function
        match self.config.num_class:
            case 2: criterion = nn.BCEWithLogitsLoss()  # OR nn.BCELoss()   For binary classification
            case 3: criterion = nn.CrossEntropyLoss()  # For multi-class classification

        
        # Define optimizer with different learning rates
        optimizer = optim.AdamW([
            {'params': model.mri_encoder.parameters(), 'lr': self.config.lr_mri, 'weight_decay': self.config.lmbda},
            {'params': model.clinical_encoder.parameters(), 'lr': self.config.lr_cl, 'weight_decay': self.config.lmbda},
            {'params': model.modality_attention.parameters(), 'lr': self.config.lr_fused, 'weight_decay': self.config.lmbda}, # for Inter_2_conat_attn
            {'params': model.classifier.parameters(), 'lr': self.config.lr_fused, 'weight_decay': self.config.lmbda}
            ])

        if len(self.config.modalities) > 1:
            modality = '+'.join(self.config.modalities)
        else:
            modality = self.config.modalities[0]

        # Define the trainer
        nn_trainer = nn_Trainer_intermediate(model, modality, criterion, optimizer,
                            config = self.config,
                            device = device,
                            train_dataloader = train_dataloader,
                            val_dataloader = val_dataloader,
                            )

        # Train the model
        nn_trainer.nn_train()


    def _test_intermediate_2(self, test_slice_df, modality, training_time_spent)-> tuple[float, float, float, float, float]:
        '''This function is the testing function for the intermediate_2 fusion method
        
        Args:
            test_slice_df (_type:Pandas DataFrame_): containing the test data in slice level
            training_time_spent (_type:dict_): the time spent on the training 

        Returns:
            acc (_type:float_): the accuracy of the model on the test dataset
            mcc (_type:float_): the Matthews correlation coefficient of the model on the test dataset
            f1_w (_type:float_): The weighted average of f1_score of the model on the test set
            recall_w (_type:float_): The weighted average of recall of the model on the test set
            precision_w (_type:float_): The weighted average of precision of the model on the test set

        '''
        # Set the random seed for reproducibility
        self.set_seed()

        # create dataloaders
        test_dataset, test_dataloader = get_dataloader_intermediate_2(metadata_df=test_slice_df, config=self.config, train_flag=0, batch_size=1)  

        # select the backbone model
        model = model_selection_intermediate(self.config)

        # Load the saved state_dict
        try:
            model.load_state_dict(torch.load(f'best_model_{self.config.fold}.pth' , weights_only = True))
        
        except FileNotFoundError:
            print("Model checkpoint not found. Ensure the path to 'best_model.pth' is correct.")
            return

        if self.config.fused_model in ['Inter_2_concat_attn']:
            y_labels, y_outputs, y_predicted, test_loss, all_weights = self._test_loop(model, test_dataloader)
        else:
            y_labels, y_outputs, y_predicted, test_loss = self._test_loop(model, test_dataloader)

        # calculate the metrics for the model and save the results in three file of .text, .json, and .csv and return
        return calculate_save_metrics_intermediate_2(self.config, modality, y_labels, y_predicted, training_time_spent, test_loss, all_weights)
  
    
    
############################################ End of code for Intermediate_2 Fusion ########################################

##################################################### Code for Late Fusion ################################################   


    def _train_late(self, train_slice_df, val_slice_df)-> None:
        '''This function handles the training process for the late fusion method.

        Args:
            train_slice_df (_type:Pandas DataFrame_): containing the training data in slice level
            val_slice_df (_type:Pandas DataFrame_): containing the validation data in slice level

        Returns:
            None

        '''
        # Set the random seed for reproducibility
        self.set_seed() 

        # training time dictionary
        training_time_dict = {}
        # Train separate models for each modality in the list of modalities (self.config.modalities) to fuse in late fusion
        for modality in self.config.modalities:

            start_time = time.time()
            print('------------------------------------------------------------------')
            print(f'Training the model for {modality} modality started:')
            
            train_df = train_slice_df
            val_df = val_slice_df
            # The classifier (AutoInt/MLP/XGBoost) for Clinical data modality at patient level
            if modality == 'Clinical':
                train_df = slice_to_patient_dataset(train_slice_df) # Change dataset from slice level to patient level
                val_df = slice_to_patient_dataset(val_slice_df)

            self._train_model_late(train_slice_df=train_df, val_slice_df=val_df, modality=modality)

            print(f'Training the model for {modality} modality completed.')
            # Save the training time for each modality
            training_time_dict[modality] = (time.time() - start_time) / 60
            print(f'Training time for {modality} modality in fold {self.config.fold} is: {training_time_dict[modality]:.2f} minutes')

        print('=========================================================')
        print('Train phase completed.')
        # Calculate the total training time
        total_training_time = sum(training_time_dict.values()) 
        print(f'Total training time for all modalities in fold {self.config.fold} is: {total_training_time:.2f} minutes')    


    def _train_model_late(self, train_slice_df, val_slice_df, modality)-> None:
        '''This function is the internal training function (modality specific) for the late fusion method
        
        Args:
            train_metadata_df (_type:Pandas DataFrame_): containing the training data in patient level
            val_metadata_df (_type:Pandas DataFrame_): containing the validation data in patient level
            modality (_type:str_): the modality to train the model on

        Returns:
            None

        '''
        # Set the random seed for reproducibility
        self.set_seed()

        if modality == 'Clinical':
            model_type = self.config.cl_model
            learning_rate = self.config.lr_cl
            batch_size = self.config.batch_size_cl
        else:
            model_type = self.config.mri_model
            learning_rate = self.config.lr_mri
            batch_size = self.config.batch_size_mri


        # create dataloaders with sampler
        train_dataset, train_dataloader = get_dataloader_late_sampler(metadata_df=train_slice_df, config=self.config, train_flag=1, modality=modality, batch_size=batch_size)
        val_dataset, val_dataloader = get_dataloader_late(metadata_df=val_slice_df, config=self.config, train_flag=0, modality=modality, batch_size=batch_size)

        # select the backbone model to be trained
        model = model_selection_late(model_type, self.config)

        if model_type == 'XGBoost': # the backbone model is a classical ML model
            # prepare X_train_np, y_train_np, X_val_np, y_val_np
            X_train_np, y_train_np = prepare_classic_classifier_dataset(train_dataset)
            X_val_np, y_val_np = prepare_classic_classifier_dataset(val_dataset)
            # cl_trainer
            cl_trainer = cl_Trainer(model, self.config, modality, X_train_np, y_train_np, X_val_np, y_val_np)
            cl_trainer.cl_train()
            
        else: # the backbone model is a Neural Network
            # prepare for (multi-device) GPU training
            device, device_ids = prepare_device(self.config.n_gpu)
            model = model.to(device)

            # Define loss function
            match self.config.num_class:
                case 2: criterion = nn.BCEWithLogitsLoss()  # OR nn.BCELoss()   For binary classification
                case 3: criterion = nn.CrossEntropyLoss()  # For multi-class classification
                    
            # Define optimizer
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=self.config.lmbda)


            nn_trainer = nn_Trainer_late(model, modality, criterion, optimizer,
                                config = self.config,
                                device = device,
                                train_dataloader = train_dataloader,
                                val_dataloader = val_dataloader,
                                )

            nn_trainer.nn_train()

   
    def _test_late(self, test_slice_df, training_time_spent)-> tuple[float, float, float, float, float]:
        '''This function is the testing function for the late fusion method
        
        Args:
            test_slice_df (_type:Pandas DataFrame_): containing the test data in slice level
            training_time_spent (_type:dict_): the time spent on the training 

        Returns:
            acc (_type:float_): the accuracy of the model on the test dataset
            mcc (_type:float_): the Matthews correlation coefficient of the model on the test dataset
            f1_w (_type:float_): The weighted average of f1_score of the model on the test set
            recall_w (_type:float_): The weighted average of recall of the model on the test set
            precision_w (_type:float_): The weighted average of precision of the model on the test set

        '''
        # Set the random seed for reproducibility
        self.set_seed()

        pred_dic = {}  # A dictionary to save the predictions for different modality models
        multi = 0 # a flag indicating the multimodality status of model (0: unimodal, 1:multimodal)

        # Find the predictions for each modality in the list of modalities (config.modalities) to fuse them in late fusion
        for modality in self.config.modalities:
        
            # calculate the predictions for each modality specific model
            pred_dic[f'label_{modality}'], pred_dic[f'output_{modality}'], pred_dic[f'predicted_{modality}'], test_loss = self._predict_with_modality_specific_model(test_slice_df, modality)            
            
            # NOTE: For the moment I put the time for training the unimodal models as -1 
            modalitiy_specific_training_time = -1
            # calculate the metrics and save the results in three file of .text, .json, and .csv
            calculate_save_metrics_late(self.config, modality, pred_dic[f'label_{modality}'], pred_dic[f'predicted_{modality}'], multi, modalitiy_specific_training_time, test_loss)


        if len(self.config.modalities) > 1:
                modality = '+'.join(self.config.modalities)
                multi = 1
                t_loss = -1
        else:
                modality = self.config.modalities[0]
                multi = 0
                t_loss = test_loss

        y_labels = pred_dic[f'label_{self.config.modalities[0]}']

        for fusion_m in ['majority_voting', 'probability_averaging']:

            self.config.fused_model = fusion_m
            # Late fusion of the predictions of models
            y_predicted_fused = late_fusion_function(self.config, pred_dic)
            
            # calculate the metrics for the late fused model and save the results in three file of .text, .json, and .csv
        return calculate_save_metrics_late(self.config, modality, y_labels, y_predicted_fused, multi, training_time_spent, t_loss) # y_labels is the y_labels for the first modality in config.modalities list 
   

    def _predict_with_modality_specific_model(self, test_slice_df, modality)-> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        ''' This function takes datafarme of test and predict the labels for the specific 'modality' 
        that fed as an argument.

        Args:
            test_df (_type:dataframe_): the dataframe contaning test dataset
            modality (_type:str_): the modality for which the model should be trained

        Returns:
            y_labels (_type:np.array_): the true labels for the test dataset
            y_outputs (_type:np.array_): the raw outputs from the model
            y_predicted (_type:np.array_): the predicted labels from the model
            test_loss (_type:float_): the loss value for the test dataset

        '''  
        # Set the random seed for reproducibility
        self.set_seed() 

        # create test dataloader
        test_dataset, test_dataloader = get_dataloader_late(metadata_df=test_slice_df, config=self.config, train_flag=0, modality=modality, batch_size=1) # batch_size=1 for testing test batch size of 1

        if modality == 'Clinical':
            model_type = self.config.cl_model
        else:
            model_type = self.config.mri_model    
        
        model = model_selection_late(model_type, self.config) 

        axis_dic = {0: "Sagittal", 1: "Coronal", 2: "Axial"}

        if model_type == 'XGBoost': # the backbone model is a classical ML model
            # prepare X_test_np and y_labels both np.array
            X_test_np, y_labels = prepare_classic_classifier_dataset(test_dataset)

            # Load the saved model
            try:
                model.load_model(f'saved_model_{modality}_{self.config.fold}.model')
            except FileNotFoundError:
                print("Model checkpoint not found. Ensure the path to 'saved_model.pth' is correct.")
                return

            # Calculate the predictions for the test data
            y_predicted = model.predict(X_test_np)

            # Get predicted probabilities (only for the positive class)
            y_predicted_probs = model.predict_proba(X_test_np)[:, 1]  # Extract probability of class 1

            # Compute training loss
            test_loss = log_loss(y_labels, y_predicted_probs)

            return y_labels, y_predicted_probs, y_predicted, test_loss
        
        else:
            # Load the saved state_dict
            try:
                model.load_state_dict(torch.load(f'best_model_{modality}_{self.config.fold}.pth' , weights_only = True))
            except FileNotFoundError:
                print("Model checkpoint not found. Ensure the path to 'best_model.pth' is correct.")
                return
            
            return self._test_loop(model, test_dataloader)