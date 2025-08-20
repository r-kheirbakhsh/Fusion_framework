import torch
import wandb
from sklearn.metrics import matthews_corrcoef
import numpy as np

from utils import move_to_device


class nn_Trainer_intermediate:
    '''The trainer class for intermediat fusion pipeline
    
    '''
    def __init__(self, model, modality, criterion, optimizer, config, device, train_dataloader, val_dataloader):
        ''' Sets the class variables

        Args:
            model (_type:class_): the model to be trained
            modality (_type:str_): the modality for which the model should be trained, it can be 'Clinical' or 'T1_bias' or 'T1c_bias' or 'T2_bias' or 'FLAIR_bias'
            criterion (_type:class_): the loss function to be used for training
            optimizer (_type:class_): the optimizer to be used for training
            config (_type:class_): it contains the parameters according which the model will be trained
            device (_type:_): the device to be used for training
            train_dataloader (_type:_): the dataloader for the training set
            val_dataloader (_type:_): the dataloader for the validation set

        '''
        self.model = model
        self.modality = modality
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.early_stopping_patience = 10


    def set_seed(self):
        '''This function sets random seeds for reproducibility
        
        '''
        # Set seeds for PyTorch to ensure consistency across runs
        torch.manual_seed(self.config.seed)

        # Using a GPU, make operations deterministic by setting:
        torch.cuda.manual_seed(self.config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def nn_train(self):
        '''this function contains full training logic for training and runs all the epochs
        
        '''
        # Set the random seed for reproducibility
        self.set_seed()

        n_epochs = self.config.n_epochs_fused

        axis_dic = {0: "Sagittal", 1: "Coronal", 2: "Axial"}  
        # Initialize lists to store training history on wandb
        wandb.init (
            # Set the wandb project where this run will be logged
            project = self.config.project_name,
        
            # Track hyperparameters and run metadata
            config= {
                'dataset': f'{axis_dic[self.config.axis]}_43_56_396_{self.config.seed}_{self.config.fold}',
                'fusion method': self.config.fusion_method,
                'modality': self.modality, 
                'number of classes': self.config.num_class,
                'architecture': self.config.fused_model,
                'pretrained': self.config.pretrained,
                'number of epochs': self.config.n_epochs_fused,
                'learning rate': self.config.lr_fused,
                'lambda': self.config.lmbda,
                'batch size': self.config.batch_size_fused,
                'seed': self.config.seed,
                'number of gpus': self.config.n_gpu
            }
        )
  
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        train_mccs = []
        val_mccs = []


        best_val_loss = float('inf')
        patience_counter = 0  # initialize early stopping counter
        best_model_path = f'best_model_{self.config.fold}.pth' # BE CARFUL WHERE YOU SAVE THE BEST MODEL, YOU SHOULD UPLOAD WEIGHTS IN TEST FROM HERE!

        # Training loop
        for epoch in range(n_epochs):

            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            labels_train_list = []
            predicted_train_list = []


            #for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            for inputs, labels in self.train_dataloader:
                #inputs, labels = inputs.to(self.device), labels.to(self.device)
                inputs = move_to_device(inputs, self.device)
                labels = move_to_device(labels, self.device)
        
                self.optimizer.zero_grad()
                outputs= self.model(inputs)

                if self.config.num_class == 2:
                    labels = labels.float()
                    outputs = torch.squeeze(outputs)  # Remove the extra dimension [batch_size, 1] -> [batch_size]
                    predicted = torch.round(torch.sigmoid(outputs))  # for Binary classification
                else:
                    _, predicted = torch.max(outputs, 1)  # for Multi-class classification
            
                loss = self.criterion(outputs, labels)

                loss.backward()
                
                self.optimizer.step()

                train_loss += loss.item()

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                labels_train_list.append(labels.cpu().detach().numpy())
                predicted_train_list.append(predicted.cpu().detach().numpy())   
           
    
            train_loss /= len(self.train_dataloader)  # it is the average loss for each batch
            train_accuracy = correct / total # total (number of instances in the dataloader) is correct, len(self.train_dataloader) is the number of batches in the dataloader
            train_mcc = matthews_corrcoef(np.concatenate(labels_train_list), np.concatenate(predicted_train_list))  # added for mcc

            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            train_mccs.append(train_mcc)
    
            # Validation
            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            labels_val_list = []
            predicted_val_list = []
    
            with torch.no_grad():
                for inputs, labels in self.val_dataloader:

                    inputs = move_to_device(inputs, self.device)
                    labels = move_to_device(labels, self.device)

                    outputs = self.model(inputs)

                    if self.config.num_class == 2:
                        labels = labels.float()
                        outputs = torch.squeeze(outputs) # Remove the extra dimension [batch_size, 1] -> [batch_size]
                        predicted = torch.round(torch.sigmoid(outputs))  # Binary classification
                    else:
                        _, predicted = torch.max(outputs, 1)  # Multi-class classification

                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    labels_val_list.append(labels.cpu().detach().numpy())               
                    predicted_val_list.append(predicted.cpu().detach().numpy()) 

            
            val_loss /= len(self.val_dataloader)
            val_accuracy = correct / total  
            val_mcc = matthews_corrcoef(np.concatenate(labels_val_list), np.concatenate(predicted_val_list))  # added for mcc

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            val_mccs.append(val_mcc)

            
            print(f'Epoch [{epoch+1}/{n_epochs}], '
                f'Modality: {self.modality}, '
                f'Train_Loss: {train_loss:.4f}, Train_Accuracy: {train_accuracy:.2%}, Train_MCC: {train_mcc:.4f}, ' 
                f'Val_Loss: {val_loss:.4f}, Val_Accuracy: {val_accuracy:.2%}, Val_MCC: {val_mcc:.4f}') 
            

            # Log metrics to wandb
            wandb.log({'Train Loss': train_loss, 'Val Loss': val_loss, 'Train Accuracy': train_accuracy, 'Val Accuracy': val_accuracy, 'Train MCC': train_mcc, 'Val MCC': val_mcc})

              
            # Early stopping logic
            if val_loss < best_val_loss - 1e-4: # a delta threshold to avoid stopping on tiny fluctuations  
                best_val_loss = val_loss
                patience_counter = 0  # reset counter if improvement
                torch.save(self.model.state_dict(), best_model_path) 
                print(f"Best model saved with validation loss: {val_loss:.5f}")

            else:
                patience_counter += 1
                print(f"No improvement in validation loss for {patience_counter} epoch(s)")
                
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        return



    



