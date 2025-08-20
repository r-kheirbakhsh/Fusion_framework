
from sklearn.model_selection import GridSearchCV


class cl_Trainer:
    '''cl_trainer class
    
    '''
    def __init__(self, model, config, modality, X_train_np, y_train_np, X_val_np, y_val_np):
        ''' Sets the class variables

        Args:
            model (_type:class_):
            config (_type:class_): it contains the parameters according which the model will be trained
            modality (_type:str_): modality of data used for training the model
            X_train_np (_type:np.array_):
            y_train_np (_type:np.array_):
            X_val_np (_type:np.array_):
            X_val_np (_type:np.array_):

        '''
        self.model = model
        self.config = config
        self.modality = modality
        self.X_train_np = X_train_np
        self.y_train_np = y_train_np
        self.X_val_np = X_val_np
        self.y_val_np = y_val_np
      


    def cl_train(self)-> None:
        '''this function contains full training logic for classifiers like: XGBoost, SVM, LDA, ..., after training 
        it saves the trained model

        Args:
            None

        Returns:
            None
        
        '''

        # prepare X_train
        self.model.fit(self.X_train_np, self.y_train_np)

        # Save the trained model
        self.model.save_model(f'saved_model_{self.modality}_{self.config.fold}.model')  
        
        return