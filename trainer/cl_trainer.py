
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
      


    def cl_train(self)->None:
        '''this function contains full training logic for classifiers like: XGBoost, SVM, LDA, ..., after training 
        it saves the trained model

        Args:
            None

        Returns:
            None
        
        '''
        # prepare X_train
        self.model.fit(self.X_train_np, self.y_train_np)

        # # Hyperparameter grid
        # param_grid = {
        #     "learning_rate": [0.01, 0.1, 0.2],
        #     "n_estimators": [100, 300, 500],
        #     "max_depth": [3, 5, 7],
        #     "min_child_weight": [1, 3, 5],
        #     "subsample": [0.6, 0.8, 1.0],
        #     "colsample_bytree": [0.6, 0.8, 1.0],
        #     "reg_lambda": [1, 5, 10]
        # }

        # # Grid Search with Cross Validation
        # grid_search = GridSearchCV(self.model, param_grid, scoring="neg_log_loss", cv=3, verbose=1, n_jobs=-1)
        # grid_search.fit(self.X_train_np, self.y_train_np)

        # # Best Parameters
        # print("Best Parameters:", grid_search.best_params_)

        # # Train with best parameters
        # best_model = grid_search.best_estimator_

        axis_dic = {0: "Sagittal", 1: "Coronal", 2: "Axial"} 
        # Save the trained model
        # best_model.save_model(f'saved_model_{self.modality}_{self.model_type}_{self.config.dataset}.model') # for grid search
        self.model.save_model(f'saved_model_{self.modality}_{self.config.fold}.model')  
        
        return