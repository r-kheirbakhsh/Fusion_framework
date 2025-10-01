
class Config:

    def __init__(self, args):
        self.project_name = args.project_name
        self.fusion_strategy = args.fusion_strategy
        self.dataset_csv_path = args.dataset_csv_path
        self.train_csv_path = args.train_csv_path
        self.val_csv_path = args.val_csv_path
        self.test_csv_path = args.test_csv_path
        self.dataset_image_path = args.dataset_image_path
        self.label = args.label
        self.num_class = args.num_class 
        self.axis = args.axis
        self.modalities = args.modalities
        self.scale_clinical_modality = args.scale_clinical_modality
        self.mri_model = args.mri_model
        self.cl_model = args.cl_model
        self.fused_model = args.fused_model
        self.pretrained = args.pretrained
        self.lr_mri = args.lr_mri
        self.lr_cl = args.lr_cl
        self.lr_fused = args.lr_fused
        self.lmbda = args.lmbda
        self.batch_size_mri = args.batch_size_mri
        self.batch_size_cl = args.batch_size_cl
        self.batch_size_fused = args.batch_size_fused
        self.n_epochs_mri = args.n_epochs_mri
        self.n_epochs_cl = args.n_epochs_cl
        self.n_epochs_fused = args.n_epochs_fused
        self.n_gpu = args.n_gpu
        self.seed = args.seed
        self.num_folds = args.num_folds
        self.fold = args.fold

        if self.dataset_csv_path is None:
            raise ValueError("missing dataset file")
            
        else:
            if self.dataset_image_path is None:
                raise ValueError("missing image files")



    def __str__(self):
        str = self.modalities
        #print(f'project name is: {self.project_name}')
        #print(f'image path is: {self.project_name}')
        #print(f'modalities is: {self.project_name}')
        return str
    
    def pprint(self):
        print(self.modalities)
        
            
