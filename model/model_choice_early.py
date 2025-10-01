
from pytorch_tabular.models.autoint import AutoIntConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig

from model.models_classes import init_weights, AutoIntModel, MLP_1024_512_256_128, CustomDenseNet121, Inter_1_concat



def model_selection_early(config, model_type): 
    ''' This function takes the model_type and the number of classes and defines the model

    Args:
        config (_type:config_): it contains the configeration of the problem, the ones used in this function:
            num_classes (_type:int_): the number of classes in the dataset for classification
            pretrained (_type:int_): 0 meants without weights (either non-pretrained or for test), 1 means with weights
        model_type (_type:str_): the name of the class of the model to be used

    Returns:
        model (_type:class_): the class of model to be trained 

    '''

    if model_type == 'AutoInt':
        head_config = LinearHeadConfig(
            layers="",
            dropout=0.1,  # default
            initialization=("kaiming"),                     
        )
        # Initialize AutoInt model with tuned parameters
        model_config = AutoIntConfig(task="classification", head="LinearHead", attn_dropouts=0.3, attn_embed_dim=4, batch_norm_continuous_input=True, embedding_dim=16, num_attn_blocks=1, num_heads=4) 
        model_config.continuous_dim = 1
        model_config.categorical_dim = 1
        model_config.categorical_cardinality = [2]
        model_config.output_cardinality = [2]
        model_config.output_dim = 1  # For binary classification
        model_config.head_config = head_config.__dict__
        model = AutoIntModel(config=model_config)        

    elif model_type == 'MLP_1024_512_256_128':
        # Initialize MLP model
        if 'Clinical' in config.modalities:
            if len(config.modalities) > 1:
                input_size = (240*240)*(len(config.modalities)-1)+2 # MRI slides are 240*240
            else:
                input_size = 2
        else:
            input_size = (240*240)*len(config.modalities)

        hidden_size = 1024  # Increased from 32 hidden size for better representation
        model = MLP_1024_512_256_128(input_size, hidden_size, config.num_class-1)
        # Apply Kaiming initialization
        init_weights(config, model, init_type="kaiming")

    elif model_type == 'denseNet121':
        model = CustomDenseNet121(config, in_channels=1, with_head=1)

    elif model_type == 'Inter_1_concat':
        model = Inter_1_concat(config=config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


    return model
