'''
    This file containes the function that chooses the model for late fusion pipeline

'''

from pytorch_tabular.models.autoint import AutoIntBackbone, AutoIntConfig
from pytorch_tabular.models.common.heads import LinearHeadConfig

from model.models_classes import AutoIntModel, CustomDenseNet121, CustomSwin_b



def model_selection_late(model_type, config): 
    ''' This function takes the model_type and the number of classes and defines the model

    Args:
        model_type (_type:str_): the name of the class of the model we want to use
        num_classes (_type:int_): the number of classes in the dataset for classification
        pretrained (_type:int_): 0 meants without weights (either non-pretrained or for test), 1 means with weights

    Returns:
        model (_type:class_): the class of model to be trained 

    '''
    
    if model_type == 'AutoInt':
        head_config = LinearHeadConfig(
            layers="",
            dropout=0.1,  
            initialization=(  # No additional layer in head, just a mapping layer to output_dim
                "kaiming"
            ),
        )
        # Initialize the AutoInt with tuned parameters
        model_config = AutoIntConfig(task="classification", head="LinearHead", attn_dropouts=0.3, attn_embed_dim=4, batch_norm_continuous_input=True, embedding_dim=16, num_attn_blocks=1, num_heads=4)
        model_config.continuous_dim = 1
        model_config.categorical_dim = 1
        model_config.categorical_cardinality = [2]
        model_config.output_cardinality = [2]
        model_config.output_dim = 1  # For binary classification
        model_config.head_config = head_config.__dict__
        model = AutoIntModel(config=model_config)

    elif model_type == 'denseNet121':
        model = CustomDenseNet121(config, in_channels=1, with_head=1)
 
    elif model_type == 'swin_b':
        model = CustomSwin_b(config, in_channels=1, with_head=1)

    else:
        raise ValueError(f"Unknown model type: {model_type}")



    return model
