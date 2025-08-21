'''
    This file containes the function that chooses the arcitecture of the CNN model for interediate fusion pipeline

'''

from model.models_classes import Inter_2_concat, Inter_2_concat_attn, Inter_2_bidirectional_crossattention, Inter_2_bi_crossattn_selfattn
                                    


def model_selection_intermediate(config): 
    ''' This function takes the model_type and the number of classes and defines the model

    Args:
        model_type (_type:str_): the name of the class of the model we want to use
        num_classes (_type:int_): the number of classes in the dataset for classification
        pretrained (_type:int_): 0 meants without weights (either non-pretrained or for test), 1 means with weights

    Returns:
        model (_type:class_): the class of model to be trained 

    '''
    
    if config.fused_model == 'Inter_2_concat':
        model = Inter_2_concat(config=config)

    elif config.fused_model == 'Inter_2_concat_attn':
        model = Inter_2_concat_attn(config=config)

    elif config.fused_model == 'Inter_2_bidirectional_crossattention':
        model = Inter_2_bidirectional_crossattention(config=config)

    elif config.fused_model == 'Inter_2_bi_crossattn_selfattn':
        model = Inter_2_bi_crossattn_selfattn(config=config)

    else:
        raise ValueError(f"Unknown model type: {config.fused_model}")

    return model
