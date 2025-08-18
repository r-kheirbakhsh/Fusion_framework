
import numpy as np
from scipy.stats import mode
from scipy.special import expit  # Numerically stable sigmoid


def majority_voting(pred_list)->np.array:
    ''' 
    This function takes predictions of different models each trained on one modality and retunes the 
    majority voting for the predictions

    Args:
        pred_list (_type:list_): a list of np.array each of then is the prediction of one model/modality

    Returns:
        final_prediction (_type:np.array): the fused prediction array produced bt majority voting 

    ''' 
    # Stack pred_list column-wise
    pred_stacked = np.stack(pred_list)
    # Compute mode for each instance
    final_prediction = mode(pred_stacked, axis=0).mode  # the result of mode() has two parts: the first is an array of the values of modes and the second is an array of the frequences of those modes; mode().mode returns the ode values
    
    return final_prediction


def probability_averaging(output_list)->np.array:
    ''' 
    This function performs late fusion for binary classification by averaging the logits (outputs before sigmoid)
    from multiple modality-specific models, applying sigmoid to get probabilities, and thresholding to get predictions.

    Args:
        output_list (_type:list_): A list of np.array, each of shape (n_samples,) or (n_samples, 1),
                            containing the logits of one model/modality.

    Returns:
        final_prediction (_type:np.array_): The fused binary prediction array of shape (n_samples,).
    ''' 

    # Ensure all outputs are 1D arrays
    output_list = [output.squeeze() for output in output_list]  # (n_samples,)
    
    # Stack outputs from all modalities: shape -> (n_modalities, n_samples)
    output_stacked = np.stack(output_list, axis=0)

    # Average across modalities: shape -> (n_samples,)
    output_avg = np.mean(output_stacked, axis=0)

    # Apply sigmoid to get probabilities
    probs = expit(output_avg)

    # Apply threshold to get binary predictions
    final_prediction = (probs >= 0.5).astype(int)

    return final_prediction



def late_fusion_function(config, pred_dic)->np.array:
    ''' This function takes a dictionary containing the predictions of different models and returns the late 
    fused prediction.

    Args:
        config (_type:Config_): the configeration of the problem, the ones used in this class:
            config.modalities (_type:str_): modalities of data to be used in the model
            config.fusion_method (_type:str_): the name of late fusion method
        pred_dic (_type:dictionary_): a dictionary of np arrays with these keys 'label_T1', 'predicted_T1_bias', 'output_T1_bias', 'label_T2', 'predicted_T2_bias', 'output_T2_bias',
        'label_T1c', 'predicted_T1c_bias', 'output_T1c_bias', 'label_FLAIR', 'predicted_FLAIR_bias', 'output_FLAIR_bias', 'label_Clinical', 'predicted_Clinical_bias', 'output_Clinical_bias'

    Returns:
        y_predicted_fused (_type:np.array): the fused prediction array

    '''

    pred_list = []
    output_list = []
    for modality in config.modalities:
        pred_list.append(pred_dic[f'predicted_{modality}'])  # it makes a list of np.arrays each contains predictions of a modality specific trained model
        output_list.append(pred_dic[f'output_{modality}'])  # it makes a list of np.arrays each contains the output of a modality specific trained model

    match config.fused_model:
        case 'majority_voting': 
            y_predicted_fused = majority_voting(pred_list)
        case 'probability_averaging':
            y_predicted_fused = probability_averaging(output_list)

    return y_predicted_fused