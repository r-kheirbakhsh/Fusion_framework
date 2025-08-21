
import time
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from trainer import Model
from config import Config, parse_args
from utils import slice_to_patient_dataset, prepare_dataset_split



def main(config):
    # Load dataset
    dtype_dict = {'ID': str, 'slice_name': str, 'sex': int, 'age': int, 'WHO_grade': int}
    dataset_slice_df = pd.read_csv(config.dataset_csv_path, dtype=dtype_dict)

    # Create the model
    model = Model(config)

    # Convert the slice-level dataset to patient-level dataset
    # This is done to ensure that each patient is represented only once in the dataset spliting
    dataset_patient_df = slice_to_patient_dataset(dataset_slice_df)

    # Defining the Kfold
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)        

    global_time, global_acc, global_mcc, global_f1_weighted_avg, global_recall_weighted_avg, global_precision_weighted_avg = [], [], [], [], [], []
    # Loop through each fold
    for fold, (train_index, test_index) in enumerate(skf.split(dataset_patient_df, dataset_patient_df['WHO_grade'])):

        # Update the fold number in the config
        config.fold = fold  
        print(f'Fold {fold}/{config.num_folds-1} in progress...')

        # prepare the train, val, and test datasets splits and save the statistics of the splited sub-datasets
        train_slice_df, val_slice_df, test_slice_df = prepare_dataset_split(config, dataset_patient_df, train_index, test_index) 

        # training time dictionary
        start_time = time.time()

        print(f'Training the model on fold {fold}...')
        # Train the model
        model.train(train_slice_df, val_slice_df, test_slice_df)

        # Save the training time for each modality
        training_time_spent = (time.time() - start_time) / 60
        # print(f'Training time for the model on fold {fold} is: {finish_time:.2f} minutes')

        # Testing phase
        print('=========================================================')
        print(f'Evaluation of {config.fusion_method} model on fold {fold} is in progress...')
        # Evaluate the model
        acc, mcc, f1_w, recall_w, precision_w = model.evaluate(test_slice_df, training_time_spent)

        # Store results
        global_time.append(training_time_spent)
        global_acc.append(acc)
        global_mcc.append(mcc)
        global_f1_weighted_avg.append(f1_w)
        global_recall_weighted_avg.append(recall_w)
        global_precision_weighted_avg.append(precision_w)

    # Print the average results across all folds
    print(f"Average Time: {sum(global_time) / len(global_time)}")
    print(f"Average Accuracy: {sum(global_acc) / len(global_acc)}")
    print(f"Average MCC: {sum(global_mcc) / len(global_mcc)}")
    print(f"Average f1_weighted_avg: {sum(global_f1_weighted_avg) / len(global_f1_weighted_avg)}")
    print(f"Average recall_weighted_avg: {sum(global_recall_weighted_avg) / len(global_recall_weighted_avg)}")
    print(f"Average precision_weighted_avg: {sum(global_precision_weighted_avg) / len(global_precision_weighted_avg)}")      

    return


if __name__ == "__main__":
    args = parse_args()
    config = Config(args)

    main(config)




   