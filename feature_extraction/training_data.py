from os.path import join
import pandas as pd
import numpy as np
import cv2
from main_flow.flow import process_single_image
from .build_features_file import read_images
import progressbar
#from main_flow.split_features import create_features_dataframe, drop_unwanted_features


def __get_features(path, full_images_df, debug=False):
    '''
    calculate the features of all the images of the dataset
    Parameters
    ----------
    full_images_df  Dataframe containing the path and class of all the images

    Returns
    -------
    total_features  feature vector of all the ROIs in the image

    '''

    total_images = len(full_images_df)
    total_labels = []

    bar = progressbar.ProgressBar(maxval=total_images,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for img_counter in range(0, total_images):
        [_, features, _] = process_single_image(path, full_images_df['File'][img_counter], debug)
        #[labels, dices] = label_findings(gt_im_path, images_name[img_counter], features, gt_images)

        if full_images_df['Class'][img_counter] == 'les':
            features.insert(0, "label", 'les', True)
        else:
            features.insert(0, "label", 'nv', True)

        img_name = full_images_df['File'][img_counter].split('/')[-1]
        features.insert(0, "Name", img_name, True)


        if img_counter == 0:
            total_features = features
        else:
            total_features = pd.concat([total_features,  features])

        bar.update(img_counter + 1)

    bar.finish()

    return [total_features]


def flow_from_directory(dataset_path):
    '''
    NOT used:  Partition of the data for testing and training
    Parameters
    ----------
    dataset_path:   String path to all the dataset

    Returns
    -------
    dataframes:     partitioned dataframes of test and train

    '''

    # Read the training and validation datasets
    train_images_df = read_images(dataset_path)
    #val_images_df = read_images(dataset_path + '/val')

    # Merge the these datasets to perform K-Fold cross validation in the classification steps
    #full_dataset_df = train_images_df.append(val_images_df)
    #full_dataset_df.index = range(len(full_dataset_df))
    return [train_images_df]


def prepate_datasets(dataset_path,output_name, debug=False):
    '''
    Prepare the feature extraction of all the dataset

    Parameters
    ----------
    dataset_path:   path to the full dataset

    Returns
    -------
    CSV files:      dataframe containing the features and dataframe containing metadata of the ROIs

    '''

    [full_images_df] = flow_from_directory(dataset_path)
    print("Preparing training set!\n")
    [training_features] = __get_features(dataset_path,full_images_df, debug)
    training_features.to_csv(join(dataset_path, output_name))
