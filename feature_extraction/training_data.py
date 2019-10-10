from os.path import join
import pandas as pd
import numpy as np
import cv2
from main_flow.flow import process_single_image
from .build_features_file import read_images
from evaluation.dice_similarity import dice_similarity
import progressbar
from main_flow.split_features import create_features_dataframe, drop_unwanted_features


def __get_optimal_dice(roi_mask, gts):
    '''
    Get the dice score of a ROI

    Parameters
    ----------
    roi_mask       numpy array binary input mask of the ROI
    gts            numpy array binary mask regions of the ground truth

    Returns
    -------
                    dice score of the region

    '''

    values = np.zeros([len(gts), 1])
    for idx in range(0, len(gts)):
        dice, _ = dice_similarity(roi_mask, gts[idx])
        values[idx] = dice

    return 0 if len(values) == 0 else values.max()


def label_findings(gt_path, filename, features, groundtruths_filenames):
    '''
    Find the label of the ROI TP(1) if dice score > 0.2, otherwise FP(0)
    Parameters
    ----------
    gt_path         path of the ground truth images
    filename        name of the image to analyse
    features        features of the image
    groundtruths_filenames  name of the ground truth image

    Returns
    -------
    labels          integer 0 or 1
    dices           reesulting DICE index of the ROI

    '''
    labels = np.zeros([len(features)])
    dices = np.zeros([len(features)])

    # Labelled as Zero for non-masses, and 1 for masses

    # if the image doesn't have any groundtruth, we label the found masses as false positives.

    if filename not in groundtruths_filenames:
        return [labels, dices]

    gt_index = groundtruths_filenames.index(filename)
    # Dilate to remove holes in the GT
    gt = cv2.imread(join(gt_path, groundtruths_filenames[gt_index]), 0)
    _, contours, _ = cv2.findContours(gt, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    gt_layers = []

    # Separate each gt mass in different layers
    for contour in contours:
        gt_mask = np.zeros_like(gt)
        cv2.drawContours(gt_mask, [contour], 0, 1, -1)
        gt_layers.append(gt_mask)

    index = 0
    for roi in features:
        roi_mask = np.zeros_like(gt)
        contour = roi.get("Contour_np")
        cv2.drawContours(roi_mask, [contour], 0, 1, -1)
        dice = __get_optimal_dice(roi_mask, gt_layers)
        dices[index] = dice

        # Positive if the Dice index is greater than 0.2
        labels[index] = 1 if dice > 0.2 else 0
        index += 1

    return [labels, dices]


def __get_features(full_images_df, debug=False):
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
    total_features = []

    bar = progressbar.ProgressBar(maxval=total_images,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    for img_counter in range(0, 5):#range(0, total_images):
        [_, features, _] = process_single_image(full_images_df['File'][img_counter], debug)
        #[labels, dices] = label_findings(gt_im_path, images_name[img_counter], features, gt_images)


        total_features.extend(features)

        label = 'n'
        if full_images_df['Class'][img_counter] == 'les':
            label = 'l'

        total_labels.extend(label)
        bar.update(img_counter + 1)

    bar.finish()

    return [total_labels, total_features]


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
    train_images_df = read_images(dataset_path + '/train')
    val_images_df = read_images(dataset_path + '/val')

    # Merge the these datasets to perform K-Fold cross validation in the classification steps
    full_dataset_df = train_images_df.append(val_images_df)
    full_dataset_df.index = range(len(full_dataset_df))
    return [full_dataset_df]


def prepate_datasets(dataset_path):
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
    [training_labels, training_features] = __get_features(full_images_df, debug=True)

    [df_features, tags] = create_features_dataframe(training_features)
    training_features = 0
    tag_df = pd.DataFrame(np.array([training_labels]).transpose(), columns=["Class"])
    df_features = drop_unwanted_features(df_features)
    # df_features = normalize_dataframe(df_features)

    df_features.to_csv(join(dataset_path, "training.csv"))
    tag_df.to_csv(join(dataset_path, "training_metadata.csv"))
