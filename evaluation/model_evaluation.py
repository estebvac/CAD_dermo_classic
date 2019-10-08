import pandas as pd
import cv2
import ast
import matplotlib.pyplot as plt
import numpy as np
import os
from evaluation.dice_similarity import dice_similarity
import progressbar
from prettytable import PrettyTable
from classifiers.cascade_random_forests import trainCascadeRandomForestClassifier, trainCascadeRandomForestClassifierFaster, applyCascade, get_probs, subdivide_dataset_k
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy import interpolate


def recover_filename(dataframe):
    '''
    Modifies the File name field of the datagrame to match with the path

    Parameters
    ----------
    dataframe   pandas dataframe
                Dataframe contains the metadata of the ROIs

    Returns     pandas dataframe
    -------

    '''
    new = dataframe['File name'].str.split("images", n=2, expand=True)[1]
    new = new.str.replace("\\", "")
    dataframe['File name'] = new
    return dataframe


def draw_predicted_image(path, dataframe, image_name, field = 'Prediction'):
    '''
    Draws an overlap of the predicted mask over the original image and returns
    both a mask of the prediction and an overlaped version of the image.

    Parameters
    ----------
    path:           String containing the path to the database
    dataframe       pandas dataframe contains the metadata of the ROIs
    image_name      String containing the image to process.

    Returns
    -------
    img             numpy array containing the original image
    mask            numpy array containing the resulting segmentation mask

    '''
    new = dataframe[dataframe['File name'] == image_name]
    new = new[new[field] == 1]
    img = cv2.imread(path + '/images/' + image_name)
    mask = np.zeros((img.shape[0], img.shape[1]))
    for number in range(len(new)):
        contours_mass = new["Contour"].iloc[number]
        contours = np.array(ast.literal_eval(contours_mass))
        cv2.drawContours(mask, [contours], -1, 255, -1)
        cv2.drawContours(img, contours, -1, (0, 255, 0), thickness = 15)
    img = cv2.normalize(img,  img, 0, 255, cv2.NORM_MINMAX)
    mask = cv2.normalize(mask,  mask, 0, 255, cv2.NORM_MINMAX)
    return img, mask


def plot_image_and_mask(img, mask):
    '''
    Plots the original image and the segmentation result

    Parameters
    ----------
    img:        numpy array of the image to plot
    mask        numpy array of the mask to plot

    Returns
    -------

    '''
    fig = plt.figure(figsize=(8, 8), dpi= 80, facecolor='w', edgecolor='k')
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Overlapped image')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('Segmentation mask')
    plt.show()


def load_ground_truth(path, image_name, img):
    '''
    Load the ground truth image

    Parameters
    ----------
    path:           String containing the path to the database
    image_name:     String containing the image to process.
    img:            Original image, used to get the size of the gt

    Returns
    -------
    gt_mask:        numpy array containing the ground truth mask

    '''
    gt_path = path + '/groundtruth/' + image_name
    exists = os.path.isfile(str(gt_path))
    gt_mask = np.zeros_like(img[:, :, 1])
    if exists:
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        _, contours, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gt_mask = cv2.drawContours(gt_mask, contours, -1, 255, -1)

    return gt_mask


def create_marker_image(mask):
    '''
    Converts a mask of an image containing ROIs to a marker version containing multiple levels

    Parameters
    ----------
    mask:       numpy array input mask to convert to marker

    Returns
    -------
    marker:     numpy array image with a different number for each labeled region

    '''
    mask = mask.astype(np.uint8())
    _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marker = np.zeros_like(mask)
    for count in range(len(contours)):
        cv2.drawContours(marker, contours, count, (count + 1), -1)

    return marker


def match_image_markers(marker_gt, marker_pred):
    '''
    Computes the TP, FP and FN of the segmentation result it compares the
    markers of the ground truth and the markers of the predicted output.
    A region is labeled as TP when the dice score is greater than 0.2.

    Parameters
    ----------
    marker_gt       numpy array containing the ground truth regions.
    marker_pred     numpy array containing the predicted output.

    Returns
    -------
    tp, fp, fn      scalars metrics of TP FP and FN

    '''
    n_masses = np.amax(marker_gt)
    n_pred_masses = np.amax(marker_pred)
    tp = 0
    if n_masses > 0:
        for mass in range(n_masses):
            mass_img = 1. * (marker_gt == (mass + 1))
            if n_pred_masses > 0:
                for pred_mass in range(n_pred_masses):
                    pred_mass_img = 1. * (marker_pred == (pred_mass + 1))
                    dice, _ = dice_similarity(pred_mass_img, mass_img)
                    # print(dice)
                    if dice > 0.2:
                        tp += 1
                        break

    fp = n_pred_masses - tp
    fn = n_masses - tp
    return tp, fp, fn


def single_image_confusion_matrix(gt_mask, predicted_mask, show=False):
    '''
    Computes and prints a simple confusion matrix of a single image


    Parameters
    ----------
    gt_mask             numpy array containing the ground truth regions.
    predicted_mask      numpy array containing the predicted output.
    show                boolean to print the resultin confusion matrix

    Returns
    -------
    tp, fp, fn          scalars metrics of TP FP and FN
    '''

    marker_gt = create_marker_image(gt_mask)
    marker_pred = create_marker_image(predicted_mask)
    match_image_markers(marker_gt, marker_pred)
    tp, fp, fn = match_image_markers(marker_gt, marker_pred)
    if show:
        x = PrettyTable()
        x.field_names = ["Type", "Number"]
        x.add_row(["True Positives", tp])
        x.add_row(["False Positives", fp])
        x.add_row(["False Negatives", fn])
        print(x)

    return tp, fp, fn


def build_confusion_matrix(path, dataframe, show=False):
    '''
    Computes and prints a  confusion matrix of all the images in a dataframe


    Parameters
    ----------
    path:            String containing the path to the database
    dataframe       pandas dataframe contains the metadata of the ROIs
    show            boolean to print the resultin confusion matrix

    Returns
    -------
                    numpy array containing the total TP,  FP and FN of the dataframe
    '''
    file_names = dataframe['File name'].unique()
    number_of_images = len(file_names)
    confusion_matrix = np.zeros((number_of_images, 3))
    bar = progressbar.ProgressBar(maxval=number_of_images,
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    bar.update(0)
    for image_number in range(number_of_images):
        image_name = file_names[image_number]
        img, predicted_mask = draw_predicted_image(path, dataframe, image_name)
        gt_mask = load_ground_truth(path, image_name, img)
        # this crop is to speed-up the calculations, does not affect the evaluation
        shape = np.array(gt_mask.shape) / 4
        dim = (shape[1].astype(np.uint16), shape[0].astype(np.uint16))
        resized_gt = cv2.resize(gt_mask, dim, interpolation=cv2.INTER_NEAREST)
        resized_pred = cv2.resize(predicted_mask, dim, interpolation=cv2.INTER_NEAREST)
        # Calculate the confusion matrix of teach image
        tp, fp, fn = single_image_confusion_matrix(resized_gt, resized_pred)
        confusion_matrix[image_number, :] = np.array((tp, fp, fn))
        bar.update(image_number + 1)
    bar.finish()

    if show:
        tp, fp, fn = np.sum(confusion_matrix, axis=0)
        x = PrettyTable()
        x.field_names = ["Type", "Number"]
        x.add_row(["True Positives", tp])
        x.add_row(["False Positives", fp])
        x.add_row(["False Negatives", fn])
        print(x)
    return np.sum(confusion_matrix, axis=0)


def calculate_FROC(path, dataframe, probability, n_samples, type_cl = 'RF'):
    '''
    Calculate the TPs, FPs and FNs of the dataframe for multiple operation points
    it returns the points that compose the FROC curve

    Parameters
    ----------
    path:           String containing the path to the database
    dataframe       pandas dataframe contains the metadata of the ROIs
    probability     numpy array prediction probability of the classifier
    n_samples       number of operating points thresholds to use

    Returns
    -------
    froc_values     operation points of the classifier

    '''
    froc_values = np.zeros((n_samples + 2, 3))
    n_conf_matrix = np.zeros((n_samples + 2, 3))
    # Set the FROC values in the Boundary:
    froc_values[n_samples, :] = np.array([1, 0, 0])
    slope = 10
    # thresholds are selected to emphasize values close to 0 and 1
    if type_cl == 'RF':
        thresholds = 1 - 2 / (np.exp(3.5 * np.arange(n_samples) / n_samples + 1))
        thresholds = np.insert(thresholds, 0, 0)
    else:
        thresholds = 0.5 + 0.5 * np.tanh(slope * np.arange(n_samples) / n_samples - slope / 2)

    for number in range(0, n_samples):
        # Get the  response at a threshold
        n_samples = np.float32(n_samples)
        thresh = thresholds[number]
        print('\n Threshold = ' + str(thresh) )
        dataframe['Prediction'] = (probability > thresh).astype('int')

        # Calculate the confusion matrix for the given threshold
        confusion_matrix = build_confusion_matrix(path, dataframe, True)
        n_conf_matrix[number, :] = confusion_matrix

        # Calculate the sensitivity and the
        [tp, fp, fn] = confusion_matrix
        sensitivity = tp / (tp + fn)
        N_images = len(dataframe['File name'].unique())
        f_pp_i = fp / N_images
        print('\n Sensitivity= ' + str(sensitivity) + ' @   FP/Image= ' + str(f_pp_i) +
              ' # Images= '+ str(N_images) + '\n')
        froc_values[number, :] = np.array([thresh, sensitivity, f_pp_i])

    return froc_values


def plot_FROC(froc_values, param='b-', alpha=1):
    '''
    plots the resulting FROC curve of the dataset

    Parameters
    ----------
    froc_values     operating points resulting from calculate_FROC
    param           parameters to plot the curve
    alpha           transparency of the curve

    Returns
    -------

    '''
    plt.plot(froc_values[:, 2], froc_values[:, 1], param, alpha=alpha)
    plt.axis([0, 2, 0, 1])
    plt.grid()
    plt.xlabel('FP/I')
    plt.ylabel('Sensitivity')


def Kfold_FROC_curve(model, folds, train_dataframe, train_metadata, path):
    '''
    Performs cross validation for the XGboost model

    Parameters
    ----------
    model               Referencing classifier model
    folds               Number of folds to be used
    train_dataframe     pandas dataframe contains the features of the ROIs
    train_metadata      pandas dataframe contains the metadata of the ROIs
    path:               String containing the path to the database

    Returns
    -------
    test_metadata_T     Resulting metadata organized according to the cross-validation
    probability_T       Resulting probability of the classifier to all the available data

    '''
    images_name = pd.DataFrame(train_metadata["File name"].unique())
    images_name["Class"] = 0
    images_name = images_name.rename(columns = {0: "File name"})
    contain_tp = train_metadata[train_metadata["Class"] == 1]["File name"].unique()
    images_name["Class"] = images_name["File name"].isin(contain_tp).astype(int)

    # Create a Cross validation object
    cv = StratifiedKFold(n_splits=folds, shuffle=True)
    fold = 0

    for train, test in cv.split(images_name["File name"], images_name["Class"]):
        # Generate the K-training set
        train_names_k = images_name.iloc[train]["File name"]
        train_selected_k = train_metadata["File name"].isin(train_names_k)
        x_train_k = train_dataframe[train_selected_k].to_numpy()
        y_train_k = train_metadata[train_selected_k]["Class"]

        # Generate the K-testing set
        test_names_k = images_name.iloc[test]["File name"]
        test_selected_k = train_metadata["File name"].isin(test_names_k)
        x_test_k = train_dataframe[test_selected_k].to_numpy()
        y_test_k = train_metadata[test_selected_k]["Class"]

        test_metadata_k = train_metadata[test_selected_k]
        test_metadata_k.index = range(len(test_metadata_k))
        print(' \n Fold #: ' + str(fold + 1))
        print('Train with ' + str(len(y_train_k)) + ' Test with ' + str(len(y_test_k)) + ' ROIs \n')

        #########################################################################
        #           DEFINE HERE THE MODEL FIT/ PREDICT
        #########################################################################

        dtrain = xgb.DMatrix(x_train_k, label=y_train_k)
        dtest = xgb.DMatrix(x_test_k)

        params, num_rounds = model

        bst = xgb.train(params, dtrain, num_rounds)
        probability = bst.predict(dtest)

        ########################################################################
        ########################################################################
        if fold == 0:
            test_metadata_T = test_metadata_k
            probability_T = probability
        else:
            test_metadata_T = pd.concat([test_metadata_T, test_metadata_k], ignore_index=True)
            probability_T = np.concatenate((probability_T, probability), axis=0)

        fold += 1

    return test_metadata_T, probability_T


def Kfold_FROC_curve_cascadeRF(folds, train_dataframe, train_metadata, path, num_layers_to_test):
    '''
    Performs cross validation for the cascade random forest model

    Parameters
    ----------
    folds                   Number of folds to be used
    train_dataframe         pandas dataframe contains the features of the ROIs
    train_metadata          pandas dataframe contains the metadata of the ROIs
    path:                   String containing the path to the database
    num_layers_to_test      the number of layers of the random forest classifier

    Returns
    -------

    '''
    images_name = pd.DataFrame(train_metadata["File name"].unique())
    images_name["Class"] = 0
    images_name = images_name.rename(columns = {0: "File name"})
    contain_tp = train_metadata[train_metadata["Class"] == 1]["File name"].unique()
    images_name["Class"] = images_name["File name"].isin(contain_tp).astype(int)

    # Create a Cross validation object
    cv = StratifiedKFold(n_splits=folds, shuffle=True)
    fold = 0

    for train, test in cv.split(images_name["File name"], images_name["Class"]):
        # Generate the K-training set
        train_names_k = images_name.iloc[train]["File name"]
        train_selected_k = train_metadata["File name"].isin(train_names_k)
        x_train_k = train_dataframe[train_selected_k].to_numpy()
        num_features = x_train_k.shape[1]
        y_train_k = train_metadata[train_selected_k]["Class"].to_numpy()

        # Generate the K-testing set
        test_names_k = images_name.iloc[test]["File name"]
        test_selected_k = train_metadata["File name"].isin(test_names_k)
        x_test_k = train_dataframe[test_selected_k].to_numpy()

        test_metadata_k = train_metadata[test_selected_k]
        test_metadata_k.index = range(len(test_metadata_k))

        # Normalize data
        scaler = StandardScaler()
        scaler.fit(x_train_k)
        x_train_k = scaler.transform(x_train_k)
        x_test_k = scaler.transform(x_test_k)

        print(' \n Fold #: ' + str(fold + 1))
        print('Train with ' + str(len(y_train_k)) + ' Test with ' + str(len(x_test_k)) + ' ROIs \n')

        #########################################################################
        #           Cascade Random Forest
        #########################################################################
        percentage_validation = 0.3
        negative_multiplier = 1
        training_features_tp, training_features_ntp, validation_features_tp, validation_features_ntp = subdivide_dataset_k(x_train_k, y_train_k, percentage_validation, negative_multiplier)
        NP = training_features_tp.shape[0]

        #Concatenate training subsets and internal validation subsets
        fullX = np.concatenate((training_features_tp,training_features_ntp))
        fullY = np.concatenate((np.ones((training_features_tp.shape[0]), dtype=np.uint32), np.zeros((training_features_ntp.shape[0]), dtype=np.uint32)))
        fullValidX = np.concatenate((validation_features_tp,validation_features_ntp))
        fullValidY = np.concatenate((np.ones((validation_features_tp.shape[0]), dtype=np.uint32), np.zeros((validation_features_ntp.shape[0]), dtype=np.uint32)))

        #Define parameters for the RF classifier
        d_ntree = 500
        d_mtry = int(np.sqrt(num_features))
        ntree_first = 100
        ntree_last = 1000
        ntree_num_elems = 10
        mtry_first = int(0.5*np.sqrt(num_features))
        mtry_last = int(2*np.sqrt(num_features))
        mtry_num_elems = 10
        parameters = {'n_estimators':np.linspace(ntree_first,ntree_last,ntree_num_elems, dtype=int), 'max_features':np.linspace(mtry_first, mtry_last, mtry_num_elems, dtype=int)}
        st = 0.9

        layers = trainCascadeRandomForestClassifierFaster(d_ntree, d_mtry, NP, parameters, st, training_features_tp, training_features_ntp, fullX, fullY, fullValidX, fullValidY, num_layers_to_test, fold)
        probability = get_probs(layers[:num_layers_to_test], x_test_k)


        ########################################################################
        ########################################################################
        if fold == 0:
            test_metadata_T = test_metadata_k
            probability_T = probability
        else:
            test_metadata_T = pd.concat([test_metadata_T, test_metadata_k], ignore_index=True)
            probability_T = np.concatenate((probability_T, probability), axis=0)

        fold += 1

    return test_metadata_T, probability_T