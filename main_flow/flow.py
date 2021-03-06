import cv2
from segmentation.segmentation import *
from feature_extraction.feature_extraction import extract_features_no_segm, extract_features
from preprocessing.preprocessing import *
from preprocessing.utils import *
import pandas as pd


def __process_features(img, roi):
    """
    Obtain features for all segmented regions in the image
    Parameters
    ----------
    img             numpy array containing the image
    roi             Roi in the image

    Returns
    -------
    features_all    Array with all features for all regions in current image

    """
    _, contours, _ = cv2.findContours(255 * roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for roi_counter in np.arange(min(len(contours), 1)):
        roi_color, boundaries = extract_ROI(contours[roi_counter], img)
        roi_bw, _ = extract_ROI(contours[roi_counter], roi)
        features = extract_features(roi_color, contours[roi_counter], roi_bw)
        if roi_counter == 0:
            features_all = features
        else:
            features_all = np.concatenate((features_all, features), axis=0)

    return features_all


def process_single_image(filename, segm_alg, debug=False):
    """
    Process a single image extracting the ROIS and the features
    Parameters
    ----------
    path            path where all the dataset is licated
    filename        file to extract the ROIS

    Returns
    -------
    all_scales      Segmentation ROIs of the image
    features        dataframe of all the features of the ROIs in the image
    img             numpy arrray containing the image

    """
    img = cv2.imread(filename)
    img_wo_hair, _ = preprocess_and_remove_hair(img)
    if segm_alg == "ws":
        img_superpixel = segment_superpixel(img_wo_hair, debug)
        roi = segment_image(img_wo_hair, img_superpixel, debug)
    elif segm_alg == "ls":
        roi = segment_with_level_sets(img_wo_hair)
    features = __process_features(img_wo_hair, roi)
    return features


def process_single_image_no_segm(filename):
    """
    Process a single image (without segmentation) extracting the ROIS and the features
    Parameters
    ----------
    filename        file to extract the ROIS

    Returns
    -------
    features        dataframe of all the features of the ROIs in the image

    """
    img = cv2.imread(filename)
    img_wo_hair, _ = preprocess_and_remove_hair(img)
    features = extract_features_no_segm(img_wo_hair)
    return features


def extract_ROI(roi_contour, img, padding=0.05):
    """

    Parameters
    ----------
    roi_contour:    Contours points
        A contour of the images generated with the function findcontours
    img:            numpy array of 12 bits depth

    Returns
    -------
    roi:            Numpy array
        Image containing the required ROI of the image
    boundaries:     Numpy array
        Array containing the coordinates of the extracted ROI in the full image coordinates

    """
    # Get the boundaries of each region of interest
    x_b, y_b, w_b, h_b = cv2.boundingRect(roi_contour)

    padd_x = np.uint8(padding * w_b)
    padd_y = np.uint8(padding * h_b)

    x_b = max(x_b - padd_x, 0)
    y_b = max(y_b - padd_y, 0)

    # Adjust the boundaries so we get a surrounding region
    img_shape = img.shape
    x_m = min(x_b + w_b + 2 * padd_x, img_shape[1])
    y_m = min(y_b + h_b + 2 * padd_y, img_shape[0])

    # Extract the region of interest of the given contours
    roi = img[y_b:y_m, x_b:x_m]
    boundaries = x_b, y_b, w_b, h_b

    return roi, boundaries


def save_mask(roi, path, filename):
    """
    Save a mask of the ROI

    Parameters
    ----------
    roi         Mask of the lesion
    path        Path to save the file
    filename    Name of the file to save

    Returns     Nothing
    -------

    """
    directory = filename.split('/')[-3:]
    directory = [path, 'mask'] + directory
    seperator = '/'
    write_path = seperator.join(directory).replace('\\', '/')
    cv2.imwrite(write_path, roi * 255)
    cv2.waitKey(1)
