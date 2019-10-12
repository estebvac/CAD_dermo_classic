from os.path import join

import cv2
import numpy as np
from segmentation.segmentation_watershed import segment_image
from feature_extraction.build_features_file import extract_features
from preprocessing.preprocessing import preprocess_and_remove_hair
from .split_features import create_entry, create_features_dataframe, drop_unwanted_features
import matplotlib.pyplot as plt
from preprocessing.utils import *

COLOURS =\
    [(255, 0, 0),
     (0, 255, 0),
     (0, 0, 255),
     (255, 255, 0),
     (255, 0, 255),
     (0, 255, 255),
     (100, 255, 0),
     (255, 100, 0),
     (255, 100, 100)]

def __generate_outputs (img, rois, output):
    '''
    Generate output according to contours in rois
    Parameters
    ----------
    img         numpy array of the input iimage
    rois        OpenCv contours
    output      output image

    Returns
    -------
    Save an image with the overlaped region of interest

    '''
    normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    normalized_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2BGR)
    for roi in rois:
        cv2.drawContours(normalized_img, roi.get('Contour'), -1, COLOURS[roi.get('Slice')], 2)

    cv2.imwrite(output, normalized_img)


def __process_features(filename, img, roi):
    '''
    Process the resulting scales of the segmentation
    Parameters
    ----------
    filename:       input filename
    img             numpy array containing the image
    all_scales      numpy array containing all the segmented ROIS in all scales

    Returns
    -------
    dataframe       dataframe of all the ROIs in the image

    '''

    dataframe = []
    _, contours, _ = cv2.findContours(255 *roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for roi_counter in np.arange(min(len(contours), 1)):
        roi_color, boundaries = extract_ROI(contours[roi_counter], img)
        roi_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        [cnt_features, textures, hu_moments, lbp, tas_features, hog_features] = \
            extract_features(roi_gray, contours[roi_counter], roi)
        entry = create_entry(
            filename, cnt_features, textures, hu_moments, lbp, tas_features, hog_features, contours[roi_counter])
        dataframe.append(entry)

    return dataframe

def process_single_image(filename, debug=False):
    '''
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

    '''
    img = cv.imread(filename)
    img_wo_hair, _ = preprocess_and_remove_hair(img)
    roi = segment_image(img_wo_hair, debug)
    features = __process_features(filename, img, roi)
    return [roi, features, img]


def segment_single_image(path, filename):
    '''
    Segment single image
    Parameters
    ----------
    path            String path to the dataset
    filename        String name of the input image

    Returns
    -------
    all_scales      segmentated ROIs

    '''
    total_features = []
    [all_scales, features, img] = process_single_image(path, False)

    total_features.extend(features)
    [df_features, tags] = create_features_dataframe(features)
    df_features = drop_unwanted_features(df_features)
    print(df_features.to_numpy())

    return all_scales

def extract_ROI(roi_contour, img,padding = 0.05):
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
    (img_max_y, img_max_x, img_chan) = img.shape
    x_m = min(x_b + w_b + 2 * padd_x, img_max_x)
    y_m = min(y_b + h_b + 2 * padd_y, img_max_y)

    # Extract the region of interest of the given contours
    roi = img[y_b:y_m, x_b:x_m]
    boundaries = x_b, y_b, w_b, h_b

    return roi, boundaries