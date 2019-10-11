from os.path import join

import cv2
import numpy as np
from candidates_detection.find_candidates import segment_image
from false_positive_reduction.false_positive_reduction import border_false_positive_reduction
from evaluation.dice_similarity import extract_ROI
from feature_extraction.build_features_file import extract_features
from .split_features import create_entry, create_features_dataframe, drop_unwanted_features

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
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    roi = segment_image(img, debug)
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

