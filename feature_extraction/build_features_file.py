from os import listdir, chdir
from os.path import isfile, join
import pandas as pd
from preprocessing.utils import bring_to_256_levels
from feature_extraction.feature_extraction import *
import os

def read_images(general_path):
    '''
    Read all the images of the dataset

    Parameters
    ----------
    general_path:   String Path containing the folders image and groundtruth

    Returns
    -------
    images_dataframe    Datafame containing the String path of the image and the class

    '''

    #  Read the dataset
    data_tuple = []
    for folder in sorted(os.listdir(general_path)):
        for file in sorted(os.listdir(general_path + '/' + folder)):
            full_path = general_path + '/' + folder + '/' + file
            data_tuple.append((full_path, folder))

    images_df = pd.DataFrame(data_tuple, columns=['File', 'Class'])
    return images_df


def extract_features(roi, contour, roi_bw):
    '''
    Extract all the features of a ROI

    Parameters
    ----------
    roi         Region of interest of the image
    contour     Contour containing the ROI
    roi_bw      Binary version of the ROI

    Returns
    -------
    feature_vector      all extracted features of a ROI

    '''
    # Contour features
    cnt_features = calculate_contour_features(contour)

    # Haralick Features
    masked_roi = np.multiply(roi,roi_bw)
    #textures = feature_extraction_haralick(bring_to_256_levels(roi))
    textures = feature_extraction_haralick(bring_to_256_levels(masked_roi))

    # Hu moments:
    hu_moments = feature_hu_moments(contour)

    # Multi-Scale Local Binary Pattern features:
    lbp = multi_scale_lbp_features(roi)

    # TAS features
    tas_features = feature_tas(roi_bw)

    # HOG features
    hog_features = features_hog (roi)

    return [cnt_features, textures, hu_moments, lbp, tas_features, hog_features]

