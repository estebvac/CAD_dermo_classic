from os import listdir, chdir
from os.path import isfile, join
import pandas as pd
from candidates_detection.find_candidates import find_candidates
from false_positive_reduction.false_positive_reduction import border_false_positive_reduction
from evaluation.dice_similarity import dice_similarity, extract_ROI
from candidates_detection.utils import bring_to_256_levels
from feature_extraction.features_extraction import *
from feature_extraction.contour_features import *


def read_images(general_path):
    '''
    Read all the images of the dataset

    Parameters
    ----------
    general_path:   String Path containing the folders image and groundtruth

    Returns
    -------
    raw_im_Path     String path pointing to the images of the dataset
    gt_im_path      String path pointing to the ground truth of the dataset
    raw_images      List of names of the original images
    gt_images       List of names of the ground truth images

    '''

    #  Read the dataset
    raw_im_Path = general_path + "images"
    gt_im_path = general_path + "groundtruth"
    raw_images =\
        [f for f in listdir(raw_im_Path) if f.endswith(".tif") and isfile(join(raw_im_Path, f))]
    gt_images =\
        [f for f in listdir(gt_im_path) if f.endswith(".tif") and isfile(join(gt_im_path, f))]

    # Paths to store the ROIs
    false_positive_path = general_path + "false_positive"
    true_positive_path = general_path + "true_positive"

    return [raw_im_Path, gt_im_path, raw_images, gt_images, false_positive_path, true_positive_path]


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

