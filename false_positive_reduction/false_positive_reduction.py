import numpy as np
import cv2 as cv2
from candidates_detection.utils import get_breast_mask, fill_holes


def border_false_positive_reduction(all_scales, original_image):
    '''
    Simple false positive reduction, removes the ROI that are on the skin of the
    mammography and the regions of interest that are smaller than 50x50 pixels.(3.5 mm)

    Parameters
    ----------
    all_scales
    original_image

    Returns
    -------

    '''
    mask_raw = get_breast_mask(original_image)
    mask = fill_holes(mask_raw)
    mask[:, :500] = 1
    mask[:, -500:] = 1
    # Create a mask around the border of the chest to remove the false positive slices
    kernel = np.ones((55, 55), np.uint8)
    eroded_foreground_mask = cv2.erode(mask, kernel, iterations=2)
    dilated_foreground_mask = cv2.dilate(mask, kernel, iterations=1)
    pheriferical_chest = (dilated_foreground_mask.astype(int) + eroded_foreground_mask.astype(int)) == 255
    pheriferical_chest = np.uint8(pheriferical_chest * 1)

    # extract regions of interest of each layer:
    for slice_counter in np.arange(all_scales.shape[2]):
        slice = all_scales[:, :, slice_counter] * 255
        markers = np.zeros_like(slice)

        # Find the contours of each ROI, check the size of each region
        _, contours, _ = cv2.findContours(slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i in range(len(contours)):
            # Get the size of the ROI
            # We set this parameters thus we are looking for masses and not for microcalsificatins
            _, _, w_b, h_b = cv2.boundingRect(contours[i])
            if( 50 < w_b < 1100 and 50 < h_b < 1100):
                cv2.drawContours(markers, contours, i, (i + 1), -1)

        # Discard the ROI that are in the boundaries
        markers_discard = np.unique(markers * pheriferical_chest)
        for i in markers_discard:
            markers[markers==i]=0

        # Set to 1 the remaining ROI
        markers[markers>0] = 1
        all_scales[:, :, slice_counter] = markers

    return all_scales