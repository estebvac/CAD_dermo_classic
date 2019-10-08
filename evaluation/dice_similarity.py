import numpy as np
import cv2 as cv2


def dice_similarity(segmented_image, ground_truth):
    """
    This function calculates the dice similarity index for the segmentation of a given
    image and its ground truth

    Parameters
    ----------
    segmented_image:   Numpy array
        binary mask( either 0-1 or 0-255) containing the segmentation
    ground_truth:       Numpy array
        binary mask( either 0-1 or 0-255) containing the ground truth

    Returns
    -------
    dice:               double
        Dice similarity similarity index of  two samples.

    """
    segmented_image = cv2.normalize(segmented_image, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    ground_truth = cv2.normalize(ground_truth, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

    intersection = np.sum(segmented_image * ground_truth)
    union = np.sum(segmented_image) + np.sum(ground_truth)
    dice = 2.0 * intersection / union
    jaccard = intersection / (union - intersection)

    '''
    print(np.amin(segmented_image), np.amax(segmented_image), np.sum(segmented_image))
    plt.subplot(1, 2, 1)
    plt.imshow(segmented_image)
    plt.subplot(1, 2, 2)
    plt.imshow(ground_truth)
    plt.show()
    '''

    return dice, jaccard


def true_positives_labelling(over_segmented_image, ground_truth):
    """
    This function is used to label the true and false positive candidates obtained
    from the candidate detection module. This is based on the dice similarity index
    if the dice index is greater or equal to 0.2 the ROI is labeled as true
    positive otherwise is considered as false positive.

    Parameters
    ----------
    over_segemented_image:   Numpy array
        Segmentation of the image after the candidate detection
    ground_truth:   Numpy array

    Returns
    -------

    """
    return 0


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
    img_max_y, img_max_x = img.shape
    x_m = min(x_b + w_b + 2 * padd_x, img_max_x)
    y_m = min(y_b + h_b + 2 * padd_y, img_max_y)

    # Extract the region of interest of the given contours
    roi = img[y_b:y_m, x_b:x_m]
    boundaries = x_b, y_b, w_b, h_b

    return roi, boundaries


