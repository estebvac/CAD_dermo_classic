import numpy as np
import cv2 as cv


def bring_to_256_levels(the_image):
    """
    Function normalize image to 0-255 range (8 bits)

    Parameters
    ----------
    the_image : numpy array Original image.

    Returns
    -------
    img_new : numpy array
        Normalized image
    """
    if the_image.max() == the_image.min():
        return the_image.astype(np.uint8)
    img_as_double = the_image.astype(float)
    normalized = np.divide((img_as_double - np.amin(img_as_double)), (np.amax(img_as_double) - np.amin(img_as_double)))
    normalized = normalized*(pow(2, 8) - 1)
    return normalized.astype(np.uint8)


def show_image(img_to_show, img_name, factor=1.0):
    """Function to display an image with a specific window name and specific resize factor

    Parameters
    ----------
    img_to_show : numpy array
        Image to be displayed
    img_name : string
        Name for the window
    factor : float
        Resize factor (between 0 and 1)

    """
    img_to_show = cv.resize(img_to_show, None, fx=factor, fy=factor)
    cv.imshow(img_name, img_to_show)


def getLinearSE(size, angle):
    """Function to create a linear SE with a specific size and angle.

    Parameters
    ----------
    image : numpy array
        Original image.
    size: int
        Size of the square that contains the linear SE
    angle : int
        Number that identifies an angle for the linear SE according to following options
                    1: 0°
                    2: 22.5°
                    3: 45°
                    4: 67.5°
                    5: 90°
                    6: 112.5
                    7: 135°
                    8: 157.5°
                    9: 11.25°
                    10: 78.75°
                    11: 101.25°
                    12: 168.75

    Returns
    -------
    SE_diff : numpy array
        Binary array of size <size> that contains linear SE with approximate angle given by <angle>
    """

    if angle == 1 or angle == 5:
        SE_horizontal = np.zeros((size, size))
        SE_horizontal[int((size - 1) / 2), :] = np.ones((1, size))
        if angle == 1:
            return SE_horizontal.astype(np.uint8)
        else: # If vertical
            return np.transpose(SE_horizontal).astype(np.uint8)

    elif angle == 3 or angle == 7:      # If 45 or 135
        SE_diagonal = np.eye(size)
        if angle == 3:
            return np.fliplr(SE_diagonal).astype(np.uint8)
        else:
            return SE_diagonal.astype(np.uint8)

    elif angle in [2,4,6,8]: #Angle more comples
        SE_diff = np.zeros((size, size))
        row = int(((size-1)/2)/2)
        col = 0
        ctrl_var = 0
        for i in range(size):
            if ctrl_var == 2:
                row = row +1
                ctrl_var = 0
            SE_diff[row, col] = 1
            col=col+1
            ctrl_var = ctrl_var + 1
        if angle == 8:
            return SE_diff.astype(np.uint8)
        elif angle == 2:
            return np.flipud(SE_diff).astype(np.uint8)
        elif angle == 4:
            return np.fliplr(np.transpose(SE_diff)).astype(np.uint8)
        else:
            return np.transpose(SE_diff).astype(np.uint8)

    elif angle in [9,10,11,12]:
        SE_diff = np.zeros((size, size))
        row = int(((size-1)/2)/2) + int( ((((size-1)/2)/2)-1)/2 )
        col = 0
        ctrl_var = 0
        for i in range(size):
            if ctrl_var == 3:
                row = row + 1
                ctrl_var = 0
            SE_diff[row, col] = 1
            col = col + 1
            ctrl_var = ctrl_var + 1
        if angle == 9:
            return np.flipud(SE_diff).astype(np.uint8)
        elif angle == 10:
            return np.fliplr(np.transpose(SE_diff)).astype(np.uint8)
        elif angle == 11:
            return np.transpose(SE_diff).astype(np.uint8)
        else:
            return SE_diff.astype(np.uint8)