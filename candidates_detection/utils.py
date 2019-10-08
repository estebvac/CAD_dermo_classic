import cv2 as opencv;
import numpy as np;

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

    img_to_show = opencv.resize(img_to_show, None, fx=factor, fy=factor)
    opencv.imshow(img_name, img_to_show)

def bring_to_256_levels(the_image):
    """Function normalize image to 0-255 range (8 bits)

    Parameters
    ----------
    image : numpy array
        Original image.

    Returns
    -------
    img_new : numpy array
        Normalized image
    """

    if(the_image.max() == the_image.min()):
        return the_image.astype(np.uint8)
    img_as_double = the_image.astype(float)
    normalized = np.divide((img_as_double - np.amin(img_as_double)), (np.amax(img_as_double) - np.amin(img_as_double)))
    normalized = normalized*(pow(2, 8) - 1)
    return normalized.astype(np.uint8)

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

    if angle==1 or angle == 5:
        SE_horizontal = np.zeros((size, size))
        SE_horizontal[int((size - 1) / 2), :] = np.ones((1, size))
        if angle==1:
            return SE_horizontal.astype(np.uint8)
        else: #If vertical
            return np.transpose(SE_horizontal).astype(np.uint8)
    elif angle == 3 or angle == 7: #If 45 or 135
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




def discard_regions_processing(the_image, connectivity, d1, d2):
    """Function to discard regions whose area is greater than the one specified by  the range [(pi/4)*d1^2, 1.3*(pi/4)*d2^2].


    Parameters
    ----------
    image : numpy array
        Original binary image.
    connectivity : int
        Connectivity for the region analysis
    d1 : int
        Minimum diameter for the regions
    d2 : int
        Maximum diameter for the regions


    Returns
    -------
    output_image : numpy array
        Image without regions that have an area that lies outside the mentioned range.
    """

    correction_factor = 1.3
    output_image = np.zeros(the_image.shape)
    nlabels, labels, stats, centroids = opencv.connectedComponentsWithStats(the_image, connectivity)
    area_range = np.array([(np.pi/4)*pow(d1,2), (np.pi/4)*pow(d2,2)])
    for i_label in range(nlabels):
        if(stats[i_label, 4]>= area_range[0] and stats[i_label, 4]<= area_range[1]):
            if(stats[i_label, 2] >= correction_factor*d2 or stats[i_label, 3] >= correction_factor*d2):
                continue
            else:
                output_image[labels==i_label] = 1
    return output_image

def fill_holes(the_image):
    """Function to fill the holes in a binary image (from https://stackoverflow.com/questions/10316057/filling-holes-inside-a-binary-object)

    Parameters
    ----------
    the_image : numpy array
        Binary image to fill holes

    Returns
    -------
    des
        Binary image with holes filled
    """

    des = the_image.copy() #cv.bitwise_not(output_image)
    contour =\
        opencv.findContours(
            des,opencv.RETR_CCOMP,opencv.CHAIN_APPROX_SIMPLE)[1]
    for cnt in contour:
        opencv.drawContours(des,[cnt],0,255,-1)
    return des

def get_breast_mask(x_input):
    """
    Removes the pixels of the mammography that does not belong to the breast

    Parameters
    ----------
    x_input     Input image

    Returns     Breast mask (binary)
    -------

    """
    # Normalise the image
    x_normalized = 255 * normalize_image(x_input)
    x_normalized = x_normalized.astype(np.uint8)

    # Threshold the image
    _, breast_bw =\
        opencv.threshold(
            x_normalized,
            x_normalized.min(),
            x_normalized.max(),
            opencv.THRESH_BINARY + opencv.THRESH_OTSU)

    return breast_bw

def normalize_image(x_input):
    """
    Function to normalize the image limits between 0 and 1
    Parameters
    ----------
    x_input         Input image to normalize

    Returns         Image with adjusted limits between [0., 1.]
    -------

    """
    n = x_input - x_input.min()
    n = n / x_input.max()
    return n


def normalize_and_equalize(x_input):
    """
    Equalizes the histogram of a grayscale of any type

    Parameters
    ----------
    x_input         Input image to equalize

    Returns         Equalized image with limits between [0, 255]
    -------

    """
    out = 255 * normalize_image(x_input)
    out = opencv.equalizeHist(out.astype(np.uint8))
    return out