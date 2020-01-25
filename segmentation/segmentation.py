
from preprocessing.utils import *
import matplotlib.pyplot as plt
from preprocessing.preprocessing import discard_small
from preprocessing.utils import show_image
from skimage.segmentation import slic
from skimage import color
from skimage.filters.thresholding import threshold_otsu
from scipy.spatial import distance
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)

def segment_image(img_orig, img_subpixel, debug=False):
    """
    Function to segment input image using information from previous superpixel segmentation

    Parameters
    ----------
    img_orig            Original image to segment

    img_subpixel        Result of superpixels

    Returns
    ----------
    roi                 Result of the segmentation

    """
    img_color = img_subpixel
    [markers, sure_fg_val, _] = create_marker(img_subpixel, debug)

    # Apply watershed to get external pixels
    markers = markers.astype(np.int32)
    markers_base = np.copy(markers)
    markers = cv.watershed(img_color, markers)

    # Create a new marker template
    markers_base[markers == -1] = np.max(markers_base) + 2
    markers = cv.watershed(img_color, markers_base)

    # Select the central blob
    roi = np.zeros_like(img_subpixel[:, :, 1])
    roi[markers == sure_fg_val] = 1

    # print the segmentation
    if debug:
        show_image(roi * 255,"Final segmentation")
        cv.waitKey(10)


    return roi


def create_marker(img, debug=False):
    """
    Function to create markers on the image that will be used then by the Watershed algorithm

    Parameters
    ----------
    img                 Input image

    Returns
    ----------
    markers             Created markers
    sure_fg_val         Points corresponding to foreground
    sure_bg_val         Points corresponding to background

    """
    # Convert to gray
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    difference = np.max(img_gray) - np.min(img_gray)
    # check the max_min levels of the image
    if difference < 150:
        # equalize the histogram of the Y channel
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_gray = clahe.apply(img_gray)

    # Otsu Thresholding
    blur = cv.GaussianBlur(img_gray, (5, 5), 5)
    # Remove the frame from the segmentation
    sure_fg = np.zeros_like(img_gray)
    it = 0
    ellipse = mask_eliptical(img, 15, False)

    while np.sum(sure_fg) < 0.1 * sure_fg.size:
        if it == 0:
            threshold = threshold_otsu(blur)
            it += 1
        else:
            threshold += 10

        thres = blur <= threshold
        thres = thres * ellipse
        Nerotion = 3

        # sure background area
        kernel = np.ones((7, 7), np.uint8)
        sure_fg = cv.erode(thres, kernel, Nerotion).astype(np.uint8)
        sure_fg = discard_small(sure_fg, 500)
        sure_fg = discard_not_centered(sure_fg)

    extreme_borders = 127 * np.ones_like(img_gray)
    border = 5
    extreme_borders[border:-border, border:-border] = 0

    sure_fg_val = 255
    sure_bg_val = 127

    markers_out = extreme_borders + sure_fg
    if debug:
        show_image(markers_out, "markers")
        cv.waitKey(10)


    return [markers_out, sure_fg_val, sure_bg_val]


def imshow_contour(img_color, thresh, window_name="Contours"):
    """
    Function to show contours of an image 
    Parameters
    ----------
    img_color       Color image
    thresh          Contour of the ROI
    window_name     Name to show in the window

    Returns
    -------

    """
    img = np.copy(img_color)
    _, contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv.drawContours(img, contour, -1, (0, 255, 0), 3)

    show_image(img, window_name)
    cv.waitKey(10)

    return


def mask_eliptical(img, border=-1, positive=True):
    """

    Parameters
    ----------
    img         Original image
    border      Border of the elipse to be created
    positive    Boolean indicating to fill with 1 or 0

    Returns
    -------
    Ellipsed image
    """
    if positive:
        ellipse_mask = np.zeros((img.shape[0], img.shape[1]))
        color = 1
    else:
        ellipse_mask = np.ones((img.shape[0], img.shape[1]))
        color = 0

    axis_major = int(img.shape[1] / 2)
    axis_minor = int(img.shape[0] / 2)
    center = (axis_major, axis_minor)

    cv.ellipse(ellipse_mask,
               center=center,
               axes=(axis_major, axis_minor),
               angle=0,
               startAngle=0,
               endAngle=360,
               color=color,
               thickness=border)

    return ellipse_mask.astype(np.uint8)


def segment_superpixel(img, debug=False):
    """
    Function to apply superpixels to image

    Parameters
    ----------
    img           Original image to segment

    Returns
    ----------
    img_labeled   Labeled image

    """
    img_color = img
    mask = mask_eliptical(img_color, -1, True)
    segments_slic = slic(img_color, n_segments=400, compactness=10, sigma=1)
    segments_slic = segments_slic * mask
    img_labeled = color.label2rgb(segments_slic, img_color, kind='avg')
    if debug:
        plt.imshow(img_labeled)
        plt.show()

    return img_labeled


def discard_not_centered(img, tx=100, ty=125, connectivity=4):
    """

    Parameters
    ----------
    img             ROI image bw
    tx              limit in the x axis
    ty              limit in the y axis
    connectivity    connectivity type either 2 or 4

    Returns
    -------
    Image removing the connected components outside a centered window

    """
    output_image = np.zeros(img.shape)
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(img, connectivity)

    # TODO: Include info about shape of the element
    for i_label in range(1, nlabels):
        if ( (img.shape[1] / 2 - ty) < centroids[i_label, 0] < (img.shape[1] / 2 + ty)
                and (img.shape[0] / 2 - tx) < centroids[i_label, 1] < (img.shape[0] / 2 + tx) ):
            # Only structures centered
            output_image[labels==i_label] = 255

    return output_image.astype(np.uint8)


def find_segmented(binary):
    """ Function to decide which of the connected components generated by level sets corresponds to the lesion. 
    Decision is made based on the distance to the center of the image

    Parameters
    ----------
    binary          Binary mask with result of level sets

    Returns
    -------
    output_image    Binary image with segmentation result

    """
    output_image = np.zeros_like(binary)
    middle_row = np.floor(binary.shape[0]/2).astype(np.int)
    middle_col = np.floor(binary.shape[1]/2).astype(np.int)
    center = np.array([middle_col, middle_row])
    #Now select only connected component with bigger area and with centroid closest to center
    _, contours, _ = cv.findContours(binary.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    distances = np.zeros(len(contours))
    i_dist = 0
    for cnt in contours:
        area_cnt = cv.contourArea(cnt)
        if(area_cnt> 450*430):
            distances[i_dist] = 10000
            i_dist += 1
            continue
        # calculate moments for each contour
        M = cv.moments(cnt)
        if(M["m00"]==0):
            distances[i_dist] = 10000
            i_dist += 1
            continue
        # calculate x,y coordinate of center
        center_cnt = np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
        dst = distance.euclidean(center_cnt, center)
        distances[i_dist] = dst
        i_dist += 1

    min_dist = np.argmin(distances)
    cv.fillPoly(output_image, pts =[contours[min_dist]], color=255)
    
    _, contours, _ = cv.findContours(output_image.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    if(len(contours)>0): #If area is too small, erase it and take a circle in the middle
        area_cnt = cv.contourArea(contours[0])
        M = cv.moments(contours[0])
        # calculate x,y coordinate of center
        if(M["m00"]!=0):
            center_cnt = np.array([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
            dst = distance.euclidean(center_cnt, center)
        else:
            dst = 10
        if(area_cnt<30 or dst>150):
            cv.fillPoly(output_image, pts =[contours[0]], color=0)
            cv.circle(output_image, (center[0], center[1]), 100, 255, -1)

    if(len(contours)==0): #If segmentation failed, take a circle in the middle of the image
        cv.circle(output_image, (center[0], center[1]), 100, 255, -1)

    #Bring to 0-1 range
    max_val = np.max(output_image.astype(np.uint8))
    output_image = output_image/max_val

    return output_image.astype(np.uint8)
    

def segment_with_level_sets(img):
    """ Function to perform segmentation with level sets. img_gray should not contain hairs nor black zones

    Parameters
    ----------
    binary          Binary mask with result of level sets

    Returns
    -------
    output_image    Binary image with segmentation result

    """
    #First, find out if it is necessary to appy inpainting to remove black zones

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    r1 = img_gray[:5,:5]
    r2 = img_gray[:5,-5:]
    r3 = img_gray[-5:,:5]
    r4 = img_gray[-5:,-5:]

    if(np.mean(r1)<50 or np.mean(r2)<50 or np.mean(r3)<50 or np.mean(r4)<50): #If zones of the corners are too dark, apply ellipse inpainting
        elliptical_mask = 1 - mask_eliptical(img_gray)
        to_process = cv.inpaint(img_gray, elliptical_mask, 21, cv.INPAINT_NS)
    else:
        to_process = img_gray

    init_ls = checkerboard_level_set(img_gray.shape, 6)

    ls = morphological_chan_vese(to_process, 35, init_level_set=init_ls, smoothing=3)

    segmented = find_segmented(ls)

    return segmented
