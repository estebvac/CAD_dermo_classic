
from preprocessing.utils import *
import matplotlib.pyplot as plt
from preprocessing.preprocessing import discard_small
from skimage.segmentation import slic
from skimage import color
from skimage.filters.thresholding import threshold_otsu


def segment_image(img_orig, img_subpixel, debug=False):
    """

    Parameters
    ----------
    img_subpixel

    Returns
    -------

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
        plt.imshow(img_orig)
        plt.contour(roi)
        plt.show()

    return roi


def create_marker(img, debug=False):

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
        plt.imshow(markers_out)
        plt.show()

    return [markers_out, sure_fg_val, sure_bg_val]


def imshow_contour(img_color, thresh, window_name="Contours"):
    """

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
