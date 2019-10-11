import cv2 as cv
from preprocessing.is_right import is_right
from preprocessing.remove_background import remove_background_and_apply_clahe
from .utils import *
import morphsnakes as ms
import matplotlib.pyplot as plt


def segment_image(img, debug=False):
    """Function to find masses candidates based on linear structuring elements and thresholding

    Parameters
    ----------
    img : numpy array
        Full resolution original with original bit depth.
    num_floors : int
        Number of floors for the gaussian pyramid. 3 is the recommended value
    debug: bool
        Boolean to activate debugging mode

    Returns
    -------
    all_scales : numpy array (3D)
        Output binary images with the regions corresponding to the candidates. Last dimension indicates scale
    """
    # PREPROPCESSING STEP MISSING HERE

    #Image Equalization
    gray_image = cv.cvtColor(bring_to_256_levels(img), cv.COLOR_BGR2GRAY)
    # = cv.equalizeHist(gray_image)

    #Segment with watershed
    roi = watershed_segment(gray_image, debug)

    return roi


def watershed_segment(img, debug=False):
    """

    Parameters
    ----------
    img

    Returns
    -------

    """

    # Get the 3 chanel image
    img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    [markers, sure_fg_val, _] = create_marker(img, debug)

    # Apply watershed to get external pixels
    markers = markers.astype(np.int32)
    markers_base = np.copy(markers)
    markers = cv.watershed(img_color, markers)

    if debug:
        plt.imshow(markers)
        plt.show()


    # Create a new marker template
    markers_base[markers == -1] = np.max(markers_base) + 2
    markers = cv.watershed(img_color, markers_base)

    # Select the central blob
    roi = np.zeros_like(img)
    roi[markers == sure_fg_val] = 1

    #print the segmentation
    if debug:
        imshow_contour(img_color, roi, "Watershed result" )

    return roi


def create_marker(img, debug=False):

    # Get the 3 chanel image
    img_color = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

    # Otsu Thresholding
    blur = cv.GaussianBlur(img, (5, 5), 5)
    _, thres = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

    # sure background area
    kernel = np.ones((15, 15), np.uint8)
    sure_fg = cv.erode(thres, kernel, 3)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)


    if ret == 2:
        sure_bg = cv.dilate(thres, kernel, 3)
        sure_bg = (sure_bg - 1) / 255
        markers[sure_bg == 1] = 2
        sure_fg_val = 1
        sure_bg_val = 2
        markers_out = markers

    else:
        extreme_borders = np.zeros_like(img)
        extreme_borders[0:5, 0:5] = 1
        extreme_borders[-5:-1, -5:-1] = 1
        extreme_borders[0:5, -5:-1] = 1
        extreme_borders[-5:-1, 0:5] = 1

        extreme_borders_markers = extreme_borders * markers

        if np.sum(extreme_borders_markers) == 0:
            background_values = np.array(np.max(markers) + 1)
            markers[extreme_borders] = background_values
            extreme_borders_markers = extreme_borders * background_values

        else:
            extreme_borders_markers = extreme_borders_markers.astype(np.int8)
            background_values = np.unique(extreme_borders_markers)

        sure_bg_val = np.max(extreme_borders_markers)
        sure_fg_val = sure_bg_val +1

        markers_out = np.copy(markers)
        for val in np.unique(markers):
            if val > 0:
                if val in  background_values:
                    markers_out[markers == val] = sure_bg_val
                else:
                    markers_out[markers == val] = sure_fg_val

    if debug:
        plt.imshow(markers_out)
        plt.show()

    return [markers_out, sure_fg_val, sure_bg_val]


def visual_callback_2d(background, fig=None):
    """
    Returns a callback than can be passed as the argument `iter_callback`
    of `morphological_geodesic_active_contour` and
    `morphological_chan_vese` for visualizing the evolution
    of the levelsets. Only works for 2D images.

    Parameters
    ----------
    background : (M, N) array
        Image to be plotted as the background of the visual evolution.
    fig : matplotlib.figure.Figure
        Figure where results will be drawn. If not given, a new figure
        will be created.

    Returns
    -------
    callback : Python function
        A function that receives a levelset and updates the current plot
        accordingly. This can be passed as the `iter_callback` argument of
        `morphological_geodesic_active_contour` and
        `morphological_chan_vese`.
        'https://github.com/pmneila/morphsnakes/blob/master/examples.py'

    """

    # Prepare the visual environment.
    if fig is None:
        fig = plt.figure()
    fig.clf()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(background, cmap=plt.cm.gray)

    ax2 = fig.add_subplot(1, 2, 2)
    ax_u = ax2.imshow(np.zeros_like(background), vmin=0, vmax=1)
    plt.pause(0.001)

    def callback(levelset):

        if ax1.collections:
            del ax1.collections[0]
        ax1.contour(levelset, [0.5], colors='r')
        ax_u.set_data(levelset)
        fig.canvas.draw()
        plt.pause(0.001)

    return callback


def imshow_contour(img_color, thresh, window_name ="Contours"):
    # roi = roi.astype(np.uint8())
    img = np.copy(img_color)
    _, contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv.drawContours(img, contour, -1, (0, 255, 0), 3)

    show_image(bring_to_256_levels(img), window_name)
    cv.waitKey(10)

    return