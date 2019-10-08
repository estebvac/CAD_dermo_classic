import cv2 as cv
from preprocessing.is_right import is_right
from preprocessing.remove_background import remove_background_and_apply_clahe
from .utils import *


def find_candidates(img, num_floors, debug=False):
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

    #Remove background from the image and apply CLAHE
    filtered_image = remove_background_and_apply_clahe(bring_to_256_levels(img)) #This returns the filtered image

    #Get side of the breast in the image
    if(is_right(filtered_image)):
        side = "r"
    else:
        side = "l"

    if debug:
        scaling_factor = 0.95
        show_image(filtered_image, "After removing background and CLAHE", 0.2)
        cv.waitKey(0)

    full_res_rows = filtered_image.shape[0]
    full_res_cols = filtered_image.shape[1]

    # Step 1: Gaussian pyramidal decomposition
    for i in range(num_floors):
        low_res = cv.pyrDown(filtered_image)
        filtered_image = low_res

    #Get mask of the breast
    img_bin = get_breast_mask(bring_to_256_levels(low_res)) #This returns a mask of the breast
    if debug:
        show_image(img_bin, "Mask", scaling_factor)
        cv.waitKey(0)

    orig_rows = low_res.shape[0]
    orig_cols = low_res.shape[1]
    #Crop image so that only breast region is analyzed
    x_b, y_b, w_b, h_b = cv.boundingRect(img_bin)
    boundaries = [x_b, y_b, w_b, h_b]
    # Crop the image
    breast_region = low_res[y_b:y_b + h_b, x_b:x_b + w_b]
    to_add = 300 #Number of pixels to add to the image

    if debug:
        show_image(breast_region, "Cropped image", scaling_factor)
        cv.waitKey(0)

    #Here we need an additional step to avoid linear SEs to give a wrong image
    #The proposed solution is to expand the image
    if(side == "l"):
        left_d = to_add
        right_d = 0
    else:
        left_d = 0
        right_d = to_add

    expanded_image = cv.copyMakeBorder(breast_region, 0, 0, left_d, right_d, cv.BORDER_REPLICATE)
    if debug:
        show_image(expanded_image, "Expanded image", scaling_factor)
        cv.waitKey(0)

    #Carefully chose the sizes for the linear structuring elements
    d1 = int(62/pow(2, num_floors-1))
    if d1%2 ==0 :
        d1 = d1+1
    d2 = int(482/pow(2, num_floors-1))
    if d2%2 == 0:
        d2 = d2+1

    #Generate sizes for d2
    big_lines = np.arange(d1, d2, 15, dtype = np.uint8)

    #Create output image
    output_image = np.zeros((orig_rows, orig_cols,1), dtype=np.uint8)

    for curr_big_line in big_lines:

        #Define d1 as 0.6*d2 (Several values were tested and this one provided a better representation)
        small_lines = int(curr_big_line*(0.6))

        #Make d1 odd:
        if(small_lines%2 == 0):
            small_lines = small_lines+1

        #Define array for storing the output image for the current scale
        total = np.zeros(expanded_image.shape)

        #For each angle of the structuring element, apply tophat followed by opening
        for i in range(12):
            curr_big = cv.morphologyEx(expanded_image, cv.MORPH_TOPHAT, getLinearSE(curr_big_line, i+1))
            curr_small = cv.morphologyEx(curr_big, cv.MORPH_OPEN, getLinearSE(small_lines, i+1))
            total = total + cv.GaussianBlur(curr_small, (5,5), 0)

        #Recover image (remove expansion)
        if(side == "l"):
            total = total[:, to_add:]
        else:
            total = total[:, :(total.shape[1] - to_add)]

        #Show image if needed
        if debug:
            show_image(bring_to_256_levels(total), "After linear SE", scaling_factor)
            cv.waitKey(0)

        #Normalize to 0-255
        total = bring_to_256_levels(total)

        #Compute Otsu's threshold
        threshold_value, _ = cv.threshold(total , 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

        #Create 16 threshold values (from the otsu threshold to the maximum)
        steps = np.linspace(threshold_value, 254, 16)

        #Define image for temporary thresholded images and define connectivity
        region_with_candidates = np.zeros(total.shape)
        connectivity = 8
        #Apply thresholds
        for curr_thresh in steps:
            #Apply current threshold
            ret, thresholded_image = cv.threshold(total, int(curr_thresh), 255, cv.THRESH_BINARY)
            thresholded_image = fill_holes(thresholded_image)
            analysed_image = discard_regions_processing(thresholded_image, connectivity, small_lines, curr_big_line)
            region_with_candidates = region_with_candidates.astype(np.uint8) | analysed_image.astype(np.uint8)

        if debug:
            show_image(bring_to_256_levels(region_with_candidates), "After MLT", scaling_factor)
            cv.waitKey(0)

        #Reconstruct image with the size of the original one (before cropping)
        complete_image_curr_scale = np.zeros_like(low_res, dtype = np.uint8) #((orig_rows, orig_cols), dtype = np.uint8)
        complete_image_curr_scale[y_b:y_b + h_b, x_b:x_b + w_b] = region_with_candidates
        if debug:
            show_image(bring_to_256_levels(complete_image_curr_scale), "Current scale candidates", scaling_factor)

        #Store results 3D image
        if(complete_image_curr_scale.min() != complete_image_curr_scale.max()):
            output_image = np.dstack((output_image, complete_image_curr_scale.astype(np.uint8)))

    #Remove first element, which contains only zeros
    output_image = output_image[:,:,1:]

    if debug:
        for i in range(output_image.shape[2]):
            show_image(bring_to_256_levels(output_image[:,:,i]), "Output image at scale " + str(i), scaling_factor)
            cv.waitKey(0)

    #Build full resolution images
    all_scales = np.zeros((full_res_rows, full_res_cols, output_image.shape[2]), dtype = np.uint8)
    for i in range(all_scales.shape[2]):
        curr_image = output_image[:,:,i]
        all_scales[:,:,i] = cv.resize(curr_image, (full_res_cols, full_res_rows))

    #all_scales is what I need to return
    return all_scales
