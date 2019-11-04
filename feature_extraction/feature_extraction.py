import numpy as np
import cv2 as cv
import mahotas as mt
from radiomics import featureextractor
import SimpleITK as sitk
from math import copysign, log10
from skimage.feature import hog
from sklearn.cluster import KMeans
from scipy.spatial import distance
from preprocessing.utils import bring_to_256_levels
import itertools
import imutils


def get_elongation(m):
    """
    Compute elongation from the moments of a shape
    (From: https://stackoverflow.com/questions/14854592/retrieve-elongation-feature-in-python-opencv-what-kind-of-moment-it-supposed-to)
    Parameters
    ----------
    m               Moments

    Returns
    -------
    -               Elongation value
    """

    x = m['mu20'] + m['mu02']
    y = 4 * m['mu11']**2 + (m['mu20'] - m['mu02'])**2
    try:
        el = (x + y**0.5) / (x - y**0.5)
    except:
        el = 0.5
    return el 


def get_num_colors(centers):
    #Centers have two components -> Euclidean distance must be computed
    distances = np.zeros((1,12))
    ii = 0
    for i in range(4): #Always 4 centers
        init = centers[i,:]
        for j in range(4):
            if(i==j):
                continue
            else:
                distances[0,ii] = distance.euclidean(init, centers[j,:])
                ii += 1
    return np.max([1, (1/12)*np.max(distances)])

def get_full_convex_hull(contours, the_image):
    if(len(contours)==0):
        return the_image, 0
    cont = np.vstack(contours[i] for i in range(len(contours)))
    hull = cv.convexHull(cont)
    area = cv.contourArea(hull)
    uni_hull = []
    uni_hull.append(hull) # <- array as first element of list
    cv.drawContours(the_image,uni_hull,-1,255,thickness=cv.FILLED)
    return the_image, area
    
def get_largest_cohesive_area(binary):
    output_image = np.zeros_like(binary)
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(binary, 8)
    sorted_area_idx = np.argsort(stats[:, 4])
    biggest_area_idx = sorted_area_idx[-1] #Largest area index
    biggest_area = stats[biggest_area_idx,4]
    output_image[labels == biggest_area_idx] = 1
    if(np.max(np.multiply(binary, output_image))==0): #If largest connected component corresponds to background
        biggest_area_idx = sorted_area_idx[-2]
        biggest_area = stats[biggest_area_idx,4]
    return biggest_area


def get_geometrical_features(contour):
    """
    Extract geometrical features of a ROI

    Parameters
    ----------
    contour     Contour containing the ROI

    Returns
    -------
    geom_feat      all extracted geometrical features of a ROI
    """

    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)
    ellipse = cv.fitEllipse(contour)
    _, axes, _ = ellipse
    major_axis_length = max(axes)
    minor_axis_length = min(axes)

    compactness = (4*np.pi*area)/(perimeter**2)
    eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)

    # Discrete compactness (https://core.ac.uk/download/pdf/82756900.pdf)
    cd = (4 * area - perimeter) / 2
    cd_min = area-1
    cd_max = (4*area - 4*np.sqrt(area))/2
    cd_n = (cd - cd_min)/(cd_max - cd_min)

    # Elongation
    m = cv.moments(contour)
    elongation = get_elongation(m)
    equi_diameter = np.sqrt(4*area/np.pi)
    geom_feat = np.array([equi_diameter, compactness, elongation, eccentricity, cd_n])
    return geom_feat


def get_color_based_features(roi_color, mask):

    if(np.max(mask)>1): #Normalize if mask is not 0 or 1
        mask = mask/255
        mask = mask.astype(np.uint8)

    roi_color_double = roi_color.astype(float)
    #First, compute number of colors and concentricity accroding to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4739011/
    #First, convert from RGB to CIEl*a*b*
    roi_lab = cv.cvtColor(roi_color, cv.COLOR_BGR2LAB)
    #Separate channels
    _, a_channel, b_channel = cv.split(roi_lab)

    #Now cluster channels a and b for 4 clusters, using k-means
    a_masked = np.multiply(a_channel, mask)
    b_masked = np.multiply(b_channel, mask)
    a_pixels = a_masked[mask>0] #Get pixels from channel a to cluster
    b_pixels = b_masked[mask>0] #Get pixels from channel b to cluster

    data_to_cluster = np.column_stack((a_pixels, b_pixels.T)) #Build vector to cluster

    #Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=0).fit(data_to_cluster)

    #Add one so that labels start from 1
    labels = kmeans.labels_ + 1
    centers = kmeans.cluster_centers_

    #Get number of colors of the lesion
    num_colors = get_num_colors(centers)

    #Build clustered image
    clustered = np.zeros_like(a_channel)
    a_pixels_unique = np.unique(a_pixels)
    for i in range(len(a_pixels_unique)):
        locs = np.where(a_masked==a_pixels_unique[i])
        clustered[locs] = labels[a_pixels == a_pixels_unique[i]]

    #Generate 4 images (One for each color)
    c0 = np.zeros_like(clustered) #Corresponding to label 1, not 0 because 0 means background
    c1 = c0.copy()
    c2 = c1.copy()
    c3 = c2.copy()

    c0[clustered==1] = 1
    _, contours_c0, _ = cv.findContours(c0, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    full_ch_c0, area_ch_c0 = get_full_convex_hull(contours_c0, c0.copy())

    c1[clustered==2] = 1
    _, contours_c1, _ = cv.findContours(c1, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    full_ch_c1, area_ch_c1 = get_full_convex_hull(contours_c1, c1.copy())

    c2[clustered==3] = 1
    _, contours_c2, _ = cv.findContours(c2, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    full_ch_c2, area_ch_c2 = get_full_convex_hull(contours_c2, c2.copy())

    c3[clustered==4] = 1
    _, contours_c3, _ = cv.findContours(c3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    full_ch_c3, area_ch_c3 = get_full_convex_hull(contours_c3, c3.copy())

    #Organize images and convex hulls according to areas of the different clusters
    full_c = np.array([c0, c1, c2, c3]) #First dimension is color
    all_full_ch = np.array([full_ch_c0, full_ch_c1, full_ch_c2, full_ch_c3])
    sorted_areas_ch = np.argsort([area_ch_c0, area_ch_c1, area_ch_c2, area_ch_c3])
    S = full_c[sorted_areas_ch, :,:]
    ch = all_full_ch[sorted_areas_ch,:,:]

    #Compute PA
    nCCmax = get_largest_cohesive_area(S[0, :, :]) #S[0] -> S1
    nS1 = len(S[0,:,:][S[0,:,:]>0])
    PA = nCCmax/nS1

    #Compute CI
    S1andHull = cv.bitwise_and(S[0,:,:], ch[1,:,:])
    nS1andHull = len(S1andHull[S1andHull>0])
    CI = nS1andHull/nS1

    #Compute HE
    S2andCore = cv.bitwise_and(S[1,:,:], ch[0,:,:])
    nS2andCore = len(S2andCore[S2andCore>0])
    nS2 = len(S[1,:,:][S[1,:,:]>0])
    HE = 1 - nS2andCore/nS2

    #Compute concentricity
    concentricity = PA*CI*HE

    #Now compute other color features according to Celebi2007
    all_color_spaces = np.zeros(( roi_color.shape[0], roi_color.shape[1], 18), dtype = np.float32) #18 because there are 6 color spaces with 3 channels each
    all_color_spaces[:,:,0:3] = roi_color_double
    #Compute normalized RGB
    the_sum = roi_color_double[:,:,2] + roi_color_double[:,:,1] + roi_color_double[:,:,0] + 0.00001 
    all_color_spaces[:,:,3] = roi_color_double[:,:,2]/the_sum*255.0
    all_color_spaces[:,:,4] = roi_color_double[:,:,1]/the_sum*255.0
    all_color_spaces[:,:,5] = roi_color_double[:,:,0]/the_sum*255.0
    #Compute HSV
    all_color_spaces[:,:,6:9] = cv.cvtColor(roi_color, cv.COLOR_BGR2HSV).astype(np.float32)
    #Compute CIEluv
    all_color_spaces[:,:,9:12] = cv.cvtColor(roi_color, cv.COLOR_BGR2Luv).astype(np.float32)
    #Compute Ohta (I1/2/3)
    all_color_spaces[:,:,12] = (1/3)*roi_color_double[:,:,2] + (1/3)*roi_color_double[:,:,1] + (1/3)*roi_color_double[:,:,0] 
    all_color_spaces[:,:,13] = (1/2)*roi_color_double[:,:,2] + (1/2)*roi_color_double[:,:,0] 
    all_color_spaces[:,:,14] = (-1)*(1/4)*roi_color_double[:,:,2] + (1/2)*roi_color_double[:,:,1] - (1/4)*roi_color_double[:,:,0]
    #Compute l1/2/3
    denominator_l = (roi_color_double[:,:,2] - roi_color_double[:,:,1])**2+(roi_color_double[:,:,2] - roi_color_double[:,:,0])**2+(roi_color_double[:,:,1] - roi_color_double[:,:,0])**2 + 0.00001
    all_color_spaces[:,:,15] = (roi_color_double[:,:,2] - roi_color_double[:,:,1])**2/(denominator_l)
    all_color_spaces[:,:,16] = (roi_color_double[:,:,2] - roi_color_double[:,:,0])**2/(denominator_l)
    all_color_spaces[:,:,17] = (roi_color_double[:,:,2] - roi_color_double[:,:,0])**2/(denominator_l)

    #Compute mean and std for each channel of each color space
    means_and_stds = np.zeros((1,18*2))
    ii = 0
    for i in range(all_color_spaces.shape[2]):
        means_and_stds[0,ii] = np.mean(all_color_spaces[:,:,i])
        means_and_stds[0,ii+1] = np.std(all_color_spaces[:,:,i])
        ii += 2

    return np.concatenate((np.array([num_colors, concentricity]), means_and_stds.reshape(means_and_stds.shape[1],)))



def get_texture_features(roi_gray, mask):
    """
    Extract texture features from ROI

    Parameters
    ----------
    roi_gray            Region of interest of the image (gray scale)
    mask                Binary version of the ROI

    Returns
    -------
    texture_features    All extracted texture features of a ROI
    """
    # First, get GLRLM features
    data_spacing = [1, 1, 1]

    # Convert numpy arrays to sitk so that extractor.execute can be employed for GLRLM features
    sitk_img = sitk.GetImageFromArray(roi_gray)
    sitk_img.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2])))
    sitk_img = sitk.JoinSeries(sitk_img)

    mask_mod = mask + 1
    sitk_mask = sitk.GetImageFromArray(mask_mod)
    sitk_mask.SetSpacing((float(data_spacing[0]), float(data_spacing[1]), float(data_spacing[2])))
    sitk_mask = sitk.JoinSeries(sitk_mask)
    sitk_mask = sitk.Cast(sitk_mask, sitk.sitkInt32)

    # Parameters for radiomics extractor
    params = {}
    params['binWidth'] = 20
    params['sigma'] = [1, 2, 3]
    params['verbose'] = True

    # For GLRLM features
    extractor = featureextractor.RadiomicsFeatureExtractor(**params)
    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('glrlm')

    result = extractor.execute(sitk_img, sitk_mask)
    glrlm_features_list = []
    for key, value in result.items():
        if 'glrlm' in key:
            glrlm_features_list.append(value.item())

    glrlm_features = np.array(glrlm_features_list)

    masked_roi = bring_to_256_levels(np.multiply(roi_gray, mask))
    textures = mt.features.haralick(masked_roi, ignore_zeros=True)
    haralick_features = textures.mean(axis=0)

    texture_features = np.concatenate((haralick_features, glrlm_features), axis=0)

    return texture_features


def get_texture_geometrical_and_asymetry_features(roi_gray, cnt, mask):

    #First, extract features for asymmetry
    num_rows = mask.shape[0]
    num_cols = mask.shape[1]
    total_area = len(mask[mask>0])
    center_row = int(np.floor(num_rows/2))
    center_col = int(np.floor(num_cols/2))

    if(len(cnt)<5): #If less than 5 points in contour, generate a contour with 8 points (rectangle)
        cnt = np.array([[center_row-np.floor(center_row/2), center_col-np.floor(center_col/2)], \
                        [center_row-np.floor(center_row/2), center_col], \
                        [center_row-np.floor(center_row/2), center_col+np.floor(center_col/2)], \
                        [center_row, center_col+np.floor(center_col/2)], \
                        [center_row+np.floor(center_row/2), center_col+np.floor(center_col/2)], \
                        [center_row+np.floor(center_row/2), center_col], \
                        [center_row+np.floor(center_row/2), center_col-np.floor(center_col/2)],
                        [center_row, center_col-np.floor(center_col/2)]], dtype = np.int32)
        cnt = cnt[:,np.newaxis,:] #Add new axis so that it's similar to normal contours
    geom_features = get_geometrical_features(cnt)
    texture_features = get_texture_features(roi_gray, mask)

    try:
        (x,y),(MA,ma),angle = cv.fitEllipse(cnt)
    except: #If some error makes cnt to have too few points
        x = center_col
        y = center_row
        angle = 90
    x_int = int(np.floor(x))
    y_int = int(np.floor(y))
    mMA = np.tan(-angle*(np.pi)/180)
    mma = np.tan(-angle*(np.pi)/180 - np.pi/2)
    output_images = [mask.copy() for i in range(8)] # 8 images to be generated
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if(mMA*(i-y) + x > j):
                output_images[0][i,j] = 0
                output_images[4][i,j] = 0
                output_images[6][i,j] = 0
            else:
                output_images[1][i,j] = 0
                output_images[5][i,j] = 0
                output_images[7][i,j] = 0
            if(mma*(i-y) + x > j):
                output_images[2][i,j] = 0
                output_images[4][i,j] = 0
                output_images[7][i,j] = 0
            else:
                output_images[3][i,j] = 0
                output_images[5][i,j] = 0
                output_images[6][i,j] = 0

    texture_features_partial_area = np.zeros((len(texture_features), 8))
    geom_features_partial_area = np.zeros((len(geom_features), 8))


    try:
        for i in range(8):
            _, curr_cnt, _ = cv.findContours(output_images[i].astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            texture_features_partial_area[:,i] = get_texture_features(roi_gray, output_images[i])/texture_features
            geom_features_partial_area[:,i] = get_geometrical_features(curr_cnt[0])/geom_features
    except: #If something goes wrong with the generation of the images, just multiply original ones by angle in rads
        for i in range(8):
            texture_features_partial_area[:,i] = texture_features*angle*(np.pi)/180
            geom_features_partial_area[:,i] = geom_features*angle*(np.pi)/180

    #Now extract asymmetry index (Stoecker)
    T = np.float32([[1, 0, center_col-y_int], [0, 1, center_row-x_int ]]) 
    translated = cv.warpAffine(mask.copy(), T, (mask.shape[1], mask.shape[0])) 
    rotated = imutils.rotate(translated, angle+90)
    upper = rotated[:center_row,:]
    bottom = rotated[center_row:,:]
    left = rotated[:,:center_col]
    right = rotated[:,center_col:]
    flipped_updown = cv.flip(upper, 0)
    flipped_lr = cv.flip(left,1)
    #solve differences in size
    if(bottom.shape == flipped_updown.shape):
        diff_vert = bottom - flipped_updown
    else:
        if(bottom.shape[0]>flipped_updown.shape[0]): #If upper has more rows
            diff_vert = bottom[:flipped_updown.shape[0],:] - flipped_updown
        else:
            diff_vert = bottom - flipped_updown[:bottom.shape[0],:]
    if(right.shape == flipped_lr.shape):
        diff_hor = right - flipped_lr
    else:
        if(right.shape[1]>flipped_lr.shape[1]):
            diff_hor = right[:, :flipped_lr.shape[1]] - flipped_lr
        else:
            diff_hor = right - flipped_lr[:,:right.shape[1]]
    area_diff_vert = len(diff_vert[diff_vert>0])
    area_diff_hor = len(diff_hor[diff_hor>0])
    if(area_diff_hor>area_diff_vert):
        asymmetry_index = area_diff_vert/total_area
    else:
        asymmetry_index = area_diff_hor/total_area

    return np.concatenate((np.array([asymmetry_index]), texture_features, geom_features, texture_features_partial_area.flatten(), geom_features_partial_area.flatten()))



def feature_hu_moments(contour):
    """
    Calculate the shape features based on HU moments

    Parameters
    ----------
    bin_roi     binary version of the ROI

    Returns
    -------
    hu_moments  numpy array containing the HU moments

    """

    hu_moments = cv.HuMoments(cv.moments(contour))
    # Log scale transform
    for i in range(0, 7):
        try:
            hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]))
        except:
            hu_moments[i] = 1
    return hu_moments.reshape(-1)


def multi_scale_lbp_features(roi):
    """
    Calculate multi scale local binary pattern based on mahotas
    Parameters
    ----------
    roi:        numpy array of The region of interest(8-bits)

    Returns
    -------
    lbp         Histogram of multi scale lbp
    """
    roi_img = cv.normalize(roi, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
    roi_or = np.copy(roi_img)
    r = 1
    R = 1
    i = 0
    lbp = np.zeros((5, 36))
    while R < 35:
        lb_p = mt.features.lbp(roi_img, np.rint(R), 8, ignore_zeros=False)
        lb_p = lb_p / np.sum(lb_p)
        lbp[i, :] = lb_p
        r_1 = r
        r = r_1 * (2 / (1 - np.sin(np.pi / 8)) - 1)
        R = (r + r_1) / 2
        if np.floor(r - r_1) % 2 == 0:
            k_size = np.int(np.ceil(r - r_1))
        else:
            k_size = np.int(np.floor(r - r_1))
        std_dev = (k_size / 2)
        roi_img = cv.GaussianBlur(roi_or, (k_size, k_size), std_dev)
        i += 1

    return lbp.reshape((180,))


def features_hog(roi):
    """
    Calculate a scale based Histogram of Oriented gradients based on ski-image

    Parameters
    ----------
    roi             numpy array of The region of interest(8-bits)

    Returns
    -------
    hog features    numpy array containing a HOG descriptor of the image

    """
    width = np.int(roi.shape[0] / 10)
    height = np.int(roi.shape[1] / 10)
    w_t = np.int((roi.shape[0] - width * 10) / 2)
    h_t = np.int((roi.shape[1] - height * 10) / 2)
    crop_roi = roi[w_t: w_t + 10*width, h_t: h_t + 10*height]
    f_hog = hog(crop_roi, orientations=8, pixels_per_cell=(width, height),
                cells_per_block=(1, 1), visualize=False, multichannel=False)
    return f_hog


def extract_features(roi_color, contour, mask):
    """
    Extract all the features of a ROI. A total of 41 features are extracted. LBP and HoG are not activated

    Parameters
    ----------
    roi_color           Region of interest of the image (RGB)
    contour             Contour containing the ROI
    mask                Binary version of the ROI

    Returns
    -------
    feature_vector      All extracted features of a ROI
    """

    #Re-estimate contour of shape
    _, contours, _ = cv.findContours(mask.astype(np.uint8), cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    contour = contours[0]

    try:
        #Geometrical features
        geom_features = get_geometrical_features(contour)
        # Hu Moments
        hu_moments = feature_hu_moments(contour)
    except:
        geom_features = np.zeros((5,))
        hu_moments = np.zeros((7,))

    roi_gray = cv.cvtColor(roi_color, cv.COLOR_BGR2GRAY)

    # Color based features
    color_features = get_color_based_features(roi_color, mask)

    # LBP
    lbp = multi_scale_lbp_features(roi_gray)

    # Texture
    texture_features = get_texture_features(roi_gray, mask)

    # HOG features
    hog_features = features_hog (roi_gray)

    return np.transpose(np.concatenate((geom_features, texture_features, color_features, hu_moments, lbp, hog_features), axis=0))
