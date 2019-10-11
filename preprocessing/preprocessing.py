# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:04:04 2019

@author: Sergio814
"""


import numpy as np
import cv2 as cv
import time
from utils import bring_to_256_levels, getLinearSE

def discard_small(input_image, connectivity):
    """Function to discard small connected components from binary image

    Parameters
    ----------
    input_image : numpy array
        Binary image.

    connectivity : int
        Type of connectivity to be used (4 or 8)

    Returns
    -------
    output_image : numpy array
        Binary image without small components
    """
    output_image = np.zeros(input_image.shape)
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(input_image, connectivity)
    print('Num labels: ', nlabels)
    #TODO: Include info about shape of the element
    for i_label in range(nlabels):
        if(stats[i_label, 4] >= 50): #Only structures bigger than 50 pixels
            output_image[labels==i_label] = 255
    return output_image.astype(np.uint8)

def only_hair(binarized):
    """Function to discard elements that are not hair-like by means of area and circularity from binary image

    Parameters
    ----------
    binarized : numpy array
        Input binary image.

    Returns
    -------
     : numpy array
        Image with structures that are most likely to be hair according to area and circularity
    """
    output = np.zeros_like(binarized)
    _, contours, _ = cv.findContours(binarized, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    #Analyze every contour
    for i_cnt in range(len(contours)):
        cnt = contours[i_cnt]
        area_cnt = cv.contourArea(cnt)
        perimeter_cnt = cv.arcLength(cnt, True)
        if(perimeter_cnt==0): 
            continue
        else:
            compactness_cnt = (4*np.pi*area_cnt)/(perimeter_cnt**2)
        if(area_cnt > 50 and compactness_cnt < 0.3):
            cv.drawContours(output, [cnt], 0, color=255, thickness=-1)
    return cv.bitwise_and(output.astype(np.uint8), binarized).astype(np.uint8)

        

def applyCriteria(base_image, to_check, connectivity):
    """Function to perform final step of the hair removal process. It analyzes input structures and analyzes them
        in comparizon with the base image to_check, which contains the pixels that are most probable to be hairs

    Parameters
    ----------
    base_image : numpy array
        Binary image with pixels that are most likely to contain hair.
    to_check : numpy array
        Input binary image to be analyzed. Only elements covered at least 30% by base_image are kept
    connectivity : int
        Connectivity to be used (4 or 8)
    Returns
    -------
    output_image : numpy array
        Output image after applying criteria
    """
    output_image = np.zeros(to_check.shape)
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(to_check, connectivity)
    print('Num labels: ', nlabels)
    for i_label in range(nlabels):
        num_pixels_element = np.count_nonzero(labels==i_label) #Number of pixels with current label
        one_element = np.zeros_like(base_image)
        one_element[labels==i_label] = 255
        and_result = np.bitwise_and(one_element.astype(np.uint8), base_image)
        num_pixels_element_after_and = np.count_nonzero(and_result>0)
        if(num_pixels_element_after_and > np.floor(0.3*num_pixels_element)): #If after AND at least 30% of the pixels remain, add element to result
            output_image[labels==i_label] = 255

    return output_image.astype(np.uint8)


def preprocess_and_remove_hair(img):
    """Function normalize image to 0-255 range (8 bits)

    Parameters
    ----------
    img : numpy array
        Original image (RGB).

    Returns
    -------
    inpainted : numpy array
        Gray scale image with removed hairs and median-filtered. 
    """
    #Apply median filter 
    img_after_median = cv.medianBlur(img, 5)

    #Convert to gray
    img_gray = cv.cvtColor(img_after_median, cv.COLOR_BGR2GRAY)

    #Add artificial hair
    img_gray[:70,480:483] = max(np.min(img_gray) - 10, 0)

    #Apply bottom hat operation
    bottom_hatted1 = cv.morphologyEx(img_gray, cv.MORPH_BLACKHAT, cv.getStructuringElement(cv.MORPH_ELLIPSE,(7,7)))
    inverted1 = cv.bitwise_not(bottom_hatted1)

    #TODO: Check whether there is hair or not. Temporary solution: hair added to all images
    #first_se_length = 30
    #for i in range(12):
    #    opened = cv.morphologyEx(bottom_hatted1, cv.MORPH_OPEN, getLinearSE(first_se_length, i+1))

    #Intermediate step: opening on bottom hatted image with linear SE
    total1 = np.zeros_like(bottom_hatted1)
    connectivity = 8
    se_lengths1 = [3]
    for se_l in se_lengths1:
        for i in range(12):
            temp = cv.morphologyEx(bottom_hatted1, cv.MORPH_TOPHAT, getLinearSE(se_l, i+1))
            #threshold, binarized = cv.threshold(bring_to_256_levels(temp), 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            binarized = cv.adaptiveThreshold(bring_to_256_levels(temp),255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, -5)
            #binarized = cv.adaptiveThreshold(bring_to_256_levels(temp),255,cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 21, -15)
            corrected = only_hair(binarized) 
            total1 = cv.bitwise_or(total1.astype(np.uint8), corrected)

    dilated = cv.morphologyEx(total1, cv.MORPH_DILATE, getLinearSE(7, i+1)) 

    #Apply bottom hat operation
    bottom_hatted2 = cv.morphologyEx(img_gray, cv.MORPH_BLACKHAT, cv.getStructuringElement(cv.MORPH_ELLIPSE,(31,31)))
    inverted2 = cv.bitwise_not(bottom_hatted2)

    total2 = np.zeros_like(bottom_hatted2) 
    connectivity = 8
    se_lengths2 = [21]
    for se_l in se_lengths2:
        for i in range(12):
            temp = cv.morphologyEx(bottom_hatted2, cv.MORPH_TOPHAT, getLinearSE(se_l, i+1))
            binarized = cv.adaptiveThreshold(bring_to_256_levels(temp),255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 21, -10)
            corrected = only_hair(binarized) 
            total2 = cv.bitwise_or(total2, applyCriteria(dilated.astype(np.uint8), corrected, connectivity))
            
    #Refinement
    binary = cv.morphologyEx(total2, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3,3)))

    #Dilate final result so that inpainting works better
    binary = cv.morphologyEx(binary, cv.MORPH_DILATE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3)))

    #Apply inpainting
    inpainted = cv.inpaint(img_gray, binary, 15, cv.INPAINT_TELEA)

    return inpainted, binary

