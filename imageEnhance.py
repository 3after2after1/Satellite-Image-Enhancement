"""
imageEnhance.py

YOUR WORKING FUNCTION

"""
import numpy as np
import cv2
from PIL import Image
from matplotlib import cm
import tensorflow as tf
import os

input_dir = 'input/input'
output_dir = 'output/output'

# you are allowed to import other Python packages above
##########################
def enhanceImage(img):
    # Inputs
    # inputImg: Input image, a 3D numpy array of row*col*3 in BGR format
    #
    # Output
    # outputImg: Enhanced image
    #
    #########################################################################
    # ADD YOUR CODE BELOW THIS LINE
    img = equalize_clahe_color_hsv(img)
    
    output1 = adjust_gamma(img,0.83)
    #for testset 1 --> saturation_factor = 1.65
    #for testset 2 --> saturation_factor = 1.35
    output2 = tf.image.adjust_saturation(output1, 1.65)
    output2 = output2.eval(session=tf.compat.v1.Session())
    cv2.imwrite('bestinput.jpg', output2)
    final = cv2.imread("bestinput.jpg")
    os.remove("bestinput.jpg")
    
    # END OF YOUR CODE
    #########################################################################
    return final
    

def equalize_clahe_color_hsv(img):
    """Equalize the image splitting it after conversion to HSV and applying CLAHE
    to the V channel and merging the channels and convert back to BGR
    """

    cla = cv2.createCLAHE(clipLimit=4.0)
    H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_V = cla.apply(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
    return eq_image

def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)