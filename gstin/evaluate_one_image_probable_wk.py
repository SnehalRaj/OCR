## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

'''
This file takes a cropped image as input(gstin box) and returns the values for (w,k) for which the 




'''
# Import packages
import os
import math
import cv2
import numpy as np
import tensorflow as tf
import sys
import datetime
import time
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
# import imbinarize
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
now = datetime.datetime.now()
# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
import re
import pytesseract
from PIL import Image, ImageEnhance
import argparse
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)
from scipy.misc import imsave
#time=datetime.datetime.now()
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
    help="path to input image")
ap.add_argument("-w","--word",type=str,
    help="expected registration number")
args = vars(ap.parse_args())
# print (args)

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = args["image"]
word = args["word"]
CWD_PATH=os.getcwd()
# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Grab path to current working directory
CWD_PATH = os.getcwd()
# Number of classes the object detector can identify
def correctnum(ch):

    if ch=="a" or ch=="A":  # a =2
        return "4"
    elif ch=="b" or ch=="B":
        return "8"
    elif ch=="d" or ch=="D":
        return "0"
    elif ch=="f" or ch=="F":
        return "7"
    elif ch=="g" or ch=="G":
        return "6"
    elif ch=="h" or ch=="H": #h="23"
        return "8"
    elif ch=="i" or ch=="I":
        return "1"
    elif ch=="j" or ch=="L":
        return "1"
    elif ch=="l" or ch=="L":
        return "1"
    elif ch=="o" or ch=="O":
        return "0"
    elif ch=="q" or ch=="Q":
        return "0"
    elif ch=="r" or ch=="R":
        return "8"
    elif ch=="s" or ch=="S":
        return "5"
    elif ch=="T":
        return "7"
    elif ch=="z" or ch=="Z":
        return "2"
    else :
        return ch

def correctchar(ch):
    if ch=="0":
        return "O"
    elif ch=="1":
        return "I"
    elif ch=="2":
        return "Z"
    elif ch=="3":
        return "B"
    elif ch=="4":
        return "A"
    elif ch=="5":
        return "S"
    elif ch=="6":
        return "G"
    elif ch=="7":
        return "Z"
    elif ch=="8":
        return "B"
    elif ch=="9":
        return "O"
    elif ch==')':
        return "J"
    else:
        return ch


def process(number):
    if len(number) <15:
        return number
    num = number[-15:]
    stra = ""
    for i in range(15):
        if i==0 or i==1 or i==7 or i==8 or i==9 or i==10 or i==12:
            stra=stra+correctnum(num[i])
        elif i==2 or i==3 or i==4 or i==5 or i==6 or i==11 :
            stra=stra+correctchar(num[i])
        elif i==13: 
            stra=stra+"Z"
        else:
            stra=stra+num[-1:]
    return stra
def FindText(img,word):
    img = img.convert('L')
    # text = tools.image_to_string(img,builder=pyocr.builders.DigitBuilder())
    # print("Image before processing")
    # print(img)    
    text = pytesseract.image_to_string(img)
    text=process(text)
    # print(text)
    searchObj1 = re.search( word, text)
    if searchObj1:
        # print(searchObj1.group()) 
        return 1,""
    else:
        return 0,text

##########################################################################################################################################################
def binarize_image(img_path, threshold=110):
    """Binarize an image."""
    image_file = img_path
    image = image_file.convert('L')  # convert image to monochrome
    image = np.array(image)
    image = binarize_array(image, threshold)
    image = Image.fromarray(image)
    return image


def binarize_array(numpy_array, threshold):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > threshold:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input",
                        dest="input",
                        help="read this file",
                        metavar="FILE",
                        required=True)
    parser.add_argument("-o", "--output",
                        dest="output",
                        help="write binarized file hre",
                        metavar="FILE",
                        required=True)
    parser.add_argument("--threshold",
                        dest="threshold",
                        default=200,
                        type=int,
                        help="Threshold when to show white")
    return parser

##########################################################################################################################################################
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
def sauvola(img):
    return threshold_sauvola(np.asarray(cropped,dtype="int64"), window_size=5, k=0.05)    
##########################################################################################################################################################
check=0
t1 = time.time()
win=3
kthresh=0.05

img = mpimg.imread(PATH_TO_IMAGE)   
image = rgb2gray(img)
while kthresh<=0.15:
    win=3
    while win<=99:
        thresh_sauvola = threshold_sauvola(image, window_size=win, k=kthresh)  
        binary_sauvola = image > thresh_sauvola
        plt.imsave("122.png",binary_sauvola, cmap=plt.cm.gray)  
        cropped = cv2.imread("122.png")
        cropped = Image.fromarray(cropped)
        # cropped.show()       
        flag,out = FindText(cropped,word)
        if flag==1:
            print(IMAGE_NAME+","+str(win)+","+str(kthresh))
        win = win + 2
    kthresh = kthresh + 0.01
# thresh_sauvola = threshold_sauvola(image, window_size=29, k=0.07)  
# binary_sauvola = image > thresh_sauvola
# plt.imsave("122.png",binary_sauvola, cmap=plt.cm.gray)  
# cropped = cv2.imread("122.png")
# cropped = Image.fromarray(cropped)
# # cropped.show()       
# flag,out = FindText(cropped,word)
# if flag is 0:
    # print(out,word)    
cv2.destroyAllWindows()
