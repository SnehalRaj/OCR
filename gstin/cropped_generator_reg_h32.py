## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py


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
args = vars(ap.parse_args())
# print (args)

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph'
IMAGE_NAME = args["image"]

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,'inference_graph_gstin/frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training_gstin','labelmap.pbtxt')

# Path to image
PATH_TO_IMAGE = os.path.join(CWD_PATH,IMAGE_NAME)

# Number of classes the object detector can identify
NUM_CLASSES = 7



# Load the label map.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
def rotate(image, center = None, scale = 1.0):
    angle=360-int(re.search('(?<=Rotate: )\d+', pytesseract.image_to_osd(image)).group(0))
    print("angle is",angle)
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def FindText(img,key):
    img = img.convert('L')
    # text = tools.image_to_string(img,builder=pyocr.builders.DigitBuilder())
    # print("Image before processing")
    # print(img)    
    text = pytesseract.image_to_string(img)
    # print(text)
    searchObj1 = re.search( r'[0-9][0-9][A-Z][A-Z][A-Z][A-Z][A-Z][0-9][0-9][0-9][0-9][A-Z][A-Z0-9][Z][A-Z0-9]', text)
    if searchObj1:
        return searchObj1.group(),1
    else:
        return "",0


def FindDOB(img,key):
    img=img.convert('L')
    text=pytesseract.image_to_string(img)
    if key==0:
        searchObj11 = re.search(r'[D][O][B]',text)
        searchObj12 = re.search(r'[Y][e][a][r] [o][f] [B][i][r][t][h]',text)
        if searchObj11 :
            searchObj1 = re.search( r'[0-3][0-9][/][0-1][0-9][/][1-2][0-9][0-9][0-9]', text)
            # print(int(searchObj1[-4:]))
            # print(type(searchObj1))
            if searchObj1:
                # print("DOB found: ",searchObj1.group())
                return searchObj1.group(),1
            else:
                return "",0
        elif searchObj12 :
            searchObj2 = re.search( r'[1-2][0-9][0-9][0-9]',text)
            if searchObj2:
                # print("DOB found: ",searchObj2.group())
                return searchObj2.group(),2
            else:
                return "",0
        else:
            # print("DOB not found in Aadhar!!!")
            return "",0
    elif key==1:
        searchObj1 = re.search( r'[0-3][0-9][/][0-1][0-9][/][1-2][0-9][0-9][0-9]', text)
        if searchObj1:
            # print("DOB found: ",searchObj1.group())
            return searchObj1.group(),1
        else:
            # print("DOB not found in PAN!!!")
            return "",0

def FindGEN(img):
    img=img.convert('L')
    text=pytesseract.image_to_string(img)
    searchObj1 = re.search( r'[M][A][L][E]', text)
    searchObj2 = re.search( r'[F][E][M][A][L][E]', text)
    searchObj3 = re.search(r'[M][a][l][e]',text)
    searchObj4 = re.search( r'[F][e][m][a][l][e]', text)

    if searchObj2 or searchObj4:
        # print("Gender found: Female")
        return "Female",1
    elif searchObj1 or searchObj3:
        # print("Gender found: male")
        return "Male",1
    else:
        # print("Gender not detected from Aadhar Card!!!")
        return "",0

def FindNAME(img,key):
    img=img.convert('L')
    text=pytesseract.image_to_string(img)
    searchObj1 = re.findall( r'.{1,100}\n',text)
    searchObj2 = re.findall( r'.{1,100}\n',text)
    if key is 0:
        # print(searchObj1)
        name=(searchObj1[1]).splitlines()
        return  (name[0]),0
    else:
        # print(searchObj2)
        name=(searchObj2[2]).splitlines()
        fname=(searchObj2[3]).splitlines()
        return name[0],fname[0],0


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

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV and
# expand image dimensions to have shape: [1, None, None, 3]
# i.e. a single-column array, where each item in the column has the pixel RGB value
image = cv2.imread(PATH_TO_IMAGE)
# image2 = cv2.imread(PATH_TO_IMAGE,0)
im1= np.copy(image)
image_expanded = np.expand_dims(image, axis=0)

# Perform the actual detection by running the model with the image as input
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

tempConfidCard = 0
tempConfidNum = 0
numb = ""
Cardtype = ""
x,y = vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.30)

print(y)
for k,v in y.items():
    words = v[0].split() #split the sentence into individual words
    # print(words)
    confid = int(words[-1][:-1])
    ymin, xmin, ymax, xmax = k
    
    image_pil = Image.fromarray(np.uint8(im1)).convert('RGB')
    # image_pil = Image.fromarray(rgb2gray(im1))
    im_width, im_height = image_pil.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,ymin * im_height, ymax * im_height)
    cropped_new = image_pil.crop( ( left*0.98, top*1.0, right*1.02, bottom*1.0))
    img = cropped_new
    hsize = 32
    wpercent = (hsize/float(img.size[1]))
    basewidth = int((float(img.size[0])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    cropped = img
    if "reg:" in words: #see if one of the words in the sentence is the word we want  
        thresh_sauvola = threshold_sauvola(image, window_size=27, k=0.7)  
        binary_sauvola = image > thresh_sauvola
        plt.imsave(IMAGE_NAME[:-4]+"cropped.png",binary_sauvola, cmap=plt.cm.gray)
        
cv2.destroyAllWindows()
