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


# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph_gstin'
IMAGE_NAME = args["image"]

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

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

def FindText(img):
    img = img.convert('L')
    text = pytesseract.image_to_string(img)
    text=text.replace(" ","")
    print(text)

    searchObj1 = re.search( r'[0-9][0-9][A-Z][A-Z][A-Z][A-Z][A-Z][0-9][0-9][0-9][0-9][A-Z][A-Z0-9][Z][A-Z0-9]', text)
    if searchObj1:
        return searchObj1.group(),1
    else:
        return "",0

'''
Global threholding on image can be done by using below functions.
Returns binarised image based on threshold.
'''

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
num=0

for k,v in y.items():
    words = v[0].split() #split the sentence into individual words
    confid = int(words[-1][:-1])
    ymin, xmin, ymax, xmax = k
    
    image_pil = Image.fromarray(np.uint8(im1)).convert('RGB')
    im_width, im_height = image_pil.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,ymin * im_height, ymax * im_height)
    cropped = image_pil.crop( ( left*0.9, top*0.9, right*1.1, bottom*1.1))
    img = cropped
    basewidth = 500
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    cropped = img
    # Saving cropped image containing reg no. for sauvola binarisation
    nam= IMAGE_NAME.split('/')[-1]
    name=nam[:-4]+"_cropped.png"
    nname=nam[:-4]+"_cropped1.png"
    cropped.save(name)
    img = mpimg.imread(name)     
    image_gray = rgb2gray(img)
    '''Comment out next section to try niblack processing on other images which was not giving good results on gstin images '''
    # thresh_niblack = threshold_niblack(image, window_size=win, k=kthresh)  
    # binary_niblack = image > thresh_niblack
    # plt.imsave(nname,binary_niblack, cmap=plt.cm.gray)
    # tempNumb,flag = FindText(cropped,1)

    tempNumb=""
    if "reg:" in words: #see if one of the words in the sentence is the word we want  
        # Appling sauvola local binarisation with parameters best suited for the dataset.
        # The parametes can be changed in next line and are more sensitive to k.
        thresh_sauvola = threshold_sauvola(image_gray, window_size=29, k=0.07)  
        binary_sauvola = image_gray > thresh_sauvola
        plt.imsave(nname,binary_sauvola, cmap=plt.cm.gray)
        cropped = cv2.imread(nname)
        cropped = Image.fromarray(cropped)
        # cropped.show()       
        #Searchng for registration number in cropped image using tesseract and regex with some preprocessing.
        tempNumb,flag = FindText(cropped)
        # If some valid number is found and confidence is higher than threshold then output gstin number.
        # Currenty threshold set to 0.65.
        if(flag==1 and confid >= 0.65):
            num=1
            numb= tempNumb
        #removing all temporary files
        os.remove(nname)
    os.remove(name)
        

if num is 0:
    print("Gstin Number not detected")
else:
    print("Number is",numb)
# All the results have been drawn on image. Now display the image.
cv2.imshow('Object detector', image)
#Time required for execution
t2 = time.time()
# print("Check is",check)
print("time taken to detect",t2-t1)
# To exit image_viewer
print("Press any key to close the image.........")
cv2.waitKey(0)

# Clean up
cv2.destroyAllWindows()
