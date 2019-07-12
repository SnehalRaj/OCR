## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py



# Import packages
import os
import cv2
import numpy as np
import tensorflow as tf
import sys

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util
import re
import pytesseract
from PIL import Image

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'inference_graph_aadhar'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training_aadhar','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 4

## Load the label map.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)
def FindText(img,key):
    img = img.convert('L')
    # text = tools.image_to_string(img,builder=pyocr.builders.DigitBuilder())
    # print(text)    
    text = pytesseract.image_to_string(img)
    # print(text)
    if key==0:
        searchObj1 = re.search( r'[0-9][0-9][0-9][0-9] [0-9][0-9][0-9][0-9] [0-9][0-9][0-9][0-9]', text)
        searchObj2 = re.search( r'[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]', text)
        if searchObj1:
           # print("Aadhar No. Found : ", searchObj1.group())
           return searchObj1.group(),1
        elif searchObj2:
           # print("Aadhar No. Found : ", searchObj2.group())
           return searchObj2.group(),1   
        else:
           # print("Aadhar No. Not found!!")
           return "",0
    elif key==1:
        searchObj1 = re.search( r'[A-Z][A-Z][A-Z][A-Z][A-Z][0-9][0-9][0-9][0-9][A-Z]', text)
        if searchObj1:
           # print("PAN No. Found : ", searchObj1.group())
           return searchObj1.group(),1
        else:
           # print("PAN No. Not found!!")
           return "",0
           
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

# Initialize webcam feed
video = cv2.VideoCapture(0)
ret = video.set(3,1280)
ret = video.set(4,720)

while(True):

    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    ret, frame = video.read()
    im1= np.copy(frame)
    frame_expanded = np.expand_dims(frame, axis=0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})
    tempConfidCard = 0
    tempConfidNum = 0
    numb = ""
    Cardtype = ""
    # Draw the results of the detection (aka 'visulaize the results')
    x,y = vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.40)
    for k,v in y.items():
        words = v[0].split() #split the sentence into individual words
        confid = int(words[-1][:-1])
        ymin, xmin, ymax, xmax = k
        
        image_pil = Image.fromarray(np.uint8(im1)).convert('RGB')
        im_width, im_height = image_pil.size
        (left, right, top, bottom) = (xmin * im_width, xmax * im_width,ymin * im_height, ymax * im_height)
        cropped = image_pil.crop( ( left*0.9, top*0.9, right*1.1, bottom*1.1))
        if "PANNo.:" in words: #see if one of the words in the sentence is the word we want            
            tempNumb,flag = FindText(cropped,1)
            if confid>tempConfidNum and flag==1:
                tempConfidNum = confid
                numb = tempNumb
        elif  "AadharNo.:" in words:
            tempNumb,flag = FindText(cropped,0)
            if confid>tempConfidNum and flag==1:
                tempConfidNum = confid
                numb = tempNumb
        if "PAN:" in words: #see if one of the words in the sentence is the word we want            
            if confid>tempConfidCard:
                tempConfidCard = confid
                Cardtype = 'PAN'
        elif  "Aadhar:" in words:
            if confid>tempConfidCard:
                tempConfidCard = confid
                Cardtype = 'Aadhar'
    if Cardtype!="":
        print(Cardtype,numb)
    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

