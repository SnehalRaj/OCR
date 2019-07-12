import os
import math
import cv2
import numpy as np
import tensorflow as tf
import sys
import datetime
import time
from matplotlib import pyplot as plt
import tempfile
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

from scipy.misc import imsave
#time=datetime.datetime.now()
# construct the argument parser and parse the arguments
def rescale_img(image_path):
    image=Image.open(image_path)
    length_x,width_y=image.size
    factor=min(1,float(1024.0/length_x))
    size=int(factor*length_x),int(factor*width_y)
    # print(size)
    # return ""
    im_resized=image.resize(size,Image.ANTIALIAS)
    temp_file=tempfile.NamedTemporaryFile(delete=False,suffix='.png')
    # temp_filename=temp_file.name
    im_resized.save("1111.png",dpi=(300,300))
    return ""
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
    help="path to input image")
args = vars(ap.parse_args())
im_name=args["image"]
image=cv2.imread(im_name)
image1=rescale_img(im_name)
cv2.imshow('Object detector',image)
cv2.waitKey(0)
cv2.destroyAllWindows()



