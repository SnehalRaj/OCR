import matplotlib.pyplot as plt
import re
import os
import glob
from PIL import Image
import pytesseract

import cv2
import numpy as np
from darkflow.net.build import TFNet

options = {
		   "model": "cfg/tiny-yolo1.cfg",
		   "load": 1500,
		   "gpu": 0.8,
		   "threshold":0.3
		   }
		   
tfnet = TFNet(options)
tfnet.load_from_ckpt()


def FindText(img,key):
	text = pytesseract.image_to_string(img)
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




def boxing(original_img, predictions):
	newImage = np.copy(original_img)
	tempConfidCard = 0
	tempConfidNum = 0
	Num=""
	Cardtype=""
	for result in predictions:
		strno = ""
		top_x = result['topleft']['x']
		top_y = result['topleft']['y']

		btm_x = result['bottomright']['x']
		btm_y = result['bottomright']['y']
		confidence = result['confidence']

		cropped = original_img[top_y-int(top_y*0.1):btm_y+int(btm_y*0.1),top_x-int(top_x*0.15):btm_x+int(btm_x*0.15),:]
		if result['label'] == "PANNo." :   
			tempNumb,flag = FindText(cropped,1)
			if confidence>tempConfidNum and flag==1:
				tempConfidNum = confidence
				Num = tempNumb
				Cardtype = 'PAN'
		elif result['label'] == "AadharNo." :
			tempNumb,flag = FindText(cropped,0)
			if confidence>tempConfidNum and flag==1:
				tempConfidNum = confidence
				Num = tempNumb
				Cardtype = 'Aadhar'	
		label = result['label'] + " " + str(round(confidence, 3))
		if confidence > 0.5:
			newImage = cv2.rectangle(newImage, (top_x, top_y), (btm_x, btm_y), (255,0,0), 3)
			newImage = cv2.putText(newImage, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.5, (0, 230, 0), 1, cv2.LINE_AA)
	return newImage,Num,Cardtype



	
#Prepare WebCamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
	#Face Detection
	ret, frame = cap.read() #BGR

	if ret == True:
		frame = np.asarray(frame)
		results = tfnet.return_predict(frame)

		new_frame,Num,CardType = boxing(frame, results)
		if Num!="":
			print(Num,CardType)
		# Display the resulting frame
		cv2.imshow('frame',new_frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
