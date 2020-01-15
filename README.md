# OCR
A repository for hosting all of the OCR code written for Aadhar, PAN and GSTIN

# Aadhaar card and PAN card Detection

## Using Tensorflow Object Detection API

Unzip the inference_graph.zip file in the inference_graph_aadhar directory.



### Demo on image
```    
python Object_detection_image.py --image <image_path> 
```
### Demo on webcam
```    
python Object_detection_webcam.py
```

## Using YOLO Detect

Unzip the ckpt.zip file in the same directory.

### Demo on image
```    
python YoloDemo_image.py --image <image_path>
```
### Demo on webcam
```    
python YoloDemo_webcam.py
```
### Training the tensorflow model for other variations
Use generate_tfrecord.py to generate tfrecord files for test and training set from labeled images.  
Use train.py to train the tensorflow model for any change in format.  
For labeling images follow the *[github link](https://github.com/tzutalin/labelImg.git).  
For detailed explanation you can follow first reference.  
The training for yolo goes similarly except for the model you choose.
The tensorflow model is trained on ssd_mobilenet_v1 model which is computationally light with pretty accurate results.  
## References

* [Tensorflow Object Detection API](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) - tutorial
* [YOLO Detect](https://towardsdatascience.com/yolov2-object-detection-using-darkflow-83db6aa5cf5f) - tutorial


