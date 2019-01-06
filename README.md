# Complex YOLO with Uncertainty
## Deep Learning Project
### [Yuanchu Dang](https://www.linkedin.com/in/yuanchu-dang-6364a562/) and Wei Luo
Our repo contains a PyTorch implementation of the [Complex YOLO](https://arxiv.org/pdf/1803.06199.pdf) model with uncertainty for object detection in 3D.  
Our code is inspired by and builds on existing implementations of Complex YOLO [implementation of 2D YOLO](https://github.com/marvis/pytorch-yolo2) and [sample Complex YOLO implementation](https://github.com/AI-liu/Complex-YOLO).   
Our further contributions are as follows:
1. Added dropout layers and incorporated uncertainty into 3D object detection while preserving average precision.
2. Projected predictions to 3D using homography.
3. Attempted to add innovative loss terms to improve the model in cases when it predicts overlapping bounding boxes.

## Data
To run the model, you need to download and unzip the following data:

* [Velodyne point clouds (29 GB)](http://www.cvlibs.net/download.php?file=data_object_velodyne.zip): Information about the
surrounding for a single frame gathered by Velodyne
HDL64 laser scanner. This is the primary data we use.

* [Left color images of object data set (12 GB)](http://www.cvlibs.net/download.php?file=data_object_image_2.zip): The cam-
eras were one color camera stereo pairs.  We use left
Images corresponding to the velodyne point clouds for
each frame.

* [Camera  calibration  matrices  of  object  data  set  (16
MB)](http://www.cvlibs.net/download.php?file=data_object_calib.zip): Used for calibrating and rectifying the data captured by the camera and sensor.

* [Training labels of object data set (5 MB)](http://www.cvlibs.net/download.php?file=data_object_label_2.zip).

The following is an visualization of a sample image and its corresponding velodyne point-cloud.
<img src="https://github.com/Yuanchu/YOLO3D/blob/master/images/data.PNG" width="800px">

## Network Architecture
<img src="https://github.com/Yuanchu/YOLO3D/blob/master/images/architecture.PNG" height="400px">

## Training
First, you need to have a train.txt under data/training that contains 6-digit indices for the images that you want in the training set. Each line corresponds to one image. See the sample file in this repo. 

A reasonable train.txt can be generate by running:
```
python generate_train_txt.py
```
This produces a train.txt that contains the first 6000 images. Toggle the optional arguments to set your desired size. 

Next, you can kick off the training process by executing:
```
python train.py
```
There are also optional arguments that control batch size, logging, learning rate, momentum, weight decay and epochs. To figure out their usage, simply read the prompts in the parser and track their usage in the script. 

## Predicting
To run predictions on images, use predict.py. Suppose the directory for your trained model is model/ComplexYOLO_epoch400, and you want to predict and draw bounding boxes on the first 100 images. Execute the following command:
```
python predict.py 1 100 model/ComplexYOLO_epoch400
```
The first argument is the starting index of the image you want to predict, and the second argument is the ending index, both inclusive. 

There are also two optional arguments:
* --mode: "train" or "eval", case insensitive, with batch normalization and dropout layers actively engaged under the "train" mode. Default is set to the "eval" mode. 
* --num_predict: number of times to evaluate each image, only meaningful for the "train" mode with active dropout layers. Default is set to 1. 

For example, if you wish to turn on batch normalization and the dropout layers, and evaluate, say image 21, 1000 times to get a sense of the uncertainty for the model prediction, you can do: 
```
python predict.py 21 21 model/ComplexYOLO_epoch400 --mode train --num_predict 1000
```

## Generating Results
The [heat folder](https://github.com/Yuanchu/YOLO3D/tree/master/heat) and [project folder](https://github.com/Yuanchu/YOLO3D/tree/master/project) contain code for generating heatmap and 3D projections, respectively. The heatmap script loads a saved .npy file containing bounding box predictions, and a .png file for the corresponding road image.  Note that running the heatmap script requires an account on plotly. After running the program, it will put the resulting image on plotly. You should change the configurations inside the script accordingly. For projection, the script loads in saved .npy files containing target and prediction boxes, as well as original road image and corresponding velodyne point-cloud with target and prediction boxes drawn.  It also needs predefined heights and fine-tuned homography anchor points to produce an accurate 3D projection.

## Sample Results
Below are sample velodyne point-cloud with box predictions, along with the corresponding heatmaps that show our model's confidence.

<img src="https://github.com/Yuanchu/YOLO3D/blob/master/images/results.PNG" height="500px">

Below is a comparison of average precision between original Complex YOLO and our Complex YOLO with uncertainty.

<img src="https://github.com/Yuanchu/YOLO3D/blob/master/images/result_table.PNG" width="500px">

You may refer to either our report or poster for more details.

## Future Work
For future work, we can train model directly on labeled 3D data to make direct predictions without having to use homography and be able to visualize uncertainty in 3D. We can also attempt to take other models such as Fast-RCNN to 3D. Yet another direction would to extend to 4D as just presented at NeurIPS 2018: [YOLO 4D](https://openreview.net/pdf?id=B1xWZic29m)!

## Acknowledgments
We would like to thank Professor Iddo Drori and Chenqin for their constructive feedbacks throughout this project!
