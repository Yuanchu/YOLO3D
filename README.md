# Complex YOLO 3D
## Deep Learning Project
### Wei Luo (wl2671), Yuanchu Dang (yd2466)
This repo contains a PyTorch implementation of the [Complex YOLO model](https://arxiv.org/pdf/1803.06199.pdf) for object detection in 3D.  
Our code is inspired by [implementation of 2D YOLO](https://github.com/marvis/pytorch-yolo2) and [sample Complex YOLO implementation](https://github.com/AI-liu/Complex-YOLO).   

## Data
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
