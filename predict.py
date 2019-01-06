"""
    Predict and draw bounding boxes on images using loaded model. 
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import cv2
import os
import json
import argparse
from scipy import misc
from torch.autograd import Variable
from utils import *


# Load json configs
with open('config.json', 'r') as f:
    config = json.load(f)
boundary = config["boundary"]


def drawRect(img, pt1, pt2, pt3, pt4, color, lineWidth):
    cv2.line(img, pt1, pt2, color, lineWidth)
    cv2.line(img, pt2, pt3, color, lineWidth)
    cv2.line(img, pt3, pt4, color, lineWidth)
    cv2.line(img, pt1, pt4, color, lineWidth)


def get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors, only_objectness = 1, validation = False):
    anchor_step = int(len(anchors) / num_anchors)
    if output.dim() == 3:
        output = output.unsqueeze(0)
    batch = output.size(0)
    assert(output.size(1) == (7 + num_classes) * num_anchors)
    
    # h: 16, w: 32
    h = output.size(2)
    w = output.size(3)
    nB = output.data.size(0)

    # num_anchors: 5, num_classes: 8, nH: 16, nW: 32
    nA = num_anchors
    nC = num_classes
    nH = output.data.size(2)
    nW = output.data.size(3)
    anchor_step = int(len(anchors) / num_anchors)
    output = output.view(nB, nA, (7 + nC), nH, nW)
    x = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([0]))).view(nB, nA, nH, nW))
    y = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([1]))).view(nB, nA, nH, nW))
    w = output.index_select(2, Variable(torch.cuda.LongTensor([2]))).view(nB, nA, nH, nW)
    l = output.index_select(2, Variable(torch.cuda.LongTensor([3]))).view(nB, nA, nH, nW)
    im = output.index_select(2, Variable(torch.cuda.LongTensor([4]))).view(nB, nA, nH, nW)
    re = output.index_select(2, Variable(torch.cuda.LongTensor([5]))).view(nB, nA, nH, nW)
    conf = torch.sigmoid(output.index_select(2, Variable(torch.cuda.LongTensor([6]))).view(nB, nA, nH, nW))
    cls = output.index_select(2, Variable(torch.linspace(7, 7 + nC - 1,nC).long().cuda()))
    cls = cls.view(nB*nA, nC, nH*nW).transpose(1, 2).contiguous().view(nB * nA * nH * nW, nC)

    pred_boxes = torch.cuda.FloatTensor((7 + nC), nB * nA * nH * nW)
    grid_x = torch.linspace(0, nW - 1, nW).repeat(nH, 1).repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
    grid_y = torch.linspace(0, nH - 1, nH).repeat(nW, 1).t().repeat(nB * nA, 1, 1).view(nB * nA * nH * nW).cuda()
    anchor_w = torch.Tensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([0])).cuda()
    anchor_l = torch.Tensor(anchors).view(nA, anchor_step).index_select(1, torch.LongTensor([1])).cuda()
    anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)
    anchor_l = anchor_l.repeat(nB, 1).repeat(1, 1, nH * nW).view(nB * nA * nH * nW)

    pred_boxes[0] = x.data.view(nB * nA * nH * nW).cuda() + grid_x
    pred_boxes[1] = y.data.view(nB * nA * nH * nW).cuda() + grid_y
    pred_boxes[2] = torch.exp(w.data).view(nB * nA * nH * nW).cuda() * anchor_w
    pred_boxes[3] = torch.exp(l.data).view(nB * nA * nH * nW).cuda() * anchor_l
    pred_boxes[4] = im.data.view(nB * nA * nH * nW).cuda()
    pred_boxes[5] = re.data.view(nB * nA * nH * nW).cuda()
    
    pred_boxes[6] = conf.data.view(nB * nA * nH * nW).cuda()
    pred_boxes[7:(7 + nC)] = cls.data.view(nC, nB * nA * nH * nW).cuda()
    pred_boxes = convert2cpu(pred_boxes.transpose(0, 1).contiguous().view(-1, (7 + nC)))   #torch.Size([2560, 15])
    
    all_boxes =[]
    for i in range(2560):
        if pred_boxes[i][6] > conf_thresh:
            all_boxes.append(pred_boxes[i])
    return all_boxes


def eval(image_idx, mode, num_predict, model_dir):
    """
        image_idx: indices of images to be evaluated
        mode: training or eval, with the distinction of active or inactive dropout layers 
        num_predict: number of predictions per box, meaningful for active dropout layers to gauge uncertainty
    """
    for idx in image_idx:
        print("Predicting image = %d" % idx)

        # Get input
        test_i = str(idx).zfill(6)
        cur_dir = os.getcwd()
        lidar_file = cur_dir + '/data/training/velodyne/' + test_i + '.bin'
        calib_file = cur_dir + '/data/training/calib/' + test_i + '.txt'
        label_file = cur_dir + '/data/training/label_2/' + test_i + '.txt'

        # Load target data
        calib = load_kitti_calib(calib_file)  
        target = get_target(label_file, calib['Tr_velo2cam'])

        # Load point cloud data
        a = np.fromfile(lidar_file, dtype = np.float32).reshape(-1, 4)
        b = removePoints(a, boundary)
        rgb_map = makeBVFeature(b, boundary, 40 / 512)
        misc.imsave('predict/eval_bv.png', rgb_map)

        # Load trained model and forward, raw input (512, 1024, 3)
        input = torch.from_numpy(rgb_map)
        input = input.reshape(1, 3, 512, 1024)
        model = torch.load(model_dir)
        model.cuda()
        
        # Set model mode to determine whether batch normalization and dropout are engaged
        if mode == "train":
            model.train()
        elif mode == "eval":
            model.eval()
        else:
            raise(ValueError("Unsupported mode = %s" % mode))

        img = cv2.imread('eval_bv.png')

        # Paint ground truth
        true_boxes = []

        # Up to 50 true boxes
        for j in range(50):
            if target[j][1] == 0:
                break
            img_y = int(target[j][1] * 1024.0)
            img_x = int(target[j][2] * 512.0)
            img_width  = int(target[j][3] * 1024.0)
            img_height = int(target[j][4] * 512.0)
            rect_top1 = int(img_y - img_width / 2)
            rect_top2 = int(img_x - img_height / 2)
            rect_bottom1 = int(img_y + img_width / 2)
            rect_bottom2 = int(img_x + img_height / 2)
            cv2.rectangle(img, (rect_top1, rect_top2), (rect_bottom1, rect_bottom2), (0, 0, 255), 1)
            box = [rect_top2, rect_bottom2, rect_top1, rect_bottom1]
            true_boxes.append(box)

        all_predicts = []
        for k in range(num_predict):
            output = model(input.float().cuda())
            conf_thresh = 0.5
            num_classes = int(8)
            num_anchors = int(5)
            all_boxes = get_region_boxes(output, conf_thresh, num_classes, anchors, num_anchors)
            for i in range(len(all_boxes)):
                print("Box predicted!") 
                
                pred_img_y = int(all_boxes[i][0] * 1024.0 / 32.0)
                pred_img_x = int(all_boxes[i][1] * 512.0 / 16.0)
                pred_img_width = int(all_boxes[i][2] * 1024.0 / 32.0)
                pred_img_height = int(all_boxes[i][3] * 512.0 / 16.0)

                rect_top1 = int(pred_img_y-pred_img_width / 2)
                rect_top2 = int(pred_img_x-pred_img_height / 2)
                rect_bottom1 = int(pred_img_y+pred_img_width / 2)
                rect_bottom2 = int(pred_img_x+pred_img_height / 2)

                cv2.rectangle(img, (rect_top1, rect_top2), (rect_bottom1, rect_bottom2), (255, 0, 0), 1)
                box = [rect_top1, rect_top2, rect_bottom1, rect_bottom2]
                all_predicts.append(box)
        
        misc.imsave('predict/eval_bv' + test_i + '.png', img)
        np.save("all_predicts_%d" % idx, all_predicts)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog = 'Predict and draw bounding boxes on images.')
    parser.add_argument('start_idx', type = int, help = 'Start index (inclusive).')
    parser.add_argument('end_idx', type = int, help = 'End index (inclusive).')
    parser.add_argument('model_dir', type = str, help = 'Directory and name of model to be loaded.')
    parser.add_argument('--mode', choices = ['train', 'eval'], default = "eval", type = lambda s : s.lower(), help = 'Run mode: train/eval.')
    parser.add_argument('--num_predict', type = int, default = 1, help = 'How many predictions per image.')

    args = parser.parse_args()

    # Clean up args
    start_idx = args.start_idx
    end_idx = args.end_idx
    image_idx = list(range(start_idx, end_idx + 1))
    model_dir = args.model_dir
    mode = args.mode
    num_predict = args.num_predict

    # Load model, run predictions and draw boxes
    eval(image_idx, mode, num_predict, model_dir)
