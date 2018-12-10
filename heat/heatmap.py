import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go

'''
This script uses plotly to draw heatmap of the bounding boxes.
Note that running this script requires an account on plotly.  
You must create an account and set the key for this script to run.

'''


def gen_freq(freq_map, boxes):
    '''
    For each pixel contained in a bounding box,
    increment frequency for that pixel by one, scaled
    by total number of predictions (1000 here).
    The final frequency is stored in freq_map.
    '''
    print(len(boxes))
    count = 0
    for box in boxes:
        if count % 100 == 0:
            print(count)
        count += 1
        x1,y1,x2,y2 = box[1], box[0],box[3],box[2]
        for i in range(x1, x2):
            for j in range(y1, y2):
                if i < 0 or i >= 512:
                    continue
                if j < 0 or j >= 1024:
                    continue
                freq_map[i][j] += 1/1000

def main():
    # load corners of predicted boxes
    boxes = np.load("all_predicts_1557.npy")

    # load point-cloud containing predicted bounding boxes
    img = cv2.imread("eval_bv001557.png")

    freq_map = np.zeros((img.shape[0], img.shape[1]),dtype=np.float32)
    gen_freq(freq_map, boxes)

    trace = go.Heatmap(z=freq_map)
    data=[trace]
    py.plot(data, filename='basic-heatmap.png')

if __name__ == "__main__":
    main()
