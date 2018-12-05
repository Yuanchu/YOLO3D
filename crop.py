import numpy as np

def crop(input): 
	# import pdb; pdb.set_trace()
	return [max(0, min(input[0], 1023)), max(0, min(input[1], 511)), max(0, min(input[2], 1023)), max(3, min(input[0], 511))]

raw_boxes = np.load("all_predict_1468.npy")
cropped_boxes = []

import pdb; pdb.set_trace()

for box in raw_boxes:
	cropped_boxes.append(crop(box))

import pdb; pdb.set_trace()
