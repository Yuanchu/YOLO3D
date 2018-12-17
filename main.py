"""
    Entry point for the main training, on 6000 images, batch size of 12, and 1000 epochs. 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import logging 
from logger import Logger
import datetime
import os

from complexYOLO import ComplexYOLO
from kitti import KittiDataset
from region_loss import RegionLoss

date_str = str(datetime.date.today())
logging_file = 'training_log_' + date_str + '.log'
logging.basicConfig(filename = logging_file, level=logging.DEBUG, format = '%(asctime)s %(message)s')

batch_size = 12

# Remove old loggings in the tensorboard folder 
ts_dir = './logs'
for ts_file in os.listdir(ts_dir):
  ts_path = os.path.join(ts_dir, ts_file)
  os.unlink(ts_path)

# dataset
dataset=KittiDataset(root = '/home/yd2466/Complex-YOLO/data', set = 'train')
data_loader = data.DataLoader(dataset, batch_size, shuffle = True, pin_memory = False)

model = ComplexYOLO()
model.cuda()

# define optimizer
optimizer = optim.SGD(model.parameters(), lr = 1e-5 ,momentum = 0.9 , weight_decay = 0.0005)

# define the number of epochs
epochs = range(1000)

# Define the loss function
region_loss = RegionLoss(num_classes = 8, num_anchors = 5)
loss_history = np.zeros((len(epochs), int(len(data_loader.dataset) / batch_size), 8))

for epoch in epochs:
   logging.info('Running epoch = %d' % epoch)

   # Learning rate varies with epoch
   for group in optimizer.param_groups:
       if(epoch >= 4 & epoch < 80):
           group['lr'] = 1e-4
       if(epoch>=80 & epoch<160):
           group['lr'] = 1e-5
       if(epoch>=160):
           group['lr'] = 1e-6

   for batch_idx, (rgb_map, target) in enumerate(data_loader): 

          logging.info("Running batch_idx = %d" % batch_idx)
         
          optimizer.zero_grad()

          rgb_map = rgb_map.view(rgb_map.data.size(0),rgb_map.data.size(3),rgb_map.data.size(1),rgb_map.data.size(2))
          output = model(rgb_map.float().cuda())

          loss = region_loss(output,target, loss_history, epoch, batch_idx)
          loss.backward()

          optimizer.step()

   # Average the loss for all batches in the same epoch and log the loss	
   loss_epoch = loss_history[epoch, :, :].mean(axis = 0)
   logging.info("Epoch loss = %s" % loss_epoch)

   # Add tensorboard looging to monitor losses in real time
   tensorboard_info = dict(zip(['x', 'y', 'w', 'l', 'conf', 'cls', 'euler'], loss_epoch))
   tensorboard_logger = Logger('./logs')
   tensorboard_logger.scalar_summary(tensorboard_info, epoch)
   
   # Save model and loss every 50 epochs
   if (epoch % 50 == 0):
       logging.info("Saving model at epoch = %d" % epoch)
       torch.save(model, "model/ComplexYOLO_epoch" + str(epoch))
       
       logging.info("Saving all losses at epoch = %d" % epoch)
       np.save("loss/complexYOLO_epoch" + str(epoch), loss_history)

# Save model and loss at the very end
logging.info("Saving model at the last epoch = %d!" % epoch)
torch.save(model, "model/ComplexYOLO_epoch" + str(epoch))

logging.info("Saving all losses at the last epoch = %d!" % epoch)
np.save("loss/complexYOLO_epoch" + str(epoch), loss_history)

