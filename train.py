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
import argparse

from model import ComplexYOLO
from kitti import KittiDataset
from region_loss import RegionLoss


num_classes = 8
num_anchors = 5


def save_model_and_loss(epoch, do_logging):
    
    # Log and save model
    if do_logging:
        logging.info("Saving model at the last epoch = %d!" % epoch)
    torch.save(model, "model/ComplexYOLO_epoch" + str(epoch))
    
    # Log and save loss
    if do_logging:
        logging.info("Saving all losses at the last epoch = %d!" % epoch)
    np.save("loss/complexYOLO_epoch" + str(epoch), loss_history)


if __name__ == "__main__":
    
    # Parse args
    parser = argparse.ArgumentParser(prog = 'Train the mode.')
    parser.add_argument('--batch_size', type = int, default = 12, help = 'Number of images per batch.')
    parser.add_argument('--do_logging', type = bool, default = True, help = 'Whether or not do logging.')
    parser.add_argument('--logging_file', type = str, default = '', help = 'Overriding logging file name.')
    parser.add_argument('--lr', type = float, default = 1e-5, help = 'Initial learning rate.')
    parser.add_argument('--momentum', type = float, default = 0.9, help = 'Momentum coefficient.')
    parser.add_argument('--weight_decay', type = float, default = 0.0005, help = 'Weight decay.')
    parser.add_argument('--epochs', type = int, default = 1000, help = 'Number of epochs.')

    args = parser.parse_args()

    # Assign args to local vars
    batch_size = args.batch_size
    do_logging = args.do_logging
    logging_file = args.logging_file
    lr = args.lr;
    momentum = args.momentum
    weight_decay = args.weight_decay
    epochs = range(args.epochs)

    # Assign default name for training log if input is empty
    if do_logging and logging_file == '': 
        date_str = str(datetime.date.today())
        logging_file = 'training_log_' + date_str + '.log'
        logging.basicConfig(filename = logging_file, level=logging.DEBUG, format = '%(asctime)s %(message)s')

    # Remove old loggings in the tensorboard folder 
    ts_dir = './logs'
    for ts_file in os.listdir(ts_dir):
      ts_path = os.path.join(ts_dir, ts_file)
      os.unlink(ts_path)

    dirname = os.path.dirname(__file__)

    # Construct training dataset
    train_data_path = os.path.join(dirname, 'data')
    dataset = KittiDataset(root = train_data_path, set = 'train')
    data_loader = data.DataLoader(dataset, batch_size, shuffle = True, pin_memory = False)

    # Initialize a model
    model = ComplexYOLO()
    model.cuda()

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)

    # Define the loss function
    region_loss = RegionLoss(num_classes = num_classes, num_anchors = num_anchors)

    # Initialize a record to store loss history: # epochs X # batches X # classes
    num_batch = int(np.ceil(len(data_loader.dataset) / batch_size))
    loss_history = np.zeros((len(epochs), num_batch, num_classes))

    # Loop over epoch
    for epoch in epochs:
        
        # Log epoch idx
        if do_logging:
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
            
            # Log batch idx
            if do_logging:
               logging.info("Running batch_idx = %d" % batch_idx)

            optimizer.zero_grad()
            rgb_map = rgb_map.view(rgb_map.data.size(0), 
                    rgb_map.data.size(3),
                    rgb_map.data.size(1),
                    rgb_map.data.size(2))
            output = model(rgb_map.float().cuda())
            loss = region_loss(output,target, loss_history, epoch, batch_idx)
            loss.backward()
            optimizer.step()

        # Average the loss for all batches in the same epoch and log the loss	
        loss_epoch = loss_history[epoch, :, :].mean(axis = 0)
        
        # Log mean batch loss per epoch
        if do_logging:
            logging.info("Epoch loss = %s" % loss_epoch)

        # Add tensorboard logging to monitor losses in real time
        tensorboard_info = dict(zip(['x', 'y', 'w', 'l', 'conf', 'cls', 'euler'], loss_epoch))
        tensorboard_logger = Logger('./logs')
        tensorboard_logger.scalar_summary(tensorboard_info, epoch)

        # Save model and loss every 50 epochs
        if (epoch % 50 == 0):
            save_model_and_loss(epoch, do_logging)

    # Save model and loss at the very end
    save_model_and_loss(epoch, do_logging)
