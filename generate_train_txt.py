"""
    Generate train.txt file which contains a list of training indices.
"""


import argparse
import os

if __name__ == "__main__":
    
    # Train ratio and max size from cmd
    parser = argparse.ArgumentParser(prog = 'Generate training idx.')
    parser.add_argument('--train_ratio', type = float, default = 0.85, help = 'Training ratio.')
    parser.add_argument('--max_size', type = int, default = 6000, help = 'Maximum number of training images.')
    args = parser.parse_args()
    train_ratio = args.train_ratio
    max_size = args.max_size

    # Grab all files and determine training size
    dirname = os.path.dirname(__file__)
    
    # Calibration path and files
    calib_path = os.path.join(dirname, 'data/training/calib')
    calib_files = [f for f in os.listdir(calib_path) if os.path.isfile(os.path.join(calib_path, f))]
   
    # Train size = min(max_size, train_ratio * total_size)
    train_size = min(max_size, int(len(calib_files) * train_ratio))

    # Write to train.txt 
    output_file = os.path.join(dirname, 'data/training/train.txt')
    output = open(output_file, 'w')
    for i in range(train_size):
            output.write(str(i).zfill(6) + '\n')
    output.close() 

