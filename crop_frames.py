#for model outputs that have the original image on the left and the depth map on the right, crop to just the depth map (output to /cropped folder)

import os
import cv2
import argparse

parser = argparse.ArgumentParser(description='Generate a txt file with images in a given dataset/id ')
parser.add_argument('--path', type=str,
                    help='path with frames')
parser.add_argument('--epoch', type=str,
                    help='specific epoch, not required if frames are directly in path dir')

args = parser.parse_args()
path = args.path
if args.epoch is not None:
    path = os.path.join(path, "epoch_" + args.epoch)

filenames = next(os.walk(path), (None, None, []))[2]

cropped_dir = os.path.join(args.path, "cropped")
if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)

for filename in filenames:
    if not filename.startswith('.'):

        img = cv2.imread(os.path.join(path, filename))
        shape = img.shape
        cropped_img = img[:,shape[1]//2:,:]
        cv2.imwrite(os.path.join(cropped_dir, filename), cropped_img)