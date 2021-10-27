# Convert .mov videos to jpeg frames for inference use with model

import sys
import argparse

import cv2
print(cv2.__version__)

def extractImages(pathIn, pathOut):
    vidcap = cv2.VideoCapture(pathIn)
    success,image = vidcap.read()
    count = 0
    success = True
    while success and (count < 75):
      success,image = vidcap.read()
      print ('Read a new frame: ', success)
      cv2.imwrite( pathOut + "/frame%d.jpg" % count, image)     # save frame as JPEG file
      count += 1

if __name__=="__main__":
    print("test")
    pathIn = "/home/adam/Desktop/repos/mannequinchallenge/test_data/custom-handheld/source/translate.mov";
    pathOut = "/home/adam/Desktop/repos/mannequinchallenge/test_data/custom-handheld/translate";
    extractImages(pathIn, pathOut)