#https://learnopencv.com/homography-examples-using-opencv-python-c/
import cv2
import numpy as np


def warp(img_path):
    # Read source image.
    im_src = cv2.imread(img_path)
    # # Four corners of the book in source image
    # pts_src = np.array([[141, 131], [480, 159], [493, 630],[64, 601]])

    # # Read destination image.
    # im_dst = cv2.imread('book1.jpg')
    # # Four corners of the book in destination image.
    # pts_dst = np.array([[318, 256],[534, 372],[316, 670],[73, 473]])

    # # Calculate Homography
    # h, status = cv2.findHomography(pts_src, pts_dst)

    h = np.array([[0.1,0.0,0.0],[0.0,0.1,0.0],[0.0,0.0,0.5]])

    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(im_src, h, (im_src.shape[1],im_src.shape[0]))
    # im_out = im_src
    
    cv2.imwrite("test_warp2_new.jpg", im_out)

warp("test_results/depth_id_4ea9094e0cbf1972_frame_5_to_5.png")

