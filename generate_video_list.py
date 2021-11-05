# Script to generate test_davis_video_list.txt
# Call to run inference with different datasets

# Run by calling ./generate_video_list
# Change directory by editing path variable
path = "/home/adam/Desktop/repos/mannequinchallenge/test_data/custom-handheld/rotate"

import os

a = open("./test_data/test_marc_video_list.txt", "w")

files = os.listdir(path)
for f in files:
    a.write(path + "/" + f + "\n")
