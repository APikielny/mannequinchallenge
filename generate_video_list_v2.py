# Script to generate list for supervision training from scratch for one video (supervision_list.txt)
# AND all training videos (train.txt)

import os

# Making supervision_list.txt for one ID
id = "00abf2d3644c4b04"
# img_path = "/home/adam/Desktop/repos/mannequin-dataset/data/00abf2d3644c4b04/images"
# depth_path = "/home/adam/Desktop/repos/mannequin-dataset/data/00abf2d3644c4b04/depth"
img_path = "/home/adam/Desktop/repos/mannequin-dataset/data/" + id + "/images"
depth_path = "/home/adam/Desktop/repos/mannequin-dataset/data/" + id + "/depth"

a = open("./test_data/supervision_list.txt", "w")

img_files = os.listdir(img_path)
depth_files = os.listdir(depth_path)

if (len(img_files) != len(depth_files)):
    print("Number of img files and depth files is different")
    exit()

num_files = len(img_files)
print("number of files for supervision_list.txt, ", num_files)
for i in range(num_files):
    # Writing img path, and then depth path
    a.write(img_path + "/" + img_files[i] + "\n")
    a.write(depth_path + "/" + depth_files[i] + "\n")
#########################################################################################
# Making train_list.txt for all ID
all_dirs = os.listdir("/home/adam/Desktop/repos/mannequin-dataset/data-half/")
print("number of ids for train_list.txt, ", len(all_dirs))
# print(all_dirs)

b = open("./test_data/temp_list.txt", "w")
count = 0

for id in all_dirs:
    img_path = "/home/adam/Desktop/repos/mannequin-dataset/data-half/" + id + "/images"
    depth_path = "/home/adam/Desktop/repos/mannequin-dataset/data-half/" + id + "/depth"

    img_files = os.listdir(img_path)
    depth_files = os.listdir(depth_path)

    if (len(img_files) != len(depth_files)):
        print("Number of img files and depth files is different - ", id, str(len(img_files)), str(len(depth_files)))
    else:
        num_files = len(img_files)
        count += num_files
        for i in range(num_files):
            # Writing img path, and then depth path
            b.write(img_path + "/" + img_files[i] + "\n")
            b.write(depth_path + "/" + depth_files[i] + "\n")

print(count)
