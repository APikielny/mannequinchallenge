import argparse
import os


parser = argparse.ArgumentParser(description='Generate a txt file with images in a given dataset/id ')
parser.add_argument('--dataset', type=str,
                    help='just the dataset, not full path')
parser.add_argument('--id', type=str,
                    help='just the id')

args = parser.parse_args()

# Making supervision_list.txt for one ID
id = args.id #912b73b5672a7c28, 6ea49c46039bf641
if id[-1] == '/':
    id = id[:-1] #chop slash at end

dataset = args.dataset
img_path = "/data/jhtlab/apikieln/mannequin-dataset/{}/{}/images".format(dataset, id)

if not os.path.exists("../test_data/accuracy_testing_txt_files/{}".format(dataset)):
    os.makedirs("../test_data/accuracy_testing_txt_files/{}".format(dataset))

file_to_write = open("../test_data/accuracy_testing_txt_files/{}/{}.txt".format(dataset, id), "w")

img_files = os.listdir(img_path)
num_files = len(img_files)
for i in range(num_files):
    # Writing img path, and then depth path
    file_to_write.write(img_path + "/" + img_files[i] + "\n")