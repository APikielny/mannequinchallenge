import torch
import numpy as np
import os
import argparse
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Metrics for consistent depth.')
parser.add_argument('--folder', type=str,
                    help='folder with frames')

# sum up difference in weights between two models


def compareModelWeights(model_a, model_b):
    module_a = model_a._modules
    module_b = model_b._modules
    if len(list(module_a.keys())) != len(list(module_b.keys())):
        return False
    a_modules_names = list(module_a.keys())
    b_modules_names = list(module_b.keys())
    sum_diff = 0
    for i in range(len(a_modules_names)):
        layer_name_a = a_modules_names[i]
        layer_name_b = b_modules_names[i]
        layer_a = module_a[layer_name_a]
        layer_b = module_b[layer_name_b]
        if hasattr(layer_a, 'weight') and hasattr(layer_b, 'weight'):
            sum_diff += abs(np.mean(layer_a.weight.data-layer_b.weight.data))
    return sum_diff

# compare L2 distance between frames and plot through time


# cut in half: if the frame has the depth map and original image, only use depth part of jpg
def L2_frame_consistency(folder, cut_in_half=True):
    img_file_names = []
    img_list = []

    for filename in os.listdir(folder):
        if filename.endswith(".jpg") and not filename.startswith("."):
            img_file_names.append(filename)
    img_file_names.sort()
    for filename in img_file_names:
        img = cv2.imread(os.path.join(folder, filename))
        if(cut_in_half):
            img_list.append(img[:, 512:, :])
        else:
            img_list.append(img)

    if(len(img_list) < 2):
        print("Error: check the input folder.")
        return

    distances = []
    for i in range(len(img_list) - 2):
        distances.append(np.sum(np.square(img_list[i] - img_list[i + 1])))

    # TODO normalize or not??
    distances = (np.array(distances) - np.min(distances))
    distances = distances / np.max(distances)

    variance = np.var(np.array(distances))

    plt.plot(distances)
    plt.xlabel("Frame")
    plt.ylabel("L2 between adjacent frames")

    name = folder.split("/")[-1]
    dataset = folder.split("/")[-2]
    if name == "":
        name = folder.split("/")[-2]
        dataset = folder.split("/")[-3]

    plt.title(name + ",\n Variance: " + str(variance)[:6])
    save_path = "L2_frame_comparisons/" + dataset + "/" + \
        name + "_L2_plot.png"
    plt.savefig(save_path)
    print(save_path)


args = parser.parse_args()
L2_frame_consistency(args.folder)
