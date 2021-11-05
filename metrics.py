import torch
import numpy as np
import os
import argparse
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Metrics for consistent depth.')
parser.add_argument('--folder', type=str,
                    help='folder with frames')

#sum up difference in weights between two models
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

#compare L2 distance between frames and plot through time
def L2_frame_consistency(folder, cut_in_half = True): #cut in half if the frame has the depth map and original image
    img_list = []
    for( directory, subdirectories, file ) in os.walk(folder):
        for f in file:
            if not f.startswith('.'): #ignore hidden files
                if(cut_in_half):
                    img_list.append(cv2.imread(os.path.join(directory,f))[:,512:,:])
                else:
                    img_list.append(cv2.imread(os.path.join(directory,f)))

    if(len(img_list) < 2):
        print("Error: check the input folder.")
        return

    distances = []
    for i in range(len(img_list) - 2):
        distances.append(np.sum(np.square(img_list[i] - img_list[i + 1])))
    
    plt.plot(distances)
    plt.xlabel("Frame")
    plt.ylabel("L2 between adjacent frames")
    save_path = "L2_frame_comparisons/" + folder.split("/")[-1] + "_L2_plot.png"
    plt.savefig(save_path)

args = parser.parse_args()
L2_frame_consistency(args.folder)