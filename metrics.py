import torch
import numpy as np
import os
import argparse
import cv2
import matplotlib.pyplot as plt
from models import hourglass
from scipy.ndimage.filters import gaussian_filter


parser = argparse.ArgumentParser(description='Metrics for consistent depth.')
parser.add_argument('--L2_folder', type=str,
                    help='folder with frames')
parser.add_argument('--weights_a', type=str,
                    help='to compare weights of two models')
parser.add_argument('--weights_b', type=str,
                    help='to compare weights of two models')

# sum up difference in weights between two models


# def compareModelWeights(model_a, model_b):
#     module_a = model_a._modules
#     module_b = model_b._modules
#     if len(list(module_a.keys())) != len(list(module_b.keys())):
#         return False
#     a_modules_names = list(module_a.keys())
#     b_modules_names = list(module_b.keys())
#     sum_diff = 0
#     for i in range(len(a_modules_names)):
#         layer_name_a = a_modules_names[i]
#         layer_name_b = b_modules_names[i]
#         layer_a = module_a[layer_name_a]
#         layer_b = module_b[layer_name_b]
#         if hasattr(layer_a, 'weight') and hasattr(layer_b, 'weight'):
#             sum_diff += abs(np.mean(layer_a.weight.data-layer_b.weight.data))
#     return sum_diff

#check whether two sets of weights are exactly equal
def check_model_equality(weights_a, weights_b):
    model_a = hourglass.HourglassModel(3)
    model_b = hourglass.HourglassModel(3)
    model_a.load_state_dict(weights_a)
    model_b.load_state_dict(weights_b)

    for p1, p2 in zip(model_a.parameters(), model_b.parameters()):
        if p1.data.ne(p2.data).sum() > 0: #checks equality
            return False
    return True

# compare L2 distance between frames and plot through time
def L2_frame_consistency(folder, cut_in_half=True): # cut in half: if the frame has the depth map and original image, only use depth part of jpg
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

    # Adding gaussian blur (NOTE: sigma is a hyperparam)
    list = img_list # temp variable, so we can avoid append
    for i in range(len(img_list)):
        img_list[i] = gaussian_filter(list[i], sigma = 5)

    distances = []
    for i in range(len(img_list) - 2):
        distances.append(np.sqrt(np.sum(np.square(img_list[i] - img_list[i + 1]))))

    # TODO normalize or not??
    min = np.min(distances)
    max = np.max(distances)
    distances = (np.array(distances) - min) / (max-min)

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
if (args.L2_folder is not None):
    L2_frame_consistency(args.L2_folder)

if (args.weights_a is not None and args.weights_b is not None):
    print(check_model_equality(torch.load(args.weights_a), torch.load(args.weights_b)))

# weights_a = torch.load(
#     'checkpoints/test_local/control_1_net_G.pth')
# weights_b = torch.load(
#     'checkpoints/test_local/control_2_net_G.pth')
