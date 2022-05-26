from curses import pair_number
import torch, torchvision
from torchvision.utils import save_image
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser(description='Generate a txt file with images in a given dataset/id ')
parser.add_argument('--dataset', type=str,
                    help='just the dataset, not full path')
parser.add_argument('--model_name', type=str,
                    help='model weights name')

args = parser.parse_args()

def compute_least_squares(mask, pred_d, gt_d):
    gt_mean = torch.mean(gt_d[mask > 0])
    pred_mean = torch.mean(pred_d[mask > 0])
    m = torch.sum((pred_d[mask > 0] - pred_mean)*(gt_d[mask > 0] - gt_mean))/torch.sum(torch.square(pred_d[mask > 0] - pred_mean))
    b = gt_mean - m*pred_mean
    return m,b

#based on mannequin Confidence_Loss()
def compute_scale(mask, pred_d, gt_d):
    m,b = compute_least_squares(mask, pred_d, gt_d)
    return m,b


def save_image_from_pair(gt_path, pred_path, dataset, model_name, id):
    MASK_THRESHOLD = 0.1

    gt_depth = torchvision.io.read_image(gt_path).float()
    pred_depth = torchvision.io.read_image(pred_path, torchvision.io.ImageReadMode.GRAY).float()
    resize_layer = torchvision.transforms.Resize((288,512))
    gt_depth_resize = resize_layer(gt_depth)
    cropped_pred_depth = pred_depth[:, :, 512:]

    # save_image(cropped_pred_depth/255, "accuracy_testing/pred_depth.jpg")
    # save_image(gt_depth_resize/255, "accuracy_testing/gt_depth.jpg")

    gt_mask = (gt_depth > MASK_THRESHOLD * 255).float()
    gt_mask = resize_layer(gt_mask)

    # save_image(gt_mask, "accuracy_testing/mask.jpg")
    frame_name = pred_path.split("/")[-1]

    m,b = compute_scale(gt_mask, cropped_pred_depth, gt_depth_resize)
    save_image((cropped_pred_depth * m + b)/255, "test_data_new/viz_predictions/aligned/{}/{}/{}/{}".format(dataset, id, model_name, frame_name))


def save_imgs_for_id(id, dataset, model_name, method = "median_scale"):

    filenames = next(os.walk('../mannequin-dataset/{}/{}/depth/'.format(dataset, id)), (None, None, []))[2]
    for file_path in filenames:
        gt_path = os.path.join('../mannequin-dataset/{}/{}/depth/{}'.format(dataset, id, file_path))
        pred_path = 'test_data/viz_predictions/accuracy_testing/{}/{}/images/{}/{}'.format(dataset, id, model_name, file_path)
        save_image_from_pair(gt_path, pred_path, dataset, model_name, id)

def generate_aligned_videos_for_a_model(model_name, dataset):
    print("Started generating frames...")
    #iterate through each id in dataset
    ids = next(os.walk('../mannequin-dataset/{}'.format(dataset)))[1]

    # sum_accuracy = 0
    accuracies = []
    
    #for each id, compute accuracy, then average them
    for id in ids:
        dir = "test_data_new/viz_predictions/aligned/{}/{}".format(dataset, model_name)
        if not os.path.exists(dir):
            os.makedirs(dir)
        save_imgs_for_id(id, dataset, model_name)
        print("Finished exporting for id: {}".format(id))
    return

generate_aligned_videos_for_a_model(args.model_name, args.dataset)
print("Finished generating frames.")