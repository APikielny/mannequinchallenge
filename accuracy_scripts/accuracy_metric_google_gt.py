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

###########################
#hack for aligning videos
parser.add_argument('--aligning_videos', action='store_true',
                    help='hack to align some frames using this code, for adam-translate data')
parser.add_argument('--epoch', type=str,
                    help='epoch to use for model')
###########################

args = parser.parse_args()

def compute_least_squares(mask, pred_d, gt_d):
    gt_mean = torch.mean(gt_d[mask > 0])
    pred_mean = torch.mean(pred_d[mask > 0])

    # print("gt mean ", gt_mean)
    # print("pred mean ", pred_mean)

    m = torch.sum((pred_d[mask > 0] - pred_mean)*(gt_d[mask > 0] - gt_mean))/torch.sum(torch.square(pred_d[mask > 0] - pred_mean))
    b = gt_mean - m*pred_mean
    # print("m, b", m, b)
    return m,b

#based on mannequin Confidence_Loss()
def compute_scaled_error(pred_confidence, mask, pred_d, gt_d):
        # EPSILON = 0
        # using least square to find scaling factor
        # N = torch.sum(mask) + EPSILON
        # N = N.item()

        # scale_factor = torch.median(
        #     gt_d[mask > 0] /
        #     (pred_d[mask > 0] + EPSILON)).item()
        # bias = torch.median(
        #     gt_d[mask > 0] -
        #     (pred_d[mask > 0] + EPSILON)).item()

        # print("scale factor: ", scale_factor)

        # save_image((pred_d * mask.data)/255, "accuracy_testing/pred_d_masked.jpg")
        # save_image((pred_d[mask>0])/255, "accuracy_testing/pred_d_masked.jpg")

        # print("scale factor: ", scale_factor)

        m,b = compute_least_squares(mask, pred_d, gt_d)

        # pred_d_aligned_scale = pred_d * scale_factor
        # pred_d_aligned_bias = pred_d + bias
        # pred_d_least_squares = pred_d * m + b
        # pred_d_aligned = pred_d_aligned_bias
        # save_image(pred_d_aligned_scale/255, "accuracy_testing/scaled_pred_d.jpg")
        # save_image(pred_d_aligned_bias/255, "accuracy_testing/bias_pred_d.jpg")
        # save_image(pred_d_least_squares/255, "accuracy_testing/least_squares_pred_d.jpg")

        # print("aligned: ", pred_d_aligned.data)
        # print("gt_d: ", gt_d.data)

        # error = torch.abs(pred_d_aligned.data -
        #                     gt_d.data) / (gt_d.data + EPSILON)
        # error = torch.exp(-error * 2.0)

        # original_difference = pred_d - gt_d
        # scale_aligned_difference = pred_d_aligned_scale - gt_d
        # bias_aligned_difference = pred_d_aligned_bias - gt_d
        # least_squares_aligned_difference = pred_d_least_squares - gt_d

        # scale_aligned_abs_diff = torch.abs(scale_aligned_difference)
        # bias_aligned_abs_difference = torch.abs(bias_aligned_difference)
        # least_squares_abs_aligned_difference = torch.abs(least_squares_aligned_difference)
        # original_abs_diff = torch.abs(original_difference)

        # save_image(difference/255, "accuracy_testing/difference.jpg")
        # save_image(scale_aligned_abs_diff/255, "accuracy_testing/scale_aligned_abs_diff.jpg")
        # save_image(bias_aligned_abs_difference/255, "accuracy_testing/bias_aligned_abs_diff.jpg")
        # save_image(least_squares_abs_aligned_difference/255, "accuracy_testing/least_squares_abs_aligned_difference.jpg")
        # save_image(original_abs_diff/255, "accuracy_testing/original_abs_diff.jpg")

        # save_image(mask*scale_aligned_abs_diff/255, "accuracy_testing/masked_scale_aligned_abs_diff.jpg")
        # save_image(mask*bias_aligned_abs_difference/255, "accuracy_testing/masked_bias_aligned_abs_diff.jpg")
        # save_image(mask*least_squares_abs_aligned_difference/255, "accuracy_testing/masked_least_squares_abs_aligned_difference.jpg")
        

        # print("error: ", error)

        # error_var = error
        # error_var = autograd.Variable(error, requires_grad=False)
        # u_loss = mask * torch.abs(pred_confidence - error_var)
        # confidence_term = torch.sum(u_loss) / N

        # print("confidence: ", confidence_term)

        # print("pred_d shape", pred_d.shape)
        # print("pred_d mean value", torch.mean(pred_d))

        # print("num valid pixels: ", torch.sum(mask[mask>0]))
        # print("sum not scaled pixels: ", torch.sum(pred_d[mask > 0]))
        # print("sum scaled pixels: ", torch.sum(pred_d[mask > 0] * scale_factor))

        # scale_error = torch.sum(pred_d[mask > 0] * scale_factor - gt_d[mask > 0]) / (255*torch.sum(mask[mask>0]))
        # bias_error = torch.sum((pred_d[mask > 0] + bias) - gt_d[mask > 0]) / (255*torch.sum(mask[mask>0]))
        # lst_sq_error_L1 = torch.sum(torch.abs((pred_d[mask > 0]*m + b) - gt_d[mask > 0])) / (255*torch.sum(mask[mask>0]))
        # lst_sq_error_L2 = torch.sqrt(torch.sum(torch.square((pred_d[mask > 0]*m + b) - gt_d[mask > 0]))) / (255*torch.sum(mask[mask>0]))
        lst_sq_MSE = torch.sum(torch.square(((pred_d[mask > 0]*m + b) - gt_d[mask > 0])/255.0)) / (pred_d[mask > 0].shape[0]) #divide by number of pixels in mask

        # print("sanity check, should be equal: ", pred_d[mask > 0].shape, torch.sum(mask[mask>0]))
        # print("sanity check, should be equal: ", pred_d[mask > 0].shape, mask[mask>0].shape)
        # print("mean mask value: ", torch.mean(mask[mask>0]))

        # print("bias calc: ", bias)
        # print("m, b: ", m,b)
        # print("bias num: ", torch.sum((pred_d[mask > 0] + bias) - gt_d[mask > 0]))
        # print("bias num abs: ", torch.sum(torch.abs((pred_d[mask > 0] + bias) - gt_d[mask > 0])))
        # print("lsq num:", torch.sum(torch.abs((pred_d[mask > 0]*m + b) - gt_d[mask > 0])))
        # print("denom:", (255*torch.sum(mask[mask>0])))
        

        # return scale_error, bias_error
        # return lst_sq_erro
        return lst_sq_MSE, m, b

def compute_error_single_pair(gt_path, pred_path):
    pred_confidence = 1

    gt_depth = torchvision.io.read_image(gt_path, torchvision.io.ImageReadMode.GRAY).float()[:, :, 512:] #need to add crop for the google input depth
    pred_depth = torchvision.io.read_image(pred_path, torchvision.io.ImageReadMode.GRAY).float()
    # print("gt depth read: ", torch.mean(gt_depth))
    # # gt_depth_resize = torch.resize(gt_depth, (1,288,512))
    resize_layer = torchvision.transforms.Resize((288,512))
    gt_depth_resize = resize_layer(gt_depth)
    cropped_pred_depth = pred_depth[:, :, 512:]

    # save_image(cropped_pred_depth/255, "accuracy_testing/pred_depth.jpg")
    # save_image(gt_depth_resize/255, "accuracy_testing/gt_depth.jpg")

    # gt_mask = (gt_depth > 0.2 * 255).float()
    gt_mask = (gt_depth > 0.1 * 255).float()
    # gt_mask = (gt_depth > -1000 ).float()
    gt_mask = resize_layer(gt_mask)

    # save_image(gt_mask, "accuracy_testing/mask.jpg")

    error, m, b = compute_scaled_error(pred_confidence, gt_mask, cropped_pred_depth, gt_depth_resize) #TODO what should pred confidence be?

    abs_diff = torch.abs((cropped_pred_depth * m + b) - gt_depth_resize)
    masked_abs_diff = gt_mask * abs_diff
    aligned_frame = cropped_pred_depth * m + b

    # print("eg path pred", pred_path)
    # print("eg path gt", gt_path)

    frame = pred_path.split("/")[-1]
    difference_dir = os.path.join(os.path.dirname(pred_path), "absolute_accuracy_difference_google_gt")
    masked_difference_dir = os.path.join(os.path.dirname(pred_path), "absolute_accuracy_masked_difference_google_gt")
    aligned_frame_dir = os.path.join(os.path.dirname(pred_path), "aligned_frames")
    if not os.path.exists(difference_dir):
        os.makedirs(difference_dir)
    if not os.path.exists(masked_difference_dir):
        os.makedirs(masked_difference_dir)
    # if not os.path.exists(aligned_frame_dir):
    #     os.makedirs(aligned_frame_dir)

    masked_diff_path = masked_difference_dir+"/"+frame
    diff_path = difference_dir+"/"+frame
    # aligned_frame_path = aligned_frame_dir+"/"+frame
    save_image(abs_diff/255, diff_path)
    save_image(masked_abs_diff/255, masked_diff_path)
    # save_image(aligned_frame/255, aligned_frame_path)

    # print("eg. path: ", abs_diff_path)

    return error, m, b

#compute accuracy using mean of accuracies of all frames in given video id
def compute_accuracy_for_id(id, dataset, model_name, method = "median_scale"):
    # if method != "median_scale":
        # print("Error, least squares not implemented yet.")
        # return

    sum_error = 0
    counter = 0
    filenames = next(os.walk('../../mannequin-dataset/{}/{}/depth/'.format(dataset, id)), (None, None, []))[2]

    if args.aligning_videos:
        ids = ["0ad91361c7a21579", "7f4dc061d49b46d2", "683d547fd69a910d", "c275e0ca7f69fd5c", "d489b3ee8cd0bbf1", "d7102c98a6fbdd24", "d95469b0511b2b60", "eaf1e44811bb8dc9", "fac61f7cfd26cac5"]
        for id in ids:
            ms = []
            bs = []
            testing_dir = '../test_data/viz_predictions/{}/{}/{}/images/{}'.format("accuracy_testing", args.dataset, id, model_name)
            for file_path in next(os.walk(testing_dir), (None, None, []))[2]:
                if not file_path.startswith('.'):
                    gt_path = '../../mannequinchallenge/test_data/viz_predictions/{}/{}/{}/images/{}/{}'.format("accuracy_testing", args.dataset, id, "best_depth_Ours_Bilinear_inc_3", file_path)
                    # gt_path = '../../mannequinchallenge/test_data/viz_predictions/{}/{}/{}'.format("adam_translate", "best_depth_Ours_Bilinear_inc_3", file_path)
                    pred_path = testing_dir + "/{}".format(file_path)
                    _, m, b = compute_error_single_pair(gt_path, pred_path)
                    ms.append(m)
                    bs.append(b)
            mean_m = np.mean(ms)
            mean_b = np.mean(bs)

            for file_path in next(os.walk(testing_dir), (None, None, []))[2]:
                if not file_path.startswith('.'):
                    pred_path = testing_dir + "/{}".format(file_path)
                    pred_depth = torchvision.io.read_image(pred_path, torchvision.io.ImageReadMode.GRAY).float()
                    cropped_pred_depth = pred_depth[:, :, 512:]
                    aligned_frame = cropped_pred_depth * mean_m + mean_b
                    aligned_frame_dir = os.path.join(os.path.dirname(pred_path), "aligned_frames")
                    if not os.path.exists(aligned_frame_dir):
                        os.makedirs(aligned_frame_dir)
                    aligned_frame_path = aligned_frame_dir+"/"+file_path
                    save_image(1.0-(aligned_frame/255), aligned_frame_path)
        counter = 1
    else:
        for file_path in filenames:
        # for file_path in [filenames[0]]:
            gt_path = '../../mannequinchallenge/test_data/viz_predictions/accuracy_testing/{}/{}/images/{}/{}'.format(dataset, id, "best_depth_Ours_Bilinear_inc_3", file_path)
            pred_path = '../test_data/viz_predictions/accuracy_testing/{}/{}/images/{}/{}'.format(dataset, id, model_name, file_path)
            pair_error, m, b = compute_error_single_pair(gt_path, pred_path)
            # print("curr error", pair_error)
            sum_error += pair_error
            counter += 1
    return sum_error/counter

# compute_accuracy_for_id('77cb2354f18d5954')
# print(compute_accuracy_for_id('data-half', '77cb2354f18d5954'))
def compute_accuracy_for_model(model_name, dataset):
    #iterate through each id in dataset
    ids = next(os.walk('../../mannequin-dataset/{}'.format(dataset)))[1]

    # sum_accuracy = 0
    accuracies = []
    
    #for each id, compute accuracy, then average them
    for id in ids:
        id_acc = compute_accuracy_for_id(id, dataset, model_name)
        # sum_accuracy += id_acc
        accuracies.append(id_acc)
        print("Id : ", id, "acc, ", id_acc)
    accuracies = np.array(accuracies)
    return np.mean(accuracies), np.std(accuracies)

if not args.aligning_videos:
    print("Started computing accuracy, using google model as GT.")
    acc, std_dev = compute_accuracy_for_model(args.model_name, args.dataset)
    print("Final Results: ")
    print("Model: ", args.model_name)
    print("Dataset: ", args.dataset)
    print("Accuracy: ", acc)
    print("Standard Deviation: ", std_dev)
else:
    compute_accuracy_for_id(-1, -1, args.model_name, "")
