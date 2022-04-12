from curses import pair_number
import torch, torchvision
from torchvision.utils import save_image
import numpy as np
import os


#based on mannequin Confidence_Loss()
def compute_scaled_error(pred_confidence, mask, pred_d, gt_d):
        EPSILON = 0
        # using least square to find scaling factor
        # N = torch.sum(mask) + EPSILON
        # N = N.item()

        scale_factor = torch.median(
            gt_d[mask > 0] /
            (pred_d[mask > 0] + EPSILON)).item()

        # print("scale factor: ", scale_factor)

        # save_image((pred_d * mask.data)/255, "accuracy_testing/pred_d_masked.jpg")
        # save_image((pred_d[mask>0])/255, "accuracy_testing/pred_d_masked.jpg")

        # print("scale factor: ", scale_factor)

        pred_d_aligned = pred_d * scale_factor
        save_image(pred_d_aligned/255, "accuracy_testing/scaled_pred_d.jpg")

        # print("aligned: ", pred_d_aligned.data)
        # print("gt_d: ", gt_d.data)

        error = torch.abs(pred_d_aligned.data -
                            gt_d.data) / (gt_d.data + EPSILON)
        error = torch.exp(-error * 2.0)

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

        return torch.sum(pred_d[mask > 0] * scale_factor - gt_d[mask > 0]) / (255*torch.sum(mask[mask>0]))

def compute_error_single_pair(gt_path, pred_path):
    pred_confidence = 1

    gt_depth = torchvision.io.read_image(gt_path).float()
    pred_depth = torchvision.io.read_image(pred_path, torchvision.io.ImageReadMode.GRAY).float()
    # print("gt depth read: ", torch.mean(gt_depth))
    # # gt_depth_resize = torch.resize(gt_depth, (1,288,512))
    resize_layer = torchvision.transforms.Resize((288,512))
    gt_depth_resize = resize_layer(gt_depth)
    cropped_pred_depth = pred_depth[:, :, 512:]

    save_image(cropped_pred_depth/255, "accuracy_testing/pred_depth.jpg")
    save_image(gt_depth_resize/255, "accuracy_testing/gt_depth.jpg")

    gt_mask = (gt_depth > 0.3 * 255).float()
    # gt_mask = (gt_depth > -1000 ).float()
    gt_mask = resize_layer(gt_mask)

    save_image(gt_mask, "accuracy_testing/mask.jpg")

    error = compute_scaled_error(pred_confidence, gt_mask, cropped_pred_depth, gt_depth_resize) #TODO what should pred confidence be?

    return error

#compute accuracy using mean of accuracies of all frames in given video id
def compute_accuracy_for_id(id, method = "median_scale"):
    if method != "median_scale":
        print("Error, least squares not implemented yet.")
        return

    sum_error = 0
    counter = 0
    filenames = next(os.walk('mannequin-dataset/data-half/{}/depth/'.format(id)), (None, None, []))[2]
    for file_path in filenames:
    # for file_path in [filenames[0]]:
        gt_path = os.path.join('mannequin-dataset/data-half/{}/depth/{}'.format(id, file_path))
        pred_path = 'alias-free-mannequinchallenge/test_data/viz_predictions/images/overfit_downsample_testing_anti_alias_upsample_radial_pad_non_crit_sampling_20_epochs_post_weight_init_fix/epoch_6/{}'.format(file_path)
        pair_error = compute_error_single_pair(gt_path, pred_path)
        print("curr error", pair_error)
        sum_error += pair_error
        counter += 1
    return sum_error/counter

# compute_accuracy_for_id('77cb2354f18d5954')
print(compute_accuracy_for_id('77cb2354f18d5954'))