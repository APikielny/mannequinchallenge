import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import cv2
import os

# uses pytorch pca to visualize layer
def visualize_layer(latent):
        normalized_latent = (latent - torch.min(latent)) / (torch.max(latent) - torch.min(latent))
        reshaped_latent = torch.reshape(normalized_latent, (-1, 2304))

        (U,S,V) = torch.pca_lowrank(reshaped_latent, q=256, center=True, niter=2)
        k = 3

        # print("U shape: ", U.shape)
        # print("S shape: ", S.shape)
        # print("V shape: ", V.shape)

        projected = torch.matmul(reshaped_latent.T, V.T[:, :k])

        ch1 = torch.reshape(projected[:,0], (36, 64)).numpy()
        ch2 = torch.reshape(projected[:,1], (36, 64)).numpy()
        ch3 = torch.reshape(projected[:,2], (36, 64)).numpy()
        rgb = np.stack((ch1, ch2, ch3))
        return rgb

# SKLEARN.DECOMP version
# INSTALL SCIKIT: pip3 install -U scikit-learn
def visualize_layer_sklearn(latent):

        # (256, w, h)
        # print(type(latent))
        # print(latent.size())
        # normalized_latent = (latent - torch.min(latent)) / torch.max(latent)

        # reshaped_latent = torch.reshape(normalized_latent, (-1, 256))
        # wxh = 2304
        reshaped_latent = torch.reshape(latent, (-1, 2304))
        # (256, wXh)
        reshaped_latent = torch.transpose(reshaped_latent, 0, 1)
        # (wXh, 256)

        pca = PCA(n_components=3)
        out = pca.fit_transform(reshaped_latent)
        # print(type(out))
        # print(out.shape)

        reshaped_out = np.reshape(out, (36, 64, -1))

        return reshaped_out

# view the shape of each activation layer's shape in our map
def view_all_activation_maps(visualisation_feature_map):
    feature_map_list = list(visualisation_feature_map.values())
    for i in range(len(feature_map_list)):
        print(feature_map_list[i].shape)

def visualize(visualisation_feature_map, input_, videoType, frameName, model_weights_name):
    #options
    visualize_model_arch = False
    view_feature_maps = True

    # pred_feature = self.seq(input_)

    # if(visualize_model_arch):
    #     print(self.seq)

    if(view_feature_maps):
        # Shows latent outputs for both Seq blocks, final output is vertically stacked
        show_both_latent_outputs = True

        # print('visualizing')
        # this is a list of all the layers that had the hook called
        # print(list(visualisation_feature_map.values())[0].size())
        # print(list(visualisation_feature_map.values())[1].size())
        if (show_both_latent_outputs) :
            # getlayer (both 0 and 1 module in Channel 1)
            latent0 = list(visualisation_feature_map.values())[0][0,:,:,:].cpu().detach()
            latent1 = list(visualisation_feature_map.values())[1][0,:,:,:].cpu().detach()#.numpy()

            # SKLearn produces more consistent results
            useSkLearn = True
            if(useSkLearn):
                latent_vis0 = visualize_layer_sklearn(latent0)
                latent_vis1 = visualize_layer_sklearn(latent1)
            else:
                latent_vis0 = visualize_layer(latent0)
                latent_vis1 = visualize_layer(latent1)

            # using np.ptp to max-min normalize
            latent_vis0 = (latent_vis0 - latent_vis0.min())/np.ptp(latent_vis0)
            latent_vis0 = latent_vis0 * 255.0

            latent_vis1 = (latent_vis1 - latent_vis1.min())/np.ptp(latent_vis1)
            latent_vis1 = latent_vis1 * 255.0

            if useSkLearn:
                img0 = cv2.merge((latent_vis0[:, :, 0], latent_vis0[:, :, 1], latent_vis0[:, :, 2]))
                img1 = cv2.merge((latent_vis1[:, :, 0], latent_vis1[:, :, 1], latent_vis1[:, :, 2]))
            else:
                img0 = cv2.merge((latent_vis0[0], latent_vis0[1], latent_vis0[2]))
                img1 = cv2.merge((latent_vis1[0], latent_vis1[1], latent_vis1[2]))

            # Upscaling so a bit easier to see
            scaleFactor = 4
            img0 = cv2.resize(img0, (64 * scaleFactor, 36 * scaleFactor))
            img1 = cv2.resize(img1, (64 * scaleFactor, 36 * scaleFactor))

            folder_path = os.path.join(os.getcwd(), "latent_images/" + model_weights_name, videoType)
            # create directory if it does not exist
            if(not os.path.exists(folder_path)):
                os.makedirs(folder_path)

            img = np.vstack((img0, img1))
            cv2.imwrite(os.path.join(folder_path, frameName + ".png"), img)
        else :
            # get a layer (Pick between 0 or 1 module in Channel 1)
            latent = list(visualisation_feature_map.values())[0][0,:,:,:].cpu().detach()
            # latent = list(visualisation_feature_map.values())[1][0,:,:,:].cpu().detach()#.numpy()

            # SKLearn produces more consistent results
            useSkLearn = True
            if(useSkLearn):
                latent_vis = visualize_layer_sklearn(latent)
            else:
                latent_vis = visualize_layer(latent)

            # using np.ptp to max-min normalize
            latent_vis = (latent_vis - latent_vis.min())/np.ptp(latent_vis)
            latent_vis = latent_vis * 255.0

            if useSkLearn:
                img = cv2.merge((latent_vis[:, :, 0], latent_vis[:, :, 1], latent_vis[:, :, 2]))
            else:
                img = cv2.merge((latent_vis[0], latent_vis[1], latent_vis[2]))

            # Upscaling so a bit easier to see
            scaleFactor = 4
            img = cv2.resize(img, (64 * scaleFactor, 36 * scaleFactor))

            folder_path = os.path.join(os.getcwd(), "latent_images", videoType)
            # create directory if it does not exist
            if(not os.path.exists(folder_path)):
                os.makedirs(folder_path)

            cv2.imwrite(os.path.join(folder_path, frameName + ".png"), img)

    return
