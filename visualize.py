import torch
import torch.nn as nn
import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
import cv2
import os

# attempt to reduce dimensionality and view visualize a layer as an image
def visualize_layer(latent):

        normalized_latent = (latent - torch.min(latent)) / torch.max(latent)

        # reshaped_latent = torch.reshape(normalized_latent, (-1, 256))
        reshaped_latent = torch.reshape(normalized_latent, (-1, 2304))
        # print("reshaped shape", reshaped_latent.shape)

        # quick test for reshaping
        # print("qck test")
        # array = np.array([
        # [[1,2], [3,4], [5,6]],
        # [[7,8], [9,10], [11,12]]
        # ]
        # )
        # array = torch.from_numpy(array)
        # print(array.size())
        # print(array)
        #
        # r_array = torch.reshape(array, (-1, 6))
        # print(r_array.size())
        # print(r_array)
        #
        # f_array = torch.reshape(r_array, (-1, 3, 2))
        # print(f_array.size())
        # print(f_array)

        ################################### PCA
        (U,S,V) = torch.pca_lowrank(reshaped_latent, q=256, center=False, niter=2) #TODO do I need to center?
        # U,S,V = torch.linalg.svd(reshaped_latent)

        k = 3

        print("U shape: ", U.shape)
        print("S shape: ", S.shape)
        print("V shape: ", V.shape)

        # projected = torch.matmul(reshaped_latent.reshape(-1, 256), U[:, :k]) #this method worked with the linalg svd #TODO I think this should be V
        # projected = torch.matmul(reshaped_latent, V[:, :k])

        projected = torch.matmul(reshaped_latent.T, V.T[:, :k])
        # print("projected size: ", projected.size())
        # 2304 x 3

        ch1 = torch.reshape(projected[:,0], (36, 64)).numpy()
        ch2 = torch.reshape(projected[:,1], (36, 64)).numpy()
        ch3 = torch.reshape(projected[:,2], (36, 64)).numpy()
        rgb = np.stack((ch1, ch2, ch3))
        # print(rgb.shape)
        return rgb

        #https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html
        #https://www.programcreek.com/python/example/101191/torch.svd



        # projected = V[:,:,:3]

        # print("projected!")

        # normalize_projected = (projected - torch.min(projected)) / torch.max(projected)# * 0.5 #not sure why mult. by 0.5 here brings it to [0,1]ish

        # print("normalized max and min: ", torch.max(normalize_projected), torch.min(normalize_projected))

        # print(normalize_projected.shape)
        ###
        # print('shape: ', projected.shape)
        # projected = torch.reshape(projected, (36, 64, -1))
        # projected = torch.reshape(projected, (-1, 36, 64))

        # return projected
        # return normalized_latent[0,:,:]
        # return latent[0,:,:]

# view the shape of each activation layer's shape in our map
def view_all_activation_maps(visualisation_feature_map):
    feature_map_list = list(visualisation_feature_map.values())
    for i in range(len(feature_map_list)):
        print(feature_map_list[i].shape)

def visualize(visualisation_feature_map, input_, frameName):
    #options
    visualize_model_arch = False
    view_feature_maps = True

    # pred_feature = self.seq(input_)

    # if(visualize_model_arch):
    #     print(self.seq)

    if(view_feature_maps):
        # print('visualizing')

        # this is a list of all the layers that had the hook called
        # print(list(visualisation_feature_map.values())[0].size())
        # print(list(visualisation_feature_map.values())[1].size())

        # get a layer (Pick between 0 or 1 module in Channel 1)
        latent = list(visualisation_feature_map.values())[0][0,:,:,:].cpu().detach()
        # latent = list(visualisation_feature_map.values())[1][0,:,:,:].cpu().detach()#.numpy()

        latent_vis = visualize_layer(latent)

        # Pick Between Clipping vs Normalizing
        # NOTE: Don't do both and normalizing looks better it seems
        # latent_vis = latent_vis.clip(min=0)
        # latent_vis = (latent_vis - latent_vis.min())/latent_vis.max()

        latent_vis = latent_vis * 255.0

        img = cv2.merge((latent_vis[0], latent_vis[1], latent_vis[2]))

        # Upscaling so a bit easier to see
        scaleFactor = 4
        img = cv2.resize(img, (64 * scaleFactor, 36 * scaleFactor))

        cv2.imwrite(os.path.join("latent_images", frameName + ".png"), img)

    return
