import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import cv2
import os

# attempt to reduce dimensionality and view visualize a layer as an image
def visualize_layer(latent):

        normalized_latent = (latent - torch.min(latent)) / torch.max(latent)

        reshaped_latent = torch.reshape(normalized_latent, (256, -1)) 
        print("reshaped shape", reshaped_latent.shape)       

        ################################### PCA
        # (U,S,V) = torch.pca_lowrank(torch.reshape(reshaped_latent, (36, 256, 64)), q=None, center=True, niter=2) #TODO do I need to center?

        U,S,V = torch.linalg.svd(reshaped_latent)
        

        k = 3

        print("U shape: ", U.shape)
        print("S shape: ", S.shape)
        print("V shape: ", V.shape)

        projected = torch.matmul(reshaped_latent.reshape(-1, 256), U[:, :k])


        # projected = V[:,:,:3]

        # print("projected!")

        # normalize_projected = (projected - torch.min(projected)) / torch.max(projected)# * 0.5 #not sure why mult. by 0.5 here brings it to [0,1]ish

        # print("normalized max and min: ", torch.max(normalize_projected), torch.min(normalize_projected))

        # print(normalize_projected.shape)
        ### 
        print('shape: ', projected.shape)
        projected = torch.reshape(projected, (36, 64, -1))

        return projected
        # return normalized_latent[0,:,:]
        # return latent[0,:,:]
    
# view the shape of each activation layer's shape in our map
def view_all_activation_maps(visualisation_feature_map):
    feature_map_list = list(visualisation_feature_map.values())
    for i in range(len(feature_map_list)):
        print(feature_map_list[i].shape)

def visualize(visualisation_feature_map, input_, input_num):
    #options
    visualize_model_arch = False
    view_feature_maps = True

    # pred_feature = self.seq(input_)

    # if(visualize_model_arch):
    #     print(self.seq)
    
    if(view_feature_maps):
        # print('visualizing')
        
        # this is a list of all the layers that had the hook called
        # print(visualisation_feature_map.keys())
        # view_all_activation_maps(visualisation_feature_map)


        # print(list(visualisation_feature_map.values())[0].shape) # = torch.Size([1, 256, 36, 64])


        # print(list(visualisation_feature_map.values())[0][0,0,:,:].shape)

        # get a layer
        latent = list(visualisation_feature_map.values())[1][0,:,:,:].cpu().detach()#.numpy()
        # for i in range(256):
        #     latent_slice = list(visualisation_feature_map.values())[0][0,i,:,:].cpu().detach()#.numpy()
        #     cv2.imwrite(os.path.join("latent_images", "slice" + str(i) + ".png"), latent_slice.numpy() * 255)

        # # print("latent shape: ", latent.shape)
        # latent = list(visualisation_feature_map.values())[0][0,0,:,:].cpu().detach()#.numpy()

        latent_vis = visualize_layer(latent)

        # latent_slice = visualize_layer(latent)

        cv2.imwrite(os.path.join("latent_images", "img" + str(input_num) + ".png"), latent_vis.numpy() * 255)
        # print(list(visualisation_feature_map.values())[0][0,0,:,:].cpu().detach().numpy())

    return
