import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import cv2
import os

# attempt to reduce dimensionality and view visualize a layer as an image
def visualize_layer(latent):

        normalized_latent = (latent - torch.min(latent)) / torch.max(latent)

        reshaped_latent = torch.reshape(normalized_latent, (-1, 256)) 
        print("reshaped shape", reshaped_latent.shape)       

        ################################### PCA
        #https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html

        # (U,S,V) = torch.pca_lowrank(reshaped_latent, q=None, center=False, niter=2) #TODO do I need to center?
        # projected = torch.matmul(reshaped_latent, V[:, :k])

        k = 6

        #https://www.programcreek.com/python/example/101191/torch.svd
        # U,S,Vh = torch.linalg.svd(reshaped_latent)
        # print("U shape: ", U.shape)
        # print("S shape: ", S.shape)
        # print("Vh shape: ", Vh.shape)


        # print("dist: ", torch.dist(reshaped_latent, torch.mm(torch.mm(U, torch.diag(S)), Vh.t())))

        # projected = torch.matmul(reshaped_latent.reshape(-1, 256), U[:, :k]) #this method worked with the linalg svd #TODO I think this should be V 

        #https://stats.stackexchange.com/questions/134282/relationship-between-svd-and-pca-how-to-use-svd-to-perform-pca
        #https://pytorch.org/docs/stable/generated/torch.svd.html
        a_big = reshaped_latent
        u, s, v = torch.svd(a_big)
        print("U shape: ", u.shape)
        print("S shape: ", s.shape)
        print("Vh shape: ", v.shape)
        print("dist", torch.dist(a_big, torch.matmul(torch.matmul(u, torch.diag_embed(s)), v.transpose(-2, -1))))

        reconstruct = torch.matmul(u[:,3:k], torch.diag_embed(s)[3:k,3:k])
        print("reconstruct shape: ", reconstruct.shape)

        #reconstruct to original dims and measure distance. With higher k value, this distance decreases
        # print("reconstruct dist", torch.dist(a_big, torch.matmul(reconstruct, v[:, :k].transpose(-2, -1))))

        return torch.reshape(reconstruct, (36, 64, -1))

        #contact harry (yuanhao) wang











        # projected = V[:,:,:3]

        # print("projected!")

        # normalize_projected = (projected - torch.min(projected)) / torch.max(projected)# * 0.5 #not sure why mult. by 0.5 here brings it to [0,1]ish

        # print("normalized max and min: ", torch.max(normalize_projected), torch.min(normalize_projected))

        # print(normalize_projected.shape)
        ### 
        # print('shape: ', projected.shape)
        # projected = torch.reshape(projected, (36, 64, -1))

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

        cv2.imwrite(os.path.join("latent_images", frameName + ".png"), latent_vis.numpy() * 255)
        # print(list(visualisation_feature_map.values())[0][0,0,:,:].cpu().detach().numpy())

    return
