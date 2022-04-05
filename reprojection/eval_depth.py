import argparse
import os
from matplotlib import image
import numpy as np
import sys
sys.path.append("..")
# from read_dense_colmap import read_array
import mvs_util
import torch
import torch.nn.functional as F
import pickle 
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.6f')
import PIL.Image as Image
# from scale import scale
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import cv2
import random
random.seed(0)
import torchvision
from torchvision.utils import save_image

#use this to import by specifying path to module
import sys
sys.path.append("/home/adam/Desktop/repos/mannequin-dataset")
from read_write_model import extractCameraInfo

def load_poses(input_dir, tgt_height, tgt_width):
    poses_w2c, poses, imgs, I = mvs_util.load_poses(os.path.join(input_dir, 'sparse/images.txt'), os.path.join(input_dir, 'images'), tgt_height, tgt_width)

    cam = mvs_util.load_cameras(os.path.join(input_dir, 'sparse/cameras.txt'), tgt_height, tgt_width).flatten()
    data = {}
    for idx in range(len(imgs)):

        data[imgs[idx].split('.')[0]] = { 
     'img': I[idx],
	'pose': poses[idx],
	'cam' : cam}

    return data

#seems like first param is a target pose, second param is a list of src poses. src is an img. tgt_depth is a depth. 
#c2w probs means camera to world
def warp(tgt_pose_c2w, src_poses_c2w, src, tgt_depth, cam):

    #width, height, focal length x and y? 
    #these are camera intrinsics I think, maybe extracted from colmap
    w, h, fx, fy =int(cam[0]), int(cam[1]), cam[2], cam[3]

    y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    x = x.cuda()
    y = y.cuda()

    #move x from [0,w] to [-w/2, w/2]
    #then mult by tgt depths? and divide by fx? idk what this means
    #i think it's taking the depth and putting it into the world space from the camera or something like that
    x_tgt = (x - w/2.0) * tgt_depth / fx
    y_tgt = (y - h/2.0) * tgt_depth / fy
    z_tgt = tgt_depth

    

    p_tgt = torch.stack((x_tgt.view( (-1, h * w) ),
                         y_tgt.view( (-1, h * w) ), 
                         z_tgt.view( (-1, h * w) ),
                         torch.ones( (x_tgt.shape[0], h * w)).cuda()), 1)
    
    src_poses_w2c = torch.inverse(src_poses_c2w)

    #K is intrinsics matrix
    K = torch.tensor([fx, 0., w/2., 0.,
		0., fy, h/2., 0.,
                 0., 0., 1., 0.]).reshape([3, 4]).cuda()
    KT = torch.matmul(K, torch.matmul(src_poses_w2c, tgt_pose_c2w))
    
    p_src = torch.matmul(KT, p_tgt)


    
    p_z = p_src[:, 2, :]
    invalid = (p_z <= 0).view(-1, h, w)
    
    p_src[:, 0, :] = p_src[:, 0, :] / (p_src[:, 2, :] + 1e-16)
    p_src[:, 1, :] = p_src[:, 1, :] / (p_src[:, 2, :] + 1e-16)

    x_norm = p_src[:, 0, :] / float(w - 1) * 2 - 1.0 # Normalize grid coordinates to [-1, 1]
    y_norm = p_src[:, 1, :] / float(h - 1) * 2 - 1.0



    grid = torch.stack((x_norm.reshape(-1, h, w), y_norm.reshape(-1, h, w)), -1)

    o = F.grid_sample(src.double(), grid.double(), align_corners=True)

#    cv2.imwrite('pre.png', src)
#    cv2.imwrite('post.png', o)

    return o, invalid

#assume same cam intrinsics src and tgt
#src_poses_c2w puts cam src points into W space
def fwd_depth(tgt_pose_c2w, src_poses_c2w, src_depths, cam):

    #cam intrinsics 
    w, h, fx, fy =int(cam[0]), int(cam[1]), cam[2], cam[3]
    
    y, x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    y = y.cuda()
    x = x.cuda()

    # x from [0,w] to [-w/2, w/2]
    # x_src, y_src, z_src are the depths in camera space
    #i think it's taking the depth and putting it into the world space from the camera
    x_src = (x - w/2.0) * src_depths / fx
    y_src = (y - h/2.0) * src_depths / fy
    z_src = src_depths
    p_src = torch.stack((x_src.view((-1, h*w)),
                         y_src.view((-1, h*w)), 
                         z_src.view((-1, h*w)),
                         torch.ones((x_src.shape[0], h*w)).cuda()), 1)
    tgt_pose_w2c = torch.inverse(tgt_pose_c2w) #take inverse to get matrix from world to target camera space
    K = torch.tensor([fx, 0, w*.5, 0,
                      0, fy, h*.5, 0,
                      0, 0, 1, 0]).reshape([3,4]).cuda()
    KT = torch.matmul(K, torch.matmul(tgt_pose_w2c, src_poses_c2w)) #matrix from src cam->world->target cam->homogeneous
    p_tgt = torch.matmul(KT, p_src)
    p_z = p_tgt[:, 2, :] #distance along ray to each point is the tgt depth

    p_x = torch.round(p_tgt[:, 0, :] / p_z).long() #get x and y by dividing by the depth
    p_y = torch.round(p_tgt[:, 1, :] / p_z).long()
    #this point^ is in continuous space

    #need to resolve occlusion and discretize points
    num_planes = 128 #chunk into depth ranges for occlusion?
    mx, mn = torch.max(p_z), torch.min(p_z)
    p_z_ord = torch.round((p_z - mn) / (mx-mn) *num_planes).int() #round the depth to the nearest plane
    
    in_bounds = torch.logical_and(torch.logical_and(p_x >= 0, p_x < w),
                                  torch.logical_and(p_y >= 0, p_y < h))
    
    o = torch.zeros(p_y.shape).cuda()#.double()
    for i in range(num_planes + 1, 0, -1):
        idx = torch.logical_and(p_z_ord == i, in_bounds)
        for j in range(idx.shape[0]): #wondering why only iterate through one dim of idx. Is it 2 dimensional?
            if not torch.any(idx[j, :]):
                continue
            o[j, p_y[j, idx[j, :]] * w + p_x[j, idx[j, :]]] = p_z[j, idx[j, :]]
    o = o.reshape((-1, h, w))
     
    return o 



'''
# w, h, f, cx, cy
cam = cam[:5] # Remove distortion parameters from camera array
cam[:2] = Is.shape[::-1][:2]
cam[3:] = [dim/2 for dim in Is.shape[::-1][:2]] # recalculate cx, cy
cam[2] = cam[2] * (Is.shape[-1] / (w if (w < h) else h)) # scale f
'''

def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    header = file.readline().decode('UTF-8').rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    # scale = float(file.readline().rstrip())
    scale = float((file.readline()).decode('UTF-8').rstrip())
    if scale < 0: # little-endian
        data_type = '<f'
    else:
        data_type = '>f' # big-endian
    data_string = file.read()
    data = np.frombuffer(data_string, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data



scenes = ["courtyard","delivery_area","electro","facade","kicker","meadow", "pipes","playground","relief","relief_2","terrace","terrains"]

dense_folder = '../data/ETH3D/train/%s/images/undistort'
gt_folder = '../data/ETH3D/gt/%s/ground_truth_depth/dslr_images'
mvs_folder = '../data/MVS/MVS'
ours_folder = '../data/ours/%s'
eth3d_ours_folder = '../data/eth3d_ours/%s'
#algos = ['colmap', 'deepmvs', 'mvsnet']
#algos = ['mvsnet_eth3d', 'mvsnet_eth3d_f', 'mvsnet_f']
#algos = ['eth3d_ours']
algos = ['colmap', 'deepmvs', 'mvsnet', 'ours', 'eth3d_ours']
h = 540
w = 810
num = 1000
eps = 1e-19
corrupted = ['DSC_0689', 'DSC_0695']
def resize(din_colmap):
    if np.any(np.isnan(din_colmap)) or np.any(np.isinf(din_colmap)):
        assert False, "not a legal resize item!"
    a_min = np.min(din_colmap)
    a_max = np.max(din_colmap)
    a_scaled = 255*(din_colmap-a_min)/(a_max-a_min) 
    din_colmap = Image.fromarray(a_scaled) 
    din_colmap = din_colmap.resize((w, h), Image.NEAREST)
    din_colmap = np.array(din_colmap)
    din_colmap = din_colmap * (a_max-a_min) / 255. + a_min 
    return din_colmap
def vis(depth_map, root, image):
    min_depth, max_depth = np.percentile(
			depth_map, [5, 95])
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth
    plt.figure()
    plt.imshow(depth_map)
    plt.title("depth map")
    plt.savefig(os.path.join(root, '%s.png' % image)) 
    
if __name__ == '__NOT_main__':
    stats = {}

    for scene in tqdm(scenes):
        stats[scene] = {}
        dinss = {}
        for algo in algos:
            dinss[algo] = {}
        for algo in algos:
            stats[scene][algo] = {}
        imagedirs = [f.split('.')[0] for f in sorted(os.listdir(os.path.join(dense_folder % scene, 'images'))) if f.endswith('png') or f.endswith('jpg') or f.endswith('.JPG') or f.endswith('jpeg')]

        with open(os.path.join(dense_folder % scene, 'mvs_id.json'), 'r') as f:
            mapper = json.load(f) 
        # data: stored essentials for reprojection 
        data = load_poses(dense_folder % scene, h, w)

        # for each image in this scene
        for sample in tqdm(sorted(list(mapper.keys()))):


            sample = sample.split('.')[0]
            if sample in corrupted:
                continue 
            # gt: a dict of depth lables
            # (i,j) -> depth
            # assume image size: 540*810
            with open(os.path.join(gt_folder % scene, "%s.JPG.pkl" % sample), 'rb') as f:
                gt = pickle.load(f, encoding='latin1')
            # which 1000 labels would participate in scaling? 
            ij = random.sample(list(gt.keys()), num)
            # for each algorith, load depth prediction for this image
            # make sure depth instead of disparity
            # resize to (540, 810)
            # scaled 

            # commenting out for import
            # if 'ours' in algos:
            #     idxx = imagedirs.index(sample)
            #     if idxx < 10:
            #         idxx = '0' + str(idxx)
            #     else:
            #         idxx = str(idxx)
            #     din_ours = np.load(os.path.join(ours_folder % scene, "%s.npy" % idxx))
            #     din_ours = resize(din_ours)
            #     din_ours = scale(din_ours, gt, ij)
            #     dinss['ours'][sample] = din_ours 
            if 'eth3d_ours' in algos:
                idxx = imagedirs.index(sample)
                if idxx < 10:
                    idxx = '0' + str(idxx)
                else:
                    idxx = str(idxx)
                din_ours = np.load(os.path.join(eth3d_ours_folder % scene, "%s.npy" % idxx))
                din_ours = resize(din_ours)
                din_ours = scale(din_ours, gt, ij)
                dinss['eth3d_ours'][sample] = din_ours 
            # commenting out due to import
            # if 'colmap' in algos:

            #     din_colmap = read_array(os.path.join(dense_folder % scene, 'stereo/depth_maps/%s.jpg.geometric.bin' % sample))

            #     din_colmap = resize(din_colmap)
            #     din_colmap = scale(din_colmap, gt, ij)
            #     dinss['colmap'][sample] = din_colmap

            # if 'deepmvs' in algos:

            #     din_deepmvs = np.load(os.path.join(dense_folder % scene, 'depth_maps/%s.jpg.output.npy' % sample))
            #     min_dis = np.min(din_deepmvs[din_deepmvs>0])
            #     din_deepmvs[din_deepmvs ==0] = min_dis 
            #     din_deepmvs = 1./(din_deepmvs )


            #     din_deepmvs = resize(din_deepmvs)
            #     din_deepmvs = scale(din_deepmvs, gt, ij) 

            #     dinss['deepmvs'][sample] = din_deepmvs

            # if 'mvsnet' in algos:

            #     pfm_dir = os.path.join(dense_folder % scene, 'depths_mvsnet/%s_init.pfm' % str(mapper['%s.jpg' % sample]))
            #     din_mvsnet = load_pfm(open(pfm_dir, 'rb'))


            #     din_mvsnet = resize(din_mvsnet)
            #     din_mvsnet = scale(din_mvsnet, gt, ij)

            #     dinss['mvsnet'][sample] = din_mvsnet
            # if 'mvsnet_f' in algos:

            #     pfm_dir = os.path.join(mvs_folder, 'MVS', scene, '%s_prob_filtered.pfm' % str(mapper['%s.jpg' % sample]))
            #     din_mvsnet_f = load_pfm(open(pfm_dir, 'rb'))


            #     din_mvsnet_f = resize(din_mvsnet_f)
            #     din_mvsnet_f = scale(din_mvsnet_f, gt, ij)

            #     dinss['mvsnet_f'][sample] = din_mvsnet_f
            # if 'mvsnet_eth3d' in algos:

            #     pfm_dir = os.path.join(mvs_folder, 'MVS_eth3d', scene, '%s_init.pfm' % str(mapper['%s.jpg' % sample]))
            #     din_mvsnet_eth3d = load_pfm(open(pfm_dir, 'rb'))


            #     din_mvsnet_eth3d = resize(din_mvsnet_eth3d)
            #     din_mvsnet_eth3d = scale(din_mvsnet_eth3d, gt, ij)

            #     dinss['mvsnet_eth3d'][sample] = din_mvsnet_eth3d
            # if 'mvsnet_eth3d_f' in algos:

            #     pfm_dir = os.path.join(mvs_folder, 'MVS_eth3d', scene, '%s_prob_filtered.pfm' % str(mapper['%s.jpg' % sample]))
            #     din_mvsnet_eth3d_f = load_pfm(open(pfm_dir, 'rb'))


            #     din_mvsnet_eth3d_f = resize(din_mvsnet_eth3d_f)
            #     din_mvsnet_eth3d_f = scale(din_mvsnet_eth3d_f, gt, ij)

            #     dinss['mvsnet_eth3d_f'][sample] = din_mvsnet_eth3d_f
            # for this scene, this sample, each algo, calculate mse, abe and Q25
            for algo in algos:
                if sample not in dinss[algo]:
                    continue
                din = dinss[algo][sample]
                mse = 0.
                abe = []

                for (i, j) in gt:
                    if din[i][j] == 0.:
                       diff = abs(1./gt[(i, j)] - 1./np.max(din))
                    else:  
                       diff = abs(1./gt[(i, j)] - 1./din[i][j])
                    mse += diff**2
                    abe.append(diff)

                if sample not in stats[scene][algo]:
                    stats[scene][algo][sample] = {}

                stats[scene][algo][sample]['mse'] = mse / float(len(abe)) 
                stats[scene][algo][sample]['abe'] = sum(abe) / float(len(abe))
                stats[scene][algo][sample]['Q25'] = np.percentile(np.array(abe), 25)
        oses = None
        rmgs = None
        for sname in dinss[algos[0]]:
            if oses is None:
                oses = torch.from_numpy(data[sname]['pose']).unsqueeze(0).cuda()
                rmgs = torch.from_numpy(data[sname]['img']).permute(2, 0, 1).unsqueeze(0).cuda()

            else:
                oses = torch.cat((oses, torch.from_numpy(data[sname]['pose']).unsqueeze(0).cuda()), dim=0)
                rmgs = torch.cat((rmgs, torch.from_numpy(data[sname]['img']).permute(2, 0, 1).unsqueeze(0).cuda()), dim=0)
        oses = oses.double()
        rmgs = rmgs.double() 
 
        # for this scene, each algo, each image, calculate photo and cons 
        for algo in algos:
            dins = dinss[algo]

            deps = None
            for sname in dins:
                if deps is None:

                    deps = torch.from_numpy(dins[sname]).unsqueeze(0).cuda()     
                else:

                    deps = torch.cat((deps, torch.from_numpy(dins[sname]).unsqueeze(0).cuda()), dim=0) 
            deps = deps.double()
            for ref_idx, sname in enumerate(dins):
                nrs_idx = list(range(oses.shape[0]))
                nrs_idx.remove(ref_idx)
                nrs_idx = torch.tensor(nrs_idx).cuda()
                depths_in_ref = fwd_depth(oses[ref_idx, ...], oses, deps, data[sname]['cam'])

                rgbs_in_ref, rgbs_invalid = warp(oses[ref_idx, ...], oses, rmgs, deps[ref_idx, ...].unsqueeze(0), data[sname]['cam'])
                

                stats[scene][algo][sname]['photo'] = float(torch.mean(torch.abs(rgbs_in_ref.permute(1, 0, 2, 3)[:, ~rgbs_invalid] - rmgs.permute(1, 0, 2, 3)[:, ~rgbs_invalid]) / 255.).cpu())

                depths_invalid = depths_in_ref<=eps

                depths_in_ref[depths_invalid] = torch.min(depths_in_ref[depths_in_ref>eps])

                depths_invalid = torch.any(depths_invalid, 0, keepdim=True)

         
                variance = torch.var(depths_in_ref, dim=0, keepdim=True)

                stats[scene][algo][sname]['const'] = float((torch.sum(variance[~depths_invalid])/torch.sum((~depths_invalid).long())).cpu())



        # store stats once a scene is done in case of shut
        with open('result.json', 'w') as f:
            json.dump(stats, f)


# warp(in_tgt_pose_c2w, in_src_poses_c2w, in_src, in_tgt_depth, in_cam)

#added by adam
#frame1/2 don't need leading zeroes, they will be added
#given ids/frames, using fwd_depth (in this file), to warp one frame into another
def depth_warp(id, frame1, frame2):
    frame1_added_zeroes = f'{frame1:04}' #fill to 4 digits
    frame2_added_zeroes = f'{frame2:04}'

    cam_intrinsic_matrix, extrinsics_per_frame = extractCameraInfo(id)
    fx = float(cam_intrinsic_matrix[0, 0])
    fy = float(cam_intrinsic_matrix[1, 1])
    w = float(cam_intrinsic_matrix[0, 2])
    h = float(cam_intrinsic_matrix[1, 2])
    cam = [w, h, fx, fy]

    #camera to world matrices for src and target frames
    extrinsics_frame1 = torch.tensor(extrinsics_per_frame[frame1+1], dtype=torch.float32).cuda() #this dict is 1-indexed 
    extrinsics_frame2 = torch.tensor(extrinsics_per_frame[frame2+1], dtype=torch.float32).cuda()

    src_depth_path = "/home/adam/Desktop/repos/mannequin-dataset/data-copy/" + id + "/depth/frame" + str(frame1_added_zeroes) + ".jpg"
    
    
    resize_transform = torchvision.transforms.Resize((int(h),int(w)), antialias = True) #antialias flag doesn't seem to do much
    #TODO doesn't make sense that i'd need to resize the depth to fit w/h
    src_depth = resize_transform(torchvision.io.read_image(src_depth_path)).cuda() 

    print(extrinsics_frame1)

    # output = fwd_depth(extrinsics_frame2, extrinsics_frame1, src_depth, cam)
    save_image(output/255, 'test_results/depth_id_{}_frame_{}_to_{}_Yiqing_version.png'.format(id, frame1, frame2))

def fwd_img(tgt_c2w, src_c2w, img, cam):
    return torch.inverse(tgt_c2w) * src_c2w * img

#added by adam
#wanted to take the depth component out of the equation. Can I warp image a into image b?
def image_warp(id, frame1, frame2):
    frame1_added_zeroes = f'{frame1:04}' #fill to 4 digits
    frame2_added_zeroes = f'{frame2:04}'

    cam_intrinsic_matrix, extrinsics_per_frame = extractCameraInfo(id)
    fx = float(cam_intrinsic_matrix[0, 0])
    fy = float(cam_intrinsic_matrix[1, 1])
    w = float(cam_intrinsic_matrix[0, 2])
    h = float(cam_intrinsic_matrix[1, 2])
    cam = [w, h, fx, fy]

    #camera to world matrices for src and target frames
    extrinsics_frame1 = torch.tensor(extrinsics_per_frame[frame1+1], dtype=torch.float32).cuda() #this dict is 1-indexed 
    extrinsics_frame2 = torch.tensor(extrinsics_per_frame[frame2+1], dtype=torch.float32).cuda()

    src_img_path = "/home/adam/Desktop/repos/mannequin-dataset/data-copy/" + id + "/images/frame" + str(frame1_added_zeroes) + ".jpg"
    
    
    resize_transform = torchvision.transforms.Resize((int(h),int(w)), antialias = True) #antialias flag doesn't seem to do much
    #TODO doesn't make sense that i'd need to resize the depth to fit w/h
    src_img = resize_transform(torchvision.io.read_image(src_img_path)).cuda() 

    output = fwd_img(extrinsics_frame2, extrinsics_frame1, src_img, cam)
    save_image(src_img/255, 'test_results/image_warp_id_{}_frame_{}_to_{}.png'.format(id, frame1, frame2))

# depth_warp("4ea9094e0cbf1972", 53, 54)
image_warp("4ea9094e0cbf1972", 53, 54)

# for i in range(1, 53):
#     depth_warp("e128dcbcc059c94f", i, 40)
    
#### trying yiqing's load poses 
# id = "e128dcbcc059c94f"
# data = load_poses("/home/adam/Desktop/repos/mannequin-dataset/data-copy/" + id, 640,  380)
# torch.from_numpy(data[sname]['pose']).unsqueeze(0).cuda()

#trying this warp fn instead

