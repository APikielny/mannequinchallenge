import numpy as np
from PIL import Image
import os

def load_pts3d(fin):
    xyz = []
    rgb = []
    idx = []
    
    with open(fin, 'r') as freader:
        for line in freader:
            tokens = line.split()

            if not tokens or tokens[0] == '#':
                continue
            else:
                idx.append(int(tokens[0]))
                xyz.append([float(i) for i in tokens[1:4]])
                rgb.append([float(i) for i in tokens[4:7]])

    idx = np.array(idx)
    xyz = np.array(xyz)
    rgb = np.array(rgb)

    return (xyz, rgb, idx)

def load_depths(fin, img_names):
    depths = []

    if img_names:
        if not [f for f in os.listdir(os.path.join(fin)) if f[-4:] == '.npy']:
            for img_name in img_names:
                dname = img_name[:-4] + '.png'
                dmap = np.asarray(Image.open( os.path.join(fin, dname) )) / (65535.0 * 0.1)
                depths.append(dmap)
        else:
            for img_name in img_names:
                dname = img_name[:-4] + '.npy'
                dmap = np.load( os.path.join(fin, dname) ) 
                depths.append(dmap)

    return np.array(depths)
    

def load_poses(fin, imfolder=None, tgt_height=-1, tgt_width=None):
    if tgt_height == -1:
        assert False, "No scale passed!"
    poses = []
    invposes = []
    qwxyz = []
    txyz = []
    imgs = []
    img_names = []
    
    with open(fin, 'r') as freader:
        skipLine = False

        for line in freader:
            tokens = line.split()

            if skipLine:
                skipLine = False
                continue
            elif not tokens or tokens[0] == '#':
                continue
            else:
                qwxyz = np.array([float(i) for i in tokens[1:5]]) # rotation as a quaternion
                txyz  = np.array([float(i) for i in tokens[5:8]]) # translation

                imfile = tokens[-1]
                if imfolder:
                    img = Image.open(os.path.join(imfolder, imfile))

                    w, h = img.size
                    scale = float(tgt_height) / float(h)
                    img = img.resize( (int(w*scale) if tgt_width is None else int(tgt_width), int(tgt_height)), Image.LANCZOS)

                    imgs.append(np.asarray(img))

                img_names.append(imfile)  
                pose = np.identity(4)
                pose[0:3, 0:3] = quaternion2mat(qwxyz)
                pose[0:3, -1]  = txyz
                poses.append(pose)

                invpose = np.linalg.inv(pose)
                invposes.append(invpose)
                
                skipLine = True

    return (np.array(poses), np.array(invposes), img_names, np.array(imgs))
def visibility4view(fin, fname):
    V = []
    with open(fin, 'r') as freader:
        read_next = False
        for line in freader:
            tokens = line.split()
            if not tokens or tokens[0] == '#':
                continue
            elif tokens[-1] == fname:
                read_next = True
                continue
            if read_next:
                pidx = np.array([i for i in range(2, len(tokens), 3)])
                P = np.array([float(i) for i in tokens])[pidx].astype(int)
                break
    P = P[P!=-1]
    return np.array(P)
'''
def visibility4view(fin, view):
    V = []
    with open(fin, 'r') as freader:
        skipLine = True
        viewIdx = 0
        
        for line in freader:
            tokens = line.split()

            if not tokens or tokens[0] == '#':
                continue
            elif skipLine:
                skipLine = False
                continue
            elif viewIdx != view :
                viewIdx += 1
                skipLine = True
                continue
            else:
                vidx = np.array([i for i in range(2, len(tokens), 3)])
                V = np.array([float(i) for i in tokens])[vidx].astype(int)
                break

    V = V[V != -1]
    return np.array(V)
'''

def quaternion2mat(q):
    rot = np.zeros((3, 3))
    rot[0, 0] = 1 - 2 * q[2] ** 2 - 2 * q[3] ** 2
    rot[0, 1] = 2 * q[1] * q[2] - 2 * q[3] * q[0]
    rot[0, 2] = 2 * q[1] * q[3] + 2 * q[2] * q[0]
    rot[1, 0] = 2 * q[1] * q[2] + 2 * q[3] * q[0]
    rot[1, 1] = 1 - 2 * q[1] ** 2 - 2 * q[3] ** 2
    rot[1, 2] = 2 * q[2] * q[3] - 2 * q[1] * q[0]
    rot[2, 0] = 2 * q[1] * q[3] - 2 * q[2] * q[0]
    rot[2, 1] = 2 * q[2] * q[3] + 2 * q[1] * q[0]
    rot[2, 2] = 1 - 2 * q[1] ** 2 - 2 * q[2] ** 2
    return rot


def load_cameras(fin, tgt_height, tgt_width=None):
    K = []

    with open(fin, 'r') as freader:

        for line in freader:
            tokens = line.split()
            if not tokens or tokens[0] == '#':
                continue
            else:
                # FULL_OPENCV camera model: resx, resy, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
                scale_y = float(tgt_height) / float(tokens[3])
                if tgt_width is None:
                    scale_x = scale_y
                else:
                    scale_x = float(tgt_width) / float(tokens[2])

                cam = [scale_x * float(tokens[2]) if tgt_width is None else tgt_width, tgt_height, scale_x * float(tokens[4]), scale_y * float(tokens[5])]
                cam += [cam[0]/2., cam[1] /2.]
                cam = np.array(cam)

                K.append(cam)

    return np.array(K)


def view2world(u, v, d, invpose, cam):

    # Image space points are obtained by subtracting the principal point (cam[3]/cam[4]),
    # and dividing by the focal length in pixels (cam[2])
    x0 = (u - cam[3]) / cam[2]
    y0 = (v - cam[4]) / cam[2]
    
    # Camera space 
    zc = d
    xc = x0 * zc
    yc = y0 * zc

    # World space
    invpose = np.expand_dims(np.expand_dims(invpose, -1), -1) # To match dimensions for broadcasting below

    x = xc * invpose[0, 0] + yc * invpose[0, 1] + zc * invpose[0, 2] + invpose[0, 3]
    y = xc * invpose[1, 0] + yc * invpose[1, 1] + zc * invpose[1, 2] + invpose[1, 3]
    z = xc * invpose[2, 0] + yc * invpose[2, 1] + zc * invpose[2, 2] + invpose[2, 3]

    return (x, y, z)


def world2view(pts, pose, cam):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    # Camera space points
    xc = x * pose[0, 0] + y * pose[0, 1] + z * pose[0, 2] + pose[0, 3]
    yc = x * pose[1, 0] + y * pose[1, 1] + z * pose[1, 2] + pose[1, 3]
    zc = x * pose[2, 0] + y * pose[2, 1] + z * pose[2, 2] + pose[2, 3]

    # Image space points
    x0 = xc / (1e-19 + zc)
    y0 = yc / (1e-19 + zc)

    # Ignoring radial and tangential distortion for now to make the inverse operation simpler
    u = cam[2] * x0 + cam[3]
    v = cam[2] * y0 + cam[4]
    #v = cam[1] - v - 1
    return np.stack( (u, v, zc), 1)

def world2views(x, y, z, poses, cam, pinhole=1):
    x = x.unsqueeze(0).expand(poses.shape[0], -1, -1)
    y = y.unsqueeze(0).expand(poses.shape[0], -1, -1)
    z = z.unsqueeze(0).expand(poses.shape[0], -1, -1)

    poses = poses.unsqueeze(-1).unsqueeze(-1) # add channels and spatial dimensions

    # Camera space points
    xc = x * poses[:, 0, 0, ...] + y * poses[:, 0, 1, ...] + z * poses[:, 0, 2, ...] + poses[:, 0, 3, ...]
    yc = x * poses[:, 1, 0, ...] + y * poses[:, 1, 1, ...] + z * poses[:, 1, 2, ...] + poses[:, 1, 3, ...]
    zc = x * poses[:, 2, 0, ...] + y * poses[:, 2, 1, ...] + z * poses[:, 2, 2, ...] + poses[:, 2, 3, ...]

    # Image space points
    x0 = xc / zc
    y0 = yc / zc
    r = x0 ** 2 + y0 ** 2

    # Ignoring radial and tangential distortion for now to make the inverse operation simpler
    if pinhole:
        u = cam[2] * x0 + cam[3]
        v = cam[2] * y0 + cam[4]
    
    else:
        u = cam[2] * x0 + cam[4]
        v = cam[3] * y0 + cam[5]

    return (u, v)

def depth2disparity(depth, f=960, b=0.126):
    return f * b / (depth + 1e-19)
