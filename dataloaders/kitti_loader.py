import os
import os.path
import glob
import fnmatch # pattern matching
import numpy as np
from numpy import linalg as LA
from  random import choice
from PIL import Image
import torch
import torch.utils.data as data
import cv2
import h5py
from dataloaders import transforms
from dataloaders.pose_estimator import get_pose_pnp
from random import choice


'''
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
'''

root_d = os.path.join('/home/lab/huangxinghong', 'depth_data')
root_rgb = os.path.join('/home/lab/huangxinghong', 'rgb_data')

#获取需要的图片和深度图路径
def get_paths_and_transform(split, args):
    if split == "train":
        glob_gt = "train/*_sync/proj_depth/velodyne_raw/image_0[2]/*.png"
        glob_gt_right = "train/*_sync/proj_depth/velodyne_raw/image_0[3]/*.png"

        def get_rgb_paths(p):
            ps = p.split('/')
            pnew = '/'.join([root_rgb]+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
            return pnew

        def get_date(p):
            ps = p.split('/')
            date = ps[-5:-4][0][0:10] #将日期取出来，因为每个日期的内参都是不一样的，所以需要修改
            return date

    elif split == "val":
         glob_gt = "val/*_sync/proj_depth/velodyne_raw/image_0[2]/*.png"
         glob_gt_right = "val/*_sync/proj_depth/velodyne_raw/image_0[3]/*.png"

         def get_rgb_paths(p):
             ps = p.split('/')
             pnew = '/'.join([root_rgb]+ps[-6:-4]+ps[-2:-1]+['data']+ps[-1:])
             return pnew

         def get_date(p):
             ps = p.split('/')
             date = ps[-5:-4][0][0:10]
             return date

    else:
        raise ValueError("Unrecognized split "+str(split))

    if glob_gt is not None:
        glob_gt = os.path.join(root_d, glob_gt) #左相机
        glob_gt_right = os.path.join(root_d, glob_gt_right)  # 右相机

        paths_gt = sorted(glob.glob(glob_gt))
        paths_gt_right = sorted(glob.glob(glob_gt_right))


        paths_rgb = [get_rgb_paths(p) for p in paths_gt]
        paths_rgb_right = [get_rgb_paths(j) for j in paths_gt_right]

        date = [get_date(p) for p in paths_gt]
    else:
        raise ValueError("The depth ground truth is empty !!")

    if len(paths_rgb) == 0 or len(paths_gt) == 0 or len(paths_rgb_right)==0 or len(paths_gt_right)==0:
        raise(RuntimeError("Found 0 images in data folders"))
    if len(paths_rgb) != len(paths_gt) or len(paths_rgb_right) != len(paths_gt_right):
        raise(RuntimeError("Produced different sizes for datasets"))

    paths = {"rgb": paths_rgb, "gt": paths_gt, "rgb_right": paths_rgb_right, "date": date}
    return paths


def load_calib():
    """
    读取内参矩阵
    """
    def intrinsics(date):
        calib = open("dataloaders/"+str(date)+".txt", "r")
        lines = calib.readlines()
        P_rect_line = lines[25]

        Proj_str = P_rect_line.split(":")[1].split(" ")[1:]
        Proj = np.reshape(np.array([float(p) for p in Proj_str]),(3,4)).astype(np.float32)
        K = Proj[:3,:3] # camera matrix

        #　由于对图片进行了裁剪，因此需要对主点的坐标进行变换
        K[0,2] = K[0,2] - 13 # from width = 1242 to 1216, with a 13-pixel cut on both sides
        K[1,2] = K[1,2] - 11.5 # from width = 375 to 352, with a 11.5-pixel cut on both sides
        return K

    date_set = ["2011_09_26", "2011_09_26", "2011_09_28", "2011_09_29", "2011_09_30", "2011_10_03"]
    K_set = {i: intrinsics(i) for i in date_set}
    return K_set

def load_transfrom( ):
    """
    读取左右相机的转换关系
    """
    def load_R_t(date):
        calib_t = open("dataloaders/" + str(date) + ".txt", "r")
        transforms = calib_t.readlines()

        lines_R_0to2 = transforms[21] #R_02
        R_0to2_str = lines_R_0to2.split(":")[1].split(" ")[1:]
        R_0to2 = np.reshape(np.array([float(p) for p in R_0to2_str]), (3,3)).astype(np.float32)

        lines_t_0to2 = transforms[22] #T_02
        t_0to2_str = lines_t_0to2.split(":")[1].split(" ")[1:]
        t_0to2 = np.reshape(np.array([float(p) for p in t_0to2_str]), (3, 1)).astype(np.float32)

        lines_R_0to3 = transforms[29] #R_03
        R_0to3_str = lines_R_0to3.split(":")[1].split(" ")[1:]
        R_0to3 = np.reshape(np.array([float(p) for p in R_0to3_str]),(3,3)).astype(np.float32)

        lines_t_0to3 = transforms[30] #T_03
        t_0to3_str = lines_t_0to3.split(":")[1].split(" ")[1:]
        t_0to3 = np.reshape(np.array([float(p) for p in t_0to3_str]), (3, 1)).astype(np.float32)

        R_0to2_inv = np.linalg.inv(R_0to2)
        R_0to3_times_R_0to2_inv = np.matmul(R_0to3, R_0to2_inv)

        R_0to3_inv = np.linalg.inv(R_0to3)
        R_0to2_times_R_0to3_inv = np.matmul(R_0to2, R_0to3_inv)

        return R_0to3_times_R_0to2_inv, t_0to2, t_0to3, R_0to2_times_R_0to3_inv

    date_set = ["2011_09_26", "2011_09_26", "2011_09_28", "2011_09_29", "2011_09_30", "2011_10_03"]
    R_0to3_times_R_0to2_inv_set = {i: load_R_t(i)[0] for i in date_set}
    t_0to2_set = {j: load_R_t(j)[1] for j in date_set}
    t_0to3_set = {k: load_R_t(k)[2] for k in date_set}
    R_0to2_times_R_0to3_inv_set = {h: load_R_t(h)[3] for h in date_set}

    return R_0to3_times_R_0to2_inv_set, t_0to2_set, t_0to3_set, R_0to2_times_R_0to3_inv_set

#读取rgb图片
def rgb_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    # rgb_png = np.array(img_file, dtype=float) / 255.0 # scale pixels to the range [0,1]
    rgb_png = np.array(img_file, dtype='uint8') # in the range [0,255]
    img_file.close()
    return rgb_png

#读取深度图,代码来自kitti官网
def depth_read(filename):
    assert os.path.exists(filename), "file not found: {}".format(filename)
    img_file = Image.open(filename)
    depth_png = np.array(img_file, dtype=int)
    img_file.close()
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert np.max(depth_png) > 255, \
        "np.max(depth_png)={}, path={}".format(np.max(depth_png),filename)

    depth = depth_png.astype(np.float) / 256.
    # depth[depth_png == 0] = -1.
    # depth = np.expand_dims(depth,-1) #(x,y) -> (1,x,y)
    return depth

oheight, owidth = 352, 1200

def drop_depth_measurements(depth, prob_keep):
    mask = np.random.binomial(1, prob_keep, depth.shape)
    depth *= mask
    return depth

def train_transform(rgb, depth, rgb_right, rgb_near, depth_near, args):
    # s = np.random.uniform(1.0, 1.5) # random scaling
    # angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
    do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

    transform_geometric = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
        transforms.HorizontalFlip(do_flip)#实现矩阵左右翻转
    ])

    depth = transform_geometric(depth)
    depth_near = transform_geometric(depth_near)

    if rgb is not None:
        brightness = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        contrast = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        saturation = np.random.uniform(max(0, 1 - args.jitter), 1 + args.jitter)
        transform_rgb = transforms.Compose([
            transforms.ColorJitter(brightness, contrast, saturation, 0),
            #transforms.LimitedEqualize(4.0),
            transform_geometric
        ])
        rgb = transform_rgb(rgb)
        rgb_right = transform_rgb(rgb_right)
        rgb_near = transform_rgb(rgb_near)

    return rgb, depth, rgb_right, rgb_near, depth_near, do_flip


def val_transform(rgb, depth, rgb_right, rgb_near, depth_near, args):
    transform = transforms.Compose([
        transforms.BottomCrop((oheight, owidth)),
        #transforms.LimitedEqualize(4.0),
    ])
    if rgb is not None:
        rgb = transform(rgb)
    if depth is not None:
        depth = transform(depth)
    if rgb_near is not None:
        rgb_near = transform(rgb_near)
    if depth_near is not None:
        depth_near = transform(depth_near)
    if rgb_right is not None:
        rgb_right = transform(rgb_right)

    do_flip = False
    return rgb, depth, rgb_right, rgb_near, depth_near, do_flip

to_tensor = transforms.ToTensor()
to_float_tensor = lambda x: to_tensor(x).float()


def get_rgb_near(path, args):
    assert path is not None, "path is None"

    def extract_frame_id(filename):
        head, tail = os.path.split(filename)
        number_string = tail[0:tail.find('.')]
        number = int(number_string)
        return head, number

    def get_nearby_filename(filename, new_id):
        head, _ = os.path.split(filename)
        new_filename = os.path.join(head, '%010d.png' % new_id)
        return new_filename

    def get_depth_paths(p):
        ps = p.split('/')
        depth_near_path = '/'.join([root_d] + ps[-5:-3] +['proj_depth']+ ['velodyne_raw'] + ps[-3:-2] + ps[-1:])
        return depth_near_path

    head, number = extract_frame_id(path) #得到当前图片的id
    max_frame_diff = 3
    candidates = [i-max_frame_diff+1 for i in range(max_frame_diff*2+1) if i-max_frame_diff+1!=0]

    random_offset = choice(candidates)
    path_near = get_nearby_filename(path, number+random_offset)
    if os.path.exists(path_near):
       depth_near = get_depth_paths(path_near)
       #print("near rgb"+str(path_near))
       #print("near_depth"+str(depth_near))
       return rgb_read(path_near), depth_read(depth_near)
    else:
       random_offset = -random_offset
       path_near = get_nearby_filename(path, number+random_offset)
       if os.path.exists(path_near):
          depth_near = get_depth_paths(path_near)
          #print(path_near)
          #print(depth_near)
          return rgb_read(path_near),depth_read(depth_near)
       else:
          raise (RuntimeError("Select nearby image wrong"))

class KittiDepth(data.Dataset):
    """A data loader for the Kitti dataset
    """
    def __init__(self, split, args):
        self.args = args
        self.split = split
        self.paths = get_paths_and_transform(split, args)
        if split == 'train':
           self.transform = train_transform
        elif split == 'val':
           self.transform = val_transform
        else:
           print("the split mode is wrong !!!")

        self.K_set = load_calib()
        self.R_0to3_times_R_0to2_inv_set, self.t_0to2_set, self.t_0to3_set, self.R_0to2_times_R_0to3_inv_set = load_transfrom()
        self.threshold_translation = 0.1
        self.num_samples = args.num_sample
        self.max_depth = 100.0

    def dense_to_sparse(self, depth):  # 稠密到稀疏点的选取
        n_keep = np.count_nonzero(depth)
        if n_keep == 0:
            return np.zeros(depth.shape)
        else:
            prob = float(self.num_samples) / n_keep
            mask = np.random.uniform(0, 1, depth.shape)< prob
            depth *= mask
            return depth

    def __getraw__(self, index):
       # print(self.paths['rgb'][index])
       # print(self.paths['gt'][index])
        rgb = rgb_read(self.paths['rgb'][index])
        rgb_right = rgb_read(self.paths['rgb_right'][index])
        depth = depth_read(self.paths['gt'][index])

        # 获取nearby rgb图片
        rgb_near, depth_near = get_rgb_near(self.paths['rgb'][index], self.args)

        # 获取矩阵内参以及投影关系
        intrinsics = self.K_set[str(self.paths['date'][index])]
        R_0to3_times_R_0to2_inv = self.R_0to3_times_R_0to2_inv_set[str(self.paths['date'][index])]
        t_0to2 = self.t_0to2_set[str(self.paths['date'][index])]
        t_0to3 = self.t_0to3_set[str(self.paths['date'][index])]
        R_0to2_times_R_0to3_inv = self.R_0to2_times_R_0to3_inv_set[str(self.paths['date'][index])]


        return rgb, depth, rgb_right, rgb_near, depth_near, intrinsics, R_0to3_times_R_0to2_inv, t_0to2, t_0to3, R_0to2_times_R_0to3_inv, self.paths['rgb'][index]

    def __getitem__(self, index):

        rgb, depth, rgb_right, rgb_near, depth_near, intrinsics, R_0to3_times_R_0to2_inv, t_0to2, t_0to3, R_0to2_times_R_0to3_inv_do_flip, path = self.__getraw__(index)
        rgb, depth, rgb_right, rgb_near, depth_near, do_flip = self.transform(rgb, depth, rgb_right, rgb_near, depth_near, self.args)

        #if do_flip:
        #    R_0to3_times_R_0to2_inv = R_0to2_times_R_0to3_inv_do_flip
        #    t_0to2_do_flip = t_0to2
        #    t_0to3_do_flip = t_0to3
        #    t_0to3 = t_0to2_do_flip
        #    t_0to2 = t_0to3_do_flip

        #print(proj_k)
        r_mat, t_vec = None, None
        candidates = {}
        if self.split == 'train':
            success, r_vec, t_vec, r_vec_inv, t_vec_inv, _, depth_curr_dilated, _, depth_near_dilated, pts_for_loss, pts_for_loss_near = get_pose_pnp(rgb, rgb_near, depth, \
                                                                                                depth_near, intrinsics)

            #print(abs(LA.norm(t_vec) - LA.norm(t_vec_inv)))
            if success:
                if abs(LA.norm(t_vec)-LA.norm(t_vec_inv))>0.08:
                   success = False

            if success and LA.norm(t_vec) > self.threshold_translation:
                r_mat, _ = cv2.Rodrigues(r_vec)
                r_mat_inv, _ = cv2.Rodrigues(r_vec_inv)
            else:
                # return the same image and no motion when PnP fails
                rgb_near = rgb
                depth_near = depth
                pts_for_loss_near = pts_for_loss
                t_vec = np.zeros((3,1))
                r_mat = np.eye(3)
                t_vec_inv = np.zeros((3, 1))
                r_mat_inv = np.eye(3)

            sparse_input = self.dense_to_sparse(depth.copy())
            sparse_input_near = self.dense_to_sparse(depth_near.copy())

            pts_for_loss = np.array(pts_for_loss)
            sparse_input = np.expand_dims(sparse_input, axis = 2)
            sparse_input_near = np.expand_dims(sparse_input_near, axis = 2)
            depth_curr_dilated = np.expand_dims(depth_curr_dilated, axis=2)
            depth_near_dilated = np.expand_dims(depth_near_dilated, axis=2)
            depth = np.expand_dims(depth, axis=2)
            depth_near = np.expand_dims(depth_near, axis=2)

            candidates = {"rgb": rgb, "d": sparse_input, "d_dilated": depth_curr_dilated, "gt": depth, "r_mat": r_mat,
                          "t_vec": t_vec, "r_mat_inv": r_mat_inv, "t_vec_inv": t_vec_inv, \
                          "rgb_near": rgb_near, "d_near": sparse_input_near, "d_near_dilated": depth_near_dilated,
                          "gt_near": depth_near, 'K': intrinsics, "rgb_right": rgb_right, \
                          "R_0to3_times_R_0to2_inv": R_0to3_times_R_0to2_inv, "t_0to2": t_0to2, \
                          "t_0to3": t_0to3}

        else:
            sparse_input = self.dense_to_sparse(depth.copy())
            sparse_input = np.expand_dims(sparse_input, axis=2)
            depth = np.expand_dims(depth, axis=2)
            candidates = {"rgb": rgb, "d": sparse_input, "gt": depth}

        items = {key: to_float_tensor(val) for key, val in candidates.items() if val is not None}
        items["path"]  =  path
        return items

    def __len__(self):
        return len(self.paths['rgb'])
