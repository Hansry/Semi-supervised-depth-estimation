import torch
import torch.nn.functional as F
import numpy as np

class Intrinsics:
    def __init__(self, width, height, fu, fv, cu=0, cv=0):
        self.height, self.width = height, width
        self.fu, self.fv = fu, fv # fu, fv: focal length along the horizontal and vertical axes

        # cu, cv: optical center along the horizontal and vertical axes
        self.cu = cu if cu>0 else (width-1) / 2.0
        self.cv = cv if cv>0 else (height-1) / 2.0

        # U, V represent the homogeneous horizontal and vertical coordinates in the pixel space
        self.U = torch.arange(start=0, end=width).expand(height, width).float()
        self.V = torch.arange(start=0, end=height).expand(width, height).t().float()

        # X_cam, Y_cam represent the homogeneous x, y coordinates (assuming depth z=1) in the camera coordinate system
        self.X_cam = (self.U-self.cu) / self.fu
        self.Y_cam = (self.V-self.cv) / self.fv

        self.is_cuda = False

    def cuda(self):
        self.X_cam.data = self.X_cam.data.cuda()
        self.Y_cam.data = self.Y_cam.data.cuda()
        self.is_cuda = True
        return self

    def scale(self, height, width):
        # return a new set of corresponding intrinsic parameters for the scaled image
        ratio_u = float(width) / self.width
        ratio_v = float(height) / self.height
        fu = ratio_u * self.fu
        fv = ratio_v * self.fv
        cu = ratio_u * self.cu
        cv = ratio_v * self.cv
        new_intrinsics = Intrinsics(width, height, fu, fv, cu, cv)
        if self.is_cuda:
            new_intrinsics.cuda()
        return new_intrinsics

    def __print__(self):
        print('size=({},{})\nfocal length=({},{})\noptical center=({},{})'.format(
            self.height, self.width, self.fv, self.fu, self.cv, self.cu
            ))

def image_to_pointcloud(depth, intrinsics):
    assert depth.dim() == 4
    assert depth.size(1) == 1

    X = torch.unsqueeze(depth[0, :, :, :], 0) * intrinsics[0].X_cam
    Y = torch.unsqueeze(depth[0, :, :, :], 0) * intrinsics[0].Y_cam

    for i in range(1, depth.shape[0]):
        X_1 = torch.unsqueeze(depth[i, :, :, :], 0) * intrinsics[i].X_cam
        Y_2 = torch.unsqueeze(depth[i, :, :, :], 0) * intrinsics[i].Y_cam
        X = torch.cat((X, X_1), dim=0)  # 遍历将逐个batch相乘得到空间点云
        Y = torch.cat((Y, Y_2), dim=0)

    return torch.cat((X, Y, depth), dim=1)

def pointcloud_to_image(pointcloud, intrinsics):
    assert pointcloud.dim() == 4

    #batch_size 的大小
    batch_size = pointcloud.size(0)
    X = pointcloud[:, 0, :, :] #.view(batch_size, -1)
    Y = pointcloud[:, 1, :, :] #.view(batch_size, -1)
    Z = pointcloud[:, 2, :, :].clamp(min=1e-3) #.view(batch_size, -1)

    #　将三维空间点重新投影到像平面上
    U_proj = intrinsics[0].fu * torch.unsqueeze(X[0, :, :], 0) / torch.unsqueeze(Z[0, :, :], 0) + intrinsics[0].cu # horizontal pixel coordinate
    V_proj = intrinsics[0].fv * torch.unsqueeze(Y[0, :, :], 0) / torch.unsqueeze(Z[0, :, :], 0) + intrinsics[0].cv # vertical pixel coordinate

    for j in range(1, batch_size):
        U_proj_1 = intrinsics[j].fu * torch.unsqueeze(X[j, :, :], 0) / torch.unsqueeze(Z[j, :, :], 0) + intrinsics[j].cu  # horizontal pixel coordinate
        V_proj_1 = intrinsics[j].fv * torch.unsqueeze(Y[j, :, :], 0) / torch.unsqueeze(Z[j, :, :], 0) + intrinsics[j].cv  # vertical pixel coordinate

        U_proj = torch.cat((U_proj, U_proj_1), dim=0)
        V_proj = torch.cat((V_proj, V_proj_1), dim=0)

    # 对于不同的intrinsics而言，尽管其fu, fv, cu, cv是不一样的，但是其　width 和　height　是一样的
    # 将投影后的坐标归一化到[-1,1]之间，torch.nn.functional.grid_sample需要
    U_proj_normalized = (2 * U_proj / (intrinsics[0].width-1) - 1).view(batch_size, -1)
    V_proj_normalized = (2 * V_proj / (intrinsics[0].height-1) - 1).view(batch_size, -1)

    # This was important since PyTorch didn't do as it claimed for points out of boundary
    # See https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    # Might not be necessary any more
    U_proj_mask = ((U_proj_normalized > 1) + (U_proj_normalized < -1)).detach()
    U_proj_normalized[U_proj_mask] = 2
    V_proj_mask = ((V_proj_normalized > 1) + (V_proj_normalized < -1)).detach()
    V_proj_normalized[V_proj_mask] = 2

    #然后把投影后的坐标resize成(batch_size, height, width, 2)的大小，返回的只是坐标而已
    pixel_coords = torch.stack([U_proj_normalized, V_proj_normalized], dim=2)  # [B, H*W, 2]
    return pixel_coords.view(batch_size, intrinsics[0].height, intrinsics[0].width, 2)

def batch_multiply(batch_scalar, batch_matrix):
    # input: batch_scalar of size b, batch_matrix of size b * 3 * 3
    # output: batch_matrix of size b * 3 * 3
    batch_size = batch_scalar.size(0)
    output = batch_matrix.clone()
    for i in range(batch_size):
        output[i] = batch_scalar[i] * batch_matrix[i]
    return output

def transform_curr_to_near(pointcloud_curr, r_mat, t_vec, intrinsics):
    # translation and rotmat represent the transformation from tgt pose to src pose
    batch_size = pointcloud_curr.size(0)
    XYZ_ = torch.bmm(r_mat, pointcloud_curr.view(batch_size, 3, -1))

    X = (XYZ_[:,0,:] + t_vec[:,0].unsqueeze(1)).view(-1,1,intrinsics[0].height, intrinsics[0].width) #注意这里需要加[0]是因为虽然不同的batch下intrinsics是不一样的,但是其尺寸是一样的
    Y = (XYZ_[:,1,:] + t_vec[:,1].unsqueeze(1)).view(-1,1,intrinsics[0].height, intrinsics[0].width)
    Z = (XYZ_[:,2,:] + t_vec[:,2].unsqueeze(1)).view(-1,1,intrinsics[0].height, intrinsics[0].width)

    pointcloud_near = torch.cat((X, Y, Z), dim=1)

    return pointcloud_near, Z

def transform_left_to_right(pointcloud_left, R_0to3_times_R_0to2_inv, t_0to2, t_0to3, intrinsics):
    # 读取kitti上的姿态,利用左相机到右相机的投影
    # P_{right} = R_{0->3} * (R_{0->2})^{-1} * (P_{left} - t_{0->2}) + t_{0->3}
    batch_size = pointcloud_left.size(0)
    XYZ_temp = pointcloud_left.view(batch_size, 3, -1) - t_0to2
    XYZ_ = torch.bmm(R_0to3_times_R_0to2_inv, XYZ_temp)

    X = (XYZ_[:, 0, :] + t_0to3[:, 0]).view(-1, 1, intrinsics[0].height, intrinsics[0].width) #注意这里需要加[0]是因为虽然不同的batch下intrinsics是不一样的,但是其尺寸是一样的
    Y = (XYZ_[:, 1, :] + t_0to3[:, 1]).view(-1, 1, intrinsics[0].height, intrinsics[0].width)
    Z = (XYZ_[:, 2, :] + t_0to3[:, 2]).view(-1, 1, intrinsics[0].height, intrinsics[0].width)

    pointcloud_near = torch.cat((X, Y, Z), dim=1)

    return pointcloud_near, Z

'''
def transform_curr_to_near_depth(pointcloud_curr, r_mat, t_vec, intrinsics):

    batch_size = pointcloud_curr.size(0)
    XYZ_ = torch.bmm(r_mat, pointcloud_curr.view(batch_size, 3, -1))
    Z = (XYZ_[:,2,:] + t_vec[:,2].unsqueeze(1)).view(-1,1,intrinsics.height,intrinsics.width)

    return transform_depth
'''

def homography_from(rgb_near, depth_curr, r_mat, t_vec, intrinsics):
    # inverse warp the RGB image from the nearby frame to the current frame

    # to ensure dimension consistency
    r_mat = r_mat.view(-1, 3, 3)
    t_vec = t_vec.view(-1, 3)

    # compute source pixel coordinate
    pointcloud_curr = image_to_pointcloud(depth_curr, intrinsics)
    pointcloud_near, _ = transform_curr_to_near(pointcloud_curr, r_mat, t_vec, intrinsics)
    pixel_coords_near = pointcloud_to_image(pointcloud_near, intrinsics)

    # the warping
    warped = F.grid_sample(rgb_near, pixel_coords_near)

    return warped


def transform_covisual(pointcloud_curr, r_mat, t_vec, intrinsics):
    # translation and rotmat represent the transformation from tgt pose to src pose
    batch_size = pointcloud_curr.size(0)
    XYZ_ = torch.bmm(r_mat, pointcloud_curr.view(batch_size, 3, -1))

    Z = (XYZ_[:,2,:] + t_vec[:,2].unsqueeze(1)).view(-1,1,intrinsics[0].height, intrinsics[0].width)

    return Z

def Covisual_from(depth_curr, depth_curr_dilated, depth_near, depth_near_dilated, r_mat, t_vec, intrinsics):

    r_mat = r_mat.view(-1,3,3)
    t_vec = t_vec.view(-1,3)

    #利用groundtruth膨胀后的真实值，利用膨胀后的点进行warping
    pointcloud_curr_dilated = image_to_pointcloud(depth_curr_dilated, intrinsics)
    pointcloud_near_dilated, depth_curr_dilated_transform  = transform_curr_to_near(pointcloud_curr_dilated, r_mat, t_vec, intrinsics)
    pixel_coords_near = pointcloud_to_image(pointcloud_near_dilated, intrinsics)

    #当前图片在t+1时刻深度的投影
    warped_depth_dilated = F.grid_sample(depth_near_dilated, pixel_coords_near)

    #当前图片预测的深度在t+1预测的深度的投影
    pointcloud_curr = image_to_pointcloud(depth_curr, intrinsics)
    depth_curr_transform  = transform_covisual(pointcloud_curr, r_mat, t_vec, intrinsics)
    warped_depth = F.grid_sample(depth_near, pixel_coords_near)


    valid_mask = (warped_depth_dilated > 0).float()*(depth_curr_dilated_transform > 0).float()

    mask = ((depth_curr_dilated_transform*valid_mask - warped_depth_dilated*valid_mask).abs()<0.08).float()
    mask = mask * valid_mask

    nonZeroCount = np.count_nonzero(mask.detach().cpu())
    #print("nonZerosCount: "+str(nonZeroCount))

    return depth_curr_transform, depth_curr_dilated_transform, warped_depth, warped_depth_dilated, mask


def left_right_from(rgb_right, depth_left, R_0to3_times_R_0to2_inv, t_0to2, t_0to3, intrinsics):

    pointcloud_left = image_to_pointcloud(depth_left, intrinsics)
    pointcloud_right, _ = transform_left_to_right(pointcloud_left, R_0to3_times_R_0to2_inv, t_0to2, t_0to3, intrinsics)
    pixel_coords_right = pointcloud_to_image(pointcloud_right, intrinsics)
    warped_right = F.grid_sample(rgb_right, pixel_coords_right)

    return warped_right