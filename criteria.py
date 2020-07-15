import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F

loss_names = ['l1', 'l2']

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class MaskedL1LogLoss(nn.Module):
    def __init__(self):
        super(MaskedL1LogLoss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        pred = pred * valid_mask
        self.loss = torch.mean(torch.abs(torch.log(target)-torch.log(pred)))
        return self.loss


class BerHu(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHu, self).__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        mask = real > 0
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        pred = pred * valid_mask
        diff = torch.abs(target - pred)
        delta = self.threshold * torch.max(diff).data.cpu().numpy()[0]

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff ** 2 - delta ** 2, 0., -delta ** 2.) + delta ** 2
        part2 = part2 / (2. * delta)

        loss = part1 + part2
        self.loss = torch.sum(loss)
        return self.loss

class SsimLoss(nn.Module):
    def __init__(self):
        super(SsimLoss, self).__init__()
        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x_sq + mu_y_sq + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)


class CovisualLoss(nn.Module):
    def __init__(self, width, height):
        super(CovisualLoss, self).__init__()
        self.loss = 0

        self.width = width
        self.height = height

        self.U = torch.arange(start=0, end=self.width).expand(self.height, self.width).float()
        self.V = torch.arange(start=0, end=self.height).expand(self.width, self.height).t().float()

    def combineImage(self, img1, img2):  # Easy to visulize
        rows1 = img1.shape[0]
        rows2 = img2.shape[0]
        if rows1 < rows2:  # Judge the size of two image
            concat = np.zeros((rows2 - rows1, img1.shape[1], 3), dtype=np.uint8)
            img1 = np.concatenate((img1, concat), axis=0)  # padding
        if rows1 > rows2:
            concat = np.zeros((rows1 - rows2, img2.shape[1], 3), dtype=np.uint8)
            img2 = np.concatenate((img2, concat), axis=0)
        combine_img = np.concatenate((img1, img2), axis=1)  # padding
        return combine_img

    def Homography(self, kp1, kp2):
        kp1 = np.array(kp1)
        kp2 = np.array(kp2)
        if (len(kp1) >= 4):
            H, mask = cv2.findHomography(kp1, kp2, cv2.RANSAC)
            if H is None:
                print('H matrix is None.')
                return [], []
            else:
                kp1 = kp1[mask.ravel() == 1]
                kp2 = kp2[mask.ravel() == 1]
        return kp1, kp2

    def image_to_pointcloud(self, depth, X_cam, Y_cam):
        assert depth.dim() == 4
        assert depth.size(1) == 1

        X = depth * X_cam
        Y = depth * Y_cam
        return torch.cat((X, Y, depth), dim=1)

    def transform_curr_to_near(self, pointcloud_curr, r_mat, t_vec):
        # translation and rotmat represent the transformation from tgt pose to src pose
        batch_size = pointcloud_curr.size(0)
        XYZ_ = torch.bmm(r_mat, pointcloud_curr.view(batch_size, 3, -1))

        Z = (XYZ_[:, 2, :] + t_vec[:, 2].unsqueeze(1)).view(-1, 1, self.height, self.width)
        return Z

    def forward(self, pts_for_loss, pts_for_loss_near, pred, pred_near, R_mat, t_vec, proj_k, batch_data, index):
        assert pred.dim() == pred_near.dim(), "inconsistent dimensions"

        '''
        pts_imshow = pts_for_loss.squeeze()
        pts_imshow_near = pts_for_loss_near.squeeze()

        kp_query, kp_train = self.Homography(pts_imshow, pts_imshow_near)

        print(pred.shape)
        rgb = np.array(pred.detach().cpu())
        rgb_near = np.array(pred_near.detach().cpu())

        rgb = rgb.squeeze()
        rgb_near = rgb_near.squeeze()


        combine_img1 = self.combineImage(rgb, rgb_near)
        combine_img = combine_img1.copy()
        for i in range(len(kp_query)):
            (x1,y1)=kp_query[i]
            (x2,y2)=kp_train[i]
            cv2.line(combine_img, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+self.width),int(np.round(y2))), (255, 0, 0), 1, lineType=cv2.LINE_AA, shift=0)

        cv2.imwrite('/home/huangxinghong/pred/'+str(index)+'.png', combine_img)
        '''

        K = torch.from_numpy(proj_k)

        fu = K[0][0]
        fv = K[1][1]
        cu = K[0][2]
        cv = K[1][2]

        X_cam = ((self.U-cu) / fu).cuda()
        Y_cam = ((self.V-cv) / fv).cuda()

        pointcloud_curr = self.image_to_pointcloud(pred, X_cam, Y_cam)
        pointcloud_near = self.transform_curr_to_near(pointcloud_curr, R_mat, t_vec) #只有深度而已

        for j in range(pts_for_loss.shape[0]): #目前来说只有一个batch，这里以后还需要修改

            pts_for_loss = pts_for_loss.squeeze()
            pts_for_loss_near = pts_for_loss_near.squeeze()

            pts_for_loss, pts_for_loss_near = self.Homography(pts_for_loss, pts_for_loss_near)

            pts_for_loss = pts_for_loss.transpose(1, 0)
            pts_for_loss_near = pts_for_loss_near.transpose(1, 0)

            pts_for_loss[[0, 1], :] = pts_for_loss[[1, 0], :]
            pts_for_loss_near[[0, 1], :] = pts_for_loss_near[[1, 0], :]

            pred_near = pred_near.squeeze()
            pointcloud_near = pointcloud_near.squeeze()

            self.loss_buffer = (pred_near[pts_for_loss_near] - pointcloud_near[pts_for_loss]).abs()

        '''
        loss_buffer = torch.zeros(pts_for_loss.shape[0]*pts_for_loss.shape[1]).cuda()#需要为loss开辟内存
        for j in range(pts_for_loss.shape[0]):
            for i in range(pts_for_loss.shape[1]):
                v_index_near = pts_for_loss[j][i][3].cpu().numpy()
                u_index_near = pts_for_loss[j][i][2].cpu().numpy()

                v_index = pts_for_loss[j][i][1].cpu().numpy()
                u_index = pts_for_loss[j][i][0].cpu().numpy()

                loss_buffer[i+j*pts_for_loss.shape[1]] = (pred_near[j][0][v_index_near][u_index_near] - pointcloud_near[j][0][v_index][u_index]).abs()
        '''
        self.loss = self.loss_buffer.mean()
        return self.loss

class CovisualLossNew(nn.Module):
    def __init__(self):
        super(CovisualLossNew, self).__init__()

    def forward(self, target, recon, mask=None, writer_tensorboard=None, tensorboard_times_count=0):
        assert recon.dim()==4, "expected recon dimension to be 4, but instead got {}.".format(recon.dim())
        assert target.dim()==4, "expected target dimension to be 4, but instead got {}.".format(target.dim())
        assert recon.size()==target.size(), "expected recon and target to have the same size, but got {} and {} instead"\
            .format(recon.size(), target.size())

        diff = (target - recon).abs()
        diff = torch.sum(diff, 1) # sum along the color channel

        # 只针对像素不为空的地方
        valid_mask = (torch.sum(recon, 1)>0).float() * (torch.sum(target, 1)>0).float()
        mask = torch.sum(mask, 1).float()
        #print(target.shape)
        if mask is not None:
            valid_mask = valid_mask * mask

        valid_mask = valid_mask.byte().detach()
        if valid_mask.numel() > 0:
            diff = diff[valid_mask]
            if diff.nelement() > 0:
                self.loss = diff.mean()
            else:
                #print("warning: diff.nelement()==0 in PhotometricLoss (this is expected during early stage of training, try larger batch size).")
                self.loss = 0.0
        else:
            print("warning: 0 valid pixel in PhotometricLoss")
            self.loss = 0.0
        return self.loss

class PhotometricLoss(nn.Module):
    def __init__(self):
        super(PhotometricLoss, self).__init__()

    def forward(self, target, recon, mask=None):

        assert recon.dim()==4, "expected recon dimension to be 4, but instead got {}.".format(recon.dim())
        assert target.dim()==4, "expected target dimension to be 4, but instead got {}.".format(target.dim())
        assert recon.size()==target.size(), "expected recon and target to have the same size, but got {} and {} instead"\
            .format(recon.size(), target.size())
        diff = (target - recon).abs()
        diff = torch.sum(diff, 1) # sum along the color channel

        # compare only pixels that are not black
        # 只针对像素不为空的地方

        valid_mask = (torch.sum(recon, 1)>0).float() * (torch.sum(target, 1)>0).float()
        #print(valid_mask)
        if mask is not None:
            #print("mask is not None")
            valid_mask = valid_mask * torch.squeeze(mask).float()
        valid_mask = valid_mask.byte().detach()
        if valid_mask.numel() > 0:
            diff = diff[valid_mask]
            if diff.nelement() > 0:
                self.loss = diff.mean()
            else:
                #print("warning: diff.nelement()==0 in PhotometricLoss (this is expected during early stage of training, try larger batch size).")
                self.loss = 0
        else:
            print("warning: 0 valid pixel in PhotometricLoss")
            self.loss = 0
        return self.loss

class SmoothnessLoss(nn.Module):
    def __init__(self):
        super(SmoothnessLoss, self).__init__()

    def forward(self, depth):
        def second_derivative(x):
            assert x.dim() == 4, "expected 4-dimensional data, but instead got {}".format(x.dim())
            horizontal = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, 1:-1, :-2] - x[:, :, 1:-1, 2:] #水平
            vertical = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, :-2, 1:-1] - x[:, :, 2:, 1:-1] #垂直
            der_2nd = horizontal.abs() + vertical.abs()
            return der_2nd.mean()
        self.loss = second_derivative(depth)
        return self.loss