import argparse
import os
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch.optim
import torch.utils.data

from dataloaders.kitti_loader import load_calib, oheight, owidth, KittiDepth
from model import DepthCompletionNet
from metrics import AverageMeter, Result
import criteria
import helper
from inverse_warp import Intrinsics, homography_from, Covisual_from, left_right_from
from tensorboardX import SummaryWriter
from vis_utils import display_warping_depth
import numpy as np

parser = argparse.ArgumentParser(description='Sparse-to-Dense')
parser.add_argument('-w', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=15, type=int, metavar='N',
                    help='number of total epochs to run (default: 11)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-c', '--criterion', metavar='LOSS', default='l1',
                    choices=criteria.loss_names,
                    help='loss function: | '.join(criteria.loss_names) + ' (default: l1)')
parser.add_argument('-b', '--batch-size', default=3, type=int,
                    help='mini-batch size (default: 1)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float,
                    metavar='LR', help='initial learning rate (default 1e-5)')
parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                    metavar='W', help='weight decay (default: 0)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-l','--layers', type=int, default=34,
                    help='use 16 for sparse_conv; use 18 or 34 for resnet')
parser.add_argument('-vn', '--valNum', type=int, default=6400, help='the number of validation set')
parser.add_argument('--jitter', type=float, default=0.1,
                    help = 'color jitter for images')
parser.add_argument('--rank-metric', type=str, default='rmse',
                    choices=[m for m in dir(Result()) if not m.startswith('_')],
                    help = 'metrics for which best result is sbatch_datacted')
parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
parser.add_argument('-tm', '--train_mode', type=str, default="rgbd", choices = ["rgb", "rgbd"], help = 'rgb | rgbd')

args = parser.parse_args()
args.use_pose = True
args.pretrained = True
args.use_covisual = True
if args.train_mode == "rgbd":
   args.use_d = True
else:
   args.use_d = False
args.result = os.path.join('..', 'results')
if args.use_pose:
    args.w1, args.w2 = 0.1, 0.1
else:
    args.w1, args.w2 = 0, 0.1

if args.use_covisual:
    args.w3 = 0.1
else:
    args.w3 = 0.0
args.num_sample = 500

print(args)

width = 1200
height = 352
tensorboard_times_count=0

# define loss functions
depth_criterion = criteria.MaskedMSELoss() if (args.criterion == 'l2') else criteria.MaskedL1Loss()
photometric_criterion = criteria.PhotometricLoss()
smoothness_criterion = criteria.SmoothnessLoss()
covisual_criterion = criteria.CovisualLossNew()
ssim_criterion = criteria.SsimLoss()

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2

cmap = plt.cm.jet


def visual_depth(depth, min_depth, max_depth):
    depth = (depth - min_depth) / (max_depth - min_depth)
    return depth

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:,:,:3] # H, W, C
    return depth.astype('uint8')

def merge_into_row(ele, index):
    def preprocess_depth(x):
        y = np.squeeze(x.data.cpu().numpy())
        return depth_colorize(y)
    # if is gray, transforms to rgb
    img_list = []

    if 'rgb' in ele:
        rgb = np.squeeze(ele['rgb'][index,...].data.cpu().numpy())
        rgb = np.transpose(rgb, (1, 2, 0))
        img_list.append(rgb)
    if 'd' in ele:
        img_list.append(preprocess_depth(ele['d'][index,...]))

    img_merge = np.hstack(img_list)
    return img_merge.astype('uint8')

def merge_into_row1(ele, pred, index):
    def preprocess_depth(x):
        y = np.squeeze(x.data.cpu().numpy())
        return depth_colorize(y)
    # if is gray, transforms to rgb
    img_list = []
    img_list.append(preprocess_depth(pred[index, ...]))
    if 'gt' in ele:
        img_list.append(preprocess_depth(ele['gt'][index, ...]))

    img_merge = np.hstack(img_list)
    return img_merge.astype('uint8')

def secondrow(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    image_to_write = cv2.cvtColor(img_merge, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, image_to_write)

def change_intrinsics(K):
    fu, fv = float(K[0, 0]), float(K[1, 1])
    cu, cv = float(K[0, 2]), float(K[1, 2])
    kitti_intrinsics = Intrinsics(owidth, oheight, fu, fv, cu, cv).cuda()
    return kitti_intrinsics

import math
def RMSE(output, target):
    valid_mask = target > 0.1

    # convert from meters to mm
    output_mm = 1e3 * output[valid_mask]
    target_mm = 1e3 * target[valid_mask]

    abs_diff = (output_mm - target_mm).abs()

    mse = float((torch.pow(abs_diff, 2)).mean())
    rmse = math.sqrt(mse)
    return  rmse

##进行迭代训练
def iterate(mode, args, loader, model, optimizer, logger, epoch, writer_tensorboard):
    global tensorboard_times_count

    block_average_meter = AverageMeter()
    average_meter = AverageMeter()
    meters = [block_average_meter, average_meter]

    # switch to appropriate mode
    assert mode in ["train", "val", "eval", "test_prediction", "test_completion"], \
        "unsupported mode: {}".format(mode)

    if mode == 'train':
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, epoch)#调整训练的学习率
    else:
        model.eval()
        lr = 0

    for i, batch_data in enumerate(loader):
        start = time.time()
        batch_path = {key: val for key,val in batch_data.items() if val is not None and key == "path"}
        batch_data = {key: val.cuda() for key,val in batch_data.items() if val is not None and key != "path"}


        rgb_input = batch_data['rgb']
        depth_input = batch_data['d']
        gt = batch_data['gt']

        if mode == 'train':
           rgb_input_near = batch_data['rgb_near']
           depth_input_near = batch_data['d_near']
           gt_near = batch_data['gt_near']

        data_time = time.time() - start

        start = time.time()

        pred = model(rgb_input, depth_input)


        if mode == 'train':
           pred_near = model(rgb_input_near, depth_input_near)

        depth_loss, photometric_loss, photometric_loss_near, photometric_loss_curr, photometric_loss_right, smooth_loss, covisual_loss, mask = 0, 0, 0, 0, 0, 0, 0, None

        if mode == 'train':

            K = batch_data['K']
            kitti_intrinsics = [change_intrinsics(k) for k in K]  # 将K的每个内参分别处理，保存在list列表中

            # Loss 1: the direct depth supervision from ground truth label
            depth_loss = depth_criterion(pred, gt)
            depth_near_loss = depth_criterion(pred_near, gt_near)
            depth_loss = 0.5*depth_loss + 0.5*depth_near_loss
            #print("depth_loss: "+str(depth_loss))

            mask = (batch_data['gt'] < 1e-3).float()#mask=1表示对应的像素没有ground truth

            # Loss 2: 重构图片损失函数()
            if args.use_pose:
                # create multi-scale pyramids
                pred_array = helper.multiscale(pred)
                pred_near_array = helper.multiscale(pred_near)
                rgb_curr_array = helper.multiscale(batch_data['rgb'])
                rgb_right_array = helper.multiscale(batch_data['rgb_right'])
                rgb_near_array = helper.multiscale(batch_data['rgb_near'])
                if mask is not None:
                    mask_array = helper.multiscale(mask)
                num_scales = len(pred_array)

                # compute photometric loss at multiple scales
                for scale in range(len(pred_array)):
                    pred_ = pred_array[scale]
                    pred_near_ = pred_near_array[scale]
                    rgb_curr_ = rgb_curr_array[scale]
                    rgb_near_ = rgb_near_array[scale]
                    rgb_right_ = rgb_right_array[scale]
                    mask_ = None
                    if mask is not None:
                        mask_ = mask_array[scale]

                    # compute the corresponding intrinsic parameters
                    height_, width_ = pred_.size(2), pred_.size(3)
                    intrinsics_ = [intrinsics_scale.scale(height_, width_) for intrinsics_scale in kitti_intrinsics]

                    # inverse warp from a nearby frame to the current frame
                    warped_curr = homography_from(rgb_near_, pred_, batch_data['r_mat'], batch_data['t_vec'], intrinsics_)
                    warped_near = homography_from(rgb_curr_, pred_near_, batch_data['r_mat_inv'], batch_data['t_vec_inv'], intrinsics_)
                    warped_curr_right = left_right_from(rgb_right_, pred_, batch_data['R_0to3_times_R_0to2_inv'],
                                                        batch_data['t_0to2'], batch_data['t_0to3'], intrinsics_)
                    photometric_loss_near += photometric_criterion(rgb_near_, warped_near, mask_) * (2 ** (scale - num_scales))
                    photometric_loss_curr += photometric_criterion(rgb_curr_, warped_curr, mask_) * (2 ** (scale - num_scales))
                    photometric_loss_right += photometric_criterion(rgb_curr_, warped_curr_right, mask_) * (2 ** (scale - num_scales))

                    if scale == len(pred_array) - 1:
                        warped_curr_save = warped_curr
                        warped_curr_right_save = warped_curr_right
                        warped_near_save = warped_near

                photometric_loss = photometric_loss_near*0.3 + photometric_loss_curr*0.3 + photometric_loss_right*0.4 if args.w1>0 else 0

                    #print(photometric_loss)

            #print("photometric_loss: "+str(photometric_loss))

            # Loss 3:　深度图平滑损失函数
            smooth_loss = smoothness_criterion(pred)
            smooth_near_loss = smoothness_criterion(pred_near)
            smooth_loss = 0.5*smooth_loss + 0.5*smooth_near_loss if args.w2>0 else 0
            #print("smooth_loss: "+str(smooth_loss))


            # Loss 4:　共视点深度损失函数
            tensorboard_times_count = tensorboard_times_count + 1
            if  args.use_covisual:
                depth_curr_transform_to_near, depth_curr_dilated_transform, warped_depth_near, warped_depth_dilated, mask_curr_to_near = Covisual_from(pred, batch_data['d_dilated'].detach(), pred_near, batch_data['d_near_dilated'].detach(), batch_data['r_mat'],\
                                                                                      batch_data['t_vec'], kitti_intrinsics)
                depth_near_transform_to_curr, _, warped_depth_curr, _ , mask_near_to_curr = Covisual_from(pred_near, batch_data['d_near_dilated'].detach(), pred, batch_data['d_dilated'].detach(), batch_data['r_mat_inv'], \
                                                                                      batch_data['t_vec_inv'], kitti_intrinsics)

                covisual_loss_near = covisual_criterion(depth_curr_transform_to_near, warped_depth_near, mask_curr_to_near, writer_tensorboard, tensorboard_times_count)
                covisual_loss_curr = covisual_criterion(depth_near_transform_to_curr, warped_depth_curr, mask_near_to_curr, writer_tensorboard, tensorboard_times_count)
                '''
                if scale == len(pred_array) - 1 and tensorboard_times_count % 50 == 0:
                    index = depth_curr_dilated_transform.shape[0] - 1
                    min_depth = torch.min(depth_curr_dilated_transform[index, :, :, :])
                    max_depth = torch.max(depth_curr_dilated_transform[index, :, :, :])

                    visual_depth_curr_transform_to_near = visual_depth(depth_curr_transform_to_near[index, :, :, :].detach(), min_depth, max_depth)
                    visual_warped_depth_near = visual_depth(warped_depth_near[index, :, :, :].detach(), min_depth,max_depth)
                    visual_depth_curr_dilated_transform = visual_depth(depth_curr_dilated_transform[index, :, :, :]* mask_curr_to_near[index, :, :, :].detach(), min_depth, max_depth)
                    visual_warped_depth_dilated = visual_depth(warped_depth_dilated[index, :, :, :]* mask_curr_to_near[index, :, :, :].detach(), min_depth,max_depth)
                    near_gt = visual_depth(batch_data['gt_near'][index, :, :, :].detach(), min_depth,max_depth)

                   
                    writer_tensorboard.add_image('depth_curr_transform_to_near', visual_depth_curr_transform_to_near, tensorboard_times_count)
                    writer_tensorboard.add_image('warped_depth_near', visual_warped_depth_near, tensorboard_times_count)
                    writer_tensorboard.add_image('depth_curr_dilated_transform', visual_depth_curr_dilated_transform, tensorboard_times_count)
                    writer_tensorboard.add_image('warped_depth_dilated', visual_warped_depth_dilated, tensorboard_times_count)
                    writer_tensorboard.add_image('near_image', batch_data['rgb_near'][index, :, :, :] / 256.0,tensorboard_times_count)
                    writer_tensorboard.add_image('near_gt', near_gt, tensorboard_times_count)
                 
                    writer_tensorboard.add_image('warped_curr_right_save',
                                                 warped_curr_right_save[index, :, :, :] / 256.0,
                                                 tensorboard_times_count)
                    writer_tensorboard.add_image('curr_image', batch_data['rgb'][index, :, :, :] / 256.0,
                                                 tensorboard_times_count)
                '''
                covisual_loss = covisual_loss_curr * 0.5 + covisual_loss_near * 0.5 if args.w3 > 0 else 0


            '''
            img_save = merge_into_row(batch_data)
            img_save1 = merge_into_row1(batch_data, warped_depth)
            img_merge = secondrow(img_save, img_save1)
            save_image(img_merge, '/home/hansry/' + str(i) + '.png')
            '''
            # backprop
            writer_tensorboard.add_scalar('depth_loss',depth_loss, tensorboard_times_count)
            writer_tensorboard.add_scalar('covisual_loss', args.w3*covisual_loss, tensorboard_times_count)
            writer_tensorboard.add_scalar('photometric', args.w1*photometric_loss, tensorboard_times_count)
            writer_tensorboard.add_scalar('smooth_loss',  args.w2*smooth_loss, tensorboard_times_count)

            loss = depth_loss + args.w3*covisual_loss + args.w1*photometric_loss + args.w2*smooth_loss
            #loss = depth_loss + args.w1 * photometric_loss + args.w2 * smooth_loss

            writer_tensorboard.add_scalar('loss', loss, tensorboard_times_count)
            optimizer.zero_grad()
            loss.backward()
            #optimizer.module.step()
            optimizer.step()

        gpu_time = time.time() - start

        # measure accuracy and record loss
        with torch.no_grad():
            mini_batch_size = next(iter(batch_data.values())).size(0)
            result = Result()

            if mode != 'test_prediction' and mode != 'test_completion':
                '''
                index = 2
                rmse = RMSE(pred[index].data, gt[index].data)
                img_save = merge_into_row(batch_data, index)
                img_save1 = merge_into_row1(batch_data, pred, index)
                img_merge = secondrow(img_save, img_save1)
                dir = batch_path["path"][index][37:63]
                image_index = batch_path["path"][index][-12:-4]
                save_image(img_merge, '/home/lab/huangxinghong/' + str(rmse) +"_" +str(dir) +"_"+str(image_index)+ '.png')
                '''
                result.evaluate(pred.data, gt.data, photometric_loss, covisual_loss)  #评价标准以mm为单位
            [m.update(result, gpu_time, data_time, mini_batch_size) for m in meters]
            logger.conditional_print(mode, i, epoch, lr, len(loader), block_average_meter, average_meter)
            logger.conditional_save_img_comparison(mode, i, batch_data, pred, epoch)
            logger.conditional_save_pred(mode, i, pred, epoch)

    avg = logger.conditional_save_info(mode, average_meter, epoch)
    is_best = logger.rank_conditional_save_best(mode, avg, epoch)
    if is_best and not (mode == "train"):
        logger.save_img_comparison_as_best(mode, epoch)
    logger.conditional_summarize(mode, avg, is_best)

    return avg, is_best

def main():
    global args, tensorboard_times_count
    checkpoint = None
    is_eval = False
    if args.evaluate: #测试
        if os.path.isfile(args.evaluate):
            print("=> loading checkpoint '{}'".format(args.evaluate))
            checkpoint = torch.load(args.evaluate)
            args = checkpoint['args']
            is_eval = True
            print("=> checkpoint loaded.")
        else:
            print("=> no model found at '{}'".format(args.evaluate))
            return
    elif args.resume: #继续训练
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']+1
            print("=> loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    print("=> creating model and optimizer...")

    #创建tensorboard节点
    writer_tensorboard = SummaryWriter()

    ##生成模型
    model = DepthCompletionNet(args).cuda()
    model_named_params = [p for _,p in model.named_parameters() if p.requires_grad]

    ##生成优化器
    optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay)

    print("=> model and optimizer created.")

    ##如果checkpoint不为空，则在checkpoint的基础上生成模型
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> checkpoint state loaded.")

    device_ids = [0,1,2]
    #optimizer = torch.nn.DataParallel(optimizer,device_ids=device_ids)
    model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()

    print("=> model transferred to multi-GPU.")

    helper.print_network(model)
    print("=> creating data loaders ...")
    ##生成训练的数据集
    if not is_eval: ##is_eval=False  not is_eval=True
        train_dataset = KittiDepth('train', args)

        from torch.utils.data.sampler import SubsetRandomSampler
        indices = np.random.randint(0, 10, 10) #一共有40702份测试集
        indices = indices.tolist()
        sampler = SubsetRandomSampler(indices)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)#shuffle为打乱的意思

    ##生成测试的数据集
    val_dataset = KittiDepth('val', args)
    from torch.utils.data.sampler import SubsetRandomSampler
    #indices = np.random.randint(0, 40702, args.valNum) #一共有40702份测试集
    #indices = indices.tolist()
    #sampler = SubsetRandomSampler(indices)
    val_loader = torch.utils.data.DataLoader(val_dataset,
        batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True) # set batch size to be 1 for validation
    print("=> data loaders created.")

    # create backups and results folder
    logger = helper.logger(args)
    if checkpoint is not None:
        logger.best_result = checkpoint['best_result']
    print("=> logger created.")

    ##测试的结果
    if is_eval:
        result, is_best = iterate("val", args, val_loader, model, None, logger, checkpoint['epoch'], writer_tensorboard)
        return

    ##进行循环训练
    for epoch in range(args.start_epoch, args.epochs):
        print("=> starting training epoch {} ..".format(epoch))
        iterate("train", args, train_loader, model, optimizer, logger, epoch, writer_tensorboard) # train for one epoch
        result, is_best = iterate("val", args, val_loader, model, None, logger, epoch, writer_tensorboard) #在验证集上进行测试
        helper.save_checkpoint({ # save checkpoint
            'epoch': epoch,
            'model': model.module.state_dict(),
            'best_result': logger.best_result,
            'optimizer': optimizer.state_dict(),
            'args' : args,
        }, is_best, epoch, logger.output_directory)

if __name__ == '__main__':
    main()
