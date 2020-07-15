import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from se_layer import SEBasicBlock, SEBottleneck
from torchvision.models import ResNet
import collections

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        m.weight.data.normal_(0, 1e-3)
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def conv_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride,
        padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.PReLU(num_parameters=1, init=0.25))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)
    return layers

class Decoder(nn.Module):

    def __init__(self, in_channels, out_channels, stride=2, output_padding = 1):
        super(Decoder, self).__init__()
        self.upper_branch = nn.Sequential(collections.OrderedDict([
              ('ConvTranspose2d', nn.ConvTranspose2d(in_channels,out_channels,kernel_size=3,stride=stride,padding=1,output_padding=output_padding)),
              ('batchnorm', nn.BatchNorm2d(out_channels)),
            ]))

        self.bottom_branch = nn.Sequential(collections.OrderedDict([
            ('Conv2d', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias = False)),
            ]))
        self.scale = 2.0
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, x):
        x1 = self.upper_branch(x)
        x2 = self.bottom_branch(x)
        x2 = nn.functional.interpolate(x2, scale_factor=self.scale, mode='bilinear', align_corners=True)
        x2 = self.bn(x2)
        x = x1+x2 #将俩个矩阵相加改为串联
        x = self.relu(x)
        return x

'''
def convt_bn_relu(in_channels, out_channels, kernel_size, \
        stride=1, padding=0, output_padding=0, bn=True, relu=True):
    bias = not bn
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
        stride, padding, output_padding, bias=bias))
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2, inplace=True))
    layers = nn.Sequential(*layers)

    # initialize the weights
    for m in layers.modules():
        init_weights(m)

    return layers
'''
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

class DepthCompletionNet(nn.Module):
    def __init__(self, args):
        super(DepthCompletionNet, self).__init__()
        self.use_d = args.use_d

        if args.use_d:
            channels_d = 16
            self.conv1_d = conv_bn_relu(1, channels_d, kernel_size=3, stride=1, padding=1)
            channels_rgb = 48
            self.conv1_img = conv_bn_relu(3, channels_rgb, kernel_size=3, stride=1, padding=1)
        else:
            channels_rgb = 64
            self.conv1_img = conv_bn_relu(3, channels_rgb, kernel_size=3, stride=1, padding=1)

        if args.layers == 18:
           se_model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=100)
        elif args.layers == 34:
           se_model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=100)
        elif args.layers == 50:
           se_model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=100)
        elif args.layers == 101:
           se_model = ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=100)
        else:
            raise TypeError('Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(args.layers))

        se_model.apply(init_weights)
        self.conv2 = se_model._modules['layer1']
        self.conv3 = se_model._modules['layer2']
        self.conv4 = se_model._modules['layer3']
        self.conv5 = se_model._modules['layer4']

        del se_model # clear memory

        # define number of intermediate channels
        num_channels = 512
        self.conv6 = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)

        # decoding layers
        kernel_size = 3
        stride = 2
        self.convt5 = Decoder(in_channels=512, out_channels=256)
        self.convt5.apply(init_weights)
        self.convt4 = Decoder(in_channels=768, out_channels=128)
        self.convt4.apply(init_weights)
        self.convt3 = Decoder(in_channels=(256+128), out_channels=64)
        self.convt3.apply(init_weights)
        self.convt2 = Decoder(in_channels=(128+64), out_channels=64)
        self.convt2.apply(init_weights)
        self.convt1 = Decoder(in_channels=128, out_channels=64, stride = 1, output_padding=0)
        self.convt1.apply(init_weights)
        self.convtf = conv_bn_relu(in_channels=128, out_channels=1, kernel_size=1, stride=1, bn=False, relu=False)

    def forward(self, rgb, depth):
        # 第一层网络
        conv1_img = self.conv1_img(rgb)

        if self.use_d:
            conv1_d = self.conv1_d(depth)
            conv1 = torch.cat((conv1_d, conv1_img),1)
        else:
            conv1 = conv1_img

        # conv1 #batchsize, 64, 352, 1216
        conv2 = self.conv2(conv1) # batchsize, 64, 352, 1216
        conv3 = self.conv3(conv2) # batchsize, 128, 176, 608
        conv4 = self.conv4(conv3) # batchsize, 256, 88, 304
        conv5 = self.conv5(conv4) # batchsize, 512, 44, 152
        conv6 = self.conv6(conv5) # batchsize, 512, 22, 76

        # decoder
        convt5 = self.convt5(conv6)       #batchsize, 256, 44 , 152
        y = torch.cat((convt5, conv5), 1) #batchsize, 768, 44, 152
        convt4 = self.convt4(y)           #batchsize, 128, 88, 304
        y = torch.cat((convt4, conv4), 1) #batchsize, 384, 88, 304

        convt3 = self.convt3(y)           #batchsize, 64, 176, 608
        y = torch.cat((convt3, conv3), 1) #batchsize, 192, 176, 608

        convt2 = self.convt2(y)           #batchsize, 64, 352, 1216
        y = torch.cat((convt2, conv2), 1) #batchsize, 128, 352, 1216

        convt1 = self.convt1(y)          #batchsize, 64, 352, 1216
        y = torch.cat((convt1,conv1), 1) #batchsize, 128, 704, 2432

        y = self.convtf(y)

        if self.training:
            return 100 * y

        else:
            min_distance = 0.9
            return F.relu(100 * y - min_distance) + min_distance # the minimum range of Velodyne is around 3 feet ~= 0.9m
