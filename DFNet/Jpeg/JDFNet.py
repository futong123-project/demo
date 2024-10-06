import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import MPNCOV

DCT = np.load('DCT.npy')
SRM_npy1 = np.load('kernels/SRM3_3.npy')
SRM_npy2 = np.load('kernels/SRM5_5.npy')

class pre_Layer_3_3(nn.Module):
    def __init__(self, stride=1, padding=1):
        super(pre_Layer_3_3, self).__init__()
        self.in_channels = 1
        self.out_channels = 25
        self.kernel_size = (3, 3)
        self.stride = (stride, stride)
        self.padding = (padding, padding)

        self.weight = Parameter(torch.Tensor(25, 1, 3, 3), requires_grad=True)
        self.bias = Parameter(torch.Tensor(25), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy1
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)


class pre_Layer_5_5(nn.Module):
    def __init__(self, stride=1, padding=2):
        super(pre_Layer_5_5, self).__init__()
        self.in_channels = 1
        self.out_channels = 5
        self.kernel_size = (5, 5)
        self.stride = (stride, stride)
        self.padding = (padding, padding)

        self.weight = Parameter(torch.Tensor(5, 1, 5, 5), requires_grad=True)
        self.bias = Parameter(torch.Tensor(5), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = SRM_npy2
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)


class Pre_Layer_jpg(nn.Module):
    def __init__(self, stride=1, padding=2):
        super(Pre_Layer_jpg, self).__init__()
        self.in_channels = 1
        self.out_channels = 4
        self.kernel_size = (5, 5)
        self.stride = (stride, stride)
        self.padding = (padding, padding)

        self.weight = Parameter(torch.Tensor(4, 1, 5, 5), requires_grad=True)
        self.bias = Parameter(torch.Tensor(4), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.weight.data.numpy()[:] = DCT
        self.bias.data.zero_()

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding)

class pre_layer(nn.Module):

    def __init__(self):
        super(pre_layer, self).__init__()

        self.conv1 = pre_Layer_3_3()
        self.conv2 = pre_Layer_5_5()
        self.conv3 = Pre_Layer_jpg()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        return torch.cat((x1, x2, x3), 1)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = 8

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()


    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


class DualPathBlock(nn.Module):
    def __init__(self, in_chs, num_1x1_a, num_3x3_b, num_1x1_c, inc, G):
        super(DualPathBlock, self).__init__()
        self.in_channel = in_chs
        self.num_1x1_c = num_1x1_c
        self.c1x1_w = self.BN_ReLU_Conv(in_chs=self.in_channel, out_chs=num_1x1_c+2*inc, kernel_size=1, stride=1)
        self.se = CoordAtt(inp=num_1x1_c+2*inc, oup=num_1x1_c+2*inc)
        self.conv = self.BN_ReLU_Conv(in_chs=num_1x1_c+3*inc, out_chs=64, kernel_size=1, stride=1)

        self.layers = nn.Sequential(OrderedDict([
            ('c1x1_a', self.BN_ReLU_Conv(in_chs=self.in_channel, out_chs=num_1x1_a, kernel_size=1, stride=1)),
            ('c3x3_b', self.BN_ReLU_Conv(in_chs=num_1x1_a, out_chs=num_3x3_b, kernel_size=3, stride=1, padding=1, groups=G)),
            ('c1x1_c', self.BN_ReLU_Conv(in_chs=num_3x3_b, out_chs=num_1x1_c+inc, kernel_size=1, stride=1)),
        ]))

    def BN_ReLU_Conv(self, in_chs, out_chs, kernel_size, stride, padding=0, groups=1):
        return nn.Sequential(OrderedDict([
            ('norm', nn.BatchNorm2d(in_chs)),
            ('relu', nn.ReLU(inplace=True)),
            ('conv', nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, groups=groups, bias=False)),
        ]))


    def forward(self, x):
        data_o = self.c1x1_w(x)
        data_o = self.se(data_o)
        data_o1 = data_o[:,:self.num_1x1_c,:,:]
        data_o2 = data_o[:,self.num_1x1_c:,:,:]

        out = self.layers(x)

        summ = data_o1 + out[:,:self.num_1x1_c,:,:]
        dense = torch.cat([data_o2, out[:,self.num_1x1_c:,:,:]], dim=1)
        fuse = torch.cat([summ, dense], dim=1)
        out = self.conv(fuse)

        return out




class Type1(nn.Module):
    '''Grouped convolution block.'''

    def __init__(self, in_planes, out_planes):
        super(Type1, self).__init__()
        self.in_channels = in_planes
        self.out_channels = out_planes
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
        )

    def forward(self, x):
        out = torch.add(x, self.block(x))
        return out



class Type3(nn.Module):
    '''Grouped convolution block.'''

    def __init__(self, in_planes, out_planes):
        super(Type3, self).__init__()
        self.in_channels = in_planes
        self.out_channels = out_planes

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=self.out_channels),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=3, stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=self.out_channels),
        )

    def forward(self, x):
        out = torch.add(self.branch1(x), self.branch2(x))
        return out


class Type4(nn.Module):
    '''Grouped convolution block.'''

    def __init__(self, in_planes, out_planes):
        super(Type4, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=in_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes, out_channels=in_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=in_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=out_planes),
        )

    def forward(self, x):
        out = torch.add(x, self.block(x))
        return out


class Type5(nn.Module):
    '''Grouped convolution block.'''

    def __init__(self, in_planes, out_planes):
        super(Type5, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_planes, out_channels=out_planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=out_planes),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.block(x)
        return out


class Type6(nn.Module):
    '''Grouped convolution block.'''

    def __init__(self, in_planes, out_planes):
        super(Type6, self).__init__()
        self.in_channels = in_planes
        self.out_channels = out_planes
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=2),
            nn.BatchNorm2d(num_features=self.out_channels),
        )
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3, stride=2, padding=1, groups=32),
            nn.BatchNorm2d(num_features=self.out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(num_features=self.out_channels),
        )


    def forward(self, x):
        out = torch.add(self.branch(x), self.block(x))
        return out

class FuNet(nn.Module):

    def __init__(self):
        super(FuNet, self).__init__()

        self.layer1 = pre_layer()
        self.dpb1 = DualPathBlock(in_chs=34, num_1x1_a = 64, num_3x3_b = 64, num_1x1_c =64, inc = 16, G =32)
        self.dpb2 = DualPathBlock(in_chs=64, num_1x1_a = 64, num_3x3_b = 64, num_1x1_c =64, inc = 16, G = 32)
        self.b31 = Type1(64, 64)
        self.b32 = Type3(64, 128)
        self.b21 = Type3(64, 128)

        self.b33 = Type3(128, 128)
        self.b34 = Type4(in_planes=128, out_planes=128)
        self.b22 = Type6(128, 128)
        self.b11 = Type5(64, 128)


        self.layer = nn.Linear(int(128 * (128 + 1) / 2), 2)


    def forward(self, x):
        x = self.layer1(x)
        x = self.dpb1(x)
        x = self.dpb2(x)
        x1 = self.b31(x)
        x1 = self.b32(x1)
        x2 = self.b21(x)
        xa = (x1 + x2)/2.0

        x3 = self.b33(xa)
        x3 = self.b34(x3)
        x4 = self.b22(xa)
        x5 = self.b11(x)
        x = (x3 + x4 + x5)/3.0


        x = MPNCOV.CovpoolLayer(x)
        x = MPNCOV.SqrtmLayer(x, 5)
        x = MPNCOV.TriuvecLayer(x)
        x = x.view(x.size(0), -1)
        x = self.layer(x)


        return x


'''
# 测试网络结构是否正确
#x = torch.rand(size=(3, 256, 256, 1))
x = torch.ones(size=(3, 1,256, 256))
print(x.shape)

net = FuNet()
print(net)

output_Y = net(x)
print('output shape: ', output_Y.shape)


model = FuNet()
#print(net)

total = sum([param.nelement() for param in model.parameters()])
print(total/1e6)
'''
