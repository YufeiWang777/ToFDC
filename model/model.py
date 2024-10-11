from abc import ABC

import torch
import torch.nn as nn
from scipy.stats import truncnorm
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

from model.modulated_deform_conv_func import ModulatedDeformConvFunction


def Conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def Conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class Basic2d(nn.Module, ABC):
    def __init__(self, in_channels, out_channels, norm_layer=None, kernel_size=3, padding=1):
        super().__init__()
        if norm_layer:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=False)
        else:
            conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                             stride=1, padding=padding, bias=True)
        self.conv = nn.Sequential(conv, )
        if norm_layer:
            self.conv.add_module('bn', norm_layer(out_channels))
        self.conv.add_module('relu', nn.ReLU(inplace=True))

    def forward(self, x):
        out = self.conv(x)
        return out


class Basic2dTrans(nn.Module, ABC):
    def __init__(self, in_channels, out_channels, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=2, padding=1, output_padding=1, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class BasicBlock(nn.Module, ABC):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None, act=True):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = Conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.act = act

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        if self.act:
            out = self.relu(out)
        return out


# Guided Feature Fusion Block
class Guide(nn.Module, ABC):

    def __init__(self, input_planes, weight_planes, norm_layer=None, weight_ks=1, input_ks=3):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.num = input_ks * input_ks
        self.stride = 1
        self.kernel_size = input_ks
        self.padding = int((input_ks - 1) / 2)
        self.dilation = 1
        self.input_planes = input_planes

        self.num_bases = 6
        bias = True

        self.conv_bases = nn.Sequential(
            Basic2d(input_planes + weight_planes, input_planes, None),
            nn.Conv2d(input_planes, self.num*self.num_bases, kernel_size=weight_ks, padding=weight_ks // 2)
        )

        self.coef = Parameter(torch.Tensor(input_planes, input_planes*self.num_bases, 1, 1))
        if bias:
            self.bias = Parameter(torch.Tensor(input_planes))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.br = nn.Sequential(
            norm_layer(num_features=input_planes),
            nn.ReLU(inplace=True),
        )
        self.conv3 = Basic2d(input_planes, input_planes, norm_layer)


    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.coef.size(1))

        nn.init.kaiming_normal_(self.coef, mode='fan_out', nonlinearity='relu')
        if self.bias is not None:
            self.bias.data.zero_()


    def forward(self, feat, weight):

        N, C, H, W = feat.shape
        H = H // self.stride
        W = W // self.stride
        drop_rate = 0.0

        weight = torch.cat((feat, weight), dim=1)
        bases = self.conv_bases(weight)

        # tmp = bases.view(N, self.num_bases, -1, H, W)
        x = F.unfold(F.dropout2d(feat, p=drop_rate, training=self.training), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding).view(N, self.input_planes, self.kernel_size*self.kernel_size, H, W)
        bases_out = torch.einsum('bmlhw, bclhw-> bcmhw', bases.view(N, self.num_bases, -1, H, W), x).reshape(N, self.input_planes*self.num_bases, H, W)
        bases_out = F.dropout2d(bases_out, p=drop_rate, training=self.training)

        out = F.conv2d(bases_out, self.coef, self.bias)

        out = self.br(out)
        out = self.conv3(out)

        return out


# Depth map optimization block
class Post_process_deconv(nn.Module, ABC):

    def __init__(self, args):
        super().__init__()

        self.dkn_residual = args.dkn_residual

        # Dummy parameters for gathering
        self.w = nn.Parameter(torch.ones((1, 1, args.kernel_size, args.kernel_size)))
        self.b = nn.Parameter(torch.zeros(1))
        self.w.requires_grad = False
        self.b.requires_grad = False

        self.stride = 1
        self.padding = int((args.kernel_size - 1) / 2)
        self.dilation = 1
        self.deformable_groups = 1
        self.im2col_step = 64

    def forward(self, depth, weight, offset):

        if self.dkn_residual:
            weight -= torch.mean(weight, 1).unsqueeze(1).expand_as(weight)
        else:
            weight /= torch.sum(weight, 1).unsqueeze(1).expand_as(weight)

        output = ModulatedDeformConvFunction.apply(
            depth, offset, weight, self.w, self.b, self.stride, self.padding,
            self.dilation, 1, self.deformable_groups, self.im2col_step
        )

        if self.dkn_residual:
            output = output + depth

        return output


class Model(nn.Module, ABC):

    def __init__(self, args, block=BasicBlock, bc=16, img_layers=(2, 2, 2, 2, 2),
                 depth_layers=(2, 2, 2, 2, 2), norm_layer=nn.BatchNorm2d, guide=Guide, weight_ks=1):
        super().__init__()
        self.args = args
        self.dep_max = None
        self.kernel_size = args.kernel_size
        self.filter_size = args.filter_size
        self.dkn_residual = args.dkn_residual
        self._norm_layer = norm_layer

        self.conv_img = Basic2d(3, bc * 2, norm_layer=norm_layer, kernel_size=5, padding=2)

        in_channels = bc * 2
        self.inplanes = in_channels
        self.layer1_img = self._make_layer(block, in_channels * 2, img_layers[0], stride=2)
        self.guide1 = guide(in_channels * 2, in_channels * 2, norm_layer, weight_ks)
        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_img = self._make_layer(block, in_channels * 4, img_layers[1], stride=2)
        self.guide2 = guide(in_channels * 4, in_channels * 4, norm_layer, weight_ks)
        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_img = self._make_layer(block, in_channels * 8, img_layers[2], stride=2)

        self.guide3 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer4_img = self._make_layer(block, in_channels * 8, img_layers[3], stride=2)

        self.guide4 = guide(in_channels * 8, in_channels * 8, norm_layer, weight_ks)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer5_img = self._make_layer(block, in_channels * 8, img_layers[4], stride=2)

        self.layer2d_img = Basic2dTrans(in_channels * 4, in_channels * 2, norm_layer)
        self.layer3d_img = Basic2dTrans(in_channels * 8, in_channels * 4, norm_layer)
        self.layer4d_img = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.layer5d_img = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)

        self.conv_lidar = Basic2d(1, bc, norm_layer=None, kernel_size=5, padding=2)
        self.conv_s2 = Basic2d(1, bc, norm_layer=norm_layer, kernel_size=5, padding=2)

        self.inplanes = in_channels
        self.layer1_lidar = self._make_layer(block, in_channels * 2, depth_layers[0], stride=2)
        self.inplanes = in_channels * 2 * block.expansion
        self.layer2_lidar = self._make_layer(block, in_channels * 4, depth_layers[1], stride=2)
        self.inplanes = in_channels * 4 * block.expansion
        self.layer3_lidar = self._make_layer(block, in_channels * 8, depth_layers[2], stride=2)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer4_lidar = self._make_layer(block, in_channels * 8, depth_layers[3], stride=2)
        self.inplanes = in_channels * 8 * block.expansion
        self.layer5_lidar = self._make_layer(block, in_channels * 8, depth_layers[4], stride=2)

        self.layer1d = Basic2dTrans(in_channels * 2, in_channels, norm_layer)
        self.layer2d = Basic2dTrans(in_channels * 4, in_channels * 2, norm_layer)
        self.layer3d = Basic2dTrans(in_channels * 8, in_channels * 4, norm_layer)
        self.layer4d = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)
        self.layer5d = Basic2dTrans(in_channels * 8, in_channels * 8, norm_layer)

        self.ref = block(bc * 2, bc * 2, norm_layer=norm_layer, act=False)
        self.conv = nn.Conv2d(bc * 2, 1, kernel_size=3, stride=1, padding=1)

        self.ref_weight_offset = Basic2d(bc * 2, bc * 2, norm_layer=None)
        self.conv_weight = nn.Conv2d(bc * 2, self.kernel_size ** 2, kernel_size=1, stride=1, padding=0)
        self.conv_offset = nn.Conv2d(bc * 2, 2 * self.kernel_size ** 2, kernel_size=1, stride=1, padding=0)

        self.Post_process = Post_process_deconv(args)

        self._initialize_weights()

    def forward(self, sample):
        img, lidar = sample['rgb'], sample['s1']
        dep = sample['dep']
        if self.args.depth_norm:
            bz = lidar.shape[0]
            self.dep_max = torch.max(lidar.view(bz,-1),1, keepdim=False)[0].view(bz,1,1,1)
            lidar = lidar/(self.dep_max +1e-4)
            dep_ = dep/(self.dep_max +1e-4)
        else:
            dep_ = dep

        img_ = self.conv_img(img)

        c1_img = self.layer1_img(img_)
        c2_img = self.layer2_img(c1_img)
        c3_img = self.layer3_img(c2_img)
        c4_img = self.layer4_img(c3_img)
        c5_img = self.layer5_img(c4_img)
        dc5_img = self.layer5d_img(c5_img)
        c4_mix = dc5_img + c4_img
        dc4_img = self.layer4d_img(c4_mix)
        c3_mix = dc4_img + c3_img
        dc3_img = self.layer3d_img(c3_mix)
        c2_mix = dc3_img + c2_img
        dc2_img = self.layer2d_img(c2_mix)
        c1_mix = dc2_img + c1_img

        c0_lidar = self.conv_lidar(lidar)
        s2_ = self.conv_s2(dep_)
        c0_lidar = torch.cat((c0_lidar, s2_), dim=1)
        c1_lidar = self.layer1_lidar(c0_lidar)
        c1_lidar_dyn = self.guide1(c1_lidar, c1_mix)
        c2_lidar = self.layer2_lidar(c1_lidar_dyn)
        c2_lidar_dyn = self.guide2(c2_lidar, c2_mix)
        c3_lidar = self.layer3_lidar(c2_lidar_dyn)
        c3_lidar_dyn = self.guide3(c3_lidar, c3_mix)
        c4_lidar = self.layer4_lidar(c3_lidar_dyn)
        c4_lidar_dyn = self.guide4(c4_lidar, c4_mix)
        c5_lidar = self.layer5_lidar(c4_lidar_dyn)
        c5 = c5_img + c5_lidar
        dc5 = self.layer5d(c5)
        c4 = dc5 + c4_lidar_dyn
        dc4 = self.layer4d(c4)
        c3 = dc4 + c3_lidar_dyn
        dc3 = self.layer3d(c3)
        c2 = dc3 + c2_lidar_dyn
        dc2 = self.layer2d(c2)
        c1 = dc2 + c1_lidar_dyn
        dc1 = self.layer1d(c1)
        c0 = dc1 + c0_lidar
        residual_feature = self.ref(c0)
        residual = self.conv(residual_feature)

        depth = lidar + residual
        if self.args.depth_norm:
            depth = depth * self.dep_max

        if self.args.preserve_input:
            mask = torch.sum(dep > 0.0, dim=1, keepdim=True).detach()
            mask = (mask > 0.0).type_as(dep)
            depth = (1.0 - mask) * depth + mask * dep

        feature = self.ref_weight_offset(c0)
        weight = self.conv_weight(feature)
        offset = self.conv_offset(feature)

        output = self.Post_process(depth, weight, offset)

        output = {'output': output}
        # output = {'an_depth': output, 'ben_depth': output, 'jin_depth': output,
        #           'ben_mask': output, 'ben_conf': output, 'jin_conf': output}

        return output

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        def truncated_normal_(num, mean=0., std=1.):
            lower = -2 * std
            upper = 2 * std
            X = truncnorm((lower - mean) / std, (upper - mean) / std, loc=mean, scale=std)
            samples = X.rvs(num)
            output = torch.from_numpy(samples)
            return output

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                data = truncated_normal_(m.weight.nelement(), mean=0, std=math.sqrt(1.3 * 2. / n))
                data = data.type_as(m.weight.data)
                m.weight.data = data.view_as(m.weight.data)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


