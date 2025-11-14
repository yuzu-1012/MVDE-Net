from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from .DnCNN import make_net

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class HRNet(nn.Module):
    def __init__(self, extra_name='w32_extra'):
        self.inplanes = 64
        w32_extra = {
            "PRETRAINED_LAYERS": ['conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1', 'stage2', 'transition2', 'stage3', 'transition3', 'stage4'],
            "FINAL_CONV_KERNEL": 1,
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": 'BASIC',
                "NUM_BLOCKS": [4, 4],
                "NUM_CHANNELS": [32, 64],
                "FUSE_METHOD": 'SUM'
            },
            "STAGE3": {
                "NUM_MODULES": 4,
                "NUM_BRANCHES": 3,
                "BLOCK": 'BASIC',
                "NUM_BLOCKS": [4, 4, 4],
                "NUM_CHANNELS": [32, 64, 128],
                "FUSE_METHOD": 'SUM'
            },
            "STAGE4": {
                "NUM_MODULES": 3,
                "NUM_BRANCHES": 4,
                "BLOCK": 'BASIC',
                "NUM_BLOCKS": [4, 4, 4, 4],
                "NUM_CHANNELS": [32, 64, 128, 256],
                "FUSE_METHOD": 'SUM'
            }
        }
        w48_extra = {
            "PRETRAINED_LAYERS": ['conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1', 'stage2', 'transition2',
                                  'stage3', 'transition3', 'stage4'],
            "FINAL_CONV_KERNEL": 1,
            "STAGE2": {
                "NUM_MODULES": 1,
                "NUM_BRANCHES": 2,
                "BLOCK": 'BASIC',
                "NUM_BLOCKS": [4, 4],
                "NUM_CHANNELS": [48, 96],
                "FUSE_METHOD": 'SUM'
            },
            "STAGE3": {
                "NUM_MODULES": 4,
                "NUM_BRANCHES": 3,
                "BLOCK": 'BASIC',
                "NUM_BLOCKS": [4, 4, 4],
                "NUM_CHANNELS": [48, 96, 192],
                "FUSE_METHOD": 'SUM'
            },
            "STAGE4": {
                "NUM_MODULES": 3,
                "NUM_BRANCHES": 4,
                "BLOCK": 'BASIC',
                "NUM_BLOCKS": [4, 4, 4, 4],
                "NUM_CHANNELS": [48, 96, 192, 384],
                "FUSE_METHOD": 'SUM'
            }
        }
        if extra_name == 'w32_extra':
            extra = w32_extra
        else:
            extra = w48_extra
        super(HRNet, self).__init__()
        self.name = 'HRNet'

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)

        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=17,
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

        # pretrained weights
        # if extra_name == 'w32_extra':
        #     self.init_weights('models/hrnet_w32-36af842e.pth')
        # else:
        #     self.init_weights('models/hrnet_w48-8ef0771d.pth')

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels, multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])

        feats = self.stage4(x_list)  # B, 32, H // 4, W // 4
        return feats

    def init_weights(self, pretrained=''):
        # logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] == '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model


class RGBDomain(nn.Module):
    def __init__(self):
        super().__init__()
        H, W = 256, 256
        self.name = 'Multi_View_RGB'
        self.net = HRNet()
        self.in_channels = [32, 64, 128, 256]
        self.gate = nn.Sequential(
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(16, 32, 1, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(32, 3),
            nn.Softmax(),
        )
        self.W2 = nn.Parameter(torch.empty(32, 32))
        nn.init.normal_(self.W2)
        self.Conv1 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(16, 32, 1, 1),
        )
        self.Conv2 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(32, 64, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 32, 1, 1)
        )
        self.Conv3 = nn.Sequential(
            nn.PixelShuffle(2),
            nn.Conv2d(64, 128, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(32, 64, 1, 1),
            nn.PixelShuffle(2),
            nn.Conv2d(16, 32, 1, 1),
        )
        self.Conv4 = nn.Conv2d(128, 128, 1, 1)

    def forward(self, x):
        x1, x2, x3, x4 = self.net(x)
        x2 = self.Conv1(x2) - x1
        x3 = self.Conv2(x3) - x2
        x4 = self.Conv3(x4) - x3
        
        B, C, H, W = x1.shape
        weights = self.mlp(self.gate(x1).view(B, C) @ self.W2).unsqueeze(-1).unsqueeze(-1).repeat(1, 32, H, W)


        out = torch.cat([x2, x3, x4], dim=1) # [B, 32*3, 256, 256]
        out = weights*out
        out = self.Conv4(torch.cat([x1, out], dim=1)) # [B, 32*4, 256, 256]
        return out
    

def get_noiseprint(path = './models/noiseprint.pth'):
    num_levels = 17
    out_channel = 1
    state_dict = torch.load(path)
    noiseprint = make_net(3, kernels=[3, ] * num_levels,
                                   features=[64, ] * (num_levels - 1) + [out_channel],
                                   bns=[False, ] + [True, ] * (num_levels - 2) + [False, ],
                                   acts=['relu', ] * (num_levels - 1) + ['linear', ],
                                   dilats=[1, ] * num_levels,
                                   bn_momentum=0.1, padding=1)

    msg = noiseprint.load_state_dict(state_dict)
    print('Loading the weights of noiseprint:', msg)
    noiseprint.requires_grad_(False)

    return noiseprint

def get_mask(mask=torch.ones([256, 256], device='cuda'), H=256, W=256, scale=0.5):
    h_start = int(H // 2 - H * scale)
    h_end = int(H //2 + H * scale)
    w_start = int(W // 2 - W * scale)
    w_end = int(W // 2 + W * scale)
    mask[h_start:h_end, w_start:w_end] = 0
    return mask.float()

def get_filtered(fft_shift, mask, B, C):
    filtered1 = fft_shift * mask[None, None, :, :]
    ifft_shift = torch.fft.ifftshift(filtered1, dim=(-2, -1))
    restored = torch.fft.ifft2(ifft_shift, dim=(-2, -1)).real
    restored_min = restored.view(B, C, -1).min(dim=2)[0][..., None, None]
    restored_max = restored.view(B, C, -1).max(dim=2)[0][..., None, None]
    return (restored - restored_min) / (restored_max - restored_min + 1e-9)


class NoiseDomain(nn.Module):
    def __init__(self, img_size, with_noise=False, H=256, W=256, in_channels=3, out_channels=3):
        super().__init__()
        # SRM
        self.srm = SRMConv2D(in_channels, out_channels)
        # Bayar
        self.bayar = BayarConv2d(in_channels=1, out_channels=out_channels)
        # Noiseprint
        self.noiseprint = get_noiseprint()
        # AvgPooling
        self.gap = nn.AvgPool2d(3, 1, 1)
        # MaxPooling
        self.map = nn.MaxPool2d(3, 1, 1)
        # FFT
        self.W1 = nn.Parameter(get_mask())
        self.sigmoid = nn.Sigmoid()
        self.scales = [0.5, 0.8]

        # GateNetwork
        self.GateNetwork = nn.Sequential(
            nn.Conv2d(3, 3, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(3, 3, 1, 1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.mlp = nn.Sequential(
            nn.Linear(15, 3),
            nn.Linear(3, 15),
            nn.Softmax(),
        )
        self.W2 = nn.Parameter(torch.empty(3, 15))
        nn.init.normal_(self.W2)
        

        self.conv1 = nn.Conv2d(18, 18, 1, 1)
        self.name = 'MIX_noiseprint'
        self.img_size = img_size

        self.with_noise = with_noise
        if self.with_noise:
            self.additional_noise = nn.Parameter(torch.empty(18, 256, 256))
            nn.init.normal_(self.additional_noise)
            self.conv2 = nn.Sequential(
                nn.Conv2d(18, 18, 1, 1),
                nn.GELU(),
                nn.Conv2d(18, 18, 1, 1)
            )
        

    def forward(self, x):
        B, C, H, W = x.shape
        
        W1 = self.GateNetwork(x).view(B, -1) # B 3
        W1 = W1 @ self.W2 # B 15
        W1 = self.mlp(W1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W) # [B, 15, H, W]


        # SRM
        srm_feature = self.srm(x)

        # Bayar
        x_gray = rgb2gray(x)
        bayar_feature = self.bayar(x_gray)

        # noiseprint
        noiseprint_feature = self.noiseprint(x).repeat(1,3,1,1)/255.0

        # AvgPooling
        avgpooling_feature = x - self.gap(x)

        # MaxPooling
        maxpooling_feature = self.map(x)

        # FFT
        
        mask1 = self.sigmoid(self.W1)

        fft = torch.fft.fft2(x, dim=(-2, -1))
        fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
        fft_feature1 = get_filtered(fft_shift, mask1, B, C) * 255.


        # return noise_feature
        noise_feature = torch.cat([srm_feature, bayar_feature, avgpooling_feature, maxpooling_feature, fft_feature1], dim=1)
        noise_feature = W1*noise_feature / 255.0
        noise_feature = torch.cat([noise_feature, noiseprint_feature], dim=1)

        Gf = self.conv1(noise_feature)
        if self.with_noise:
            additional_noise = self.additional_noise.unsqueeze(0).repeat(B, 1, 1, 1)
            Gf = torch.cat([Gf, self.conv2(additional_noise)], dim=1) #[B, 18+18, H, W]

        return Gf


def rgb2gray(rgb):
    b, g, r = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]
    gray = 0.2989*r + 0.5870*g + 0.1140*b
    gray = torch.unsqueeze(gray, 1)
    return gray


class BayarConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.minus1 = (torch.ones(self.in_channels, self.out_channels, 1) * -1.000)

        super(BayarConv2d, self).__init__()
        # only (kernel_size ** 2 - 1) trainable params as the center element is always -1
        self.kernel = nn.Parameter(torch.rand(self.in_channels, self.out_channels, kernel_size ** 2 - 1),
                                   requires_grad=True)


    def bayarConstraint(self):
        self.kernel.data = self.kernel.permute(2, 0, 1)
        self.kernel.data = torch.div(self.kernel.data, self.kernel.data.sum(0))
        self.kernel.data = self.kernel.permute(1, 2, 0)
        ctr = self.kernel_size ** 2 // 2
        real_kernel = torch.cat((self.kernel[:, :, :ctr], self.minus1.to(self.kernel.device), self.kernel[:, :, ctr:]), dim=2)
        real_kernel = real_kernel.reshape((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        return real_kernel

    def forward(self, x):
        output = F.conv2d(x, self.bayarConstraint(), stride=self.stride, padding=self.padding)
        return output


class SRMConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=2):
        super(SRMConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.SRMWeights = nn.Parameter(
            self._get_srm_list(), requires_grad=False)

    def _get_srm_list(self):
        # srm kernel 1
        srm1 = [[0,  0, 0,  0, 0],
                [0, -1, 2, -1, 0],
                [0,  2, -4, 2, 0],
                [0, -1, 2, -1, 0],
                [0,  0, 0,  0, 0]]
        srm1 = torch.tensor(srm1, dtype=torch.float32) / 4.

        # srm kernel 2
        srm2 = [[-1, 2, -2, 2, -1],
                [2, -6, 8, -6, 2],
                [-2, 8, -12, 8, -2],
                [2, -6, 8, -6, 2],
                [-1, 2, -2, 2, -1]]
        srm2 = torch.tensor(srm2, dtype=torch.float32) / 12.

        # srm kernel 3
        srm3 = [[0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 1, -2, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]]
        srm3 = torch.tensor(srm3, dtype=torch.float32) / 2.

        return torch.stack([torch.stack([srm1, srm1, srm1], dim=0), torch.stack([srm2, srm2, srm2], dim=0), torch.stack([srm3, srm3, srm3], dim=0)], dim=0)

    def forward(self, X):
        # X1 =
        return F.conv2d(X, self.SRMWeights, stride=self.stride, padding=self.padding)



if __name__ == '__main__':
    # from torchinfo import summary
    # # x = torch.ones([1, 1024, 256, 256])
    # m = FOCAL_HRNet().cuda()
    # summary(m, (1, 3, 1024, 1024))
    # # y = m(x)
    x = torch.ones([3, 3, 256, 256]).cuda()
    m = NoiseDomain(256).cuda()
    y = m(x)
    exit(0)