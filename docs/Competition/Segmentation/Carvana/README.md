# Carvana Image Masking Challenge

该竞赛内容就是将给定的2K图片进行车辆的语义分割（是一个单目标的语义分割任务）

>本文仅作为入门学习部分，所以仅使用语义分割任务最经典的FCN和U-net网络架构实现，最后Dice评分分别为0.982和0.989。

## Dataset构建

>[!tip]由于给定的数据集只有train和test，所以我们需要人为的将train数据集拆分成训练集和验证集，基于前面的学习我更习惯于用文本的方式进行划分，用文本来存储每个样本的名字，因为对应的真实掩码和输入图片都是以样本名字命名的。所以首先构建一个spilt_train_and_eval.py脚本

```python
import os
from pathlib import Path
import random

def split_data(data_dir, label_dir, ratio: float = 0.2):
    '''将所有的图片按照图片名字按比例随机划分成训练集和验证集文档'''
    data_paths = os.listdir(data_dir)
    data_len = len(data_paths)
    valid_data_size = int(data_len * ratio)
    train_data_size = data_len - valid_data_size
    train_data_paths = random.sample(data_paths, train_data_size)
    valid_data_paths = list(set(data_paths).difference(set(train_data_paths)))
    with open('../data/train.txt', 'w') as f:
        for train_data_path in train_data_paths:
            f.write(train_data_path + '\n')
    with open('../data/valid.txt', 'w') as f:
        for valid_data_path in valid_data_paths:
            f.write(valid_data_path + '\n')



if __name__ == '__main__':
    root_dir = Path('../data/train')
    root_masks_dir = Path('../data/train_mask')
    random.seed(520)

    split_data(root_dir, root_masks_dir, ratio=0.2)
```

此时我们已经生成了对应的train.txt和valid.txt，我们可以实现自己的Dataset通过访问对应的txt文件来读取对应的数据集。

```python
import os

import torch
from torch.utils.data import Dataset
from PIL import Image

class CarvanaSegmentation(Dataset):
    def __init__(self, root_dir, transforms=None, mode='train', txt_name: str = 'train.txt'):
        super(CarvanaSegmentation, self).__init__()
        # 确认一下root路径是否存在
        assert os.path.exists(root_dir), "path '{}' dose not exist.".format(root_dir)
        self.mode = 'train' if mode == 'valid' else mode
        img_dir = os.path.join(root_dir, self.mode)
        mask_dir = os.path.join(root_dir, self.mode + '_mask')
        txt_path = os.path.join(root_dir, txt_name)
        assert os.path.exists(txt_path), "txt_name '{}' dose not exist.".format(txt_name)

        with open(txt_path, 'r') as f:
            file_names = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        self.images = [os.path.join(img_dir + '/' + file_name) for file_name in file_names]
        if self.mode != 'test':
            self.masks = [os.path.join(mask_dir + '/' + file_name[:-4] + '_mask.gif') for file_name in file_names]
            assert (len(self.masks) == len(self.images))
        self.transforms = transforms

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        # 如果是训练集和验证集
        if self.mode != 'test':
            mask = Image.open(self.masks[idx]).convert('L')
            if self.transforms is not None:
                # 如果有数据增强，必须连带着mask一起增强，不然标签对应不上
                img, mask = self.transforms(img, mask)
            return img, mask
        else:
            # 否则作为测试集只有image
            if self.transforms is not None:
                img = self.transforms(img)
            return img

    def __len__(self):
        return len(self.images)
```
>[!note]一般写完dataset之后，我们都会调一个dataloader测试一下我们手写的Dataset的可用性，然后用PLT或者Opencv展示一下数据情况。
## 网络搭建

### FCN网络

>[!tips] 下述代码源于霹雳吧啦Wz(https://space.bilibili.com/18161609)

**ResNet50作为Backbone**

```python
import torch
import torch.nn as nn


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        '''与之前的resnet源码最大的区别，就是引入了replace_stride_with_dilation参数，这个参数用于膨胀卷积的上采样'''
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False] # 三个false就是默认用传统的resnet
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the models by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride # stage_2 的时候因为传入的stride是2， 所以dilation也就是2
            stride = 1 # 修改stride = 1， 是因为fcn的bottlneck当中的残差分支不需要下采样，所以步长应该从原本的2降到1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    model = ResNet(block, layers, **kwargs)
    return model


def resnet50(**kwargs):
    r"""ResNet-50 models from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a models pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)
```

**FCN**

```python
from collections import OrderedDict
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import functional as F

from .Backbone import resnet50

class IntermediateLayerGetter(nn.ModuleDict):

    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        # 如果model当中没有return_layers所需要的层，则报错
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("模型中并没有返回层所需要的层结构")
        orig_return_layers = return_layers # 保存下原有返回层，不受后序使用干扰
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        # 总有序字典顺序存储需要保留的层结构
        layers = OrderedDict()
        for name, layer in model.named_children():
            layers[name] = layer
            # 知道return_layers删空时，说明所有需要的层结构以及访问到了，直接break返回
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
        # 直接调用父类的初始化函数，通过MoudleDict来构建一个Model
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # 因为需要最后返回的是一个字典，这样便于辅助分支所需要的数据
        out = OrderedDict()
        for name, layer in self.items():
            x = layer(x)
            if name in self.return_layers:
                out[self.return_layers[name]] = x
        return out


class FCN(nn.Module):

    def __init__(self, backbone, classifier, aux_classifier=None):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor):
        # 先记录下输入的分辨率
        input_shape = x.shape[-2:]
        feature = self.backbone(x)
        result = OrderedDict()
        x = feature['out']
        x = self.classifier(x)
        # 插值（mode调整为双线性插值）
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)

        result['out'] = x
        if self.aux_classifier is not None:
            x = feature['aux']
            x = self.aux_classifier(x)
            x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
            result['aux'] = x

        return result

class FCNHead(nn.Sequential):
    '''结构比较简单的话，直接用Sequential，调用父类的初始化就行了'''
    def __init__(self, in_channels, channels):
        layers = []
        inter_channels = in_channels // 4
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=inter_channels, kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(inter_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        layers.append(nn.Conv2d(in_channels=inter_channels, out_channels=channels, kernel_size=1, stride=1))
        super(FCNHead, self).__init__(*layers)

def fcn_resnet50(aux: bool, num_classes: int, pretrain_backbone: bool=False, pretrain_dir: str=None):
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])
    if pretrain_backbone:
        # map_location :
        # 具体来说，map_location参数是用于重定向，比如此前模型的参数是在cpu中的，我们希望将其加载到cuda:0中。
        # 或者我们有多张卡，那么我们就可以将卡1中训练好的模型加载到卡2中，这在数据并行的分布式深度学习中可能会用到。
        backbone.load_state_dict(torch.load(pretrain_dir, map_location='cpu'))

    # FCNhead 的输入通道数
    out_inplanes = 2048
    aux_inplanes= 1024

    layers = {'layer4': 'out'}
    if aux:
        layers['layer3'] = 'aux'

    # resnet50的原始结构中存在一些不必要的层次，比如在stage4之后的全局平均池化之类的
    backbone = IntermediateLayerGetter(backbone, return_layers = layers)
    aux_classifer = None
    if aux:
        aux_classifer = FCNHead(aux_inplanes, num_classes)
    classifier = FCNHead(out_inplanes, num_classes)

    model = FCN(backbone, classifier, aux_classifer)
    return model
```

### U-net网络

```python
import torch
from torch import nn
from torch.nn import functional as F

from typing import Dict
from torchsummary import summary

class DoubleConv(nn.Sequential):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            # inplace = False(默认)时,不会修改输入对象的值,而是返回一个新创建的对象,所以打印出对象存储地址不同,类似于C语言的值传递
            # inplace = True 时,会修改输入对象的值,所以打印出对象存储地址相同,类似于C语言的址传递,节省反复申请与释放内存的空间与时间,只是将原来的地址传递,效率更好
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class Down(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels),
        )


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear: bool=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        x1 = self.up(x1)
        # 得到两个拼接图片的高宽差值
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        # 对输出的x1进行填充，使得两个concat的图片大小相同
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        # 在通道数上拼接
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x



class Unet(nn.Module):

    def __init__(self, in_channels, num_classes, bilinear: bool=True, base_c=64):
        '''
        :param in_channels:
        :param num_classes:
        :param bilinear: 是否使用双线性插值代替转置卷积
        :param base_c: 基准channel数
        '''
        super(Unet, self).__init__()
        self.head = DoubleConv(in_channels, base_c)
        self.down_1 = Down(base_c, base_c * 2)
        self.down_2 = Down(base_c * 2, base_c * 4)
        self.down_3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down_4 = Down(base_c * 8, base_c * 16 // factor)

        self.up_1 = Up(base_c * 16, base_c * 8 // factor)
        self.up_2 = Up(base_c * 8, base_c * 4 // factor)
        self.up_3 = Up(base_c * 4, base_c * 2 // factor)
        self.up_4 = Up(base_c * 2, base_c)
        self.classify = nn.Conv2d(base_c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.head(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)
        x = self.up_1(x5, x4)
        x = self.up_2(x, x3)
        x = self.up_3(x, x2)
        x = self.up_4(x, x1)
        logits = self.classify(x)
        return {'out': logits}


def cal_h_w(hin, win, k_size, padding, stride):
    hout = (hin + 2 * padding - k_size) / stride + 1
    wout = (win + 2 * padding - k_size) / stride + 1
    return hout, wout


if __name__ == '__main__':
    # 用summary打印一下网络结构
    net = Unet(in_channels=3, num_classes=1)
    net.to('cuda')
    print(summary(net, (3, 480, 480)))
```

## 模型训练

>[!note]使用的是FCN网络进行训练，如果想用U-net需要稍微修改一下train脚本就行了，主要是模型的输出格式不一样（实际可以改成一样，我偷懒了）

```python
import logging
import os.path
import warnings

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import functional as F
import torchvision.utils as vutils


import numpy as np
import matplotlib.pyplot as plt
from datasets import CarvanaSegmentation
from models import fcn_resnet50
from utils import transforms as T
from utils import create_lr_scheduler, train_one_epoch, evaluate

class SegmentationPresetTrain:

    def __init__(self, hfilp_prob=0.5, vfilp_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        trans = [T.Resize(256, 256)]
        if hfilp_prob > 0:
            trans.append(T.RandomHorizontalFlip(hfilp_prob))
        if vfilp_prob > 0:
            trans.append(T.RandomVerticalFlip(vfilp_prob))
        trans.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        self.transforms = T.Compose(trans)

    def __call__(self, img, target):
        return self.transforms(img, target)


class SegmentationPresetValid:

    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = T.Compose([
            T.Resize(256, 256),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    def __call__(self, img, target):
        return self.transforms(img, target)


def get_transforms(train: bool):
    return SegmentationPresetTrain(hfilp_prob=0, vfilp_prob=0) if train else SegmentationPresetValid()


def create_model(aux: bool, num_classes: int, pretrain_backbone: bool=False, backbone_dir=None,pretrain_model: bool=False, checkpoint=None):
    model = fcn_resnet50(aux=aux, num_classes=num_classes, pretrain_backbone=pretrain_backbone, pretrain_dir=backbone_dir)

    if pretrain_model and checkpoint is not None:
        weight_dict = torch.load(checkpoint, map_location='cpu')

        missing_keys, unexpected_keys = model.load_state_dict(weight_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    return model


def create_logger():
    # 设置训练logger
    logger = logging.getLogger(name='training_log')
    logger.setLevel(logging.INFO)

    # 输出控制台的处理器
    consoleHandler = logging.StreamHandler()
    # 输出到文件中的处理器
    fileHandler = logging.FileHandler(filename='Carvana.log', mode='w')

    standard_formatter = logging.Formatter('%(asctime)s %(name)s [%(pathname)s line:(lineno)d] %(levelname)s %(message)s]')
    simple_formatter = logging.Formatter('%(levelname)s %(message)s')

    consoleHandler.setFormatter(standard_formatter)
    fileHandler.setFormatter(simple_formatter)

    logger.addHandler(consoleHandler)
    logger.addHandler(fileHandler)

    return logger


def main(args):
    # torch.device 可以之指明创建那个设备对象
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 设置打印日志
    logger = create_logger()
    print(logger)

    batch_size = args.batch_size
    num_classes = 1 if args.num_classes == 1 else args.num_classes + 1


    train_dataset = CarvanaSegmentation(args.data_path, get_transforms(train=True))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

    valid_dataset = CarvanaSegmentation(args.data_path, get_transforms(train=False), 'valid', 'valid.txt')
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=args.num_workers)

    model = create_model(aux=True, num_classes=num_classes, pretrain_backbone=True, backbone_dir=args.backbone_path, pretrain_model=False)
    model.to(device)

    params_to_optimize = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model.classifier.parameters() if p.requires_grad]}
    ]

    if args.aux:
        params = [p for p in model.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})

    optimizer = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=args.weight_decay)

    criterion = nn.BCEWithLogitsLoss()

    # 自动混合精度，用于加速运算，节约显存
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    lr_scheduler = create_lr_scheduler(optimizer, len(train_dataloader), args.num_epochs, warmup=True)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # 训练设置
    train_losses = []
    val_losses = []
    lr_rates = []

    min_loss = 0x3f3f3f3f
    max_score = 0.

    for epoch in range(args.start_epoch, args.num_epochs):
        logger.info("epoch : {}".format(epoch + 1))
        # 训练一个epoch
        train_epoch_loss = train_one_epoch(model, optimizer, criterion, train_dataloader, device, lr_scheduler, scaler)
        train_losses.append(train_epoch_loss)

        # 验证一个epoch
        val_epoch_loss, dice_epoch_score = evaluate(model, criterion, valid_dataloader, device)
        val_losses.append(val_epoch_loss)

        lr_rates.append(optimizer.param_groups[0]['lr'])

        logger.info(f"epoch - {epoch + 1}/{args.num_epochs}:\n"
                    f"{' ' * 5}dice score - {dice_epoch_score}\n"
                    f"{' ' * 5}train loss - {train_epoch_loss}\n"
                    f"{' ' * 5}val loss - {val_epoch_loss}\n"
                    f"{' ' * 5}LR rate - {lr_scheduler.get_last_lr()}")

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}

        if (val_epoch_loss <= min_loss and dice_epoch_score >= max_score):
            min_loss = val_epoch_loss
            max_score = dice_epoch_score
            torch.save(save_file, args.save_weight_path + "model_weight.pth")
            logger.info(f"epoch - {epoch + 1} save the model which dice score : {dice_epoch_score} and val loss : {val_epoch_loss}")

    return {
        'lr': lr_rates,
        'train_loss': train_losses,
        'valid_loss': val_losses,
    }

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="pytorch fcn training")

    parser.add_argument("--data-path", default="./data/", help="Data root")
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--aux", default=True, type=bool, help="auxilier loss")
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("-b", "--batch-size", default=12, type=int)
    parser.add_argument("--num-epochs", default=30, type=int, metavar="N",
                        help="number of total epochs to train")
    parser.add_argument("--num-workers", default=8, type=int, help='num of worker')

    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start_epoch')
    # Mixed precision training parameters
    parser.add_argument("--amp", default=False, type=bool,
                        help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--backbone-path", default='./weight/resnet50.pth', help='backbone pretrain weight path')
    parser.add_argument("--save-weight-path", default='./weight/', help='save weight path')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_args()

    if not os.path.exists("./weight"):
        os.mkdir("./weight")

    main(args)
```

## 推理脚本

```python
import csv
import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from torchvision import transforms as T

from models import fcn_resnet50
from datasets import CarvanaSegmentation
from utils import mask2rle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

def plot_pred_segment(img, mask = None, pred = None, epoch = 0, root_dir = './data/pred/'):

    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    img = img.to('cpu')
    if mask is not None:
        mask = mask.to('cpu')
    pred = pred.to('cpu')

    if mask is None:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(14, 4))
        fig.tight_layout()

        ax1.axis('off')
        ax1.set_title('images')
        ax1.imshow(np.transpose(vutils.make_grid(img, padding=2).numpy(), (1, 2, 0)))

        ax2.axis('off')
        ax2.set_title('pred masks')
        pred = torch.sigmoid(pred)
        mask_np = vutils.make_grid(pred, padding=2).numpy()
        ax2.imshow(np.transpose(mask_np, (1, 2, 0)), cmap='gray')
        plt.savefig(root_dir + f'epoch_{epoch}_pred.jpg')
        plt.show()
        plt.close(fig)
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, figsize=(14, 6))
        fig.tight_layout()

        ax1.axis('off')
        ax1.set_title('images')
        ax1.imshow(np.transpose( vutils.make_grid(img, padding=2).numpy(), (1, 2, 0)))

        ax2.axis('off')
        ax2.set_title('ground trues')
        ax2.imshow(np.transpose(vutils.make_grid(mask, padding=2).numpy(), (1, 2, 0)), cmap='gray')

        ax3.axis('off')
        ax3.set_title('pred masks')
        pred = torch.sigmoid(pred)
        mask_np = vutils.make_grid(pred, padding=2).numpy()
        ax3.imshow(np.transpose(mask_np, (1, 2, 0)), cmap='gray')
        plt.savefig(root_dir + f'epoch_{epoch}_pred.jpg')
        plt.show()
        plt.close(fig)


def main():
    
    batch_size = 4
    submission_path = './data/submission.csv'
    weight_path = './weight/model_weight.pth'
    root_dir = './data/'
    assert os.path.exists(weight_path), f"weights {weight_path} not found"
    assert os.path.exists(root_dir), f"img root {root_dir} not found"

    model = fcn_resnet50(aux=True, num_classes=1, pretrain_backbone=False)

    model_weight = torch.load(weight_path)['model']
    model.load_state_dict(model_weight)

    transforms = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

    test_dataset = CarvanaSegmentation(root_dir=root_dir, transforms=transforms, mode='test', txt_name='test.txt')
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print("生成csv文件")
    with open(submission_path, encoding="utf-8", mode='w', newline='') as f:
        csv_f = csv.writer(f)
        csv_f.writerow(['rle_mask'])
        with torch.no_grad():
            for idx, imgs in enumerate(tqdm(test_dataloader)):
                imgs = imgs.to(device)
                pred = model(imgs)
                pred = pred['out']
                # plot_pred_segment(imgs, pred=pred, epoch=idx, root_dir='./data/pred/')
                pred = torch.sigmoid(F.interpolate(pred, (1280, 1918), mode='bilinear', align_corners=False))
                pred = pred > 0.5
                pred_np = pred.cpu().detach().numpy().astype('uint8')
                masks = [[mask2rle(item)] for item in pred_np]
                csv_f.writerows(masks)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"生成结束，共生成有{batch_size * (idx + 1)}行数据")
    mask = pd.read_csv(submission_path)
    with open('./data/test.txt', 'r') as f:
        mask.insert(0, 'img', f.readlines())
        mask['img'] = mask['img'].apply(lambda x: x[:-1])
    mask.to_csv(submission_path, index=False)


if __name__ == '__main__':
    main()

```

>[!tip] 其实刚开始做Kaggle最痛苦的就是我模型训练好了，数据都跑出来了，但是不会生成提交结构的csv文件，下面给出这个项目的提交文件所需的mask转rle编码格式的方式

```python
import numpy as np

def mask2rle(mask_image):
    pixels = mask_image.flatten()
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return ' '.join(str(x) for x in runs)
```

>[!warning]上述代码都仅供初学者参考，代码并不优美（我觉得是我自己写的垃圾山，但是毕竟是入门的项目，能独立写完跑通就挺好），也不能直接运行，毕竟你的文件存放格式和我的有所不同，路径需要变换。比如我在实现的过程中我的训练和推理脚本在本地pycharm里可以在子目录中运行，但是放在服务器端则无法正确运行（显示找不到部分子包内函数的调用路径）所以只能将训练和推理脚本拉到根目录当中才行，还没搞懂为什么。