# HOOK
>[!note] 获取神经网络特征和梯度的有效工具

## 简述

在实验过程中，我们通常需要观察模型训练得到的卷积核、特征图或者梯度等信息，这在CNN可视化研究中经常用到。其中，卷积核最易获取，将模型参数保存即可得到；特征图是中间变量，所对应的图像处理完即会被系统清除，否则将严重占用内存；梯度跟特征图类似，除了叶子结点外，其它中间变量的梯度都被会内存释放，因而不能直接获取。
最容易想到的获取方法就是改变模型结构，在forward的最后不但返回模型的预测输出，还返回所需要的特征图等信息。

> [!tip] Pytorch的hook编程可以在不改变网络结构的基础上有效获取、改变模型中间变量以及梯度等信息。

常见有三个hook函数函数可以实现上述功能：
- Tensor.register_hook(hook_fn)，
- nn.Module.register_forward_hook(hook_fn)，
- nn.Module.register_backward_hook(hook_fn).

## Tensor.register_hook

> 功能： 注册一个反向传播hook函数，用于自动记录Tensor的梯度。

pytorch对于中间变量和非叶子节点的梯度，在梯度反传完之后会自动释放，来减缓内存占用，对于非中间变量和叶子节点，我们可以直接通过访问其梯度得到，而对于中间变量和非叶子节点，我们只能用hook来实现。

```python
def hook_fn(grad):
    print(grad)

tensor_k.register_hook(hook_fn)  # tensor_k是中间节点, 此时成功注册了hook之后，在梯度反传时就会直接打印该节点梯度值
```
>[!note] 自定义的hook_fn函数的函数名可以是任取的，它的参数是grad，表示Tensor的梯度。
><br>这里需要注意的是，如果要将梯度值装在一个列表或字典里，那么首先要定义一个同名的全局变量的列表或字典，即使是局部变量，也要在自定义的hook函数外面。另一个需要注意的点就是如果要改变梯度值，hook函数要有返回值，返回改变后的梯度。

## nn.Module.register_forward_hook and nn.Module.register_backward_hook

>[!note] 需要注意的是，这两个的操作对象都是nn.Module类，如神经网络中的卷积层(nn.Conv2d)，全连接层(nn.Linear)，池化层(nn.MaxPool2d, nn.AvgPool2d)，激活层(nn.ReLU)或者nn.Sequential定义的小模块等,而上述的Tensor.register_hook是针对单个tensor节点，, 

```python
# 其hook_fn的函数定义有3个参数，分别表示：模块，模块的输入，模块的输出
def forward_hook(module, input, output):
    operations

def backward_hook(module, grad_in, grad_out):
    operations
```

## 利用hook获取模型的中间特征图

```python
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
from torchvision.transforms.functional import normalize, resize, to_tensor, to_pil_image

model = models.vgg16_bn(pretrained=True).eval()

# 建立一个全局变量的字典，将特征图放在其中
feature_map = {}

# 构建hook函数
def forward_hook(module, inp, outp):
    feature_map['features18'] = outp

# 获取第18层的layer
features = list(model.children())[0]
hook_layer = features[18]

# 对第18层的layer注册hook
hook_layer.register_forward_hook(forward_hook)

# 进行一次前向传播，此时对应的第18层的特征图就出现在feature_map里了
with torch.no_grad():
    score = model(input_img)
```