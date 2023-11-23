# timm

> 是Hugging Face开源网站所提供的第三方库，内置实现了大量的计算机视觉领域的SOTA模型、关键层结构、工具、优化器、学习率规划器、数据增强方式等等。

## quick start

```
pip install timm
```

需要注意现在pip拉下来的timm都是最新的0.9.10版本，可能会与很多论文的源码有冲突，所以如果需要使用早期版本，基本上使用如下方式即可
```
pip install timm==0.6.13
```

### 快速创建一个预训练模型
```python
import timm

# 使用create_model函数可以快速创建timm所提供的预训练模型。并且支持修改分类头为设定的num_classes值
m = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=10)
# 进入eval模式进行模型推理
m.eval()
```

>[!tip] 如果想要知道timm内提供了哪些预训练模型，可以使用timm.list_models()函数实现，内部接受一个string字符串作为过滤。
>```python
>model_list = timm.list_models('*resnet*',pretrained=True)
>print(*model_list, sep="\n")
>```

### 快速微调一个预训练模型

```python
import timm

# 其实就是刚刚说的修改分类头就是最快速的微调
m = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=10)
```

### 使用一个预训练模型坐特征提取

```python
x = torch.randn(1, 3, 224, 224)
# 使用create_model函数可以快速创建timm所提供的预训练模型。并且支持修改分类头为设定的num_classes值
model = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=10)
features = model.forward_features(x)
# torch.Size([1, 960, 7, 7])
```
如此可以自动的省去最后的分类头，而是获得我们分类头前的特征图

### 图像增广

```python
>>> timm.data.create_transform((3, 224, 224))
Compose(
    Resize(size=256, interpolation=bilinear, max_size=None, antialias=None)
    CenterCrop(size=(224, 224))
    ToTensor()
    Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
)
```
可以快速创建Transform，减少工作量。

同时可以结合模型预训练的配置参数来生成transorm

```python
>>> data_cfg = timm.data.resolve_data_config(model.pretrained_cfg) # 得到模型预训练的配置
>>> transform = timm.data.create_transform(**data_cfg)
>>> transform
Compose(
    Resize(size=256, interpolation=bicubic, max_size=None, antialias=None)
    CenterCrop(size=(224, 224))
    ToTensor()
    Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
)
```