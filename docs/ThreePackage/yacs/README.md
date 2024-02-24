# yacs

> 常用于创建默认配置，然后从命令行传递 --config-file参数来覆盖默认配置，进行统一的配置管理

## argparse

> 统一的配置控制通常都使用yacs和argparse共同管理。

argparse 是python自带的命令行选项、参数和子命令解析器，可以用来方便地读取命令行参数。

范例解析：

```py
import argparse  #导入argparse模块

parser = argparse.ArgumentParser(description='argparse learning')  # 创建解析器
parser.add_argument('--integers', type=int, default=0, help='input an integer')  # 添加参数

args = parser.parse_args()  # 解析参数
print(args)
```

>[!note] 其中比较常用的就是创建解析器的一些参数：<br> ArgumentParser方法的最常用参数主要就是description和formatter_class，前者表示解析器的说明，后者表示自定义帮助文档<br> 最为关键的还是add_argument方法

add_argument 的主要参数：

1. name of flags：

    该方法必须确认当前这个参数是否是形如'-f'或'--foo'的一个可选参数，或者是一个位置参数。需要注意当'-'和'--'参数同时出现的时候，系统默认后者为参数名，但是在命令行输入的时候没有区别。

2. action:

    将命令行参数和某一个动作进行关联。具体用到再去查，比较少用。

3. default：默认值

4. required：该参数是否必须设置

5. type: 参数的类型，如果传入类型不符合会报错

6. choices: 参数值只能从固定的选项里选择

7. help: 指定参数的说明信息

8. dest：设置参数在代码中的变量名（此时在代码中只能用变量名来访问参数）

## yacs

### 引入库

首先需要创建一个配置未见：default.py。这个文件是所有可配置选项的参考点，需要有很好的文档记录性，并且为每个选项提供合理的默认值。

```py
from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 8
# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4

_C.TRAIN = CN()
# A very important hyperparameter
_C.TRAIN.HYPERPARAMETER_1 = 0.1
# The all important scales for the stuff
_C.TRAIN.SCALES = (2, 4, 8, 16)

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  return _C.clone()

```

此时对于每一次实验，我们都需要创建一个YMAL文件，在文件中写出需要修改的参数即可。

```py
def update_config(cfg, args):
    cfg.defrost()
    cfg.merge_from_file(args.cfg)  # update cfg
    cfg.freeze()
```