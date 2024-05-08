# <center>A²PI²-PyTorch version2.2</center>

* 包含PyTorch生态的软件包.

# 1.timm

| 版本   | 描述             | 注意 | 适配M1 |
| ------ | ---------------- | ---- | ------ |
| 0.9.16 | PyTorch图像模型. | -    | 是     |

## 1.1.create_model()

创建一个模型.|`torch.nn.Module`

```python
import timm

model = timm.create_model(model_name='efficientnet_b0',  # str|模型的名称.
                          pretrained=False,  # bool|False|是否加载预训练的ImageNet-1k权重.
                          num_classes=6,  # int|1000|分类类别的数量.
                          in_chans=1)  # int|3|图像的输入通道数.
```

# 2.torch

| 版本  | 描述          | 注意 | 适配M1 |
| ----- | ------------- | ---- | ------ |
| 2.2.1 | 深度学习框架. | -    | 是     |

## 2.1.arange()

返回指定区间的一维张量.|`torch.Tensor`

```python
from torch import arange

tensor = arange(start=0,  # Number|0|点集的起始值.
                end=10,  # Number|点集的结束值.
                step=2)  # Number|1|步长.
```

## 2.2.argmax()

返回指定维度最大值的索引.|`torch.Tensor`

```python
from torch import argmax, Tensor

arr = Tensor([1, 2, 3, 4])
tensor = argmax(input=arr,  # Tensor|输入的数据.
                dim=0)  # int(可选)|None|维度的位置.
```

## 2.3.BoolTensor()

初始化一个`BoolTensor`, 数据类型为`torch.bool`.

```python
from torch import BoolTensor

tensor = BoolTensor(data=[0, 1, 1, 0])  # array-like|输入的数据.
```

## 2.4.backends

| 版本 | 描述                   | 注意 |
| ---- | ---------------------- | ---- |
| -    | Torch对不同后端的支持. | -    |

### 2.4.1.mps

#### 2.4.1.1.is_available()

判断系统是否支持Metal GPU(MPS).|`bool`

```python
from torch.backends.mps import is_available

is_available()
```

#### 2.4.1.2.is_built()

判断系统构建了Metal GPU(MPS)支持.|`bool`

```python
from torch.backends.mps import is_built

is_built()
```

## 2.5.cat()

按照指定维度合并多个张量.|`torch.Tensor`

```python
import torch

a = torch.Tensor([1, 2])
tensor = torch.cat(tensors=[a, a],  # sequence of Tensors|要合并的张量.
                   dim=0)  # int(可选)|0|沿指定维度合并.
```

## 2.6.cuda

| 版本 | 描述               | 注意 |
| ---- | ------------------ | ---- |
| -    | Torch对CUDA的支持. | -    |

### 2.6.1.is_available()

判断系统是否支持CUDA.|`bool`

```python
from torch import cuda

cuda.is_available()
```

## 2.7.cos()

返回输入张量的余弦值.|`torch.Tensor`

```python
import torch

tensor = torch.cos(input=torch.Tensor([0, 1, 0.5]))  # Tensor|输入的数据.
```

## 2.8.device()

分配一个计算设备.|`torch.device`

```python
import torch

device = torch.device('mps')
```

## 2.9.distributions

| 版本 | 描述            | 注意 |
| ---- | --------------- | ---- |
| -    | Torch的分布API. | -    |

### 2.9.1.Categorical()

实例化分类分布.

```python
import torch
from torch.distributions import Categorical

probs = torch.Tensor([0.1, 0.9])
d = Categorical(probs=probs)  # torch.Tensor|每一类的概率.
```

#### 2.9.1.1.log_prob()

构建等效对数损失函数.|`torch.Tensor`

```python
import torch
from torch.distributions import Categorical

probs = torch.Tensor([0.1, 0.9])
d = Categorical(probs=probs)
d.log_prob(value=d.sample())
```

#### 2.9.1.2.sample()

进行采样, 返回类别的索引.|`torch.Tensor`

```python
import torch
from torch.distributions import Categorical

probs = torch.Tensor([0.1, 0.9])
d = Categorical(probs=probs)
d.sample()
```

## 2.10.eq()

逐元素计算张量的数值是否相等.|`torch.Tensor`

```python
import torch

tensor = torch.Tensor([1, 2, 3, 4])
tensor = torch.eq(input=tensor,  # Tensor|需要比较的张量.
                  other=2)  # Tensor or float|需要比较的张量或者数值.
```

## 2.11.exp()

返回输入张量的指数值.|`torch.Tensor`

```python
import torch

tensor = torch.exp(input=torch.Tensor([0, 1, 0.5]))  # Tensor|输入的数据.
```

## 2.12.from_numpy()

从`numpy.ndarray`中创建`Tensor`.

```python
import numpy as np
from torch import from_numpy

arr = np.asarray([1, 2])
tensor = from_numpy(arr)  # np.ndarray|输入的数据.
```

## 2.13.full()

创建一个指定形状的张量填充指定数值.|`torch.Tensor`

```python
import torch

tensor = torch.full(size=[2, 3],  # sequence of ints|张量的形状.
                    fill_value=2024,  # Number|填充张量的数值.
                    dtype=torch.int32)  # torch.dtype(可选)|张量的数据类型.
```

## 2.14.IntTensor()

初始化一个`IntTensor`, 数据类型为`torch.int32`.

```python
from torch import IntTensor

tensor = IntTensor(data=[1, 2])  # array-like|输入的数据.
```

## 2.15.jit

### 2.15.1.trace()

返回序列化模型用于在没有Python的环境中运行.｜`torch.jit._trace.TopLevelTracedModule`

```python
import torch

traced_model = torch.jit.trace(func=model,  # torch.nn.Module|要转换的模型.
                               example_inputs=example_input)  # torch.Tensor|模型输入的实例张量.
```

## 2.16.load()

加载模型.

```python
from torch import load

model = load(f='./model.pt',  # str or a file-like|文件路径.
             map_location=None)  # str or torch.device(可选)|指定重新映射保存模型的位置.
```

## 2.17.log()

返回输入张量的自然对数值.|`torch.Tensor`

```python
import torch

tensor = torch.log(input=torch.Tensor([1, 2, 2.71828]))  # Tensor|输入的数据.
```

## 2.18.manual_seed()

设置随机种子.

```python
import torch

torch.manual_seed(seed=2022)  # int|随机种子.
```

## 2.19.masked_fill()

将张量中的元素填充为指定值.|`torch.Tensor`

```python
import torch

tensor = torch.Tensor([[1, 2, 3],
                       [4, 5, 6]])
mask = torch.BoolTensor([[1, 0, 1],
                         [0, 0, 0]])

tensor = torch.masked_fill(input=tensor,  # torch.Tensor|输入的张量.
                           mask=mask,  # torch.BoolTensor|掩码(布尔)张量.
                           value=9.5)  # float|要填充的值.
```

## 2.20.matmul()

两个张量的矩阵乘积.|`torch.Tensor`

```python
import torch

tensor0 = torch.Tensor([[1, 2, 3]])
tensor1 = torch.Tensor([[1], [2], [3]])

x = torch.matmul(input=tensor0,  # torch.Tensor|第一个张量.
                 other=tensor1)  # torch.Tensor|第二个张量.
```

## 2.21.mean()

返回输入张量所有元素的均值.|`torch.Tensor`

```python
import torch

tensor = torch.Tensor([[1, 2, 3],
                       [4, 5, 6]])

value = torch.mean(input=tensor,  # torch.Tensor|输入的张量.
                   dim=1,  # int|None|所沿的维度.
                   keepdim=False)  # bool|False|输出张量是否保留输入张量的维度.
```

## 2.22.ne()

逐元素计算张量的数值是否不等.|`torch.Tensor`

```python
import torch

tensor = torch.Tensor([1, 2, 3, 4])
tensor = torch.ne(input=tensor,  # Tensor|需要比较的张量.
                  other=2)  # Tensor or float|需要比较的张量或者数值.
```

## 2.23.nn

| 版本 | 描述                       | 注意 |
| ---- | -------------------------- | ---- |
| -    | Torch的计算图的基本构建块. | -    |

### 2.23.1.Conv2d()

实例化2D卷积层.

```python
from torch.nn import Conv2d

layer = Conv2d(in_channels=1,  # int|输入图片的色彩通道数量.
               out_channels=32,  # int|卷积核的数量.
               kernel_size=3,  # int or tuple|卷积核的尺寸.
               padding='same')  # int, tuple or {'same', 'valid'}(可选)|0|填充方式.
```

### 2.23.2.CrossEntropyLoss()

实例化交叉熵损失函数.

```python
from torch.nn import CrossEntropyLoss

loss = CrossEntropyLoss(ignore_index=-100)  # int(可选)|-100|忽略该值参与梯度计算.
```

### 2.23.3.Dropout()

实例化Dropout层.

```python
from torch import nn

layer = nn.Dropout(p=0.5)  # float|0.5|随机丢弃比例.
```

### 2.23.4.Embedding()

实例化嵌入层.

```python
from torch import nn

layer = nn.Embedding(num_embeddings=100,  # int|嵌入字典的大小.
                     embedding_dim=10)  # int|每个嵌入向量的大小.
```

### 2.23.5.Flatten()

实例化展平层.

```python
from torch import nn

layer = nn.Flatten()
```

### 2.23.6.functional

#### 2.23.6.1.binary_cross_entropy_with_logits()

计算带有sigmoid的二分类交叉熵的值.|`torch.Tensor`

```python
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

y = torch.Tensor([1, 1, 1, 1, 1])
y_pred = torch.Tensor([1, 1, 1, 1, 0])
loss = binary_cross_entropy_with_logits(input=y_pred,  # torch.Tensor|预测值.
                                        target=y)  # torch.Tensor|真实值.
```

#### 2.23.6.2.kl_div()

计算KL散度损失值.|`torch.Tensor`

```python
import torch
from torch.nn.functional import kl_div

y_pred = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([1.0, 2.0, 2.0])
tensor = kl_div(input=y_pred,  # torch.Tensor|预测值.
                target=y,  # torch.Tensor|真实值.
                reduction='batchmean')  # {'none', 'batchmean', 'sum' or 'mean'}|'mean'|损失归约方式.
```

#### 2.23.6.3.log_softmax()

应用对数softmax函数在输入的张量上.|`torch.Tensor`

```python
import torch
from torch.nn.functional import log_softmax

tensor = torch.Tensor([-2., -1., 0., 1., 2.])
tensor = log_softmax(input=tensor,  # torch.Tensor|输入的张量.
                     dim=0)  # int|指定的维度.
```

#### 2.23.6.4.relu()

应用relu函数在输入的张量上.|`torch.Tensor`

```python
import torch
from torch.nn.functional import relu

tensor = torch.Tensor([-2., -1., 0., 1., 2.])
tensor = relu(input=tensor)  # torch.Tensor|输入的张量.
```

#### 2.23.6.5.softmax()

应用softmax函数在输入的张量上.|`torch.Tensor`

```python
import torch
from torch.nn.functional import softmax

tensor = torch.Tensor([-2., -1., 0., 1., 2.])
tensor = softmax(input=tensor,  # torch.Tensor|输入的张量.
                 dim=0)  # int|指定的维度.
```

### 2.23.7.init

#### 2.23.7.1.xavier_uniform_()

使用均匀分布的Xavier方法初始化输入张量.|`torch.Tensor`

```python
import torch
from torch.nn.init import xavier_uniform_

tensor = torch.ones(3, 4)
tensor = xavier_uniform_(tensor=tensor)  # torch.Tensor|输入的张量.
```

### 2.23.8.Linear()

实例化全连接层.

```python
from torch import nn

layer = nn.Linear(in_features=32,  # int|输入神经元的数量.
                  out_features=32)  # int|神经元的数量.
```

### 2.23.9.LogSoftmax()

实例化LogSoftmax层.

```python
from torch import nn

layer = nn.LogSoftmax(dim=0)  # int(可选)|None|指定的维度.
```

### 2.23.10.MaxPool2d()

实例化2D最大池化层.

```python
from torch.nn import MaxPool2d

layer = MaxPool2d(kernel_size=2)  # int or tuple|池化窗口.
```

### 2.23.11.Module()

实例化`Module`.

```python
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 初始化神经网络层.
        self.flatten_layer = nn.Flatten()
        self.dense_layer = nn.Linear(in_features=28 * 28, out_features=128)
        self.output_layer = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        # 实现模型的前向传播.
        x = self.flatten_layer(x)
        x = self.dense_layer(x)
        x = self.dropout_layer(x)

        return self.output_layer(x)
```

#### 2.23.11.1.eval()

设置模块为评估模式.

```python
model.eval()
```

#### 2.23.11.2.load_state_dict()

加载模块的权重.

```python
model.load_state_dict(state_dict)  # dict|参数字典.
```

#### 2.23.11.3.named_parameters()

返回模块名称参数组成元组的迭代器.

```python
model.named_parameters()
```

#### 2.23.11.4.parameters()

返回模块参数迭代器.

```python
model.parameters()
```

#### 2.23.11.5.register_buffer()

向模块添加缓冲区, 缓冲区不被视为模型参数, 但是模块状态的一部分.

```python
model.register_buffer(name='buffer',  # str|缓冲区的名称.
                      tensor=torch.zeros(2, 3))  # Tensor or None|要注册的缓冲区.
```

#### 2.23.11.6.state_dict()

返回模块参数字典.

```python
model.state_dict()
```

#### 2.23.11.7.train()

设置模块为训练模式.

```python
model.train()
```

### 2.23.12.ModuleList()

实例化`ModuleList`.

```python
from torch import nn

linear_layer = nn.Linear(16, 32)
layers = nn.ModuleList(modules=(linear_layer,
                                linear_layer,
                                linear_layer))  # list of Module or tuple of Module|Module组成的列表.
```

### 2.23.13.Parameter()

实例化Parameter层, 可将张量视为一个`Module`.

```python
import torch
from torch import nn

tensor = torch.randn(3, 4)
layer = nn.Parameter(data=tensor)  # torch.Tensor|参数张量.
```

### 2.23.14.ReLU()

实例化ReLU层.

```python
from torch import nn

layer = nn.ReLU()
```

### 2.23.15.Sequential()

实例化`Sequential`.

```python
from torch import nn

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=28 * 28, out_features=128),
    nn.Linear(in_features=128, out_features=10),
)
```

#### 2.23.15.1.add_module()

添加一个模块到`Sequential`结尾, 使用给定名称.

```python
model.add_module(name='flatten_layer',  # str|模块名称.
                 module=nn.Flatten())  # nn.Module|模块.
```

#### 2.23.15.2.append()

添加一个模块到`Sequential`结尾.

```python
model.append(module=nn.Flatten())  # nn.Module|模块.
```

### 2.23.16.Softmax()

实例化Softmax层.

```python
from torch import nn

layer = nn.Softmax(dim=0)  # int(可选)|None|指定的维度.
```

### 2.23.17.utils

#### 2.23.17.1.rnn

##### 2.23.17.1.1.pad_sequence()

对不同长度的`Tensor`列表进行填充.|`torch.Tensor`

```python
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

a = Tensor([1, 2])
b = Tensor([3, 4, 5])
c = Tensor([7, 8, 9, 10])
tensor_list = [a, b, c]

new_tensor_list = pad_sequence(sequences=tensor_list,  # list of torch.Tensor|不同长度的Tensor列表.
                               batch_first=True,  # bool(可选)|False|按照批次优先(B, T, *)或最长长度优先(T, B, *).
                               padding_value=0.0)  # float(可选)|0.0|填充元素的值.
```

## 2.24.no_grad()

禁用梯度计算的上下文管理器(可以减少内存消耗).

```python
from torch import no_grad

with no_grad():
    # 代码.
```

## 2.25.ones()

生成全一张量.|`torch.Tensor`

```python
import torch

tensor = torch.ones(size=[2, 3])  # sequence of ints|张量的形状.
```

## 2.26.optim

| 版本 | 描述              | 注意                                |
| ---- | ----------------- | ----------------------------------- |
| -    | Torch的优化器API. | 1.优化器相同的类方法都写在`Adam`里. |

### 2.26.1.Adam()

实例化`Adam`优化器.

```python
from torch.optim import Adam

optimizer = Adam(params,  # 需要优化的参数.
                 lr=1e-3,  # float(可选)|1e-3|学习率.
                 betas=(0.9, 0.999),  # tuple of floats(可选)|(0.9, 0.999)|用于计算梯度及其平方滑动平均的系数.
                 eps=1e-8)  # float(可选)|1e-8|提高数值稳定性的常小数.
```

#### 2.26.1.1.param_groups

优化器的参数组, 可以指定优化哪些张量以及特定的优化选项.|`list`

```python
optimizer.param_groups
```

#### 2.26.1.2.step()

更新梯度.

```python
optimizer.step()
```

#### 2.26.1.3.zero_grad()

将梯度设置为零.

```python
optimizer.zero_grad()
```

### 2.26.2.AdamW()

实例化`AdamW`优化器.

```python
from torch.optim import AdamW

optimizer = AdamW(params,  # 需要优化的参数.
                  lr=1e-3,  # float(可选)|1e-3|学习率.
                  weight_decay=1e-2)  # float(可选)|1e-2|权重衰减.
```

### 2.26.3.SGD()

实例化随机梯度下降优化器.

```python
from torch.optim import SGD

optimizer = SGD(params,  # 需要优化的参数.
                lr=1e-2)  # float|学习率.
```

## 2.27.rand()

生成均匀分布随机张量.|`torch.Tensor`

```python
from torch import rand

tensor = rand(3, 4)  # sequence of ints|张量的形状.
```

## 2.28.randn()

生成正态分布随机张量.|`torch.Tensor`

```python
from torch import randn

tensor = randn(3, 4)  # sequence of ints|张量的形状.
```

## 2.29.Size()

实例化张量的形状.

```python
import torch

size = torch.Size([3, 4])
```

## 2.30.save()

保存模型或模型参数.

```python
from torch import nn
from torch import save

model = nn.Module()
save(obj=model,  # 要保存的模型.
     f='./model.pt')  # str or a file-like|文件路径.
```

## 2.31.sin()

返回输入张量的正弦值.|`torch.Tensor`

```python
import torch

tensor = torch.sin(input=torch.Tensor([0, 1, 0.5]))  # Tensor|输入的数据.
```

## 2.32.std()

返回输入张量所有元素的标准差.|`torch.Tensor`

```python
import torch

tensor = torch.Tensor([[1, 2, 3],
                       [4, 5, 6]])

value = torch.std(input=tensor,  # torch.Tensor|输入的张量.
                  dim=1,  # int|None|所沿的维度.
                  keepdim=False)  # bool|False|输出张量是否保留输入张量的维度.
```

## 2.33.Tensor()

初始化一个`Tensor`.

```python
from torch import Tensor

tensor = Tensor(data=[1, 2])  # array-like|输入的数据.
```

### 2.33.1.backward()

计算张量的梯度(反向传播).

```python
tensor.backward()
```

### 2.33.2.clip()

逐元素裁切张量.|`torch.Tensor`

```python
from torch import Tensor

tensor = Tensor(data=[0, 1, 2, 3, 4, 5, 6])
tensor = tensor.clip(min=1,  # int or float|None|最小值.
                     max=5)  # int or float|None|最大值.
```

### 2.33.3.detach()

禁用张量的梯度.|`torch.Tensor`

```python
tensor = tensor.detach()
```

### 2.33.4.device

张量的存储设备.|`torch.device`

```python
tensor.device
```

### 2.33.5.dim()

张量的维度.|`int`

```python
tensor.dim()
```

### 2.33.6.dtype

张量的数据类型.|`torch.dtype`

```python
tensor.dtype
```

### 2.33.7.expand()

增加张量的维度.|`torch.Tensor`

```python
import torch

tensor = torch.Tensor([1, 2])
tensor = tensor.expand(size=[1, 1, 2])  # sequence of ints|期望增加后的张量形状.
```

### 2.33.8.grad

张量的梯度.|`torch.Tensor`

```python
tensor.grad
```

### 2.33.9.grad_fn

张量的梯度函数.|`class`

```python
tensor.grad_fn
```

### 2.33.10.item()

将张量的值转换为Python数字.|`float`

```python
import torch
 
tensor = torch.Tensor(data=[1])
tensor.item()
```

### 2.33.11.requires_grad

张量是否需要返回梯度.|`bool`

```python
tensor.requires_grad
```

### 2.33.12.requires_grad_()

设置张量是否需要返回梯度.

```python
tensor.requires_grad_(requires_grad=True)  # bool|True|是否需要返回梯度.
```

### 2.33.13.reshape()

指定张量的形状(如果的可能的话返回是输入的视图).|`torch.Tensor`

```python
import torch

tensor = torch.Tensor([1, 2, 3, 4])
tensor = tensor.reshape((2, 2))  # tuple of ints|新的形状.
```

### 2.33.14.shape

张量的形状.|`torch.Size`

```python
tensor.shape
```

### 2.33.15.to()

执行张量的设备转换.|`torch.Tensor`

```python
from torch import cuda
from torch import Tensor

tensor = Tensor([1, 2])
if cuda.is_available():
    tensor = tensor.to(device='cuda')  # {'cpu', 'cuda', 'xpu', 'mkldnn', 'opengl',
                                       #  'opencl', 'ideep', 'hip', 've', 'ort', 'mlc',
                                       #  'xla', 'lazy', 'vulkan', 'meta', 'hpu'}|转换到的目标设备.
```

### 2.33.16.transpose()

对张量维度`dim0`和`dim1`进行转置操作.|`torch.Tensor`

```python
import torch

tensor = torch.Tensor([[1, 2],
                       [3, 4]])
tensor = tensor.transpose(dim0=1,  # int|要被转置的第一个维度.
                          dim1=0)  # int|要被转置的第二个维度.
```

### 2.33.17.type_as()

将张量转换为给定张量的数据类型.|`torch.Tensor`

```python
import torch

tensor = torch.Tensor([1, 2])
tensor = tensor.type_as(other=torch.tensor([1], dtype=torch.float16))  # Tensor|给定的张量.
```

### 2.33.18.unsqueeze()

增加张量的维度.|`torch.Tensor`

```python
from torch import Tensor

tensor = Tensor(data=[1, 2, 3])
tensor = tensor.unsqueeze(dim=1)  # int|添加新维度的位置.
```

### 2.33.19.view()

指定张量的形状(返回是输入的视图).|`torch.Tensor`

```python
import torch

tensor = torch.Tensor([1, 2, 3, 4])
tensor = tensor.view((2, 2))  # tuple of ints|新的形状.
```

## 2.34.triu()

返回二维张量的上三角部分.|`torch.Tensor`

```python
import torch

tensor = torch.triu(input=torch.rand(3, 3),  # torch.Tensor|输入的张量.
                    diagonal=0)  # int(可选)|0|考虑的对角线.
```

## 2.35.utils

### 2.35.1.data

| 版本 | 描述                 | 注意 |
| ---- | -------------------- | ---- |
| -    | Torch的数据加载工具. | -    |

#### 2.35.1.1.DataLoader()

实例化数据加载器.

```python
import numpy as np
from torch import Tensor
from torch.utils.data import DataLoader

arr = np.asarray([[1], [2], [3], [4]])

def collate_fn(batch):
    """对每个样本值乘2."""
    for sample in batch:
        sample *= 2

    return Tensor(batch)

dataloader = DataLoader(dataset=arr,  # array-like|要加载的数据集.
                        batch_size=2,  # int(可选)|1|批次大小.
                        shuffle=False,  # bool(可选)|False|是否打乱数据.
                        num_workers=0,  # int(可选)|0|用于数据加载的子进程数量, 0表示数据将在主进程中加载.
                        collate_fn=collate_fn,  # callable(可选)|None|整理函数.
                        drop_last=False)  # bool(可选)|False|是否丢弃最后一个不完整的批次.
```

#### 2.35.1.2.Dataset()

自定义一个数据集.

```python
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,
                 image_dir,  # 图片文件夹的路径.
                 annotations_file,  # 标签文件的位置.
                 transform=None,  # 对数据集进行预处理转换.
                 target_transform=None):  # 对标签进行预处理转换.
        super(MyDataset, self).__init__()
        # 初始化代码.

    def __len__(self):
        # 数据集的大小.

    def __getitem__(self, index):
        # 给定索引加载一个数据和标签.
        return feature, label
```

## 2.36.zeros()

生成全零张量.|`torch.Tensor`

```python
import torch

tensor = torch.zeros(size=[2, 3])  # sequence of ints|张量的形状.
```

# 3.torchvision

| 版本   | 描述                           | 注意 | 适配M1 |
| ------ | ------------------------------ | ---- | ------ |
| 0.13.1 | Torch的图像和视频数据集和模型. | -    | 是     |

## 3.1.datasets

| 版本 | 描述                     | 注意 |
| ---- | ------------------------ | ---- |
| -    | Torchvision的内置数据集. | -    |

### 3.1.1.MNIST()

实例化mnist数据集.

```python
from torchvision.datasets import MNIST

training_data = MNIST(root='./data',  # str|数据集保存的目录.
                      train=True,  # bool(可选)|True|是否是训练集.
                      transform=None,  # function(可选)|None|对数据集进行预处理转换.
                      download=False)  # bool(可选)|False|是否从网络下载数据集.
```

## 3.2.io

###  3.2.1.read_image()

将JPEG或PNG图像读入为张量.|`torch.Tensor`

```python
from torchvision.io import read_image

tensor = read_image(path='img.jpeg')  # str|图像的路径.
```

## 3.3.transforms

| 版本 | 描述                      | 注意 |
| ---- | ------------------------- | ---- |
| -    | Torchvision的数据转换API. | -    |

### 3.3.1.Lambda()

应用自定义的数据转换.|`torch.Tensor`

```python
from torch import Tensor
from torchvision.transforms import 

tensor = Tensor(data=[1, 2, 3, 4])
tensor = Lambda(lambd=lambda x: x * 10)(tensor)  # lambda or function|自定义的数据转换函数.
```

### 3.3.2.Resize()

将输入图像调整为给定的尺寸.|`torch.Tensor`

```python
import torch
from torchvision.transforms import Resize

image = torch.tensor([[[1, 2], [3, 4]]])
tensor = Resize(size=[3, 3],  # sequence of ints|输出张量的形状.
                antialias=True)(img=image)  # bool(可选)|True|是否应用抗锯齿.
```

### 3.3.3.ToPILImage()

将张量或numpy.ndarray转换为PIL Image.|`PIL.Image.Image`

```python
from torch import Tensor
from torchvision.transforms import ToPILImage

tensor = Tensor(data=[[1, 2], [3, 4]])
image = ToPILImage()(pic=tensor)  # Tensor or numpy.ndarray|要转换的张量.
```

### 3.3.4.ToTensor()

将PIL Image或numpy.ndarray转换为张量.|`torch.Tensor`

```python
import numpy as np
from torchvision.transforms import ToTensor

arr = np.asarray([[1, 2, 3]])
tensor = ToTensor()(pic=arr)  # PIL Image or numpy.ndarray|要转换的图像.
```

# 4.stable_baselines3

| 版本  | 描述                             | 注意 | 适配M1 |
| ----- | -------------------------------- | ---- | ------ |
| 1.6.2 | Torch的强化学习Stable Baselines. | -    | 是     |

## 4.1.A2C()

实例化优势动作评价算法.

```python
from stable_baselines3 import A2C


model = A2C(policy='MlpPolicy',  # {'MlpPolicy', 'CnnPolicy'}|使用的策略.
            env=env,  # gym.env|Gym环境.
            learning_rate=7e-4,  # float|7e-4|学习率.
            n_steps=5,  # int|5|每轮环境的时间步数.
            gamma=0.99,  # float|0.99|折扣系数.
            gae_lambda=1.0,  # float|1.0|广义优势估计器的偏差与方差权衡因子.
            ent_coef=0.0,  # float|0.0|损失计算的熵系数.
            vf_coef=0.5,  # float|0.5|损失计算的价值函数系数.
            max_grad_norm=0.5,  # float|0.5|梯度标准化的最大值.
            use_rms_prop=True,  # bool|True|是否使用RMSprop或Adam作为优化器.
            use_sde=False,  # bool|False|是否使用广义状态依赖探索(gSDE)而不是动作噪声探索.
            normalize_advantage=False,  # bool|False|是否标准化优势.
            tensorboard_log=None,  # str|None|日志的保存位置.
            policy_kwargs=dict(log_std_init=2,  # dict|None|用于创建策略的其他参数.
                               ortho_init=False),
            verbose=1)  # {0, 1, 2}|0|日志显示模式.
```

### 4.1.1.learn()

训练模型.

```python
model.learn(total_timesteps=2000000)  # int|训练步数.
```

### 4.1.3.save()

保存模型到zip文件.

```python
model.save(path='./a2c-AntBulletEnv-v0')  # str|文件名.
```

## 4.2.common

### 4.2.1.env_util

#### 4.2.1.1.make_vec_env()

创建一组并行环境.|`stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv`

```python
from stable_baselines3.common.env_util import make_vec_env

envs = make_vec_env(env_id='LunarLander-v2',  # str|环境id.
                    n_envs=16)  # int|1|并行的环境数量.
```

### 4.2.2.evaluation

#### 4.2.2.1.evaluate_policy()

评估模型并返回平均奖励.|`tuple`

```python
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(model=model,  # base_class.BaseAlgorithm|你想评估的模型.
                                          env=env,  # gym.env|Gym环境.
                                          n_eval_episodes=10,  # int|10|评估周期.
                                          deterministic=True)  # bool|True|使用确定动作还是随机动作.
```

### 4.2.3.vec_env

#### 4.2.3.1.DummyVecEnv()

创建向量化环境包装器, Python进程将逐一调用.|`stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv`

```python
import gym
from stable_baselines3.common.vec_env import DummyVecEnv

eval_env = DummyVecEnv(env_fns=[lambda: gym.make('LunarLander-v2')])  # list of functions|环境生成函数列表.
```

#### 4.2.3.2.VecNormalize()

并行环境的滑动平均、标准化装饰器.|`stable_baselines3.common.vec_env.vec_normalize.VecNormalize`

```python
from stable_baselines3.common.vec_env import VecNormalize

env = VecNormalize(venv=env,  # VecEnv|并行环境.
                   training=True,  # bool|True|是否更新滑动平均.
                   norm_obs=True,  # bool|True|是否标准化可观察空间.
                   norm_reward=True,  # bool|True|是否标准化奖罚.
                   clip_obs=10.0)  # float|10.0|可观察空间的最大值.
```

##### 4.2.3.2.1.load()

加载环境.|`stable_baselines3.common.vec_env.vec_normalize.VecNormalize`

```python
env = VecNormalize.load(load_path='./vec.pkl',  # str|文件名.
                        venv=env)  # VecEnv|并行环境.
```

##### 4.2.3.2.2.save()

保存环境到pickle文件.

```python
env.save(save_path='./vec.pkl')  # str|文件名.
```

## 4.3.PPO()

实例化近端策略算法.

```python
model = PPO(policy='MlpPolicy',  # {'MlpPolicy', 'CnnPolicy'}|使用的策略.
            env=envs,  # gym.env|Gym环境.
            n_steps=1024,  # int|2048|每轮环境的时间步数.
            batch_size=64,  # int|64|批次大小.
            n_epochs=4,  # int|10|优化代理损失的轮数.
            gamma=0.999,  # float|0.99|折扣系数.
            gae_lambda=0.98,  # float|0.95|广义优势估计器的偏差与方差权衡因子.
            ent_coef=0.01,  # float|0.0|损失计算的熵系数.
            verbose=1,  # {0, 1, 2}|0|日志显示模式.
            device='auto')  # torch.device or str|'auto'|分配的硬件设备(Torch支持的硬件设备).
```

### 4.3.1.learn()

训练模型.

```python
model.learn(total_timesteps=200000)  # int|训练步数.
```

### 4.3.2.load()

加载模型.|`stable_baselines3.ppo.ppo.PPO`

```python
model = PPO.load(path='ppo-LunarLander-v2',  # str|文件名.
                 print_system_info=True)  # bool|False|打印保存模型的系统信息和当前的系统信息.
```

### 4.3.3.save()

保存模型到zip文件.

```python
model.save(path='./ppo-LunarLander-v2')  # str|文件名.
```

