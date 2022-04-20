# <center>A²PI²-PyTorch version2.2</center>

* 包含PyTorch生态的软件包.

# 1.torch

| 版本   | 描述          | 注意 | 适配M1 |
| ------ | ------------- | ---- | ------ |
| 1.11.0 | 深度学习框架. | -    | 是     |

## 1.1.argmax()

返回指定维度最大值的索引.|`torch.Tensor`

```python
from torch import argmax, Tensor

arr = Tensor([1, 2, 3, 4])
tensor = argmax(input=arr,  # Tensor|输入的数据.
                dim=0)  # int(可选)|None|维度的位置.
```

## 1.2.cuda

| 版本 | 描述               | 注意 |
| ---- | ------------------ | ---- |
| -    | Torch对CUDA的支持. | -    |

### 1.2.1.is_available()

判断系统是否支持CUDA.|`bool`

```python
from torch import cuda

cuda.is_available()
```

## 1.3.from_numpy()

从`numpy.ndarray`中创建`Tensor`.

```python
import numpy as np
from torch import from_numpy

arr = np.asarray([1, 2])
tensor = from_numpy(arr)  # np.ndarray|输入的数据.
```

## 1.4.load()

加载模型.

```python
from torch import load

model = load(f='./model.pt')  # str or a file-like|文件路径.
```

## 1.5.nn

| 版本 | 描述                       | 注意 |
| ---- | -------------------------- | ---- |
| -    | Torch的计算图的基本构建块. | -    |

### 1.5.1.CrossEntropyLoss()

实例化交叉熵损失函数.

```python
from torch.nn import CrossEntropyLoss

loss = CrossEntropyLoss()
```

### 1.5.2.Dropout()

实例化Dropout层.

```python
from torch import nn

layer = nn.Dropout(p=0.5)  # float|0.5|随机丢弃比例.
```

### 1.5.3.Flatten()

实例化展平层.

```python
from torch import nn

layer = nn.Flatten()
```

### 1.5.4.Linear()

实例化全连接层.

```python
from torch import nn

layer = nn.Linear(in_features=32,  # int|输入神经元的数量.
                  out_features=32)  # int|神经元的数量.
```

### 1.5.5.Module()

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

#### 1.5.5.1.eval()

设置模块为评估模式.

```python
model.eval()
```

#### 1.5.5.2.load_state_dict()

加载模块的权重.

```python
model.load_state_dict(state_dict)  # dict|参数字典.
```

#### 1.5.5.3.parameters()

返回模块参数迭代器.

```python
model.parameters()
```

#### 1.5.5.4.train()

设置模块为训练模式.

```python
model.train()
```

### 1.5.6.ReLU()

实例化ReLU层.

```python
from torch import nn

layer = nn.ReLU()
```

### 1.5.7.Sequential()

实例化`Sequential`.

```python
from torch import nn

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=28 * 28, out_features=128),
    nn.Linear(in_features=128, out_features=10),
)
```

#### 1.5.7.1.add_module()

添加一个模块到`Sequential`结尾, 使用给定名称.

```python
model.add_module(name='flatten_layer',  # str|模块名称.
                 module=nn.Flatten())  # nn.Module|模块.
```

#### 1.5.7.2.append()

添加一个模块到`Sequential`结尾.

```python
model.append(module=nn.Flatten())  # nn.Module|模块.
```

### 1.5.8.Softmax()

实例化Softmax层.

```python
from torch import nn

layer = nn.Softmax()
```

## 1.6.no_grad()

禁用梯度计算的上下文管理器(可以减少内存消耗).

```python
from torch import no_grad

with no_grad():
    # 代码.
```

## 1.7.optim

| 版本 | 描述              | 注意                                |
| ---- | ----------------- | ----------------------------------- |
| -    | Torch的优化器API. | 1.优化器相同的类方法都写在`Adam`里. |

### 1.7.1.Adam()

实例化`Adam`优化器.

```python
from torch.optim import Adam

optimizer = Adam(params,  # 需要优化的参数.
                 lr=1e-3)  # float(可选)|1e-3|学习率.
```

#### 1.7.1.1.step()

更新梯度.

```python
optimizer.step()
```

#### 1.7.1.2.zero_grad()

将梯度设置为零.

```python
optimizer.zero_grad()
```

## 1.8.rand()

生成均匀分布随机张量.|`torch.Tensor`

```python
from torch import rand

tensor = rand(3, 4)  # int|张量的形状.
```

## 1.9.save()

保存模型或模型参数.

```python
from torch import nn
from torch import save

model = nn.Module()
save(obj=model,  # 要保存的模型.
     f='./model.pt')  # str or a file-like|文件路径.
```

## 1.10.Tensor()

初始化一个`Tensor`.

```python
from torch import Tensor

tensor = Tensor(data=[1, 2])  # array-like|输入的数据.
```

### 1.10.1.backward()

计算张量的梯度(反向传播).

```python
tensor.backward()
```

### 1.10.2.clip()

逐元素裁切张量.|`torch.Tensor`

```python
from torch import Tensor

tensor = Tensor(data=[0, 1, 2, 3, 4, 5, 6])
tensor = tensor.clip(min=1,  # int or float|None|最小值.
                     max=5)  # int or float|None|最大值.
```

### 1.10.3.device

张量的存储设备.|`torch.device`

```python
tensor.device
```

### 1.10.4.dtype

张量的数据类型.|`torch.dtype`

```python
tensor.dtype
```

### 1.10.5.shape

张量的形状.|`torch.Size`

```python
tensor.shape
```

### 1.10.6.to()

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

### 1.10.7.unsqueeze()

增加张量的维度.|`torch.Tensor`

```python
from torch import Tensor

tensor = Tensor(data=[1, 2, 3])
tensor = tensor.unsqueeze(dim=1)  # int|添加新维度的位置.
```

## 1.11.utils

### 1.11.1.data

| 版本 | 描述                 | 注意 |
| ---- | -------------------- | ---- |
| -    | Torch的数据加载工具. | -    |

#### 1.11.1.1.DataLoader()

实例化数据加载器.

```python
import numpy as np
from torch.utils.data import DataLoader

arr = np.asarray([[1], [2], [3], [4]])
dataloader = DataLoader(dataset=arr,  # array-like|要加载的数据集.
                        batch_size=2,  # int(可选)|1|批次大小.
                        shuffle=False)  # bool(可选)|False|是否打乱数据.
```

#### 1.11.1.2.Dataset()

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

# 2.torchvision

| 版本   | 描述                           | 注意 | 适配M1 |
| ------ | ------------------------------ | ---- | ------ |
| 0.12.0 | Torch的图像和视频数据集和模型. | -    | 是     |

## 2.1.datasets

| 版本 | 描述                     | 注意 |
| ---- | ------------------------ | ---- |
| -    | Torchvision的内置数据集. | -    |

### 2.1.1.MNIST()

实例化mnist数据集.

```python
from torchvision.datasets import MNIST

training_data = MNIST(root='./data',  # str|数据集保存的目录.
                      train=True,  # bool(可选)|True|是否是训练集.
                      transform=None,  # function(可选)|None|对数据集进行预处理转换.
                      download=False)  # bool(可选)|False|是否从网络下载数据集.
```

## 2.2.io

###  2.2.1.read_image()

将JPEG或PNG图像读入为张量.|`torch.Tensor`

```python
from torchvision.io import read_image

tensor = read_image(path='img.jpeg')  # str|图像的路径.
```

## 2.3.transforms

| 版本 | 描述                      | 注意 |
| ---- | ------------------------- | ---- |
| -    | Torchvision的数据转换API. | -    |

### 2.3.1.Lambda()

应用自定义的数据转换.|`torch.Tensor`

```python
from torch import Tensor
from torchvision.transforms import 

tensor = Tensor(data=[1, 2, 3, 4])
tensor = Lambda(lambd=lambda x: x * 10)(tensor)  # lambda or function|自定义的数据转换函数.
```

### 2.3.2.ToPILImage()

将张量或numpy.ndarray转换为PIL Image.|`PIL.Image.Image`

```python
from torch import Tensor
from torchvision.transforms import ToPILImage

tensor = Tensor(data=[[1, 2], [3, 4]])
image = ToPILImage()(pic=tensor)  # Tensor or numpy.ndarray|要转换的张量.
```

### 2.3.3.ToTensor()

将PIL Image或numpy.ndarray转换为张量.|`torch.Tensor`

```python
import numpy as np
from torchvision.transforms import ToTensor

arr = np.asarray([[1, 2, 3]])
tensor = ToTensor()(pic=arr)  # PIL Image or numpy.ndarray|要转换的图像.
```