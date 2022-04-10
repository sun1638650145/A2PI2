# <center>A²PI²-PyTorch version2.2</center>

* 包含PyTorch生态的软件包.

# 1.torch

| 版本   | 描述          | 注意 | 适配M1 |
| ------ | ------------- | ---- | ------ |
| 1.11.0 | 深度学习框架. | -    | 是     |

## 1.1.cuda

| 版本 | 描述               | 注意 |
| ---- | ------------------ | ---- |
| -    | Torch对CUDA的支持. | -    |

### 1.1.1.is_available()

判断系统是否支持CUDA.|`bool`

```python
from torch import cuda

cuda.is_available()
```

## 1.2.from_numpy()

从`numpy.ndarray`中创建`Tensor`.

```python
import numpy as np
from torch import from_numpy

arr = np.asarray([1, 2])
tensor = from_numpy(arr)  # np.ndarray|输入的数据.
```

## 1.3.load()

加载模型.

```python
from torch import load

model = load(f='./model.pt')  # str or a file-like|文件路径.
```

## 1.4.nn

| 版本 | 描述                       | 注意 |
| ---- | -------------------------- | ---- |
| -    | Torch的计算图的基本构建块. | -    |

### 1.4.1.CrossEntropyLoss()

实例化交叉熵损失函数.

```python
from torch.nn import CrossEntropyLoss

loss = CrossEntropyLoss()
```

### 1.4.2.Dropout()

实例化Dropout层.

```python
from torch import nn

layer = nn.Dropout(p=0.5)  # float|0.5|随机丢弃比例.
```

### 1.4.3.Flatten()

实例化展平层.

```python
from torch import nn

layer = nn.Flatten()
```

### 1.4.4.Linear()

实例化全连接层.

```python
from torch import nn

layer = nn.Linear(in_features=32,  # int|输入神经元的数量.
                  out_features=32)  # int|神经元的数量.
```

### 1.4.5.Module()

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

#### 1.4.5.1.eval()

设置模块为评估模式.

```python
model.eval()
```

#### 1.4.5.2.parameters()

返回模块参数迭代器.

```python
model.parameters()
```

#### 1.4.5.3.train()

设置模块为训练模式.

```python
model.train()
```

### 1.4.6.ReLU()

实例化ReLU层.

```python
from torch import nn

layer = nn.ReLU()
```

## 1.5.no_grad()

禁用梯度计算的上下文管理器(可以减少内存消耗).

```python
from torch import no_grad

with no_grad():
    # 代码.
```

## 1.6.optim

| 版本 | 描述              | 注意                                |
| ---- | ----------------- | ----------------------------------- |
| -    | Torch的优化器API. | 1.优化器相同的类方法都写在`Adam`里. |

### 1.6.1.Adam()

实例化`Adam`优化器.

```python
from torch.optim import Adam

optimizer = Adam(params,  # 需要优化的参数.
                 lr=1e-3)  # float(可选)|1e-3|学习率.
```

#### 1.6.1.1.step()

更新梯度.

```python
optimizer.step()
```

#### 1.6.1.2.zero_grad()

将梯度设置为零.

```python
optimizer.zero_grad()
```

## 1.7.save()

保存模型或模型参数.

```python
from torch import nn
from torch import save

model = nn.Module()
save(obj=model,  # 要保存的模型.
     f='./model.pt')  # str or a file-like|文件路径.
```

## 1.8.Tensor()

初始化一个`Tensor`.

```python
from torch import Tensor

tensor = Tensor(data=[1, 2])  # array-like|输入的数据.
```

### 1.8.1.backward()

计算张量的梯度(反向传播).

```python
tensor.backward()
```

### 1.8.2.device

张量的存储设备.|`torch.device`

```python
tensor.device
```

### 1.8.3.dtype

张量的数据类型.|`torch.dtype`

```python
tensor.dtype
```

### 1.8.4.shape

张量的形状.|`torch.Size`

```python
tensor.shape
```

### 1.8.5.to()

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

## 1.9.utils

### 1.9.1.data

| 版本 | 描述                 | 注意 |
| ---- | -------------------- | ---- |
| -    | Torch的数据加载工具. | -    |

#### 1.9.1.1.DataLoader()

实例化数据加载器.

```python
import numpy as np
from torch.utils.data import DataLoader

arr = np.asarray([[1], [2], [3], [4]])
dataloader = DataLoader(dataset=arr,  # array-like|要加载的数据集.
                        batch_size=2)  # int(可选)|1|批次大小.
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

## 2.2.transforms

| 版本 | 描述                      | 注意 |
| ---- | ------------------------- | ---- |
| -    | Torchvision的数据转换API. | -    |

### 2.2.1.ToTensor()

将PIL Image或numpy.ndarray转换为张量.|`torch.Tensor`

```python
import numpy as np
from torchvision.transforms import ToTensor

arr = np.asarray([[1, 2, 3]])
tensor = ToTensor()(pic=arr)  # PIL Image or numpy.ndarray|转换成张量的图像.
```