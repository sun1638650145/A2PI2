# <center>A²PI²-TensorFlow version2.2</center>

* 包含TensorFlow生态的软件包.

# 1.tensorflow

| 版本  | 描述          | 注意                                                         | 适配M1 |
| ----- | ------------- | ------------------------------------------------------------ | ------ |
| 2.5.0 | 深度学习框架. | 1. macOS的安装方式请参考[链接](https://developer.apple.com/metal/tensorflow-plugin/). | 是     |

## 1.1.config

### 1.1.1.experimental

#### 1.1.1.1.set_memory_growth()

设置物理设备可以使用的内存量.

```python
import tensorflow as tf

tf.config.experimental.set_memory_growth(device,  # tensorflow.python.eager.context.PhysicalDevice|TensorFlow可识别的物理设备.
                                         enable=True)  # bool|启用内存增长.
```

### 1.1.2.experimental_connect_to_cluster()

连接到计算集群.

```python
import tensorflow as tf

tf.config.experimental_connect_to_cluster(cluster_spec_or_resolver)  # `ClusterSpec` or `ClusterResolver` describing the cluster|计算集群.
```

### 1.1.3.list_physical_devices()

返回所有可用的物理设备.|list

```python
import tensorflow as tf

devices = tf.config.list_physical_devices(device_type=None)  # str(可选)|None|设备类型.
```

## 1.2.constant()

创建常张量.|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf

tensor = tf.constant(value=2021)  # int, float or list|输入的数据.
```

## 1.3.data

| 版本 | 描述                        | 注意 |
| ---- | --------------------------- | ---- |
| -    | TensorFlow的数据输入流水线. | -    |

### 1.3.1.AUTOTUNE

自动调整常量.

```python
import tensorflow as tf

autotune = tf.data.AUTOTUNE
```

### 1.3.2.Dataset

#### 1.3.2.1.as_numpy_iterator()

返回numpy迭代器,将元素转换为numpy.|tensorflow.python.data.ops.dataset_ops._NumpyIterator

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(10)
iterator = dataset.as_numpy_iterator()
```

#### 1.3.2.2.batch()

为数据集划分批次.|tensorflow.python.data.ops.dataset_ops.BatchDataset

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(6)
dataset = dataset.batch(batch_size=3)  # int|批次大小.
```

#### 1.3.2.3.from_tensor_slices()

从张量切片中创建数据集.|tensorflow.python.data.ops.dataset_ops.TensorSliceDataset

```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices(tensors=([1, 2], [3, 4]))  # array-like|输入的数据.
```

#### 1.3.2.4.map()

对数据应用处理.|tensorflow.python.data.ops.dataset_ops.MapDataset or tensorflow.python.data.ops.dataset_ops.ParallelMapDataset

```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices(tensors=([1., 2.], [3., 4.]))  # array-like|输入的数据.
dataset = dataset.map(map_func=lambda x, y: (x + 0.5, y - 0.5),  # function or lambda|处理函数.
                      num_parallel_calls=tf.data.AUTOTUNE)  # int|None|并行处理的数量.
```

#### 1.3.2.5.padded_batch()

为数据集划分批次(按照规则进行填充).|tensorflow.python.data.ops.dataset_ops.PaddedBatchDataset

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(1, 4, output_type=tf.int32)
dataset = dataset.map(lambda x: tf.fill([x], x))
dataset = dataset.padded_batch(batch_size=2,  # int|批次大小.
                               padded_shapes=4,  # int(可选)|None|填充后的形状.
                               padding_values=-1)  # int(可选)|0|填充的值.
```

#### 1.3.2.6.prefetch()

对数据集进行预加载.|tensorflow.python.data.ops.dataset_ops.PrefetchDataset

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(1, 4, output_type=tf.int32)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # int|预加载缓冲区的大小.
```

#### 1.3.2.7.range()

创建指定范围的数据集.|tensorflow.python.data.ops.dataset_ops.RangeDataset

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(10,
                                output_type=tf.int32)  # tensorflow.python.framework.dtypes.DType|元素的数据类型.
```

#### 1.3.2.8.shuffle()

对数据集进行打乱.|tensorflow.python.data.ops.dataset_ops.ShuffleDataset

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(10, output_type=tf.int32)
dataset = dataset.shuffle(buffer_size=2)  # int|打乱缓冲区的大小.
```

#### 1.3.2.9.skip()

跳过指定个数数据创建新数据集.|tensorflow.python.data.ops.dataset_ops.SkipDataset

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(10)
dataset = dataset.skip(count=3)  # int|跳过的个数.
```

#### 1.3.2.10.take()

取出指定个数数据创建新数据集.|tensorflow.python.data.ops.dataset_ops.TakeDataset

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(10)
dataset = dataset.take(count=3)  # int|取出的个数.
```

### 1.3.3.experimental

#### 1.3.3.1.make_csv_dataset()

读取CSV文件.|tensorflow.python.data.ops.dataset_ops.PrefetchDataset

```python
import tensorflow as tf

dataset = tf.data.experimental.make_csv_dataset(file_pattern='ds.csv',  # str|CSV文件路径.
                                                batch_size=1,  # int|批次大小.
                                                column_names=['C1', 'C2'],   # list of str(可选)|None|列名.
                                                label_name='C2',   # str(可选)|None|标签列名.
                                                num_epochs=1)  # int|None|数据集重复加载的次数， None则是一直重复加载.
```

## 1.4.distribute

| 版本 | 描述                    | 注意 |
| ---- | ----------------------- | ---- |
| -    | TensorFlow的分布式策略. | -    |

### 1.4.1.cluster_resolver

#### 1.4.1.1.TPUClusterResolver()

实例化TPU集群.

```python
import tensorflow as tf

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
```

### 1.4.2.MirroredStrategy()

```python
import tensorflow as tf

# 实例化单机多GPU策略.
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # 模型构建代码.
    ...
```

### 1.4.3.TPUStrategy()

实例化TPU策略.

```python
import tensorflow as tf

strategy = tf.distribute.TPUStrategy()
with strategy.scope():
    # 模型构建代码.
    ...
```

## 1.5.einsum()

爱因斯坦求和约定.|tensorflow.python.framework.ops.EagerTensor

```python
import numpy as np
import tensorflow as tf

a = np.asarray([[1], [2]])
b = np.asarray([[1, 2]])
res = tf.einsum('ij,jk->ik',  # str|描述公式.
                a, b)  # array-like|输入的数据.
```

## 1.6.feature_column

### 1.6.1.categorical_column_with_vocabulary_list()

创建分类列.|tensorflow.python.feature_column.feature_column_v2.VocabularyListCategoricalColumn

```python
import tensorflow as tf

column = tf.feature_column.categorical_column_with_vocabulary_list(key='sex',  # str|特征名称.
                                                                   vocabulary_list=['male', 'female'])  # list of str|属性名称.
```

### 1.6.2.indicator_column()

将分类列进行one-hot表示.|tensorflow.python.feature_column.feature_column_v2.IndicatorColumn

```python
import tensorflow as tf

column = tf.feature_column.categorical_column_with_vocabulary_list(key='sex',
                                                                   vocabulary_list=['male', 'female'])
column = tf.feature_column.indicator_column(categorical_column=column)  # CategoricalColumn|分类列.
```

### 1.6.3.numeric_column()

```python
import tensorflow as tf

# 创建数值列.|tensorflow.python.feature_column.feature_column_v2.NumericColumn
column = tf.feature_column.numeric_column(key='age')  # str|特征名称.
```

## 1.7.GradientTape()

实例化梯度带.

```python
import tensorflow as tf

tape = tf.GradientTape()
```

### 1.7.1.gradient()

计算梯度.|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf

x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = tf.multiply(2, x)

grad = tape.gradient(target=y,  # Tensors|`sources`关于`target`的梯度.
                     sources=x)
```

## 1.8.image

| 版本 | 描述                  | 注意 |
| ---- | --------------------- | ---- |
| -    | TensorFlow的图像操作. | -    |

### 1.8.1.convert_image_dtype()

转换图像的数据类型.|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf

arr = [[[1., 2., 3.], [4., 5., 6.]],
       [[7., 8., 9.], [1., 2., 3.]]]
img = tf.image.convert_image_dtype(image=arr,  # array-like|图像.
                                   dtype=tf.uint8)  # tensorflow.python.framework.dtypes.DType|转换后的数据类型.
```

### 1.8.2.decode_image()

转换BMP、GIF、JPEG或者PNG图片为张量.|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf

tensor = tf.io.read_file(f'./img.jpg')
tensor = tf.image.decode_image(contents=tensor,  # A `Tensor` of type `string`|图片的字节流.
                               channels=None,  # int|0|色彩通道数.
                               dtype=tf.uint8)  # tensorflow.python.framework.dtypes.DType|转换后的数据类型.
```

### 1.8.3.decode_jpeg()

转换JPEG图片为张量.|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf

tensor = tf.io.read_file('./img.jpg')
tensor = tf.image.decode_jpeg(contents=tensor,  # A `Tensor` of type `string`|JPEG图片的字节流.
                              channels=0)  # int|0|色彩通道数.
```

### 1.8.4.decode_png()

转换PNG图片为张量.|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf

tensor = tf.io.read_file('./img.png')
tensor = tf.image.decode_png(contents=tensor,  # A `Tensor` of type `string`|PNG图片的字节流.
                             channels=0)  # int|0|色彩通道数.
```

### 1.8.5.resize()

修改图片的尺寸.|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf

tensor = tf.io.read_file('./img.jpg')
tensor = tf.image.decode_jpeg(tensor, 3)
tensor = tf.image.resize(tensor,  # 4-D Tensor or #-D Tensor|输入的图片.
                         size=[200, 200])  # list of int|修改后的尺寸.
```

## 1.9.io

| 版本 | 描述                 | 注意 |
| ---- | -------------------- | ---- |
| -    | TensorFlow的I/O操作. | -    |

### 1.9.1.read_file()

读取文件.|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf

tensor = tf.io.read_file(filename='./img.jpg')  # str|文件路径.
```

## 1.10.keras

| 版本  | 描述                         | 注意 |
| ----- | ---------------------------- | ---- |
| 2.5.0 | TensorFlow的高阶机器学习API. | -    |

### 1.10.1.applications

| 版本 | 描述                      | 注意                               |
| ---- | ------------------------- | ---------------------------------- |
| -    | 提供带有预训练权重的模型. | 1. 默认的缓存路径是~/.keras/models |

#### 1.10.1.1.efficientnet

| 版本 | 描述 | 注意                                      |
| ---- | ---- | ----------------------------------------- |
| -    | -    | 1. `efficientnet`包提供的模型包括`B0-B7`. |

##### 1.10.1.1.1.EfficientNetB0()

EfficientNetB0的预训练模型.|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.efficientnet import EfficientNetB0

model = EfficientNetB0(include_top=True,  # bool|True|是否包含全连接输出层.
                       weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
                       input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

##### 1.10.1.1.2.preprocess_input()

按ImageNet格式预处理数据.|numpy.array or tf.Tensor

```python
from tensorflow.keras.applications.efficientnet import preprocess_input

tensor = preprocess_input(x=tensor)  # numpy.array or tf.Tensor|输入的数据.
```

#### 1.10.1.2.imagenet_utils

##### 1.10.1.2.1.preprocess_input()

按ImageNet格式预处理数据.|numpy.array or tf.Tensor

```python
from tensorflow.keras.applications.imagenet_utils import preprocess_input

tensor = preprocess_input(x=tensor)  # numpy.array or tf.Tensor|输入的数据.
```

#### 1.10.1.3.inception_resnet_v2

##### 1.10.1.3.1.InceptionResNetV2()

InceptionResNetV2的预训练模型.|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

model = InceptionResNetV2(include_top=True,  # bool|True|是否包含全连接输出层.
                          weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
                          input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

#### 1.10.1.4.inception_v3

##### 1.10.1.4.1.InceptionV3()

InceptionV3的预训练模型.|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3

model = InceptionV3(include_top=True,  # bool|True|是否包含全连接输出层.
                    weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
                    input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

#### 1.10.1.5.mobilenet_v2

##### 1.10.1.5.1.MobileNetV2()

MobileNetV2的预训练模型.|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

model = MobileNetV2(include_top=True,  # bool|True|是否包含全连接输出层.
                    weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
                    input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

#### 1.10.1.6.resnet

| 版本 | 描述 | 注意                                       |
| ---- | ---- | ------------------------------------------ |
| -    | -    | 1. `resnet`包提供的模型包括`50, 101, 152`. |

##### 1.10.1.6.1.ResNet50()

ResNet50的预训练模型.|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.resnet import ResNet50

model = ResNet50(include_top=True,  # bool|True|是否包含全连接输出层.
                 weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
                 input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

#### 1.10.1.7.resnet_v2

| 版本 | 描述 | 注意                                                |
| ---- | ---- | --------------------------------------------------- |
| -    | -    | 1. `resnet_v2`包提供的模型包括`50V2, 101V2, 152V2`. |

##### 1.10.1.7.1.ResNet50V2()

ResNet50V2的预训练模型.|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

model = ResNet50V2(include_top=True,  # bool|True|是否包含全连接输出层.
                   weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
                   input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

#### 1.10.1.8.vgg19

##### 1.10.1.8.1.vgg19()

VGG19的预训练模型.|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.vgg19 import vgg19

model = VGG19(include_top=True,  # bool|True|是否包含全连接输出层.
              weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
              input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

#### 1.10.1.9.xception

##### 1.10.1.9.1.Xception()

Xception的预训练模型.|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.xception import Xception

model = Xception(include_top=True,  # bool|True|是否包含全连接输出层.
                 weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
                 input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

### 1.10.2.backend

| 版本 | 描述                | 注意 |
| ---- | ------------------- | ---- |
| -    | tf.keras的后端函数. | -    |

#### 1.10.2.1.cast()

转换张量元素的数据类型.|tensorflow.python.framework.ops.EagerTensor

```python
import numpy as np
from tensorflow.keras import backend as K

arr = np.asarray([1, 2, 3], dtype=np.float32)
tensor = K.cast(x=arr,  # array-like or tf.Tensor|输入的数据.
                dtype='float16')  # str or tensorflow.python.framework.dtypes.DType|转换后的数据类型.
```

#### 1.10.2.2.clear_session()

重置计算图.

```python
from tensorflow.keras import backend as K

K.clear_session()
```

#### 1.10.2.3.clip()

逐元素裁切张量.|tensorflow.python.framework.ops.EagerTensor

```python
import numpy as np
from tensorflow.keras import backend as K

arr = np.arange(1, 10)
tensor = K.clip(x=arr,  # array-like or tf.Tensor|输入的数据.
                min_value=2,  # int or float|最小值.
                max_value=8)  # int or float|最大值.
```

#### 1.10.2.4.ctc_batch_cost()

逐批次计算ctc损失.|tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras import backend as K

loss = K.ctc_batch_cost(y_true,  # tf.Tensor(samples, max_string_length)|真实的标签.
                        y_pred,  # tf.Tensor(samples, time_steps, num_categories)|预测的标签.
                        input_length,  # tf.Tensor(samples, 1)|预测的长度.
                        label_length)  # tf.Tensor(samples, 1)|真实的长度.
```

#### 1.10.2.5.ctc_decode()

解码CTC输出.|tuple of tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras import backend as K

tensor = K.ctc_decode(y_pred,  # tf.Tensor(samples, time_steps, num_categories)|预测的标签.
                      input_length,  # tf.Tensor(samples, )|预测的长度.
                      greedy=True)  # bool|True|是否使用贪心解码.
```

#### 1.10.2.6.expand_dims()

增加张量的维度.|tensorflow.python.framework.ops.EagerTensor

```python
import numpy as np
from tensorflow.keras import backend as K

arr = np.asarray([1, 2, 3])
tensor = K.expand_dims(x=arr,  # tf.Tensor or array-like|输入的数组.
                       axis=0)  # int|添加新维度的位置.
```

#### 1.10.2.7.get_value()

获取变量的值.|numpy.ndarray

```python
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

model = Model()
model.compile(optimizer='adam')
value = K.get_value(x=model.optimizer)  # 输入的变量.
```

#### 1.10.2.8.one_likes()

创建输入张量形状相同形状的全一张量.|tensorflow.python.framework.ops.EagerTensor

```python
import numpy as np
from tensorflow.keras import backend as K

arr = np.asarray([[1, 2], [3, 4]])
tensor = K.ones_like(x=arr)  # tf.Tensor or array-like|输入的张量.
```

#### 1.10.2.9.set_value()

设置数值变量的值.

```python
from tensorflow.keras import backend as K

K.set_value(x,  # 被设置的变量.
            value)  # numpy.ndarray|设置的值.  
```

#### 1.10.2.10.shape()

返回张量的形状.|tensorflow.python.framework.ops.EagerTensor

```python
import numpy as np
from tensorflow.keras import backend as K

arr = np.asarray([1, 2, 3])
tensor_shape = K.shape(x=arr)  # tf.Tensor or array-like|输入的张量.
```

#### 1.10.2.11.sigmoid()

逐元素计算sigmoid的值.|tensorflow.python.framework.ops.EagerTensor

```python
import numpy as np
from tensorflow.keras import backend as K

arr = np.asarray([1., 2., 3.])
tensor = K.sigmoid(x=arr)  # tf.Tensor or array-like|输入的张量.
```

#### 1.10.2.12.zeros_like()

创建输入张量形状相同形状的全零张量.|tensorflow.python.framework.ops.EagerTensor

```python
import numpy as np
from tensorflow.keras import backend as K

arr = np.asarray([[1, 2], [3, 4]])
tensor = K.zeros_like(x=arr)  # tf.Tensor or array-like|输入的张量.
```

### 1.10.3.callbacks

| 版本 | 描述                | 注意 |
| ---- | ------------------- | ---- |
| -    | tf.keras的回调函数. | -    |

#### 1.10.3.1.EarlyStopping()

实例化EarlyStopping, 用以提前停止训练避免过拟合.

```python
from tensorflow.keras.callbacks import EarlyStopping

CALLBACKS = [
    EarlyStopping(monitor='val_loss',  # str|'val_loss'|监控的评估值.
                  min_delta=0,  # float|0|监控信息的最小变化量.
                  patience=0,  # int|0|监控信息的容忍轮数, 到达后将停止训练.
                  verbose=0,  # int|0|日志显示模式.
                  restore_best_weights=False)  # bool|False|恢复最佳的监测值的权重.
]
```

#### 1.10.3.2.LearningRateScheduler()

实例化LearningRateScheduler, 用以定时调整学习率.

```python
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
CALLBACKS = [
    LearningRateScheduler(schedule=scheduler,  # function|学习率调整函数(以当前轮数和学习率为输入, 新学习率位输出).
                          verbose=0)  # int|0|日志显示模式.
]
```

#### 1.10.3.3.ModelCheckpoint()

实例化ModelCheckpoint, 用以保存模型的权重.

```python
from tensorflow.keras.callbacks import ModelCheckpoint

CALLBACKS = [
    ModelCheckpoint(filepath,  # str or PathLike|保存的路径.
                    monitor='val_loss',  # str|'val_loss'|监控的评估值.
                    verbose=0,  # int|0|日志显示模式.
                    period=1)  # int|1|定时保存的频率.
]
```

#### 1.10.3.4.ReduceLROnPlateau()

实例化ReduceLROnPlateau, 用以在评估值不变时降低学习率.

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau

CALLBACKS = [
    ReduceLROnPlateau(monitor='val_loss',  # str|'val_loss'|监控的评估值.
                      factor=0.1,  # float|0.1|学习率衰减因子.
                      patience=10,  # int|0|监控信息的容忍轮数, 到达后将停止训练.
                      verbose=0,  # int|0|日志显示模式.
                      min_delta=1e-4,  # float|1e-4|评估值的最小变化量.
                      min_lr=0)  # float|0|学习率的下界.
]
```

#### 1.10.3.5.TensorBoard()

实例化TensorBoard, 可视化训练信息.

```python
from tensorflow.keras.callbacks import TensorBoard

CALLBACKS = [
    TensorBoard(log_dir='logs',  # str|'logs'|日志保存的路径.
                histogram_freq=0,  # {0, 1}|0|是否计算直方图.
                write_graph=True,  # bool|True|是否绘制计算图.
                update_freq='epoch')  # {'epoch', 'batch'}|'epoch'|更新的频率.
]
```

### 1.10.4.datasets

| 版本 | 描述                  | 注意                                 |
| ---- | --------------------- | ------------------------------------ |
| -    | tf.keras的内置数据集. | 1. 默认的缓存路径是~/.keras/datasets |

#### 1.10.4.1.mnist

##### 1.10.4.1.1.load_data()

加载mnist数据集.|tuple

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_val, y_val) = mnist.load_data()
```

### 1.10.5.layers

| 版本 | 描述                 | 注意 |
| ---- | -------------------- | ---- |
| -    | tf.keras的网络层API. | -    |

#### 1.10.5.1.Activation()

实例化激活函数层.

```python
from tensorflow.keras import layers

layer = layers.Activation(activation='relu')  # str or keras.activations or tf.nn |激活函数.
```

#### 1.10.5.2.Add()

实例化矩阵加法层.

```python
from tensorflow.keras import layers

x1 = layers.Dense(16)
x2 = layers.Dense(16)
layer = layers.Add()([x1, x2])  # list of keras.layers.Layer|形状相同的网络层列表.
```

#### 1.10.5.3.AdditiveAttention()

实例化Bahdanau注意力层.

```python
from tensorflow.keras import layers

layer = layers.AdditiveAttention()
```

#### 1.10.5.4.BatchNormalization()

实例化批标准化层.

```python
from tensorflow.keras import layers

layer = layers.BatchNormalization()
```

#### 1.10.5.5.Bidirectional()

实例化循环层的双向封装器.

```python
from tensorflow.keras import layers

lstm_layer = layers.LSTM(256)
layer = layers.Bidirectional(layer=lstm_layer)  # keras.layers.RNN|循环层.
```

#### 1.10.5.6.Concatenate()

实例化合并层.

```python
from tensorflow.keras import layers

x1 = layers.Dense(16)
x2 = layers.Dense(16)
layer = layers.Concatenate(axis=-1)([x1, x2])  # int|-1|合并所沿的维度, 除此形状相同.
```

#### 1.10.5.7.Conv1D()

实例化1D卷积层.

```python
from tensorflow.keras import layers

layer = layers.Conv1D(filters=32,  # int|卷积核的数量.
                      kernel_size=1,  # int|卷积核的尺寸.
                      strides=1,  # int|1|滑动步长.
                      padding='valid',  # {'valid', 'same' or 'causal'}|'valid'|填充方式.
                      data_format='channels_last',  # {'channels_last' or 'channels_first'}|'channels_last'|数据格式.
                      activation=None,  # str or keras.activations|None|激活函数.
                      use_bias=True,  # bool|True|是否使用偏置.
                      kernel_initializer='glorot_uniform',  # str or keras.initializers|'glorot_uniform'|权重初始化方式.
                      bias_initializer='zeros')  # str or keras.initializers|'zeros'|偏置初始化方式.
```

#### 1.10.5.8.Conv2D()

实例化2D卷积层.

```python
from tensorflow.keras import layers

layer = layers.Conv2D(filters=32,  # int|卷积核的数量.
                      kernel_size=1,  # tuple/list of 2 integers or int|卷积核的尺寸.
                      strides=(1, 1),  # tuple/list of 2 integers or int|(1, 1)|滑动步长.
                      padding='valid',  # {'valid', 'same' or 'causal'}|'valid'|填充方式.
                      data_format='channels_last',  # {'channels_last' or 'channels_first'}|'channels_last'|数据格式.
                      activation=None,  # str or keras.activations|None|激活函数.
                      use_bias=True,  # bool|True|是否使用偏置.
                      kernel_initializer='glorot_uniform',  # str or keras.initializers|'glorot_uniform'|权重初始化方式.
                      bias_initializer='zeros')  # str or keras.initializers|'zeros'|偏置初始化方式.
```

#### 1.10.5.9.Conv2DTranspose()

实例化2D转置卷积层.

```python
from tensorflow.keras import layers

layer = layers.Conv2DTranspose(filters=32,  # int|卷积核的数量.
                               kernel_size=1,  # tuple/list of 2 integers or int|卷积核的尺寸.
                               strides=(1, 1),  # tuple/list of 2 integers or int|(1, 1)|滑动步长.
                               padding='valid',  # {'valid', 'same' or 'causal'}|'valid'|填充方式.
                               use_bias=True)  # bool|True|是否使用偏置.
```

#### 1.10.5.10.Dense()

实例化全连接层.

```python
from tensorflow.keras import layers

layer = layers.Dense(units=32,  # int|神经元的数量.
                     use_bias=True,  # bool|True|是否使用偏置.
                     input_shape)  # tuple of int|模型的第一层将需要指出输入的形状.
```

#### 1.10.5.11.DenseFeatures()

实例化DenseFeatures.

```python
from tensorflow.keras import layers

layer = layers.DenseFeatures(feature_columns)  # list of tensorflow.python.feature_column|特征列.
```

#### 1.10.5.12.Dot()

实例化点积层.

```python
from tensorflow.keras import layers

x1 = layers.Dense(16)
x2 = layers.Dense(16)
layer = layers.Dot(axes=1)(x1, x2)  # int|点积所沿的轴.
```

#### 1.10.5.13.Dropout()

实例化Dropout层.

```python
from tensorflow.keras import layers

layer = layers.Dropout(rate=0.5)  # float|随机丢弃比例.
```

#### 1.10.5.14.Embedding()

实例化嵌入层.

```python
from tensorflow.keras import layers

layer = layers.Embedding(input_dim=128,  # int|输入的维度.
                         output_dim=64,  # int|嵌入矩阵的维度.
                         embeddings_initializer='uniform',  # str or keras.initializers|'uniform'|权重初始化方式.
                         embeddings_regularizer=None)  # keras.regularizers|None|是否使用正则化器.
```

#### 1.10.5.15.experimental

##### 1.10.5.15.1.preprocessing

###### 1.10.5.15.1.1.StringLookup()

实例化词汇到索引的映射工具.

```python
from tensorflow.keras import layers

vocab = ['a', 'b', 'c', 'd', 'a']
char2num = layers.experimental.preprocessing.StringLookup(max_tokens=None,  # int|None|词汇表的最大范围.
                                                          num_oov_indices=1,  # int|1|超出词汇范围使用的索引.
                                                          vocabulary=['a', 'b', 'c'],  # list|None|词汇表.
                                                          invert=False)  # bool|False|翻转操作.
```

###### 1.10.5.15.1.2.get_vocabulary()

获取词汇表.|list

```python
from tensorflow.keras import layers

char2num = layers.experimental.preprocessing.StringLookup(max_tokens=None,
                                                          num_oov_indices=1,
                                                          vocabulary=['a', 'b', 'c'],
                                                          invert=False)
vocab = char2num.get_vocabulary()
```

#### 1.10.5.16.Flatten()

实例化展平层.

```python
from tensorflow.keras import layers

layer = layers.Flatten()
```

#### 1.10.5.17.GlobalAveragePooling1D()

实例化全局1D平均池化层.

```python
from tensorflow.keras import layers

layer = layers.GlobalAveragePooling1D()
```

#### 1.10.5.18.GlobalMaxPooling1D()

实例化全局1D最大池化层.

```python
from tensorflow.keras import layers

layer = layers.GlobalMaxPooling1D()
```

#### 1.10.5.19.GRU()

实例化门控循环网络层.

```python
from tensorflow.keras import layers

layer = layers.GRU(units=256,  # int|神经元的数量.
                   return_sequences=True)  # bool|False|是否返回全部序列.
```

#### 1.10.5.20.Input()

实例化输入层.

```python
from tensorflow.keras import layers

layer = layers.Input(shape=(224, 224, 3),  # tuple|输入张量的形状.
                     name=None,  # str|None|网络层的名称.
                     dtype=None)  # str|None|期望的数据类型.
```

#### 1.10.5.21.Lambda()

将一个函数封装称网络层.

```python
from tensorflow.keras import layers

layer = layers.Lambda(function=lambda x: x**2,  # lambda or function|要封装的函数.
                      output_shape=None,  # tuple|None|期望的输出形状.
                      name=None)  # str|None|网络层的名称.
```

#### 1.10.5.22.Layer()

自定义一个符合tf.keras接口的层.

```python
from tensorflow.keras import layers

class MyLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        # 初始化代码.

    def call(self, inputs, *args, **kwargs):
        # 处理代码.
        return outputs
```

#### 1.10.5.23.LeakyReLU()

实例化带泄漏的ReLU层.

```python
from tensorflow.keras import layers

layer = layers.LeakyReLU(alpha=0.3)  # float|0.3|负斜率系数(泄漏率).
```

#### 1.10.5.24.LSTM()

实例化长短时记忆层.

```python
from tensorflow.keras import layers

layer = layers.LSTM(units=256,  # int|神经元的数量.
                    return_sequences=True,  # bool|False|是否返回全部序列.
                    dropout=0.)  # float|0.|随机丢弃比例.
```

#### 1.10.5.25.MaxPooling1D()

实例化1D最大池化层.

```python
from tensorflow.keras import layers

layer = layers.MaxPooling1D(pool_size=2,  # int|2|池化窗口.
                            strides=None,  # int|None|滑动步长.
                            padding='valid')  # {'valid', 'same'}|'valid'|填充方式.
```

#### 1.10.5.26.MaxPooling2D()

实例化2D最大池化层.

```python
from tensorflow.keras import layers

layer = layers.MaxPooling2D(pool_size=(2, 2),  # int or tuple of 2 int|(2, 2)|池化窗口.
                            strides=None,  # int or tuple of 2 int|None|滑动步长.
                            padding='valid')  # {'valid', 'same'}|'valid'|填充方式.
```

#### 1.10.5.27.Reshape()

实例化变形层.

```python
from tensorflow.keras import layers

layer = layers.Reshape(target_shape=(None, 10))  # tuple of int|目标形状.
```

#### 1.10.5.28.SeparableConv2D()

实例化深度可分离2D卷积层.

```python
from tensorflow.keras import layers

layer = layers.SeparableConv2D(filters=32,  # int|卷积核的数量.
                               kernel_size=1,  # tuple/list of 2 integers or int|卷积核的尺寸.
                               strides=(1, 1),  # tuple/list of 2 integers or int|(1, 1)|滑动步长.
                               padding='valid')  # {'valid', 'same' or 'causal'}|'valid'|填充方式.
```

#### 1.10.5.29.SimpleRNN()

实例化循环网络层.

```python
from tensorflow.keras import layers

layer = layers.SimpleRNN(units=256,  # int|神经元的数量.
                         dropout=0.,  # float|0.|随机丢弃比例.
                         return_sequences=True)  # bool|False|是否返回全部序列.
```

#### 1.10.5.30.TimeDistributed()

实例化时间片封装器.

```python
from tensorflow.keras import layers

layer = layers.Dense(32)
layer = layers.TimeDistributed(layer=layer)  # keras.layers|需要分片的网络层.
```

#### 1.10.5.31.UpSampling2D()

```python
from tensorflow.keras import layers

# 实例化2D上采样层.
layer = layers.UpSampling2D(size=(2, 2))  # int or tuple of 2 int|(2, 2)|上采样因子.
```

#### 1.10.5.32.ZeroPadding2D()

实例化2D零填充层.

```python
from tensorflow.keras import layers

layer = layers.ZeroPadding2D(size=(1, 1))  # int or tuple of 2 int|(1, 1)|填充数.
```

### 1.10.6.losses

| 版本 | 描述                   | 注意 |
| ---- | ---------------------- | ---- |
| -    | tf.keras的损失函数API. | -    |

#### 1.10.6.1.BinaryCrossentropy()

实例化二分类交叉熵损失函数.

```python
from tensorflow.keras.losses import BinaryCrossentropy

loss = BinaryCrossentropy(from_logits=False)  # bool|False|是否将预测值解释为张量.
```

#### 1.10.6.2.CategoricalCrossentropy()

实例化多分类交叉熵损失函数(one-hot编码).

```python
from tensorflow.keras.losses import CategoricalCrossentropy

loss = CategoricalCrossentropy(from_logits=False)  # bool|False|是否将预测值解释为张量.
```

#### 1.10.6.3.MeanAbsoluteError()

实例化平均绝对误差损失函数.

```python
from tensorflow.keras.losses import MeanAbsoluteError

loss = MeanAbsoluteError()
```

#### 1.10.6.4.SparseCategoricalCrossentropy()

实例化多分类交叉熵损失函数(稀释编码).

```python
from tensorflow.keras.losses import SparseCategoricalCrossentropy

loss = SparseCategoricalCrossentropy(from_logits=False)  # bool|False|是否将预测值解释为张量.
```

### 1.10.7.metrics

| 版本 | 描述                   | 注意 |
| ---- | ---------------------- | ---- |
| -    | tf.keras的评估函数API. | -    |

#### 1.10.7.1.MAE()

实例化平均绝对误差损失函数.

```python
from tensorflow.keras.metrics import MAE

metric = MAE()
```

## 1.11.Variable()

创建变量.|tensorflow.python.ops.resource_variable_ops.ResourceVariable

```python
import tensorflow as tf

tensor = tf.Variable(2021)
```

# 2.tensorflow.js

# 3.tensorflow_datasets

# 4.tensorflow_hub

