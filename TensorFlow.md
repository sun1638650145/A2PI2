# <center>A²PI²-TensorFlow version2.2</center>

* 包含TensorFlow生态的软件包.

# 1.tensorflow

| 版本  | 描述          | 注意                                                         | 适配M1 |
| ----- | ------------- | ------------------------------------------------------------ | ------ |
| 2.5.0 | 深度学习框架. | 1. macOS的安装方式请参考[链接](https://developer.apple.com/metal/tensorflow-plugin/). | 是     |

## 1.1.clip_by_value()

逐元素裁切张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
import tensorflow as tf

arr = np.arange(1, 10)
tensor = tf.clip_by_value(t=arr,  # array-like or tf.Tensor|输入的数据.
                          clip_value_min=2,  # int or float|最小值.
                          clip_value_max=8)  # int or float|最大值.
```

## 1.2.concat()

按照指定维度合并多个张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

a = tf.constant([1., 2.])
tensor = tf.concat(values=[a, a],  # list of Tensor|要合并的张量.
                   axis=0)  # int|沿指定维度合并.
```

## 1.3.config

### 1.3.1.experimental

#### 1.3.1.1.set_memory_growth()

设置物理设备可以使用的内存量.

```python
import tensorflow as tf

tf.config.experimental.set_memory_growth(device,  # tensorflow.python.eager.context.PhysicalDevice|TensorFlow可识别的物理设备.
                                         enable=True)  # bool|启用内存增长.
```

### 1.3.2.experimental_connect_to_cluster()

连接到计算集群.

```python
import tensorflow as tf

tf.config.experimental_connect_to_cluster(cluster_spec_or_resolver)  # `ClusterSpec` or `ClusterResolver` describing the cluster|计算集群.
```

### 1.3.3.list_physical_devices()

返回所有可用的物理设备.|`list`

```python
import tensorflow as tf

devices = tf.config.list_physical_devices(device_type=None)  # str(可选)|None|设备类型.
```

## 1.4.constant()

创建常张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

tensor = tf.constant(value=2021)  # int, float or list|输入的数据.
```

## 1.5.constant_initializer()

将张量初始化为指定常量.|`tensorflow.python.ops.init_ops_v2.Constant`

```python
import tensorflow as tf

initializer = tf.constant_initializer(value=0)  # int, float, list, tuple or numpy.ndarray|1|常量值.
```

## 1.6.convert_to_tensor()

将输入的值转换为张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
import tensorflow as tf

arr = np.arange(1, 10)
tensor = tf.convert_to_tensor(value=arr,  # array-like|输入的数据.
                              dtype=None)  # tensorflow.python.framework.dtypes.DType(可选)|None|张量的数据类型.
```

## 1.7.data

| 版本 | 描述                        | 注意 |
| ---- | --------------------------- | ---- |
| -    | TensorFlow的数据输入流水线. | -    |

### 1.7.1.AUTOTUNE

自动调整常量.

```python
import tensorflow as tf

autotune = tf.data.AUTOTUNE
```

### 1.7.2.Dataset

#### 1.7.2.1.apply()

对数据集整体应用处理.|`tensorflow.python.data.experimental.ops.error_ops._IgnoreErrorsDataset`

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(10)
dataset = dataset.apply(transformation_func=tf.data.experimental.ignore_errors())  # function or lambda|处理函数.
```

#### 1.7.2.2.as_numpy_iterator()

返回`numpy`迭代器,将元素转换为`numpy`.|`tensorflow.python.data.ops.dataset_ops._NumpyIterator`

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(10)
iterator = dataset.as_numpy_iterator()
```

#### 1.7.2.3.batch()

为数据集划分批次.|`tensorflow.python.data.ops.dataset_ops.BatchDataset`

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(6)
dataset = dataset.batch(batch_size=3)  # int|批次大小.
```

#### 1.7.2.4.experimental

##### 1.7.2.4.1.ignore_errors()

忽略创建数据集过程中的一切错误.

```python
import tensorflow as tf

tf.data.experimental.ignore_errors()
```

#### 1.7.2.5.from_tensor_slices()

从张量切片中创建数据集.|`tensorflow.python.data.ops.dataset_ops.TensorSliceDataset`

```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices(tensors=([1, 2], [3, 4]))  # array-like|输入的数据.
```

#### 1.7.2.6.map()

对数据应用处理.|`tensorflow.python.data.ops.dataset_ops.MapDataset` or `tensorflow.python.data.ops.dataset_ops.ParallelMapDataset`

```python
import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices(tensors=([1., 2.], [3., 4.]))  # array-like|输入的数据.
dataset = dataset.map(map_func=lambda x, y: (x + 0.5, y - 0.5),  # function or lambda|处理函数.
                      num_parallel_calls=tf.data.AUTOTUNE)  # int|None|并行处理的数量.
```

#### 1.7.2.7.padded_batch()

为数据集划分批次(按照规则进行填充).|`tensorflow.python.data.ops.dataset_ops.PaddedBatchDataset`

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(1, 4, output_type=tf.int32)
dataset = dataset.map(lambda x: tf.fill([x], x))
dataset = dataset.padded_batch(batch_size=2,  # int|批次大小.
                               padded_shapes=4,  # int(可选)|None|填充后的形状.
                               padding_values=-1)  # int(可选)|0|填充的值.
```

#### 1.7.2.8.prefetch()

对数据集进行预加载.|`tensorflow.python.data.ops.dataset_ops.PrefetchDataset`

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(1, 4, output_type=tf.int32)
dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)  # int|预加载缓冲区的大小.
```

#### 1.7.2.9.range()

创建指定范围的数据集.|`tensorflow.python.data.ops.dataset_ops.RangeDataset`

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(10,
                                output_type=tf.int32)  # tensorflow.python.framework.dtypes.DType|元素的数据类型.
```

#### 1.7.2.10.shuffle()

对数据集进行打乱.|`tensorflow.python.data.ops.dataset_ops.ShuffleDataset`

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(10, output_type=tf.int32)
dataset = dataset.shuffle(buffer_size=2)  # int|打乱缓冲区的大小.
```

#### 1.7.2.11.skip()

跳过指定个数数据创建新数据集.|`tensorflow.python.data.ops.dataset_ops.SkipDataset`

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(10)
dataset = dataset.skip(count=3)  # int|跳过的个数.
```

#### 1.7.2.12.take()

取出指定个数数据创建新数据集.|`tensorflow.python.data.ops.dataset_ops.TakeDataset`

```python
import tensorflow as tf

dataset = tf.data.Dataset.range(10)
dataset = dataset.take(count=3)  # int|取出的个数.
```

### 1.7.3.experimental

#### 1.7.3.1.make_csv_dataset()

读取CSV文件.|`tensorflow.python.data.ops.dataset_ops.PrefetchDataset`

```python
import tensorflow as tf

dataset = tf.data.experimental.make_csv_dataset(file_pattern='ds.csv',  # str|CSV文件路径.
                                                batch_size=1,  # int|批次大小.
                                                column_names=['C1', 'C2'],   # list of str(可选)|None|列名.
                                                label_name='C2',   # str(可选)|None|标签列名.
                                                num_epochs=1)  # int|None|数据集重复加载的次数， None则是一直重复加载.
```

## 1.8.distribute

| 版本 | 描述                    | 注意 |
| ---- | ----------------------- | ---- |
| -    | TensorFlow的分布式策略. | -    |

### 1.8.1.cluster_resolver

#### 1.8.1.1.TPUClusterResolver()

实例化TPU集群.

```python
import tensorflow as tf

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
```

### 1.8.2.MirroredStrategy()

```python
import tensorflow as tf

# 实例化单机多GPU策略.
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # 模型构建代码.
    ...
```

### 1.8.3.TPUStrategy()

实例化TPU策略.

```python
import tensorflow as tf

strategy = tf.distribute.TPUStrategy()
with strategy.scope():
    # 模型构建代码.
    ...
```

## 1.9.einsum()

爱因斯坦求和约定.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
import tensorflow as tf

a = np.asarray([[1], [2]])
b = np.asarray([[1, 2]])
res = tf.einsum('ij,jk->ik',  # str|描述公式.
                a, b)  # array-like|输入的数据.
```

## 1.10.feature_column

### 1.10.1.categorical_column_with_vocabulary_list()

创建分类列.|`tensorflow.python.feature_column.feature_column_v2.VocabularyListCategoricalColumn`

```python
import tensorflow as tf

column = tf.feature_column.categorical_column_with_vocabulary_list(key='sex',  # str|特征名称.
                                                                   vocabulary_list=['male', 'female'])  # list of str|属性名称.
```

### 1.10.2.indicator_column()

将分类列进行one-hot表示.|`tensorflow.python.feature_column.feature_column_v2.IndicatorColumn`

```python
import tensorflow as tf

column = tf.feature_column.categorical_column_with_vocabulary_list(key='sex',
                                                                   vocabulary_list=['male', 'female'])
column = tf.feature_column.indicator_column(categorical_column=column)  # CategoricalColumn|分类列.
```

### 1.10.3.numeric_column()

```python
import tensorflow as tf

# 创建数值列.|tensorflow.python.feature_column.feature_column_v2.NumericColumn
column = tf.feature_column.numeric_column(key='age')  # str|特征名称.
```

## 1.11.GradientTape()

实例化梯度带.

```python
import tensorflow as tf

tape = tf.GradientTape()
```

### 1.11.1.gradient()

计算梯度.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = tf.multiply(2, x)

grad = tape.gradient(target=y,  # Tensors|`sources`关于`target`的梯度.
                     sources=x)
```

## 1.12.image

| 版本 | 描述                  | 注意 |
| ---- | --------------------- | ---- |
| -    | TensorFlow的图像操作. | -    |

### 1.12.1.convert_image_dtype()

转换图像的数据类型.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

arr = [[[1., 2., 3.], [4., 5., 6.]],
       [[7., 8., 9.], [1., 2., 3.]]]
img = tf.image.convert_image_dtype(image=arr,  # array-like|图像.
                                   dtype=tf.uint8)  # tensorflow.python.framework.dtypes.DType|转换后的数据类型.
```

### 1.12.2.decode_image()

转换BMP、GIF、JPEG或者PNG图片为张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

tensor = tf.io.read_file('./img.jpg')
tensor = tf.image.decode_image(contents=tensor,  # A `Tensor` of type `string`|图片的字节流.
                               channels=None,  # int|0|色彩通道数.
                               dtype=tf.uint8)  # tensorflow.python.framework.dtypes.DType|转换后的数据类型.
```

### 1.12.3.decode_jpeg()

转换JPEG图片为张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

tensor = tf.io.read_file('./img.jpg')
tensor = tf.image.decode_jpeg(contents=tensor,  # A `Tensor` of type `string`|JPEG图片的字节流.
                              channels=0)  # int|0|色彩通道数.
```

### 1.12.4.decode_png()

转换PNG图片为张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

tensor = tf.io.read_file('./img.png')
tensor = tf.image.decode_png(contents=tensor,  # A `Tensor` of type `string`|PNG图片的字节流.
                             channels=0)  # int|0|色彩通道数.
```

### 1.12.5.flip_left_right()

从左到右水平翻转图像.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

tensor = tf.io.read_file('./img.png')
tensor = tf.image.decode_image(contents=tensor, dtype=tf.uint8)
tensor = tf.image.flip_left_right(image=tensor)  # 4-D Tensor or 3-D Tensor|输入的图片.
```

### 1.12.6.pad_to_bounding_box()

使用零填充图像到指定的和尺寸.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

tensor = tf.io.read_file('./img.png')
tensor = tf.image.decode_image(contents=tensor, dtype=tf.uint8)
tensor = tf.image.pad_to_bounding_box(image=tensor,  # 4-D Tensor or 3-D Tensor|输入的图片.
                                      offset_height=0,  # int|高度上的偏移量.
                                      offset_width=0,  # int|宽度上的偏移量.
                                      target_height=1000,  # int|目标尺寸的高度.
                                      target_width=1000)  # int|目标尺寸的宽度.
```

### 1.12.7.resize()

修改图片的尺寸.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

tensor = tf.io.read_file('./img.jpg')
tensor = tf.image.decode_jpeg(tensor, 3)
tensor = tf.image.resize(images=tensor,  # 4-D Tensor or 3-D Tensor|输入的图片.
                         size=[200, 200])  # list of int|修改后的尺寸.
```

## 1.13.io

| 版本 | 描述                 | 注意 |
| ---- | -------------------- | ---- |
| -    | TensorFlow的I/O操作. | -    |

### 1.13.1.read_file()

读取文件.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

tensor = tf.io.read_file(filename='./img.jpg')  # str|文件路径.
```

## 1.14.keras

| 版本  | 描述                         | 注意 |
| ----- | ---------------------------- | ---- |
| 2.5.0 | TensorFlow的高阶机器学习API. | -    |

### 1.14.1.activations

| 版本 | 描述                   | 注意 |
| ---- | ---------------------- | ---- |
| -    | tf.keras的激活函数API. | -    |

#### 1.14.1.1.relu()

应用relu函数在输入的张量上.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf
from tensorflow.keras.activations import relu

tensor = tf.constant([-2., -1., 0.0, 1., 2.])
tensor = relu(x=tensor)  # tf.Tensor|输入的张量.
```

### 1.14.2.applications

| 版本 | 描述                      | 注意                               |
| ---- | ------------------------- | ---------------------------------- |
| -    | 提供带有预训练权重的模型. | 1. 默认的缓存路径是~/.keras/models |

#### 1.14.2.1.efficientnet

| 版本 | 描述 | 注意                                      |
| ---- | ---- | ----------------------------------------- |
| -    | -    | 1. `efficientnet`包提供的模型包括`B0-B7`. |

##### 1.14.2.1.1.EfficientNetB0()

EfficientNetB0的预训练模型.|`tensorflow.python.keras.engine.functional.Functional`

```python
from tensorflow.keras.applications.efficientnet import EfficientNetB0

model = EfficientNetB0(include_top=True,  # bool|True|是否包含全连接输出层.
                       weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
                       input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

##### 1.14.2.1.2.preprocess_input()

按ImageNet格式预处理数据.|`numpy.array` or `tf.Tensor`

```python
from tensorflow.keras.applications.efficientnet import preprocess_input

tensor = preprocess_input(x=tensor)  # numpy.array or tf.Tensor|输入的数据.
```

#### 1.14.2.2.imagenet_utils

##### 1.14.2.2.1.preprocess_input()

按ImageNet格式预处理数据.|`numpy.array` or `tf.Tensor`

```python
from tensorflow.keras.applications.imagenet_utils import preprocess_input

tensor = preprocess_input(x=tensor)  # numpy.array or tf.Tensor|输入的数据.
```

#### 1.14.2.3.inception_resnet_v2

##### 1.14.2.3.1.InceptionResNetV2()

InceptionResNetV2的预训练模型.|`tensorflow.python.keras.engine.functional.Functional`

```python
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

model = InceptionResNetV2(include_top=True,  # bool|True|是否包含全连接输出层.
                          weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
                          input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

#### 1.14.2.4.inception_v3

##### 1.14.2.4.1.InceptionV3()

InceptionV3的预训练模型.|`tensorflow.python.keras.engine.functional.Functional`

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3

model = InceptionV3(include_top=True,  # bool|True|是否包含全连接输出层.
                    weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
                    input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

#### 1.14.2.5.mobilenet_v2

##### 1.14.2.5.1.MobileNetV2()

MobileNetV2的预训练模型.|`tensorflow.python.keras.engine.functional.Functional`

```python
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

model = MobileNetV2(include_top=True,  # bool|True|是否包含全连接输出层.
                    weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
                    input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

#### 1.14.2.6.resnet

| 版本 | 描述 | 注意                                       |
| ---- | ---- | ------------------------------------------ |
| -    | -    | 1. `resnet`包提供的模型包括`50, 101, 152`. |

##### 1.14.1.6.1.ResNet50()

ResNet50的预训练模型.|`tensorflow.python.keras.engine.functional.Functional`

```python
from tensorflow.keras.applications.resnet import ResNet50

model = ResNet50(include_top=True,  # bool|True|是否包含全连接输出层.
                 weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
                 input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

#### 1.14.2.7.resnet_v2

| 版本 | 描述 | 注意                                                |
| ---- | ---- | --------------------------------------------------- |
| -    | -    | 1. `resnet_v2`包提供的模型包括`50V2, 101V2, 152V2`. |

##### 1.14.2.7.1.ResNet50V2()

ResNet50V2的预训练模型.|`tensorflow.python.keras.engine.functional.Functional`

```python
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

model = ResNet50V2(include_top=True,  # bool|True|是否包含全连接输出层.
                   weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
                   input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

#### 1.14.2.8.vgg19

##### 1.14.2.8.1.vgg19()

VGG19的预训练模型.|`tensorflow.python.keras.engine.functional.Functional`

```python
from tensorflow.keras.applications.vgg19 import vgg19

model = VGG19(include_top=True,  # bool|True|是否包含全连接输出层.
              weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
              input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

#### 1.14.2.9.xception

##### 1.14.2.9.1.Xception()

Xception的预训练模型.|`tensorflow.python.keras.engine.functional.Functional`

```python
from tensorflow.keras.applications.xception import Xception

model = Xception(include_top=True,  # bool|True|是否包含全连接输出层.
                 weights='imagenet',  # None, str or 'imagenet'|'imagenet'|初始化权重的加载方式.
                 input_tensor=None)  # tf.Tensor(可选)|None|输入层张量.
```

### 1.14.3.backend

| 版本 | 描述                | 注意 |
| ---- | ------------------- | ---- |
| -    | tf.keras的后端函数. | -    |

#### 1.14.3.1.abs()

逐元素计算张量的绝对值.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
import tensorflow.keras.backend as K

arr = np.asarray([-1., 2., 3.])
tensor = K.abs(x=arr)   # array-like or tf.Tensor|输入的数据.
```

#### 1.14.3.2.argmax()

返回指定维度最大值的索引.|`tensorflow.python.framework.ops.EagerTensor`

```python
from tensorflow.keras import backend as K

arr = [1, 2, 3, 1]
tensor = K.argmax(x=arr,  # array_like|输入的数据.
                  axis=-1)  # int|-1|筛选所沿的维度.
```

#### 1.14.3.3.cast()

转换张量元素的数据类型.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
from tensorflow.keras import backend as K

arr = np.asarray([1, 2, 3], dtype=np.float32)
tensor = K.cast(x=arr,  # array-like or tf.Tensor|输入的数据.
                dtype='float16')  # str or tensorflow.python.framework.dtypes.DType|转换后的数据类型.
```

#### 1.14.3.4.clear_session()

重置计算图.

```python
from tensorflow.keras import backend as K

K.clear_session()
```

#### 1.14.3.5.clip()

逐元素裁切张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
from tensorflow.keras import backend as K

arr = np.arange(1, 10)
tensor = K.clip(x=arr,  # array-like or tf.Tensor|输入的数据.
                min_value=2,  # int or float|最小值.
                max_value=8)  # int or float|最大值.
```

#### 1.14.3.6.ctc_batch_cost()

逐批次计算ctc损失.|`tensorflow.python.framework.ops.EagerTensor`

```python
from tensorflow.keras import backend as K

loss = K.ctc_batch_cost(y_true,  # tf.Tensor(samples, max_string_length)|真实的标签.
                        y_pred,  # tf.Tensor(samples, time_steps, num_categories)|预测的标签.
                        input_length,  # tf.Tensor(samples, 1)|预测的长度.
                        label_length)  # tf.Tensor(samples, 1)|真实的长度.
```

#### 1.14.3.7.ctc_decode()

解码CTC输出.|`tuple of tensorflow.python.framework.ops.EagerTensor`

```python
from tensorflow.keras import backend as K

tensor = K.ctc_decode(y_pred,  # tf.Tensor(samples, time_steps, num_categories)|预测的标签.
                      input_length,  # tf.Tensor(samples, )|预测的长度.
                      greedy=True)  # bool|True|是否使用贪心解码.
```

#### 1.14.3.8.equal()

逐元素比较两个张量是否相等.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow.keras.backend as K

arr = [1, 2]
tensor = K.equal(x=arr,  # tf.Tensor|比较的张量.
                 y=arr)  # tf.Tensor|比较的张量.
```

#### 1.14.3.9.expand_dims()

增加张量的维度.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
from tensorflow.keras import backend as K

arr = np.asarray([1, 2, 3])
tensor = K.expand_dims(x=arr,  # tf.Tensor or array-like|输入的数组.
                       axis=0)  # int|添加新维度的位置.
```

#### 1.14.3.10.gather()

取出张量指定索引处的元素组成新张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
from tensorflow.keras import backend as K

ref = [1, 2, 3, 4]
indices = [0, 3]
tensor = K.gather(reference=ref,  # tf.Tensor|参考张量.
                  indices=indices)  # array-like|索引数组.
```

#### 1.14.3.11.get_value()

获取变量的值.|`numpy.ndarray`

```python
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

model = Model()
model.compile(optimizer='adam')
value = K.get_value(x=model.optimizer)  # 输入的变量.
```

#### 1.14.3.12.greater()

逐元素比较第一个张量是否大于第二个张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow.keras.backend as K

arr1 = [1, 2]
arr2 = [3, 4]
tensor = K.greater(x=arr1,  # tf.Tensor|比较的张量.
                   y=arr2)  # tf.Tensor|比较的张量.
```

#### 1.14.3.13.less()

逐元素比较第一个张量是否小于第二个张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow.keras.backend as K

arr1 = [1, 2]
arr2 = [3, 4]
tensor = K.less(x=arr1,  # tf.Tensor|比较的张量.
                y=arr2)  # tf.Tensor|比较的张量.
```

#### 1.14.3.14.max()

返回张量中的最大值.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow.keras.backend as K

arr = [1, 2, 3, 2]
tensor = K.max(x=arr)  # tf.Tensor|输入的张量.
```

#### 1.14.3.15.min()

返回张量中的最小值.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow.keras.backend as K

arr = [1, 2, 3, 2]
tensor = K.min(x=arr)  # tf.Tensor|输入的张量.
```

#### 1.14.3.16.one_hot()

对整数张量进行独热编码.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow.keras.backend as K

arr = [0, 1, 1, 2]
tensor = K.one_hot(indices=arr,  # tf.Tensor(batch_size, dim1, dim2, ... dim(n-1))|张量.
                   num_classes=3)  # int|类别总数.
```

#### 1.14.3.17.one_likes()

创建输入张量形状相同形状的全一张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
from tensorflow.keras import backend as K

arr = np.asarray([[1, 2], [3, 4]])
tensor = K.ones_like(x=arr)  # tf.Tensor or array-like|输入的张量.
```

#### 1.14.3.18.pow()

对张量逐元素求幂.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
import tensorflow.keras.backend as K

arr = np.asarray([-1., 2., 3.])
tensor = K.pow(x=arr,   # array-like or tf.Tensor|输入的数据.
               a=2)  # int|幂次.
```

#### 1.14.3.19.set_value()

设置数值变量的值.

```python
from tensorflow.keras import backend as K

K.set_value(x,  # 被设置的变量.
            value)  # numpy.ndarray|设置的值.  
```

#### 1.14.3.20.shape()

返回张量的形状.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
from tensorflow.keras import backend as K

arr = np.asarray([1, 2, 3])
tensor_shape = K.shape(x=arr)  # tf.Tensor or array-like|输入的张量.
```

#### 1.14.3.21.sigmoid()

逐元素计算sigmoid的值.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
from tensorflow.keras import backend as K

arr = np.asarray([1., 2., 3.])
tensor = K.sigmoid(x=arr)  # tf.Tensor or array-like|输入的张量.
```

#### 1.14.3.22.square()

对张量逐元素求平方.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
import tensorflow.keras.backend as K

arr = np.asarray([1., 2., 3.])
tensor = K.square(x=arr)   # array-like or tf.Tensor|输入的数据.
```

#### 1.14.3.23.stack()

将秩R的矩阵堆叠成R+1的矩阵.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
import tensorflow.keras.backend as K

a = np.asarray([1, 2])
b = np.asarray([3, 4])
tensor = K.stack(x=[a, b],  # list of tf.Tensor|张量列表.
                 axis=-1)  # int|0|堆叠时的维度.
```

#### 1.14.3.24.sum()

对张量沿指定轴求和.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow.keras.backend as K

arr = [[0, 1], [2, 2]]
tensor = K.sum(x=arr,  # tf.Tensor or array-like|输入的张量.
               axis=1)  # int|None|沿指定维度合并.
```

#### 1.14.3.25.zeros_like()

创建输入张量形状相同形状的全零张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
from tensorflow.keras import backend as K

arr = np.asarray([[1, 2], [3, 4]])
tensor = K.zeros_like(x=arr)  # tf.Tensor or array-like|输入的张量.
```

### 1.14.4.callbacks

| 版本 | 描述                | 注意 |
| ---- | ------------------- | ---- |
| -    | tf.keras的回调函数. | -    |

#### 1.14.4.1.EarlyStopping()

实例化`EarlyStopping`, 用以提前停止训练避免过拟合.

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

#### 1.14.4.2.LearningRateScheduler()

实例化`LearningRateScheduler`, 用以定时调整学习率.

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

#### 1.14.4.3.ModelCheckpoint()

实例化`ModelCheckpoint`, 用以保存模型的权重.

```python
from tensorflow.keras.callbacks import ModelCheckpoint

CALLBACKS = [
    ModelCheckpoint(filepath,  # str or PathLike|保存的路径.
                    monitor='val_loss',  # str|'val_loss'|监控的评估值.
                    verbose=0,  # int|0|日志显示模式.
                    period=1)  # int|1|定时保存的频率.
]
```

#### 1.14.4.4.ReduceLROnPlateau()

实例化`ReduceLROnPlateau`, 用以在评估值不变时降低学习率.

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

#### 1.14.4.5.TensorBoard()

实例化`TensorBoard`, 可视化训练信息.

```python
from tensorflow.keras.callbacks import TensorBoard

CALLBACKS = [
    TensorBoard(log_dir='logs',  # str|'logs'|日志保存的路径.
                histogram_freq=0,  # {0, 1}|0|是否计算直方图.
                write_graph=True,  # bool|True|是否绘制计算图.
                update_freq='epoch')  # {'epoch', 'batch'}|'epoch'|更新的频率.
]
```

### 1.14.5.datasets

| 版本 | 描述                  | 注意                                 |
| ---- | --------------------- | ------------------------------------ |
| -    | tf.keras的内置数据集. | 1. 默认的缓存路径是~/.keras/datasets |

#### 1.14.5.1.mnist

##### 1.14.5.1.1.load_data()

加载mnist数据集.|`tuple`

```python
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_val, y_val) = mnist.load_data()
```

### 1.14.6.initializers

| 版本 | 描述                   | 注意 |
| ---- | ---------------------- | ---- |
| -    | tf.keras的初始化器API. | -    |

#### 1.14.6.1.RandomNormal()

实例化正态分布初始化器.

```python
from tensorflow.keras.initializers import RandomNormal

initializer = RandomNormal(mean=0.0,  # float|0.0|均值.
                           stddev=0.05)  # float|0.05|标准差.
```

### 1.14.7.layers

| 版本 | 描述                 | 注意 |
| ---- | -------------------- | ---- |
| -    | tf.keras的网络层API. | -    |

#### 1.14.7.1.Activation()

实例化激活函数层.

```python
from tensorflow.keras import layers

layer = layers.Activation(activation='relu')  # str or keras.activations or tf.nn |激活函数.
```

#### 1.14.7.2.Add()

实例化矩阵加法层.

```python
from tensorflow.keras import layers

x1 = layers.Dense(16)
x2 = layers.Dense(16)
layer = layers.Add()([x1, x2])  # list of keras.layers.Layer|形状相同的网络层列表.
```

#### 1.14.7.3.AdditiveAttention()

实例化Bahdanau注意力层.

```python
from tensorflow.keras import layers

layer = layers.AdditiveAttention()
```

#### 1.14.7.4.BatchNormalization()

实例化批标准化层.

```python
from tensorflow.keras import layers

layer = layers.BatchNormalization()
```

#### 1.14.7.5.Bidirectional()

实例化循环层的双向封装器.

```python
from tensorflow.keras import layers

lstm_layer = layers.LSTM(256)
layer = layers.Bidirectional(layer=lstm_layer)  # keras.layers.RNN|循环层.
```

#### 1.14.7.6.Concatenate()

实例化合并层.

```python
from tensorflow.keras import layers

x1 = layers.Dense(16)
x2 = layers.Dense(16)
layer = layers.Concatenate(axis=-1)([x1, x2])  # int|-1|合并所沿的维度, 除此形状相同.
```

#### 1.14.7.7.Conv1D()

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

#### 1.14.7.8.Conv2D()

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

#### 1.14.7.9.Conv2DTranspose()

实例化2D转置卷积层.

```python
from tensorflow.keras import layers

layer = layers.Conv2DTranspose(filters=32,  # int|卷积核的数量.
                               kernel_size=1,  # tuple/list of 2 integers or int|卷积核的尺寸.
                               strides=(1, 1),  # tuple/list of 2 integers or int|(1, 1)|滑动步长.
                               padding='valid',  # {'valid', 'same' or 'causal'}|'valid'|填充方式.
                               use_bias=True)  # bool|True|是否使用偏置.
```

#### 1.14.7.10.Dense()

实例化全连接层.

```python
from tensorflow.keras import layers

layer = layers.Dense(units=32,  # int|神经元的数量.
                     use_bias=True,  # bool|True|是否使用偏置.
                     input_shape)  # tuple of int|模型的第一层将需要指出输入的形状.
```

#### 1.14.7.11.DenseFeatures()

实例化DenseFeatures.

```python
from tensorflow.keras import layers

layer = layers.DenseFeatures(feature_columns)  # list of tensorflow.python.feature_column|特征列.
```

#### 1.14.7.12.Dot()

实例化点积层.

```python
from tensorflow.keras import layers

x1 = layers.Dense(16)
x2 = layers.Dense(16)
layer = layers.Dot(axes=1)(x1, x2)  # int|点积所沿的轴.
```

#### 1.14.7.13.Dropout()

实例化Dropout层.

```python
from tensorflow.keras import layers

layer = layers.Dropout(rate=0.5)  # float|随机丢弃比例.
```

#### 1.14.7.14.Embedding()

实例化嵌入层.

```python
from tensorflow.keras import layers

layer = layers.Embedding(input_dim=128,  # int|输入的维度.
                         output_dim=64,  # int|嵌入矩阵的维度.
                         embeddings_initializer='uniform',  # str or keras.initializers|'uniform'|权重初始化方式.
                         embeddings_regularizer=None)  # keras.regularizers|None|是否使用正则化器.
```

#### 1.14.7.15.experimental

##### 1.14.7.15.1.preprocessing

###### 1.14.7.15.1.1.StringLookup()

实例化词汇到索引的映射工具.

```python
from tensorflow.keras import layers

vocab = ['a', 'b', 'c', 'd', 'a']
char2num = layers.experimental.preprocessing.StringLookup(max_tokens=None,  # int|None|词汇表的最大范围.
                                                          num_oov_indices=1,  # int|1|超出词汇范围使用的索引.
                                                          vocabulary=['a', 'b', 'c'],  # list|None|词汇表.
                                                          invert=False)  # bool|False|翻转操作.
```

###### 1.14.7.15.1.2.get_vocabulary()

获取词汇表.|`list`

```python
from tensorflow.keras import layers

char2num = layers.experimental.preprocessing.StringLookup(max_tokens=None,
                                                          num_oov_indices=1,
                                                          vocabulary=['a', 'b', 'c'],
                                                          invert=False)
vocab = char2num.get_vocabulary()
```

#### 1.14.7.16.Flatten()

实例化展平层.

```python
from tensorflow.keras import layers

layer = layers.Flatten()
```

#### 1.14.7.17.GlobalAveragePooling1D()

实例化全局1D平均池化层.

```python
from tensorflow.keras import layers

layer = layers.GlobalAveragePooling1D()
```

#### 1.14.7.18.GlobalMaxPooling1D()

实例化全局1D最大池化层.

```python
from tensorflow.keras import layers

layer = layers.GlobalMaxPooling1D()
```

#### 1.14.7.19.GRU()

实例化门控循环网络层.

```python
from tensorflow.keras import layers

layer = layers.GRU(units=256,  # int|神经元的数量.
                   return_sequences=True)  # bool|False|是否返回全部序列.
```

#### 1.14.7.20.Input()

实例化输入层.

```python
from tensorflow.keras import layers

layer = layers.Input(shape=(224, 224, 3),  # tuple|输入张量的形状.
                     name=None,  # str|None|网络层的名称.
                     dtype=None)  # str|None|期望的数据类型.
```

#### 1.14.7.21.InputLayer()

实例化输入层.

```python
from tensorflow.keras import layers

layer = layers.InputLayer(input_shape=(224, 224, 3))  # tuple|输入张量的形状.
```

#### 1.14.7.22.Lambda()

将一个函数封装称网络层.

```python
from tensorflow.keras import layers

layer = layers.Lambda(function=lambda x: x**2,  # lambda or function|要封装的函数.
                      output_shape=None,  # tuple|None|期望的输出形状.
                      name=None)  # str|None|网络层的名称.
```

#### 1.14.7.23.Layer()

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

##### 1.14.7.23.1.output

网络层输出张量.|`tensorflow.python.keras.engine.keras_tensor.KerasTensor`

```python
tensor = layer.output
```

#### 1.14.7.24.LeakyReLU()

实例化带泄漏的ReLU层.

```python
from tensorflow.keras import layers

layer = layers.LeakyReLU(alpha=0.3)  # float|0.3|负斜率系数(泄漏率).
```

#### 1.14.7.25.LSTM()

实例化长短时记忆层.

```python
from tensorflow.keras import layers

layer = layers.LSTM(units=256,  # int|神经元的数量.
                    return_sequences=True,  # bool|False|是否返回全部序列.
                    dropout=0.)  # float|0.|随机丢弃比例.
```

#### 1.14.7.26.MaxPooling1D()

实例化1D最大池化层.

```python
from tensorflow.keras import layers

layer = layers.MaxPooling1D(pool_size=2,  # int|2|池化窗口.
                            strides=None,  # int|None|滑动步长.
                            padding='valid')  # {'valid', 'same'}|'valid'|填充方式.
```

#### 1.14.7.27.MaxPooling2D()

实例化2D最大池化层.

```python
from tensorflow.keras import layers

layer = layers.MaxPooling2D(pool_size=(2, 2),  # int or tuple of 2 int|(2, 2)|池化窗口.
                            strides=None,  # int or tuple of 2 int|None|滑动步长.
                            padding='valid')  # {'valid', 'same'}|'valid'|填充方式.
```

#### 1.14.7.28.ReLU()

实例化ReLU层.

```python
from tensorflow.keras import layers

layer = layers.ReLU()
```

#### 1.14.7.29.Reshape()

实例化变形层.

```python
from tensorflow.keras import layers

layer = layers.Reshape(target_shape=(None, 10))  # tuple of int|目标形状.
```

#### 1.14.7.30.SeparableConv2D()

实例化深度可分离2D卷积层.

```python
from tensorflow.keras import layers

layer = layers.SeparableConv2D(filters=32,  # int|卷积核的数量.
                               kernel_size=1,  # tuple/list of 2 integers or int|卷积核的尺寸.
                               strides=(1, 1),  # tuple/list of 2 integers or int|(1, 1)|滑动步长.
                               padding='valid')  # {'valid', 'same' or 'causal'}|'valid'|填充方式.
```

#### 1.14.7.31.SimpleRNN()

实例化循环网络层.

```python
from tensorflow.keras import layers

layer = layers.SimpleRNN(units=256,  # int|神经元的数量.
                         dropout=0.,  # float|0.|随机丢弃比例.
                         return_sequences=True)  # bool|False|是否返回全部序列.
```

#### 1.14.7.32.TimeDistributed()

实例化时间片封装器.

```python
from tensorflow.keras import layers

layer = layers.Dense(32)
layer = layers.TimeDistributed(layer=layer)  # keras.layers|需要分片的网络层.
```

#### 1.14.7.33.UpSampling2D()

```python
from tensorflow.keras import layers

# 实例化2D上采样层.
layer = layers.UpSampling2D(size=(2, 2))  # int or tuple of 2 int|(2, 2)|上采样因子.
```

#### 1.14.7.34.ZeroPadding2D()

实例化2D零填充层.

```python
from tensorflow.keras import layers

layer = layers.ZeroPadding2D(size=(1, 1))  # int or tuple of 2 int|(1, 1)|填充数.
```

### 1.14.8.losses

| 版本 | 描述                   | 注意 |
| ---- | ---------------------- | ---- |
| -    | tf.keras的损失函数API. | -    |

#### 1.14.8.1.BinaryCrossentropy()

实例化二分类交叉熵损失函数.

```python
from tensorflow.keras.losses import BinaryCrossentropy

loss = BinaryCrossentropy(from_logits=False)  # bool|False|是否将预测值解释为张量.
```

#### 1.14.8.2.CategoricalCrossentropy()

实例化多分类交叉熵损失函数(one-hot编码).

```python
from tensorflow.keras.losses import CategoricalCrossentropy

loss = CategoricalCrossentropy(from_logits=False)  # bool|False|是否将预测值解释为张量.
```

#### 1.14.8.3.Loss()

自定义一个符合tf.keras接口的损失函数.

```python
from tensorflow.keras.losses import Loss

class MyLoss(Loss):
    def __init__(self,
                 reduction='auto',  # {'auto'|'none'|'sum'|'sum_over_batch_size'}(可选)|'auto'|损失函数减少类型.
                 **kwargs):
        super(MyLoss, self).__init__(reduction, **kwargs)
        # 初始化代码.

    def call(self,
             y_true,  # array-like[batch_size, d0, .. dN]|真实值.
             y_pred):  # array-like[batch_size, d0, .. dN]|预测值.
        # 处理代码.
        return loss
```

#### 1.14.8.4.MeanAbsoluteError()

实例化平均绝对误差损失函数.

```python
from tensorflow.keras.losses import MeanAbsoluteError

loss = MeanAbsoluteError()
```

#### 1.14.8.5.SparseCategoricalCrossentropy()

实例化多分类交叉熵损失函数(稀释编码).

```python
from tensorflow.keras.losses import SparseCategoricalCrossentropy

loss = SparseCategoricalCrossentropy(from_logits=False)  # bool|False|是否将预测值解释为张量.
```

### 1.14.9.metrics

| 版本 | 描述                   | 注意 |
| ---- | ---------------------- | ---- |
| -    | tf.keras的评估函数API. | -    |

#### 1.14.9.1.MAE()

实例化平均绝对误差损失函数.

```python
from tensorflow.keras.metrics import MAE

metric = MAE()
```

### 1.14.10.models

| 版本 | 描述         | 注意                                                         |
| ---- | ------------ | ------------------------------------------------------------ |
| -    | 模型构建API. | 1.tf.keras支持两种模型`Model(Function API)`和`Sequential`, 相同的类方法都写在`Model`里. |

#### 1.14.10.1.load_model()

加载模型.|`tensorflow.python.keras.engine.training.Model` or `tensorflow.python.keras.engine.sequential.Sequential`

```python
from tensorflow.keras.models import load_model

model = load_model(filepath='model.h5')  # str or pathlib.Path|文件路径.
```

#### 1.14.10.2.Model()

实例化`Model`.

```python
from tensorflow.keras.models import Model

model = Model(inputs,  # keras.layers.Input|输入层.
              outputs)  # keras.layers|输出层.
```

##### 1.14.10.2.1.build()

构建模型.

```python
model.build(input_shape)  # single tuple, TensorShape, or list/dict of shapes|输入层的形状.
```

##### 1.14.10.2.2.compile()

编译模型, 配置模型训练参数.

```python
model.compile(optimizer='rmsprop',  # str or keras.optimizers|'rmsprop'|优化器.
              loss=None,  # str or keras.losses|None|损失函数.
              metrics=None)  # str or keras.metrics|None|评估函数.
```

##### 1.14.10.2.3.evaluate()

在测试模型下评估损失和准确率.

```python
model.evaluate(x=None,  # Numpy array, TensorFlow tensor, `tf.data` dataset, generator or `keras.utils.Sequence`|None|特征数据.
               y=None,  # Numpy array, TensorFlow tensor(如果是dataset或generator或Sequence则y都是None)|None|标签.
               batch_size=None,  # int|None|批次大小.
               verbose=1)  # int|0|日志显示模式.
```

##### 1.14.10.2.4.fit()

训练模型.|`keras.callbacks.History`

```python
model.fit(x=None,  # Numpy array, TensorFlow tensor, `tf.data` dataset, generator or `keras.utils.Sequence`|None|特征数据.
          y=None,  # Numpy array, TensorFlow tensor(如果是dataset或generator或Sequence则y都是None)|None|标签.
          batch_size=None,  # int|None|批次大小.
          epochs=1,  # int|None|轮数.
          verbose='auto',  # str or int|'auto'|日志显示模式.
          callbacks=None,  # list of tf.keras.callbacks|None|回调函数函数列表.
          validation_split=0.,  # float|0.|划分验证集的比例.
          validation_data=None,  # tuple (x_val, y_val), `tf.data` dataset or generator or `keras.utils.Sequence`|None|验证集.
          shuffle=True,  # bool|True|是否打乱数据.
          class_weight=None,  # dict(可选)|None|类别权重字典(只在训练时有效).
          initial_epoch=0,  # int|0|初始化训练的轮数.
          steps_per_epoch=None,  # int|None|每轮的步数(样本数/批次大小).
          workers=1,  # int|1|使用的线程数(仅适用`keras.utils.Sequence`).
          use_multiprocessing=False)  # bool|False|是否使用多线程(仅适用`keras.utils.Sequence`).
```

##### 1.14.10.2.5.inputs

模型的输入层对象|`keras.Input` or `list of keras.Input`

```python
inputs = model.inputs
```

##### 1.14.10.2.6.get_layer()

根据网络层名称检索网络层.|`tensorflow.python.keras.layers`

```python
layer = model.get_layer(name)  # str|网络层名称.
```

##### 1.14.10.2.7.load_weights()

加载模型的权重.

```python
model.load_weights(filepath)  # str or pathlib.Path|文件路径.
```

##### 1.14.10.2.8.predict()

使用模型进行预测.|`numpy.ndarray`

```python
y_pred = model.predict(x,  # Numpy array, TensorFlow tensor, `tf.data` dataset, generator or `keras.utils.Sequence`|None|特征数据.
                       batch_size=None,  # int|None|批次大小.
                       verbose=0)  # int|0|日志显示模式.
```

##### 1.14.10.2.9.output_shape

模型输出层的形状.|`tuple`

```python
shape = model.output_shape
```

##### 1.14.10.2.10.save()

保存模型.

```python
model.save(filepath='./model.h5',  # str or pathlib.Path|文件路径.
           save_format=None)  # {'tf', 'h5'}|None|保存文件格式.
```

##### 1.14.10.2.11.summary()

打印模型的摘要.

```python
model.summary()
```

#### 1.14.10.3.Sequential()

实例化`Sequential`.

```python
from tensorflow.keras.models import Sequential

model = Sequential()
```

##### 1.14.10.3.1.add()

添加一个网络层到`Sequential`的栈顶.

```python
model.add(layer=layers.Input(shape=(224, 224, 3)))  # keras.layers|网络层.
```

### 1.14.11.optimizers

| 版本 | 描述                 | 注意                                |
| ---- | -------------------- | ----------------------------------- |
| -    | tf.keras的优化器API. | 1.优化器相同的类方法都写在`Adam`里. |

#### 1.14.11.1.Adam()

实例化Adam优化器.

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam(learning_rate=0.001)  # float|0.001|学习率.
```

##### 1.14.11.1.1.apply_gradients()

`GradientTape`更新的参数赋值给优化器.

```python
from tensorflow.keras.optimizers import Adam

optimizer = Adam()
optimizer.apply_gradients(grads_and_vars=zip(grads, vars))  # list of (gradient, variable) pairs|梯度和变量对. 
```

#### 1.14.11.2.SGD()

实例化随机梯度下降优化器.

```python
from tensorflow.keras.optimizers import SGD

optimizer = SGD(learning_rate=0.01)  # float|0.001|学习率.
```

### 1.14.12.preprocessing

| 版本 | 描述                     | 注意 |
| ---- | ------------------------ | ---- |
| -    | tf.keras的数据预处理API. | -    |

#### 1.14.12.1.image

##### 1.14.12.1.1.array_to_img()

将数组转换为PIL图像.|`PIL.Image.Image`

```python
import numpy as np
from tensorflow.keras.preprocessing.image import array_to_img

arr = np.ones([128, 128, 3])
img = array_to_img(x=arr)  # numpy.ndarray|输入的数组.
```

##### 1.14.12.1.2.ImageDataGenerator()

实例化`ImageDataGenerator`, 对图像进行实时增强.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator(rotation_range=0,  # int|0|随机旋转的度数.
                               width_shift_range=0.,  # float|0.|水平位移范围.
                               height_shift_range=0.,  # float|0.|垂直位移范围.
                               shear_range=0.,  # float|0.|裁切角度.
                               zoom_range=0.,  # float|0.|随机缩放倍数.
                               channel_shift_range=0.,  # float|0.|随机色彩通道位移.
                               fill_mode='nearest',  # {'constant', 'nearest', 'reflect', 'wrap'}|'nearest'|填充模式.
                               horizontal_flip=False,  # bool|False|随机水平翻转.
                               vertical_flip=False)  # bool|False|随机垂直翻转.
```

###### 1.14.12.1.2.1.class_indices

类名称和索引映射字典.|`dict`

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator()
class_indices = generator.flow_from_dataframe(x).class_indices
```

###### 1.14.12.1.2.2.flow()

对数据进行增强.|`yield`

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator()
generator.flow(x,  # numpy array of rank 4 or tuple|输入的数据.
               y=None,  # array-like|None|标签.
               batch_size=32,  # int|32|批次大小.
               shuffle=True)  # bool|True|是否打乱.
```

###### 1.14.12.1.2.3.flow_from_dataframe()

从dataframe中读取数据, 并对数据进行增强.|`yield`

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator()
generator.flow_from_dataframe(dataframe,  # pandas.DataFrame|描述图片位置的dataframe.
                              directory=None,  # str|None|图片文件夹路径.
                              x_col='filename',  # str|'filename'|图片文件列.
                              y_col='class',  # str|'class'|标签列.
                              target_size=(256, 256),  # (height, width)|(256, 256)|读入图片的大小.
                              color_mode='rgb',  # {'grayscale', 'rgb', 'rgba'}|'rgb'|色彩空间.
                              classes=None,  # list of str|None|类名称列表.
                              class_mode='categorical',  # {'binary', 'categorical', 'input', 'multi_output', 'raw', 'sparse'}|'categorical'|标签数组类型.
                              batch_size=32,  # int|32|批次大小.
                              shuffle=True,  # bool|True|是否打乱.
                              interpolation='nearest',  # {'nearest', 'bilinear', 'bicubic', 'lanczos', 'box', 'hamming'}|'nearest'|插值方式.
                              validate_filenames=True)  # bool|True|是否检查文件可靠性.
```

###### 1.14.12.1.2.4.flow_from_directory()

从文件夹中读取数据(每个类别是个单独的文件夹), 并对数据进行增强.|`yield`

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

generator = ImageDataGenerator()
generator.flow_from_directory(directory,  # str|图片文件夹路径.
                              target_size=(256, 256),  # (height, width)|(256, 256)|读入图片的大小.
                              color_mode='rgb',  # {'grayscale', 'rgb', 'rgba'}|'rgb'|色彩空间.
                              classes=None,  # list of str|None|类名称列表.
                              class_mode='categorical',  # {'binary', 'categorical', 'input', 'multi_output', 'raw', 'sparse'}|'categorical'|标签数组类型.
                              batch_size=32,  # int|32|批次大小.
                              shuffle=True,  # bool|True|是否打乱.
                              interpolation='nearest')  # {'nearest', 'bilinear', 'bicubic', 'lanczos', 'box', 'hamming'}|'nearest'|插值方式.
```

##### 1.14.12.1.3.img_to_array()

将PIL图片转换为numpy数组.|`numpy.ndarray`

```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array

img = load_img(path='./img.png')
arr = img_to_array(img=img)  # PIL.Image|输入的图片.
```

##### 1.14.12.1.4.load_img()

加载图片.|`PIL.Image.Image`

```python
from tensorflow.keras.preprocessing.image import load_img

img = load_img(path='./img.png',  # str|图片的路径.
               target_size=None)  # (img_height, img_width)|None|读入图片的大小.
```

#### 1.14.12.2.timeseries_dataset_from_array()

从数组中创建时间序列数据集.|`tensorflow.python.data.ops.dataset_ops.BatchDataset`

```python
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

dataset = timeseries_dataset_from_array(data,  # numpy.ndarray or eager tensor|输入数据.
                                        targets,  # numpy.ndarray or eager tensor|标签.
                                        sequence_length,  # int|输出的序列⻓度.
                                        sequence_stride=1,  # int|1|连续输出序列之间的周期.
                                        sampling_rate=1,  # int|1|连续时间步之间的时间间隔.
                                        batch_size=128,  # int|128|批次大小.
                                        shuffle=False)  # bool|False|是否打乱.
```

### 1.14.13.regularizers

| 版本 | 描述                   | 注意 |
| ---- | ---------------------- | ---- |
| -    | tf.keras的正则化器API. | -    |

#### 1.14.13.1.L2()

实例化L2正则化器.

```python
from tensorflow.keras.regularizers import L2

regularizer = L2(l2=0.01)  # float|0.01|L2正则化因子.
```

### 1.14.14.utils

| 版本 | 描述               | 注意 |
| ---- | ------------------ | ---- |
| -    | tf.keras的工具API. | -    |

#### 1.14.14.1.get_file()

从指定URL下载文件.|`str`

```python
from tensorflow.keras.utils import get_file

file = get_file(fname,  # str|保存的文件名.
                origin,  # str|文件的URL.
                extract=False)  # bool|False|是否解压tar或zip文件.
```

#### 1.14.14.2.plot_model()

绘制模型网络图.

```python
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.utils import plot_model

plot_model(model=ResNet50(),  # keras.models|要绘制的模型.
           to_file='model.png',  # str|'model.png'｜保存的文件名.
           show_shapes=False,  # bool|False|显示网络层的形状.
           show_layer_names=True,  # bool|True|显示网络层的名称.
           rankdir='TB',  # {'TB', 'LR'}|'TB'|
           dpi=96)  # int|96|DPI值.
```

#### 1.14.14.3.Sequence()

实现数据序列(`__getitem__`和`__len__`必须实现).

```python
from tensorflow.keras.utils import Sequence

class DataSequence(Sequence):
    def __init__(self, **kwargs):
        super(DataSequence, self).__init__(**kwargs)
        self.on_epoch_end()

    def __getitem__(self, item):
        """获取一个批次的数据."""

    def __len__(self):
        """批次的数量."""

    def on_epoch_end(self):
        """每轮训练结束后对数据进行某种操作."""
```

#### 1.14.14.4.to_categorical()

将离散编码的标签转换为one-hot编码.|`numpy.ndarray`

```python
from tensorflow.keras.utils import to_categorical

label = [1, 2, 3]
y = to_categorical(y=label,  # array-like|标签.
                   num_classes=4)  # int|None|类别总数.
```

## 1.15.logical_not()

逐元素返回逻辑非的值.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
import tensorflow as tf

arr = np.asarray([0, 1, 2])
tensor = tf.logical_not(x=arr)  # tf.Tensor of type bool|输入的张量.
```

## 1.16.logical_or()

逐元素返回逻辑或的值.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
import tensorflow as tf

arr0 = np.asarray([0, 1, 0])
arr1 = np.asarray([1, 1, 0])
tensor = tf.logical_or(x=arr0,  # tf.Tensor of type bool|输入的张量.
                       y=arr1)  # tf.Tensor of type bool|输入的张量.
```

##。1.17.math

| 版本 | 描述                  | 注意 |
| ---- | --------------------- | ---- |
| -    | TensorFlow的数学操作. | -    |

### 1.17.1.ceil()

逐元素向上取整.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

arr = [-1.1, 2.4, 3.5]
tensor = tf.math.ceil(x=arr)  # tf.Tensor|输入的张量.
```

### 1.17.2.divide_no_nan()

安全除法, 遇到除零时返回值为零.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
import tensorflow as tf

arr1 = np.asarray([1., 2., 3.])
arr2 = np.asarray([1., 0., 3.])
tensor = tf.math.divide_no_nan(x=arr1,  # tf.Tensor or array-like(float32 or float64)|被除数.
                               y=arr2)  # tf.Tensor or array-like|除数.
```

### 1.17.3.sqrt()

逐元素计算平方根.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

arr = [1., 4., 9.]
tensor = tf.math.sqrt(x=arr)  # tf.Tensor|输入的张量.
```

## 1.18.meshgrid()

生成坐标矩阵.|`list of tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
import tensorflow as tf

x_coord = np.linspace(0, 4, 5)
y_coord = np.linspace(0, 4, 5)
tensor = tf.meshgrid(x_coord, y_coord)  # array_like|坐标向量.
```

## 1.19.nn

| 版本 | 描述                            | 注意 |
| ---- | ------------------------------- | ---- |
| -    | TensorFlow的神经网络操作修饰器. | -    |

### 1.19.1.sigmoid()

逐元素计算sigmoid的值.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
import tensorflow as tf

arr = np.asarray([1., 2., 3.])
tensor = tf.nn.sigmoid(x=arr)  # tf.Tensor or array-like|输入的张量.
```

### 1.19.2.sigmoid_cross_entropy_with_logits()

逐元素计算带有sigmoid的交叉熵的值.|`tensorflow.python.framework.ops.EagerTensor`

```python
import numpy as np
import tensorflow as tf

arr = np.asarray([1., 0., 1.])
tensor = tf.nn.sigmoid_cross_entropy_with_logits(labels=arr,  # tf.Tensor or array-like|真实值.
                                                 logits=arr)  # tf.Tensor or array-like|标签值.
```

## 1.20.ones()

创建全一张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

tensor = tf.ones(shape=(3, 2),  # list/tuple of int|张量的形状.
                 dtype='int32')  # str|dtypes.float32|元素数据类型.
```

## 1.21.ones_like()

创建一个输入数组形状相同的全一张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

arr = [[1, 2, 3], [4, 5, 6]]
tensor = tf.ones_like(input=arr)  # array-like|输入的数组. 
```

## 1.22.py_function()

将Python函数修饰成TensorFlow的操作, 并在Eager模式下运行.

```python
import tensorflow as tf

def get_max(a, b):
    return max(a, b)
a = tf.constant(5)
b = tf.constant(-6)
max_value = tf.py_function(func=get_max,  # function|Python函数.
                           inp=[a, b],  # list of Tensor|输入的张量.
                           Tout=tf.int32)  # tensorflow.python.framework.dtypes.DType|返回数据的数据类型.
```

## 1.23.random

### 1.23.1.normal()

生成标准正态分布的张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

tensor = tf.random.normal(shape=[2, 3])  # array-like|张量的形状.
```

### 1.23.2.uniform()

生成均匀分布的张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

tensor = tf.random.uniform(shape=[2, 3])  # array-like|张量的形状.
```

## 1.24.range()

创建一个序列张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

tensor = tf.range(start=5,  # int|序列的最大值.
                  dtype=tf.float32)  # tensorflow.python.framework.dtypes.DType(可选)|None|张量的数据类型.
```

## 1.25.reduce_max()

返回张量中的最大值.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

arr = [1, 2, 3, 2]
tensor = tf.reduce_max(input_tensor=arr)  # tf.Tensor or array-like|输入的张量.
```

## 1.26.reduce_min()

返回张量中的最小值.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

arr = [1, 2, 3, 2]
tensor = tf.reduce_min(input_tensor=arr)  # tf.Tensor or array-like|输入的张量.
```

## 1.27.reduce_sum()

对张量沿指定轴求和.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

arr = [[0, 1], [2, 2]]
tensor = tf.reduce_sum(input_tensor=arr,  # tf.Tensor or array-like|输入的张量.
                       axis=1)  # int|None|沿指定维度合并.
```

## 1.28.reshape()

改变张量的形状.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

tensor = tf.constant([1., 2., 3., 4.])
tensor = tf.reshape(tensor=tensor,  # tf.Tensor|要改变形状的张量.
                    shape=[2, 2])  # list or tuple|改变后的形状.
```

## 1.29.shape()

返回包含输入张量形状的张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

tensor = tf.constant([[1.], [2.]])
shape = tf.shape(input=tensor)  # tf.Tensor|输入的张量.
```

## 1.30.strings

| 版本 | 描述               | 注意 |
| ---- | ------------------ | ---- |
| -    | tf的字符串操作API. | -    |

### 1.30.1.reduce_join()

拼接字符串.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

str = ['12', 'ab', '3', 'c']
tensor = tf.strings.reduce_join(inputs=str)  # str|输入的字符串.
```

### 1.30.2.unicode_split()

将字符串转换为Unicode编码的字节.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

str = 'cat'
tensor = tf.strings.unicode_split(input=str,  # str|输入的字符串.
                                  input_encoding='UTF-8')  # str|输入字符串的编码.
```

## 1.31.TensorArray()

实例化`TensorArray`.

```python
import tensorflow as tf

tensor_arr = tf.TensorArray(dtype=tf.float32,  # tensorflow.python.framework.dtypes.DType|张量的数据类型.
                            size=4,  # int(可选)|None|TensorArray的大小.
                            dynamic_size=True)  # bool(可选)|False|是否可以将TensorArray增长到超过其初始大小.
```

### 1.31.1.write()

在`TensorArray`指定索引位置写入值.|`tensorflow.python.ops.tensor_array_ops.TensorArray`

```python
import tensorflow as tf

ta = tf.TensorArray(dtype=tf.float32, size=4, dynamic_size=True)
ta = ta.write(index=2,  # int|写入处的索引.
              value=10)  # tf.Tensor of type `dtype`|写入的值.
```

## 1.32.tensordot()

沿指定维度点乘.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

a = [[1], [2]]
b = [[2, 1]]
tensor = tf.tensordot(a=a,  # tf.Tensor|输入的张量.
                      b=b,  # tf.Tensor|输入的张量.
                      axes=1)  # int|维度.
```

## 1.33.tpu

### 1.33.1.experimental

#### 1.33.1.1.initialize_tpu_system()

初始化TPU系统.

```python
import tensorflow as tf

tf.tpu.experimental.initialize_tpu_system()
```

## 1.34.transpose()

对张量进行转置操作.|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf

a = [[[1, 1, 1], [2, 2, 2]]]
tensor = tf.transpose(a=a,  # tf.Tensor|输入的张量.
                      perm=[1, 2, 0])  # list|None|轴的排列顺序.
```

## 1.35.Variable()

创建变量.|`tensorflow.python.ops.resource_variable_ops.ResourceVariable`

```python
import tensorflow as tf

tensor = tf.Variable(2021)
```

## 1.36.where()

根据判断条件, 真值返回`x`, 假值返回`y`.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

a = 1
b = 2
tensor = tf.where(condition=(a > b),  # tf.Tensor of type bool|判断条件.
                  x=True,  # tf.Tensor|None|情况为真的返回值.
                  y=False)  # tf.Tensor|None|情况为假的返回值.
```

## 1.37.zeros_like()

创建一个输入数组形状相同的全零张量.|`tensorflow.python.framework.ops.EagerTensor`

```python
import tensorflow as tf

arr = [[1, 2, 3], [4, 5, 6]]
tensor = tf.zeros_like(input=arr)  # array-like|输入的数组. 
```

# 2.tensorflow.js

| 版本  | 描述                                 | 注意                             | 适配M1 |
| ----- | ------------------------------------ | -------------------------------- | ------ |
| 3.8.0 | TensorFlow 的 JavaScript 机器学习库. | 1. TensorFlow.js 使用ES2017语法. | 是     |

## 2.1.browser

### 2.1.1.fromPixels()

从图片中创建`tf.Tensor`.|`tf.Tensor3D`

```javascript
import * as tf from '@tensorflow/tfjs';

let tensor = tf.browser.fromPixels(image);  // pixels: PixelData, ImageData, HTMLImageElement, HTMLCanvasElement, HTMLVideoElement or ImageBitmap|输入的图片.
```

## 2.2.div()

张量逐元素除法.|`tf.Tensor`

```javascript
import * as tf from '@tensorflow/tfjs';

let a = tf.scalar(5);
let b = tf.scalar(2);

let c = tf.div(a,  // a: tf.Tensor, TypedArray or Array|被除数.
               b);  // b: tf.Tensor, TypedArray or Array|除数.
```

## 2.3.expandDims()

增加张量的维度.|`tf.Tensor`

```javascript
import * as tf from '@tensorflow/tfjs';

let a = tf.tensor([5]);

let b = tf.expandDims(a,  // x: tf.Tensor, TypedArray or Array|输入的数组.
                      0);  // axis: number(可选)|0|添加新维度的位置.
```

## 2.4.image

### 2.4.1.resizeBilinear()

使用双线性插值修改图片的尺寸.|`tf.Tensor3D` 或 `tf.Tensor4D`

```javascript
import * as tf from '@tensorflow/tfjs';

var tensor = tf.tensor([[[1, 2],
                         [1, 2]]]);
tensor = tf.image.resizeBilinear(tensor,  // images: tf.Tensor3D, tf.Tensor4D, TypedArray or Array|输入的图像.
                         			   [2, 3]);  // size: [number, number]|修改后的尺寸.
```

## 2.5.LayersModel

### 2.5.1.predict()

使用模型进行预测.|`tf.Tensor` 或 `tf.Tensor[]`

```javascript
import * as tf from '@tensorflow/tfjs';

let model = await tf.loadLayersModel(
    'https://storage.googleapis.com/tfjs-models/tfjs/iris_v1/model.json');
model.predict(tf.ones([1, 4])).print();  // x: tf.Tensor, tf.Tensor[]|特征数据.
```

### 2.5.2.summary()

打印模型的摘要.

```javascript
model.summary();
```

## 2.6.loadLayersModel()

加载`LayersModel`.

```javascript
import * as tf from '@tensorflow/tfjs';

let model = await tf.loadLayersModel(
     'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');  // pathOrIOHandler: string, io.IOHandler|模型的路径.
```

## 2.7.ones()

生成全一数组.|`tf.Tensor`

```javascript
import * as tf from '@tensorflow/tfjs';

let tensor = tf.ones([1, 4]);  // shape: number[]|数组的形状.
```

## 2.8.print()

将信息输出在控制台.

```javascript
import * as tf from '@tensorflow/tfjs';

let a = tf.tensor([5]);

tf.print(a);
```

## 2.9.scalar()

实例化`tf.Tensor`常量.

```javascript
import * as tf from '@tensorflow/tfjs';

let tensor = tf.scalar(10.0,  // value: number or boolean or string or Uint8Array|输入的数据.
                       'float32');  // dtype: 'float32','int32', 'bool', 'complex64' or 'string'(可选)|数据类型.
```

## 2.10.sub()

张量逐元素减法.|`tf.Tensor`

```javascript
import * as tf from '@tensorflow/tfjs';

let a = tf.scalar(5);
let b = tf.scalar(2);

let c = tf.sub(a,  // a: tf.Tensor, TypedArray or Array|被减数.
               b);  // b: tf.Tensor, TypedArray or Array|减数.
```

## 2.11.tensor()

实例化`tf.Tensor`张量.

```javascript
import * as tf from '@tensorflow/tfjs';

let tensor = tf.tensor(10.0);  // values: TypedArray or Array|输入的数据.
```

### 2.11.1.data()

异步获取的`tf.Tensor`值.|`Promise<DataTypeMap[NumericDataType]>`

```javascript
import * as tf from '@tensorflow/tfjs';

let tensor = tf.tensor(10.0);
let value = tensor.data();
```

## 2.12.tidy()

自动清理除返回值外的全部中间变量, 避免内存泄漏.

```javascript
import * as tf from '@tensorflow/tfjs';

function func() {
    let a = tf.scalar(2);
    let b = tf.scalar(2);
    
    return tf.add(a, b);
}

let c = tf.tidy(func);  // nameOrFn: string or Function|输入的函数.
```

# 3.tensorflow_datasets

| 版本  | 描述                    | 注意                                                         | 适配M1 |
| ----- | ----------------------- | ------------------------------------------------------------ | ------ |
| 4.3.0 | TensorFlow的官方数据集. | 1. 默认的缓存路径是~/tensorflow_datasets.                                                                          2. 视网络情况使用代理. | 是     |

## 3.1.load()

加载数据集.|`dict of tf.data.Datasets`

```python
import tensorflow_datasets as tfds

ds_train, ds_test = tfds.load(name='mnist',  # str|数据集的注册名称.
                              split=['train', 'test'],  # str or list{'train', ['train', 'test'], 'train[80%:]'}(可选)|None|是否拆分测试集.
                              shuffle_files=True,  # bool(可选)|False|是否打乱数据.
                              as_supervised=True)  # bool(可选)|False|是否返回标签.
```

# 4.tensorflow_hub

| 版本   | 描述                    | 注意                                                         | 适配M1 |
| ------ | ----------------------- | ------------------------------------------------------------ | ------ |
| 0.12.0 | TensorFlow的官方模型库. | 1. 推荐使用环境变量`TFHUB_CACHE_DIR`指定模型保存位置.                                                       2. [TensorFlow Hub 国内镜像](https://hub.tensorflow.google.cn/) | 是     |

## 4.1.KerasLayer()

将模型修饰为Keras的网络层.|`tensorflow_hub.keras_layer.KerasLayer`

```python
from tensorflow_hub import KerasLayer

layer = KerasLayer(handle='https://hub.tensorflow.google.cn/google/efficientnet/b0/classification/1',  # str|模型的路径或者(URL).
                   trainable=False,  # bool(可选)|False|是否可以训练.
                   output_shape=None,  # tuple|None|网络层的输出形状.
                   input_shape,  # tuple|None|期望的形状.
                   dtype='float32')  # tensorflow.python.framework.dtypes.DType|'float32'|期望的数据类型.
```

## 4.2.load()

加载模型.|`tensorflow.python.training.tracking.tracking.AutoTrackable`

```python
from tensorflow_hub import load

model = load(handle='https://hub.tensorflow.google.cn/google/efficientnet/b0/classification/1')  # str|模型的路径或者(URL).
```
