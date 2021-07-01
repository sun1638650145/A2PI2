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

## 1.10.Variable()

创建变量.|tensorflow.python.ops.resource_variable_ops.ResourceVariable

```python
import tensorflow as tf

tensor = tf.Variable(2021)
```

# 2.tensorflow.js

# 3.tensorflow_datasets

# 4.tensorflow_hub

