# A²PI²

## 1.imageio

### 1.1.imread()

读取图片并返回一个numpy数组

```python
import imageio
image = imageio.imread(uri)# 文件对象或http地址，图片的路径
```

## 2.keras

### 2.1.applications

#### 2.1.1.inception_v3

##### 2.1.1.1.InceptionV3()

InceptionV3的预训练模型

```python
from keras.applications.inception_v3 import InceptionV3
model = InceptionV3(include_top,# 是否包含全连接的输出层
                    weights,# 权重，可以是随机初始化，也可以加载'imagenet'的权重，或者自定权重的路径
                    input_tensor)# 输入层，需要使用keras.layers.Input()
```

#### 2.1.2.resnet50

##### 2.1.2.1.ResNet50()

ResNet50的预训练模型

```python
from keras.applications.resnet50 import ResNet50
model = ResNet50(include_top,# 是否包含全连接的输出层
                 weights,# 权重，可以是随机初始化，也可以加载'imagenet'的权重，或者自定权重的路径
                 input_tensor)# 输入层，需要使用keras.layers.Input()
```

####  2.1.3.xception

##### 2.1.3.1.Xception()

Xception的预训练模型

```python
from keras.applications.xception import Xception
model = Xception(include_top,# 是否包含全连接的输出层
                 weights,# 权重，可以是随机初始化，也可以加载'imagenet'的权重，或者自定权重的路径
                 input_tensor)# 输入层，需要使用keras.layers.Input()
```

### 2.2.layers

#### 2.2.1.Add()

将多个层通过矩阵加法合并

```python
from keras.layers import Add
layer = Add()(_Merge)# 相同形状的张量（层）列表
```

#### 2.2.2.Concatenate()

连接输入的层

```python
from keras.layers import Concatenate
layer = Concatenate(axis)(_Merge)
"""
axis 维度
_Merge 张量（层）列表
"""
```

#### 2.2.3.Dense()

全连接层，参看tf.keras.layers.Dense()

#### 2.2.4.Dropout()

在训练阶段按照比例随机丢弃神经元，参看tf.keras.layers.Dropout()

#### 2.2.5.Flatten()

将输入展平，参看tf.keras.layers.Flatten()

#### 2.2.6.Input()

输入层

```python
from keras.layers import Input
input_tensor = Input(shape)# 整数，形状元组
```

### 2.3.models

####2.3.1.Model()

keras自定义模型对象

```python
from keras.models import Model
model = Model(inputs,# 输入层
              outputs)# 输出层
```

##### 2.3.1.1.fit_generator()

生成批次训练数据，按批训练数据

```python
model.fit_generator(generator,# 数据生成器，比如ImageDataGenerator()
                    steps_per_epoch,# 整数，每批次步数
                    epochs,# 整数，轮数
                    verbose)# 日志显示模式 0=安静模型 1=进度条 2每轮显示
```

##### 2.3.1.2.load_model()

```python
model = load_model(filepath)# 文件路径
```

##### 2.3.1.3.save()

将模型保存为SavedModel或者HDF5文件

```python
model.save(filepath)# 保存路径
```

### 2.4.optimizers

#### 2.4.1.Adam()

Adam优化器

```python
from tensorflow.python.keras.optimizers import Adam
optimizer = Adam(lr)# 学习率
```

### 2.5.preprocessing

#### 2.5.1.image

##### 2.5.1.1.ImageDataGenerator()

对图片数据进行实时的数据增强类

```python
from keras.preprocessing.image import ImageDataGenerator
data_generator = ImageDataGenerator(rotation_range,# 整数，随机旋转度数
                                    width_shift_range,# 浮点数，水平偏移范围
                                    height_shift_range,# 浮点数，垂直偏移范围
                                    shear_range,# 浮点数（角度），裁切角范围
                                    zoom_range,# 浮点数，随机缩放倍数 
                                    channel_shift_range,# 浮点数，随机色彩通道移位
                                    fill_mode，# 填充模式，'constant''nearest''reflecto''wrap'
                                    horizontal_flip)# 布尔值，水平随机翻转                 
```

###### 2.5.1.1.1.flow_from_directory()

从给定路径读入数据并增强

```python
data_generator.flow_from_directory(directory,# 路径
                                   target_size,# 元组，调整后大小
                                   class_mode,# 返回标签数组类型，默认'categorical'
                                   batch_size,# 整数，批次大小，默认32
                                   shuffle,# 布尔值，打乱顺序
                                   interpolation)# 插值，调整尺寸 'nearest''bilinear''bicubic'
```

### 2.6.utils

#### 2.6.1.multi_gpu_model()

多GPU并行训练模型

```python
from keras.utils import multi_gpu_model
parallel_model = multi_gpu_model(model,# 模型
                                 gpus)# 整数（大于等于2），并行GPU数量
```

## 3.PIL

### 3.1.Image

#### 3.1.1.fromarray()

从输入的图片返回一个数组

```python
from PIL import Image
array = Image.fromarray(obj)# 图片对象
```

#### 3.1.2.resize()

返回调整大小后的图像的副本

```python
from PIL import Image
new_image = image.resize(size)# 有宽度和高度的二元组
```

## 4.protobuf

### 4.1.SerializeToString()

将protobuf数据转换为二进制字符串

```python
# GraphDef就是一种protobuf
graphde.SerializeToString()
```

## 5.sklearn

### 5.1.linear_model

#### 5.1.1.LogisticRegression()

构建一个对数几率回归模型

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```

##### 5.1.1.1.fit()

以给定数据训练模型

```python
model.fit(X,# 训练数据 
          y,# 标签 
          sample_weight)# 数组，类别权重，默认为None
```

##### 5.1.1.2.predict()

生成预测结果

```python
model.predict(X)# 测试数据
```

### 5.2.metrics

#### 5.2.1.accuracy_score()

计算准确率

```python
import sklearn
sklearn.metrics.accuracy_score(y_true,# 真实标签
                               y_pred,# 预测结果
                               sample_weight)# 数组，类别权重，默认为None
```

### 5.3.model_selection

#### 5.3.1.cross_val_predict()

使用交叉验证法验证

```python
import sklearn
sklearn.model_selection.cross_val_predict(estimator,# 训练的模型对象
                                          X,# 训练数据
                                          y,# 标签
                                          cv)# 整数，划分数，默认为3
```

#### 5.3.2.LeaveOneOut()

使用留一法验证

```python
import sklearn
LOO = sklearn.model_selection.LeaveOneOut() # 返回一个BaseCrossValidator对象
```

##### 5.3.2.1.split()

按照具体BaseCrossValidator对象将数据划分为训练和测试集

```python
LOO.split(X)# 训练数据
```

## 6.tensorflow r1.x

### 6.1.concats()

按某个维度连接多个张量

```python
import tensorflow as tf
tensor_a = [[1, 2], [3, 4]]
tensor_b = [[5, 6], [7, 8]]
tensor_c = tf.concat(value=[tensor_a, tensor_b], axis=1, name="concat")
"""
value待合并的张量，axis按某个维度合并，name张量的名字
"""
```

### 6.2.ConfigProto()

用于配置会话的选项

```python
import tensorflow as tf
tf.ConfigProto(gpu_option)# 配置显存
```

### 6.3.gfile

#### 6.3.1.FastGFile()

没有线程锁的文件I/O封装器

```python
import tensorflow as tf
fp = tf.gfile.FastGfile(name,# 保存名称 
                        mode)# 模式，默认'r'
```

###6.4.global_variables()

返回默认会话中所有的全局变量

```python
import tensorflow as tf
var = tf.global_variables()
```

### 6.5.global_variables_initializer()

初始化全局的变量

```python
import tensorflow as tf 
init = tf.global_variables_initializer()
```

### 6.6.GPUOptions()

使用GPU时对显存使用的控制选项

```python
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth)# 布尔值，设置为True时，显存按需分配（tensorflow默认是占用全部显存）
```

### 6.7.GraphDef()

tensorflow的信息协议

```python
import tensorflow as tf 
graph_def = tf.GraphDef()
```

#### 6.7.1.ParseFromString()

将序列化的数据转换成GraphDef

```python
graph_def.ParseFromString(serialized)# 序列化数据，数据流 
```

### 6.8.import_graph_def()

将GraphDef实例导入默认的Graph计算图

```python
import tensorflow as tf
tensor = tf.import_graph_def(graph_def,# graph_def实例
                     				 input_map,# 字典，输入张量的名称和对应的张量
                     				 return_elements)# 字符串列表，输出张量的名称
```

### 6.9.keras

#### 6.9.1.layers

##### 6.9.1.1.Conv1D()

1D卷积层（例如时序卷积）

```python
keras.layers.Conv1D(filters,# 整数，卷积核的数量 
                    kernel_size,# 整数，卷积核的大小
                    strides,# 整数、或者列表或者元组，滑动步长
                    padding,# 'same''valid''causal'，是否使用全零填充
                    data_format,# 'channels_first''channels_last' 数据格式CBHW或BHWC
                    activation,# 'relu''sigmoid' 激活函数
                    use_bias,# 布尔值 是否使用偏置
                    kernel_initializer,# 权值初始化
                    bias_initializer)# 偏置项初始化
```

##### 6.9.1.2.Dense()

全连接层

```python
keras.layers.Dense(units,# 整数，神经元数量
									 activation,# 'relu''sigmoid' 激活函数 
                   use_bias,# 布尔值 是否使用偏置
                   kernel_initializer,# 权值初始化
                   bias_initializer)# 偏置项初始化
```

##### 6.9.1.3.Dropout()

在训练阶段按照比例随机丢弃神经元

```python
keras.layers.Dropout(rate)# 浮点数 丢弃率
```

##### 6.9.1.4.Flatten()

将输入展平，不影响批次大小

```python
keras.layers.Flatten()
```

##### 6.9.1.5.MaxPooling1D()

对时序数据进行最大池化

```python
keras.layers.MaxPooling1D(pool_size,# 整数，池化核数量
													strides,# 整数、或者列表或者元组，滑动步长
                          padding)# 'same''valid''causal'，是否使用全零填充
```

#### 6.9.2.models

##### 6.9.2.1.Sequential()

构建一个线性堆叠的网络模型，不兼容TensorFlow r1.*和r2版本以及原生keras，最简单和使用的基础模型，可以构造AlexNet、VGGNet一类的线性模型，Inception带有残差网络的无法构造

```python
from tensorflow import keras
model = keras.models.Sequential()
```

###### 6.9.2.1.1.add() 

将一个具体的单层神经网络加入模型

```python
# 一维卷积层
model.add(keras.layers.Conv1D(input_shape, filters, kernel_size, strides, padding))
"""
input_shape只有第一层需要，filters卷积数，kernel_size卷积核大小，strides滑动步长，                        padding是否全零填充
"""
# 全连接层
model.add(keras.layers.Dense(units, input_dim, activation)) 
"""
input_dim只有第一层需要，activation不填就是默认a(x)=x线性激活函数
"""
# Dropout层
model.add(keras.layers.Dropout(rate))
# Flatten层，用于将卷积层和全连接层连接
model.add(keras.layers.Flatten())
# 一维最大池化层
model.add(keras.layers.MaxPooling1D(pool_size, strides, padding))
"""
pool_size池化核大小，strides滑动步长，padding是否全零填充
"""
```

######6.9.2.1.2.compile()

用于配置训练模型

```python
model.compile(optimizer,# 优化器
              loss,# 损失函数
              metrics)# 评估标准['accuracy']
```

###### 6.9.2.1.3.evaluate()

在测试模式下返回损失值和准确率

```python
model.evaluate(x,# 训练数据
               y,# 标签
               batch_size,# 整数，批次大小，默认32
               verbose)# 日志显示模式 0=安静模型 1=进度条 2每轮显示
```

###### 6.9.2.1.4.fit()

以给定批次训练模型

```python
history = model.fit(x,# 训练数据
          y,# 标签
          batch_size,# 整数，批次大小，默认32
          epochs,# 整数，轮数
          verbose,# 日志显示模式 0=安静模型 1=进度条 2每轮显示
          callbacks,# 回调函数
          validation_split,# 浮点数，验证集可以从训练集中划分
          validation_data,# 元组(x,y) 验证集数据可以直接指定，会直接覆盖validation_split
          shuffle)# 布尔值，打乱数据
```

###### 6.9.2.1.5.load_weights()

加载所有的神经网络层的参数

```python
model.load_weights(filepath)# 检查点文件路径
```

###### 6.9.2.1.6.predict()

生成预测结果

```python
model.predict(x,# 测试数据
              batch_size,# 整数，批次大小，默认32
              verbose)# 日志显示模式 0=安静模型 1=进度条
```

###### 6.9.2.1.7.summary()

查看模型的各层参数

```python
model.summary()
```

### 6.10.nn

#### 6.10.1.avg_pool()

均值池化层

```python
import tensorflow as tf
tf.nn.avg_pool(value,# 输入张量
               ksize,# 整数，池化核数量
               strides,# 整数、或者列表或者元组，滑动步长
               padding,# 'SAME''VALID'，是否使用全零填充
               data_format="NHWC",# 数据格式，默认"NHWC"
               name)# 名称
```

#### 6.10.2.dropout()

在训练阶段按照比例随机丢弃神经元

```python
import tensorflow as tf 
tf.nn.dropout(x,# 输入张量
              keep_prob,# 保留概率
							name)# 整数
```

####6.10.3.lrn()

局部响应归一化层(Local Response Normalization)
$$
b^i_{x,y}=\frac{a^i_{x,y}}{\Big(k+\alpha\sum\limits^{min(N-1,i+\frac{n}{2})}_{j=max(0,i-\frac{n}{2})}(a^j_{x,y})^2\Big)^\beta}
$$
其中$a^i_{x,y}$是input，$\frac{n}{2}$是depth_radius，$k$是bias，$\alpha$是alpha，$\beta$是beta

AlexNet使用的一种类似Dropout的减少过拟合方法，不改变size；个人认为是一种类似于池化但不改变size大小的方法，在采样半径下，输入对输出的和的标准化

```python
import tensorflow as tf
tf.nn.lrn(input,# 输入张量
          depth_radius,# 采样半径，默认5 
          bias,# 超参数，默认1
          alpha,# 超参数，默认1
          beta,# 超参数，默认0.5
          name)# 名称
```

#### 6.10.4.max_pool()

最大池化层

```python
import tensorflow as tf
tf.nn.max_pool(value,# 输入张量
               ksize,# 整数，池化核数量
               strides,# 整数、或者列表或者元组，滑动步长
               padding,# 'SAME''VALID'，是否使用全零填充
               data_format="NHWC",# 数据格式，默认"NHWC"
               name)# 名称
```

#### 6.10.5.softmax()

softmax激活函数

```python
import tensorflow as tf
tf.nn.softmax(logits)# 输入张量（非空）
```

### 6.11.placeholder()

添加一个占位符

```python
import tensorflow as tf
x = tf.placeholder(dtype,# 自负床数据类型
                   shape,# 张量形状
                   name)# 名称
```

### 6.12.python

#### 6.12.1.framework

##### 6.12.1.1.graph_util

###### 6.12.1.1.1.convert_variables_to_constants

将计算图中的变量转换为常量

```python
from tensorflow.python.framework.graph_util import convert_variables_to_constants
output_graph = convert_variables_to_constants(sess,# 需要转换变量的会话
                                   						input_graph_def,# 会话的graph_def对象
                                   						output_node_names)#字符串列表，输出层的名称
```

### 6.13.saved_model

#### 6.13.1.builder

#####6.13.1.1.SavedModelBuilder()

构建一个生成SavedModel的实例

```python
from tensorflow.saved_model import builder
builder = builder.SavedModelBuilder(export_dir)# SavedModel的保存路径
```

######6.13.1.1.1.add_meta_graph_and_variables()

添加图结构和变量信息

```python
builder.add_meta_graph_and_variables(sess,# 会话
                                   	 tags,# 标签，自定义
                                   	 signature_def_map)# 预测签名字典
```

###### 6.13.1.1.2.save()

将SaveModel写入磁盘

```python
builder.save()
```

#### 6.13.2.loader

##### 6.13.2.1.load()

从标签指定的SavedModel加载模型

```python
import tensorflow as tf
tf.saved_model.loader.load(sess,# 模型还原到的会话
                           tags,# 字符串，标签，参见tf.saved_model.tag_constants
                           export_dir)# 待还原的SavedModel目录
```

#### 6.13.3.signature_def_utils

##### 6.13.3.1.predict_signature_def()

构建预测签名

```python
from tensorflow.saved_model import signature_def_utils
signature = signatures_def_utils.predict_signature_def(inputs,# 字典，输入变量
                                                       outputs)# 字典，输出变量
```

#### 6.13.4.simple_save()

使用简单方法构建SavedModel用于服务器

```python
from tensorflow.saved_model import simple_save
simple_save(session,# 会话
            export_dir,# SavedModel的保存目录
            inputs,# 字典，输入变量
            outputs)# 字典，输出变量
```

#### 6.13.5.tag_constants

SaveModel的标签

```python
from tensorflow.saved_model import tag_constants
tags = tag_constants.SERVING
"""
标签有TPU SERVING GPU TRAINING
"""
```

### 6.14.Session()

生成一个tensorflow的会话

```python
import tensorflow as tf
sess = tf.Session(config)# 使用ConfigProto配置会话
```

#### 6.14.1.close()

关闭当前会话

```python
sess.close()
```

#### 6.14.2.graph

##### 6.14.2.1.get_tensor_by_name()

根据名称返回张量，可以使用多个线程同时调用

```python
sess.graph.get_tensor_by_name(name)# 张量的名称
```

#### 6.14.3.run()

运行传入会话的操作，返回结果张量

```python
sess.run(fetches,# 待计算的操作 'Operation''Tensor' 
         feed_dict# 输入的值，默认为None
```

### 6.15.split()

将张量按某个维度拆分成多个张量

```python
import tensorflow as tf
tensor = [[1, 2, 5, 6], [3, 4, 7, 8]]
tensor_list = tf.split(value=tensor, num_or_size_splits=2, axis=1, name="split")
"""
value需要拆分的张量，num_or_size_splits要拆分的数量，axis按某个维度拆分，name张量的名字
"""
```

### 6.16.train

#### 6.16.1.AdamOptimizer()

Adam优化器

```python
import tensorflow as tf
optimizer = tf.train.AdamOptimizer(learning_rate)# 学习率
```

#### 6.16.2.GradientDescentOptimizer()

梯度下降优化器

```python
import tensorflow as tf
optimizer = tf.train.GradientDescentOptimizer(learning_rate)# 学习率
```

#### 6.16.3.latest_checkpoint()

查找最近的保存点文件

```python
import tensorflow as tf
checkpoint = tf.train.latest_checkpoint(checkpoint_dir)# 保存点路径
```

#### 6.16.3.Saver()

生成用于保存和还原计算图的实例

```python
import tensorflow as tf
saver = tf.train.Saver(var_list)
# 将被保存和恢复的变量列表或者变量字典，默认为None（保存全部） 
```

##### 6.16.3.1.restore()

恢复保存的变量

```python
saver.restore(sess,# 会话，eager模式为None
              save_path)# 检查点文件的路径
```

### 6.17.variable_scope()

用于定义变量操作的上下文管理器

``` python
import tensorflow as tf
with tf.variable_scope(name_or_scope):# 字符串，作用域
```

## 7.tensorflow r2.x

### 7.1.data

#### 7.1.1.Datasets

##### 7.1.1.1.batch()

给数据集划分批次

```python
import tensorflow as tf
dataset = tf.data.Dataset.range(6)
dataset_shuffle = dataset.batch(batch_size=3)# 批次的大小
```

##### 7.1.1.2.from_tensor_slices()

返回一个数据集对象

```python
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices(tensor)# 张量，必须有相同的第一维
```

##### 7.1.1.3.shuffle()

随机打乱数据集对象的元素

```python
import tensorflow as tf
dataset = tf.data.Dataset.range(3)
dataset_shuffle = dataset.shuffle(buffer_size=3)# 数据集元素的数量
```

### 7.2.keras

#### 7.2.1.datasets

##### 7.2.1.1.mnist

keras自带的数据集之一

###### 7.2.1.1.1.load_data()

加载mnist数据集

```python
from tensorflow.keras import mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

#### 7.2.2.layers

##### 7.2.2.1.BatchNormalization()

批量标准化层

```python
from tensorflow.keras.layers import BatchNormalization
layer = BatchNormalization()
```

##### 7.2.2.2.Conv2D()

2D卷积层（例如时序卷积）

```python
from tensorflow.keras.layers import Conv2D
layer = Conv2D(filters,# 整数，卷积核的数量 
               kernel_size,# 整数，卷积核的大小
               strides,# 整数、或者列表或者元组，滑动步长
               padding,# 'same''valid'，是否使用全零填充
               input_shape)# 元组，第一层需要指定输入
```

##### 7.2.2.3.Conv2DTranspose()

转置卷积层 (有时被成为反卷积)，将具有卷积输出尺寸的东西 转换为具有卷积输入尺寸的东西

```python
from tensorflow.keras.layers import Conv2DTranspose
layer = Conv2DTranspose(filters,# 整数，输出空间的维度
                    		kernel_size,# 整数，卷积核的大小
                    		strides,# 整数、或者列表或者元组，滑动步长
                    		padding,# 'same''valid''，是否使用全零填充
                    		use_bias)# 布尔值 是否使用偏置
```

##### 7.2.2.4.Dense()

全连接层

```python
from tensorflow.keras.layers import Dense
layer = Dense(units,# 整数，神经元数量
              use_bias,# 布尔值 是否使用偏置
              input_shape)# 元组，第一层需要指定输入
```

##### 7.2.2.5.Dropout()

在训练阶段按照比例随机丢弃神经元

```python
from tensorflow.keras.layers import Dropout
layer = Dropout(rate)# 丢弃率
```

##### 7.2.2.6.Flatten()

将输入展平，不影响批次大小

```python
from tensorflow.keras.layers import Flatten
layer = Flatten()
```

##### 7.2.2.7.LeakyReLU()

带泄漏的 ReLU层

```python
from tensorflow.keras.layers import LeakyReLU
layer = LeakyReLU(alpha)# 负斜率系数，默认为0.3
```

##### 7.2.2.8.Reshape()

将输入重新调整为特定的尺寸

```python
from tensorflow.keras.layers import Reshape
layer = Reshape(target_shape)# 整数元组，目标尺寸
```

#### 7.2.3.losses

##### 7.2.3.1.BinaryCrossentropy()

计算真实标签和预测值标签的的交叉熵损失

```python
from tensorflow.keras.losses import BinaryCrossentropy
cross_entropy = BinaryCrossentropy(from_logits)# 是否将y_pred解释为张量
```

#### 7.2.4.optimizers

##### 7.2.4.1.Adam()

Adam优化器

```python
from tensorflow.python.keras.optimizers import Adam
optimizer = Adam(lr)# 学习率
```

#### 7.2.5.Sequential()

构建一个线性堆叠的网络模型

```python
from tensorflow.keras import Sequential
model = Sequnential()
```

##### 7.2.5.1.add()

将一个具体的单层神经网络加入模型

```python
model.add(layer)# keras layer对象实例
```

##### 7.2.5.2.output_shape()

返回模型的输出层形状

```python
print(model.output_shape())
```

### 7.3.ones_like()

创建一个全1的张量

```python
import tensorflow as tf
tensor = tf.ones_like(input)# 张量
```

### 7.4.random

#### 7.4.1.normal()

生成一个正态分布的张量

```python
import tensorflow as tf
tensor = tf.random.normal(shape)# 张量的形状
```

### 7.5.zeros_like()

创建一个全0的张量

```python
import tensorflow as tf
tensor = tf.zeros_like(input)# 张量
```

## 8.tensorflow.js@0.x

### 8.1.dispose()

手动释放显存，推荐使用tf.tidy()

```javascript
import * as tf from '@tensorflow/tfjs';
const t = tf.tensor([1, 2]);
t.dispose();
```

### 8.2.fromPixels()

从一张图像创建一个三维张量

```javascript
import * as tf from '@tensorflow/tfjs';
const image = tf.fromPixels(pixels, numChannels);
/*
pixels 输入图像
numChannels 输入图像的通道数（可选）
*/
```

### 8.3.image

#### 8.3.1.resizeBilinear()

使用双线性法改变图片的尺寸

```javascript
import * as tf from '@tensorflow/tfjs';
const resized = tf.image.resizeBilinear(images, size, alignCorners);
/*
images 输入图像的张量 tf.Tensor3D|tf.Tensor4D|TypedArray|Array  
size 改变后尺寸 [number, number]
alignCorners 布尔值，对齐角落（可选）
*/
```

### 8.4.layers

#### 8.4.1.dense()

全连接层

```javascript
tf.layers.dense({units, activation, inputShape});
/*
units 整数，神经元数量
activation 激活函数 relu'|'sigmoid'|'softmax'|'tanh'
inputShape 此参数只在模型第一层使用
*/
```

### 8.5.loadFrozenModel()

通过url加载固化的模型（异步执行）

```javascript
import * as tf from '@tensorflow/tfjs';
var cmodel;
tf.loadFrozenModel(modelUrl, weightsManifestUrl).then((model) => {cmodel = model;});
/*
modelUrl pb模型的url
weightsManifestUrl json权重的url（可选）
*/
```

### 8.6.scalar()

创建一个标量（tf.tensor()可替代）

```javascript
import * as tf from '@tensorflow/tfjs';
const s = tf.scalar(value, dtype);
/*
value 标量的值 number|boolean|string|Uint8Array
dtype 数据类型（可选） 'float32'|'int32'|'bool'|'complex64'|'string'
*/
```

### 8.7.sequential()

构建一个线性堆叠的网络模型，模型拓扑是简单的层“堆栈”，没有分支或跳过。

```javascript
import * as tf from '@tensorflow/tfjs';
const model = tf.sequential(tf.layers.dense({}));
```

#### 8.7.1.add()

将一个具体的单层神经网络加入模型

```javascript
// 全连接层
model.add(tf.layers.dense({units, activation, inputShape}));
```

#### 8.7.2.compile()

用于配置训练模型

```javascript
model.compile(args);// args 配置参数包括optimizer、loss、metrics
```

#### 8.7.3.fit()

以给定批次训练模型

```javascript
model.fit(x, y, args);
/*
x 训练数据
y 标签
args(可选) batchSize 批次大小，默认32
          epochs 训练轮数
          verbose 日志显示模式 0=安静模型 1=进度条 2每轮显示，默认1
          callbacks 回调
          validationSplit 浮点数，验证集可以从训练集中划分
          validationData 元组[x,y] 验证集数据可以直接指定，会直接覆盖validationSplit
*/
```

#### 8.7.4.predict()

生成预测结果

```javascript
model.predict(x);// 测试数据，需要是张量或者张量数组
```

#### 8.7.5.summary()

查看模型的各层参数

```javascript
model.summary();
```

### 8.8.ones()

创建一个元素值全为一的张量

```javascript
import * as tf from '@tensorflow/tfjs';
const t = tf.ones(shape, dtype);
/*
shape 张量的形状
dtype 数据类型（可选）'float32'|'int32'|'bool'|'complex64'|'string'
*/
```

### 8.9.tensor()

创建一个张量，注意张量的值一经创建不可改变

```javascript
import * as tf from '@tensorflow/tfjs';
const t = tf.tensor(values, shape, dtype);
/*
values 张量的值 
shape 张量的形状（可选）
dtype 数据类型（可选）'float32'|'int32'|'bool'|'complex64'|'string'
*/
```

#### 8.9.1.dataSync()

同步数据，此时阻塞线程直到同步完成

```javascript
const new_t = t.dataSync();
```

#### 8.9.2.expandDims()

增加张量的维度

```javascript
t.expandDims(axis);// 维度（可选）
```

#### 8.9.3.toFloat()

将张量的数据类型转换为float32

```javascript
t.toFloat();
```

### 8.10.tensor1d()

创建一个一维张量（tf.tensor()可替代）

```javascript
import * as tf from '@tensorflow/tfjs';
const t = tf.tensor1d(values, dtype);
/*
values 张量的值 
dtype 数据类型（可选）'float32'|'int32'|'bool'|'complex64'|'string'
*/
```

###8.11.tensor2d()

创建一个二维张量（tf.tensor()可替代）

```javascript
import * as tf from '@tensorflow/tfjs';
const t = tf.tensor2d(values, dtype);
/*
values 张量的值 
dtype 数据类型（可选）'float32'|'int32'|'bool'|'complex64'|'string'
*/
```

### 8.12.tidy()

执行传入的函数后，自动清除除返回值以外的系统分配的所有的中间张量，防止内存泄露

```javascript
import * as tf from '@tensorflow/tfjs';
const result = tf.tidy(fn);// 传入一个箭头函数
```

### 8.13.train

#### 8.13.1adam()

Adam优化器

```javascript
import * as tf from '@tensorflow/tfjs';
optimizer = tf.train.adam(learningRate);// 学习率
```

### 8.14.variable()

创建一个变量

```javascript
import * as tf from '@tensorflow/tfjs';
const v = tf.variable(initialValue, trainable, name, dtype);
/*
initialValue 初始值，必须是一个tf.Tensor
trainable 可训练的（可选）'bool'
name 名称（可选）
dtype 数据类型（可选）'float32'|'int32'|'bool'|'complex64'|'string'
*/
```

#### 8.14.1.assign()

给变量赋予新值

```javascript
v.assign(newValue);//newValue 新值，必须是一个tf.Tensor
```

#### 8.14.2.print()

输出变量的值在控制台

```javascript
v.print();
```

### 8.15.zeros()

创建一个元素值全为零的张量

```javascript
import * as tf from '@tensorflow/tfjs';
const t = tf.zeros(shape, dtype);
/*
shape 张量的形状
dtype 数据类型（可选）'float32'|'int32'|'bool'|'complex64'|'string'
*/
```

## 9.matplotlib

### 9.1.pyplot

#### 9.1.1.plot()

绘制函数

```python
import matplotlib
matplotlib.pyplot.plot(x,# 自变量的值
                       y)# 因变量的值
```

#### 9.1.2.scatter()

绘制散点图

```python
import matplotlib
matplotlib.pyplot.scatter(x,# X轴数据
                          y)# Y轴数据
```

####9.1.3.show()

显示图像

## 10.numpy

### 10.1.argmax()

返回最大值的索引

```python
import numpy as np
a = [1, 2, 3]
max = np.argmax(a)# 输入可以是lists, tuples, ndarrays
```

### 10.2.asarray()

将输入转化为ndarray

```python
import numpy as np
a = [[1, 2, 3]]
a = np.asarray(a,# 输入可以是lists, tuples, ndarrays
               dtype)# 数据类型，可选
```

### 10.3.astype()

强制转换成新的数据类型

```python
import numpy as np
a = [1.0, 2.0]
new_a = a.astype(dtype)# 数据类型
```

### 10.4.expand_dims()

增加ndarray的维度

```python
import numpy as np
a = [[1, 2], [3, 4]]
a = np.expand_dims(a,# 输入可以是lists, tuples, ndarrays
                   axis)# 维度
```

###10.5.linspace()

生成一个等差数列

```python
import numpy as np
a = np.linspace(start,# 序列的起始值 
                stop,# 序列的结束值
                num)# 生成样本数，默认50
```

### 10.6.load()

从npy或者npz文件中加载数组

```python
import numpy as np
np.load(file,# 文件路径
        allow_pickle,# 使用pickle，默认False
        encoding)# 编码格式，默认ASCII
```

###10.7.mat()

从列表或者数组生成一个矩阵对象

```python
import numpy as np
a = [[1, 2, 3]]
a = np.mat(a)
```

###10.8.matmul()

矩阵乘法

```python
import numpy as np
a1 = [[1, 2, 3]]
a2 = [[1], [2], [3]]
a = np.matmul(a1, a2)
```

### 10.9.mean()

按照指定的维度计算算术平均值

```python
import numpy as np
np.mean(a,# 待计算均值的列表、矩阵
				axis)# 维度
```

###10.10.transpose()

对矩阵进行转置

```python
import numpy as np
a = [[1, 2], [3, 4]]
a_t = np.transpose(a)
```

### 10.11.reshape()

在不改变数据内容的情况下，改变数据形状

```python
import numpy as np
a = [1, 2, 3, 4]
a = np.asarray(a)
a.reshape((2, 2))# 将a转换成2行2列的二维数组
b = [[1, 2], [3, 4]]
b = np.asarray(b)
b = b.reshape((-1, 2, 1))# 第一个为-1，将按照后面的输入增加一个维度
```

### 10.12.split()

将张量按某个维度拆分成多个张量

```python
import numpy as np
tensor = np.asarray([[1, 2, 5, 6], [3, 4, 7, 8]])
tensor_list = np.split(ary=tensor,# 需要拆分的张量 
                       indices_or_sections=2,# 要拆分的数量
                       axis=1)# axis按某个维度拆分
```

## 11.pandas

###11.1.DataFrame()

将其他数据格式转换为DataFrame

```python
import pandas as pd
df = {'index': [0, 1, 2], 'value': [1, 2, 3]}
df = pd.DataFrame(df)
```

#### 11.1.1.replace()

新值替换旧值

```python
df.replace(to_replace,# 旧值
           value,# 新值
           inplace)# 布尔值，默认False，修改源文件
```

### 11.2.read_csv()

读取csv文件，返回一个DataFrame对象

```python
import pandas as pd
df = pd.read_csv(filepath_or_buffer, # 文件或者缓冲区路径
                 header)# 列名，默认是0，否则是None
```

### 11.3.to_csv()

将DataFrame生成csv文件

```python
import pandas as pd
df.to_csv(path_or_buf,# 保存的文件和路径
          header,# 列名，默认是True
          index,# 索引，默认是True
          encoding)# 编码方式，默认是‘utf-8’
```

