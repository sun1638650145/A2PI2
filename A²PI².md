#A²PI² version2

第一版的文档的格式有些混乱，尤其是keras引用的地方，现在做修正,

文档主要是Python描述，涉及小部分JavaScript语言

#1.imageio

##1.1.imread()

读取图片并返回一个numpy数组

```python
import imageio
image = imageio.imread(uri)# 文件对象或http地址，图片的路径
```

#2.keras

keras在tensorflow r1.x和r2.1以及plaidml中版本均是2.2.4；除了模型初始化的策略有些许不同，不能混合续训，但是代码完全兼容，且可以混用执行预测和部署

##2.1.applications

###2.1.1.inception_v3

####2.1.1.1.InceptionV3()

InceptionV3的预训练模型

```python
from keras.applications.inception_v3 import InceptionV3
model = InceptionV3(include_top,# 是否包含全连接的输出层
                    weights,# 权重，可以是随机初始化，也可以加载'imagenet'的权重，或者自定权重的路径
                    input_tensor)# 输入层，需要使用keras.layers.Input()
```

###2.1.2.resnet50

####2.1.2.1.ResNet50()

ResNet50的预训练模型

```python
from keras.applications.resnet50 import ResNet50
model = ResNet50(include_top,# 是否包含全连接的输出层
                 weights,# 权重，可以是随机初始化，也可以加载'imagenet'的权重，或者自定权重的路径
                 input_tensor)# 输入层，需要使用keras.layers.Input()
```

###2.1.3.xception

####2.1.3.1.Xception()

Xception的预训练模型

```python
from keras.applications.xception import Xception
model = Xception(include_top,# 是否包含全连接的输出层
                 weights,# 权重，可以是随机初始化，也可以加载'imagenet'的权重，或者自定权重的路径
                 input_tensor)# 输入层，需要使用keras.layers.Input()
```

##2.2.backend

###2.2.1.ones_like()

创建一个全1的张量

```python
from keras.backend import ones_like
tensor = ones_like(x)# 张量
```

###2.2.2.zeros_like()

创建一个全0的张量

```python
from keras.backend import zeros_like
tensor = zeros_like(x)# 张量
```

## 2.3.callbacks

###2.3.1.ModelCheckpoint()

保存模型（主要是断点）

```python
from keras.callbacks import ModelCheckpoint
callbacks = [ModelCheckpoint(filepath,# 模型保存的路径
                             period)]# 保存间隔
```

## 2.4.datasets

### 2.4.1.mnist

keras自带的数据集之一

####2.4.1.1.load_data()

加载mnist数据集

```python
from keras import mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

## 2.5.layers

keras提供Model（函数式API）和Sequential（普通API），如果函数名相同则不赘述

### 2.5.1.Add()

将多个层通过矩阵加法合并

```python
from keras.layers import Add
layer = Add()(_Merge)# 相同形状的张量（层）列表
```

###2.5.2.BatchNormalization()

批量标准化层

```python
from keras.layers import BatchNormalization
layer = BatchNormalization()
```

###2.5.3.Concatenate()

连接输入的层

```python
from keras.layers import Concatenate
layer = Concatenate(axis)(_Merge)
"""
axis 维度
_Merge 张量（层）列表
"""
```

###2.5.4.Conv1D()

1D卷积层（例如时序卷积）

```python
from keras.layers import Conv1D
layer = Conv1D(filters,# 整数，卷积核的数量 
               kernel_size,# 整数，卷积核的大小
               strides,# 整数、或者列表或者元组，滑动步长
               padding,# 'same''valid''causal'，是否使用全零填充
               data_format,# 'channels_first''channels_last' 数据格式CBHW或BHWC
               activation,# 'relu''sigmoid' 激活函数
               use_bias,# 布尔值 是否使用偏置
               kernel_initializer,# 权值初始化
               bias_initializer)# 偏置项初始化
```

###2.5.5.Conv2D()

2D卷积层（例如时序卷积）

```python
from keras.layers import Conv2D
layer = Conv2D(filters,# 整数，卷积核的数量 
               kernel_size,# 整数，卷积核的大小
               strides,# 整数、或者列表或者元组，滑动步长
               padding,# 'same''valid'，是否使用全零填充
               input_shape)# 元组，第一层需要指定输入
```

###2.5.6.Conv2DTranspose()

转置卷积层 (有时被成为反卷积)，将具有卷积输出尺寸的东西 转换为具有卷积输入尺寸的东西

```python
from keras.layers import Conv2DTranspose
layer = Conv2DTranspose(filters,# 整数，输出空间的维度
                    		kernel_size,# 整数，卷积核的大小
                    		strides,# 整数、或者列表或者元组，滑动步长
                    		padding,# 'same''valid''，是否使用全零填充
                    		use_bias)# 布尔值 是否使用偏置
```

###2.5.7.Dense()

全连接层

```python
from keras.layers import Dense
layer = Dense(units,# 整数，神经元数量
              use_bias,# 布尔值 是否使用偏置
              input_shape)# 元组，第一层需要指定输入
```

###2.5.8.Dropout()

在训练阶段按照比例随机丢弃神经元

```python
from keras.layers import Dropout
layer = Dropout(rate)# 丢弃率
```

###2.5.9.Flatten()

将输入展平，不影响批次大小

```python
from keras.layers import Flatten
layer = Flatten()
```

###2.5.10.Input()

输入层

```python
from keras.layers import Input
input_tensor = Input(shape)# 整数，形状元组
```

### 2.5.11.LeakyReLU()

带泄漏的 ReLU层

```python
from keras.layers import LeakyReLU
layer = LeakyReLU(alpha)# 负斜率系数，默认为0.3
```

###2.5.12.MaxPooling1D()

对时序数据进行最大池化

```python
from keras.layers import MaxPooling1D
layer = MaxPooling1D(pool_size,# 整数，池化核数量
										 strides,# 整数、或者列表或者元组，滑动步长
                     padding)# 'same''valid''causal'，是否使用全零填充
```

###2.5.13.Reshape()

将输入重新调整为特定的尺寸

```python
from keras.layers import Reshape
layer = Reshape(target_shape)# 整数元组，目标尺寸
```

## 2.6.losses

###2.6.1.BinaryCrossentropy()

计算真实标签和预测值标签的的交叉熵损失

```python
from keras.losses import BinaryCrossentropy
cross_entropy = BinaryCrossentropy(from_logits)# 是否将y_pred解释为张量
```

## 2.7.models

###2.7.1.Model()

keras自定义模型对象

```python
from keras.models import Model
model = Model(inputs,# 输入层
              outputs)# 输出层
```

####2.7.1.1.fit_generator()

生成批次训练数据，按批训练数据

```python
model.fit_generator(generator,# 数据生成器，比如ImageDataGenerator()
                    steps_per_epoch,# 整数，每批次步数
                    epochs,# 整数，轮数
                    verbose)# 日志显示模式 0=安静模型 1=进度条 2每轮显示
```

####2.7.1.2.load_model()

加载模型

```python
from keras.models import load_model
model = load_model(filepath)# 文件路径，可以是saved model或者h5py
```

####2.7.1.3.save()

将模型保存为SavedModel或者HDF5文件

```python
model.save(filepath)# 保存路径
```

###2.7.2.Sequential()

构建一个线性堆叠的网络模型

```python
from keras.models import Sequential
model = Sequential()
```

####2.7.2.1.add() 

将一个具体的单层神经网络加入模型

```python
model.add(layer)
```

####2.7.2.2.compile()

用于配置训练模型

```python
model.compile(optimizer,# 优化器
              loss,# 损失函数
              metrics)# 评估标准['accuracy']
```

####2.7.2.3.evaluate()

在测试模式下返回损失值和准确率

```python
model.evaluate(x,# 训练数据
               y,# 标签
               batch_size,# 整数，批次大小，默认32
               verbose)# 日志显示模式 0=安静模型 1=进度条 2每轮显示
```

####2.7.2.4.fit()

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

####2.7.2.5.load_weights()

加载所有的神经网络层的参数

```python
model.load_weights(filepath)# 检查点文件路径
```

####2.7.2.6.predict()

生成预测结果

```python
model.predict(x,# 测试数据
              batch_size,# 整数，批次大小，默认32
              verbose)# 日志显示模式 0=安静模型 1=进度条
```

#### 2.7.2.7.output_shape()

返回模型的输出层形状

```python
print(model.output_shape())
```

####2.7.2.8.summary()

查看模型的各层参数

```python
model.summary()
```

##2.8.optimizers

###2.8.1.Adam()

Adam优化器

```python
from keras.optimizers import Adam
optimizer = Adam(lr)# 学习率
```

### 2.8.2.apply_gradients()

将梯度带计算出的梯度赋给优化器

```python
optimizer.apply_gradients(grads_and_vars)# 对应梯度和变量组成的列表
```

## 2.9.preprocessing

### 2.9.1.image

####2.9.1.1.ImageDataGenerator()

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

#####2.9.1.1.1.flow_from_directory()

从给定路径读入数据并增强

```python
data_generator.flow_from_directory(directory,# 路径
                                   target_size,# 元组，调整后大小
                                   class_mode,# 返回标签数组类型，默认'categorical'
                                   batch_size,# 整数，批次大小，默认32
                                   shuffle,# 布尔值，打乱顺序
                                   interpolation)# 插值，调整尺寸 'nearest''bilinear''bicubic'
```

######2.9.1.1.1.1.class_indices

返回训练数据的索引

```python
from keras.preprocessing.image import ImageDataGenerator
batches = ImageDataGenerator().flow_from_directory()
print(bathes.class_indices)
```

####2.9.1.2.load_image()

按照PIL的格式加载图像

```python
from keras.preprocessing import image
img = image.load_img(path,# 路径
                     target_size)# 元组，调整后大小
```

##2.10.utils

###2.10.1.multi_gpu_model()

多GPU并行训练模型

```python
from keras.utils import multi_gpu_model
parallel_model = multi_gpu_model(model,# 模型
                                 gpus)# 整数（大于等于2），并行GPU数量
```

#3.PIL

##3.1.Image

###3.1.1.fromarray()

从输入的图片返回一个数组

```python
from PIL import Image
array = Image.fromarray(obj)# 图片对象
```

###3.1.2.resize()

返回调整大小后的图像的副本

```python
from PIL import Image
new_image = image.resize(size)# 有宽度和高度的二元组
```

#4.sklearn

##4.1.linear_model

###4.1.1.LogisticRegression()

构建一个对数几率回归模型

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```

####4.1.1.1.fit()

以给定数据训练模型

```python
model.fit(X,# 训练数据 
          y,# 标签 
          sample_weight)# 数组，类别权重，默认为None
```

####4.1.1.2.predict()

生成预测结果

```python
model.predict(X)# 测试数据
```

##4.2.metrics

###4.2.1.accuracy_score()

计算准确率

```python
import sklearn
sklearn.metrics.accuracy_score(y_true,# 真实标签
                               y_pred,# 预测结果
                               sample_weight)# 数组，类别权重，默认为None
```

##4.3.model_selection

###4.3.1.cross_val_predict()

使用交叉验证法验证

```python
import sklearn
sklearn.model_selection.cross_val_predict(estimator,# 训练的模型对象
                                          X,# 训练数据
                                          y,# 标签
                                          cv)# 整数，划分数，默认为3
```

###4.3.2.LeaveOneOut()

使用留一法验证

```python
import sklearn
LOO = sklearn.model_selection.LeaveOneOut() # 返回一个BaseCrossValidator对象
```

####4.3.2.1.split()

按照具体BaseCrossValidator对象将数据划分为训练和测试集

```python
LOO.split(X)# 训练数据
```

##4.4.tree

###4.4.1.DecisionTreeClassifier()

生成一个分类决策树实例

```python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion,# 划分条件 'gini''entropy'
                              random_state)
															# 随机状态 默认是None（通过np.random自动生成），也可以赋一个整数
```

####4.4.1.1.fit()

以给定数据训练模型

```python
tree.fit(X,# 训练数据 
         y)# 标签
```

###4.4.2.export_graphviz()

以dot文件导出决策树

```python
from sklearn.tree import export_graphviz
dot_data = export_graphviz(decision_tree,# 决策树
                           out_file,#生成dot文件 默认为None（返回str）
                           feature_names,# 属性名称
                           class_names)# 分类名称
```

#5.tensorflow r1.x

tensorflow r1.x将逐步停止支持，可能只有为了支持tensorflow.js@0.x会用到

##5.1.concats()

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

##5.2.ConfigProto()

用于配置会话的选项

```python
import tensorflow as tf
tf.ConfigProto(gpu_option)# 配置显存
```

##5.3.gfile

###5.3.1.FastGFile()

没有线程锁的文件I/O封装器

```python
import tensorflow as tf
fp = tf.gfile.FastGfile(name,# 保存名称 
                        mode)# 模式，默认'r'
```

##5.4.global_variables()

返回默认会话中所有的全局变量

```python
import tensorflow as tf
var = tf.global_variables()
```

##5.5.global_variables_initializer()

初始化全局的变量

```python
import tensorflow as tf 
init = tf.global_variables_initializer()
```

##5.6.GPUOptions()

使用GPU时对显存使用的控制选项

```python
import tensorflow as tf
gpu_options = tf.GPUOptions(allow_growth)# 布尔值，设置为True时，显存按需分配（tensorflow默认是占用全部显存）
```

##5.7.GraphDef()

tensorflow的信息协议

```python
import tensorflow as tf 
graph_def = tf.GraphDef()
```

###5.7.1.ParseFromString()

将序列化的数据转换成GraphDef

```python
graph_def.ParseFromString(serialized)# 序列化数据，数据流 
```

##5.8.import_graph_def()

将GraphDef实例导入默认的Graph计算图

```python
import tensorflow as tf
tensor = tf.import_graph_def(graph_def,# graph_def实例
                     				 input_map,# 字典，输入张量的名称和对应的张量
                     				 return_elements)# 字符串列表，输出张量的名称
```

##5.10.nn

###5.10.1.avg_pool()

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

###5.10.2.dropout()

在训练阶段按照比例随机丢弃神经元

```python
import tensorflow as tf 
tf.nn.dropout(x,# 输入张量
              keep_prob,# 保留概率
							name)# 整数
```

###5.10.3.lrn()

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

###5.10.4.max_pool()

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

###5.10.5.softmax()

softmax激活函数

```python
import tensorflow as tf
tf.nn.softmax(logits)# 输入张量（非空）
```

##5.11.placeholder()

添加一个占位符

```python
import tensorflow as tf
x = tf.placeholder(dtype,# 自负床数据类型
                   shape,# 张量形状
                   name)# 名称
```

##5.12.python

###5.12.1.framework

####5.12.1.1.graph_util

#####5.12.1.1.1.convert_variables_to_constants

将计算图中的变量转换为常量

```python
from tensorflow.python.framework.graph_util import convert_variables_to_constants
output_graph = convert_variables_to_constants(sess,# 需要转换变量的会话
                                   						input_graph_def,# 会话的graph_def对象
                                   						output_node_names)#字符串列表，输出层的名称
```

######5.12.1.1.1.1.SerializeToString()

将protobuf数据转换为二进制字符串

```python
# GraphDef就是一种protobuf
graphdef.SerializeToString()
```

##5.13.saved_model

###5.13.1.builder

####5.13.1.1.SavedModelBuilder()

构建一个生成SavedModel的实例

```python
from tensorflow.saved_model import builder
builder = builder.SavedModelBuilder(export_dir)# SavedModel的保存路径
```

#####5.13.1.1.1.add_meta_graph_and_variables()

添加图结构和变量信息

```python
builder.add_meta_graph_and_variables(sess,# 会话
                                   	 tags,# 标签，自定义
                                   	 signature_def_map)# 预测签名字典
```

#####5.13.1.1.2.save()

将SaveModel写入磁盘

```python
builder.save()
```

###5.13.2.loader

####5.13.2.1.load()

从标签指定的SavedModel加载模型

```python
import tensorflow as tf
tf.saved_model.loader.load(sess,# 模型还原到的会话
                           tags,# 字符串，标签，参见tf.saved_model.tag_constants
                           export_dir)# 待还原的SavedModel目录
```

###5.13.3.signature_def_utils

####5.13.3.1.predict_signature_def()

构建预测签名

```python
from tensorflow.saved_model import signature_def_utils
signature = signatures_def_utils.predict_signature_def(inputs,# 字典，输入变量
                                                       outputs)# 字典，输出变量
```

###5.13.4.simple_save()

使用简单方法构建SavedModel用于服务器

```python
from tensorflow.saved_model import simple_save
simple_save(session,# 会话
            export_dir,# SavedModel的保存目录
            inputs,# 字典，输入变量
            outputs)# 字典，输出变量
```

###5.13.5.tag_constants

SaveModel的标签

```python
from tensorflow.saved_model import tag_constants
tags = tag_constants.SERVING
"""
标签有TPU SERVING GPU TRAINING
"""
```

##5.14.Session()

生成一个tensorflow的会话

```python
import tensorflow as tf
sess = tf.Session(config)# 使用ConfigProto配置会话
```

###5.14.1.close()

关闭当前会话

```python
sess.close()
```

###5.14.2.graph

####5.14.2.1.get_tensor_by_name()

根据名称返回张量，可以使用多个线程同时调用

```python
sess.graph.get_tensor_by_name(name)# 张量的名称
```

###5.14.3.run()

运行传入会话的操作，返回结果张量

```python
sess.run(fetches,# 待计算的操作 'Operation''Tensor' 
         feed_dict# 输入的值，默认为None
```

##5.15.split()

将张量按某个维度拆分成多个张量

```python
import tensorflow as tf
tensor = [[1, 2, 5, 6], [3, 4, 7, 8]]
tensor_list = tf.split(value=tensor, num_or_size_splits=2, axis=1, name="split")
"""
value需要拆分的张量，num_or_size_splits要拆分的数量，axis按某个维度拆分，name张量的名字
"""
```

##5.16.train

###5.16.1.AdamOptimizer()

Adam优化器

```python
import tensorflow as tf
optimizer = tf.train.AdamOptimizer(learning_rate)# 学习率
```

###5.16.2.GradientDescentOptimizer()

梯度下降优化器

```python
import tensorflow as tf
optimizer = tf.train.GradientDescentOptimizer(learning_rate)# 学习率
```

###5.16.3.latest_checkpoint()

查找最近的保存点文件

```python
import tensorflow as tf
checkpoint = tf.train.latest_checkpoint(checkpoint_dir)# 保存点路径
```

###5.16.3.Saver()

生成用于保存和还原计算图的实例

```python
import tensorflow as tf
saver = tf.train.Saver(var_list)
# 将被保存和恢复的变量列表或者变量字典，默认为None（保存全部） 
```

####5.16.3.1.restore()

恢复保存的变量

```python
saver.restore(sess,# 会话，eager模式为None
              save_path)# 检查点文件的路径
```

##5.17.variable_scope()

用于定义变量操作的上下文管理器

``` python
import tensorflow as tf
with tf.variable_scope(name_or_scope):# 字符串，作用域
```

#6.tensorflow r2.x

##6.1.data

###6.1.1.Datasets

####6.1.1.1.batch()

给数据集划分批次

```python
import tensorflow as tf
dataset = tf.data.Dataset.range(6)
dataset_shuffle = dataset.batch(batch_size=3)# 批次的大小
```

####6.1.1.2.from_tensor_slices()

返回一个数据集对象

```python
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices(tensor)# 张量，必须有相同的第一维
```

####6.1.1.3.shuffle()

随机打乱数据集对象的元素

```python
import tensorflow as tf
dataset = tf.data.Dataset.range(3)
dataset_shuffle = dataset.shuffle(buffer_size=3)# 数据集元素的数量
```

##6.2.GradientTape()

创建一个梯度带，可用于自动求导

```python
import tensorflow as tf
with tf.GradientTape() as tape:
		# 在tf.GradientTape()的上下文管理器内的计算都将被用于求导
```

###6.2.1.gradient()

在梯度带内计算导数

```python
		grad = tape.gradient(target, sources)# 计算target关于sources的导数
```

##6.4.ones_like()

创建一个全1的张量

```python
import tensorflow as tf
tensor = tf.ones_like(input)# 张量
```

##6.5.random

###6.5.1.normal()

生成一个正态分布的张量

```python
import tensorflow as tf
tensor = tf.random.normal(shape)# 张量的形状
```

##6.6.zeros_like()

创建一个全0的张量

```python
import tensorflow as tf
tensor = tf.zeros_like(input)# 张量
```

#7.tensorflow.js@0.x

tensorflow.js@0.x版本必须和tensorflow r1.x生成的SavedModel模型配套，不兼容tensorflow r2.x

##7.1.dispose()

手动释放显存，推荐使用tf.tidy()

```javascript
import * as tf from '@tensorflow/tfjs';
const t = tf.tensor([1, 2]);
t.dispose();
```

##7.2.fromPixels()

从一张图像创建一个三维张量

```javascript
import * as tf from '@tensorflow/tfjs';
const image = tf.fromPixels(pixels, numChannels);
/*
pixels 输入图像
numChannels 输入图像的通道数（可选）
*/
```

##7.3.image

###7.3.1.resizeBilinear()

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

##7.4.layers

###7.4.1.dense()

全连接层

```javascript
tf.layers.dense({units, activation, inputShape});
/*
units 整数，神经元数量
activation 激活函数 relu'|'sigmoid'|'softmax'|'tanh'
inputShape 此参数只在模型第一层使用
*/
```

##7.5.loadFrozenModel()

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

##7.6.scalar()

创建一个标量（tf.tensor()可替代）

```javascript
import * as tf from '@tensorflow/tfjs';
const s = tf.scalar(value, dtype);
/*
value 标量的值 number|boolean|string|Uint8Array
dtype 数据类型（可选） 'float32'|'int32'|'bool'|'complex64'|'string'
*/
```

##7.7.sequential()

构建一个线性堆叠的网络模型，模型拓扑是简单的层“堆栈”，没有分支或跳过。

```javascript
import * as tf from '@tensorflow/tfjs';
const model = tf.sequential(tf.layers.dense({}));
```

###7.7.1.add()

将一个具体的单层神经网络加入模型

```javascript
// 全连接层
model.add(tf.layers.dense({units, activation, inputShape}));
```

###7.7.2.compile()

用于配置训练模型

```javascript
model.compile(args);// args 配置参数包括optimizer、loss、metrics
```

###7.7.3.fit()

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

###7.7.4.predict()

生成预测结果

```javascript
model.predict(x);// 测试数据，需要是张量或者张量数组
```

###7.7.5.summary()

查看模型的各层参数

```javascript
model.summary();
```

##7.8.ones()

创建一个元素值全为一的张量

```javascript
import * as tf from '@tensorflow/tfjs';
const t = tf.ones(shape, dtype);
/*
shape 张量的形状
dtype 数据类型（可选）'float32'|'int32'|'bool'|'complex64'|'string'
*/
```

##7.9.tensor()

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

###7.9.1.dataSync()

同步数据，此时阻塞线程直到同步完成

```javascript
const new_t = t.dataSync();
```

###7.9.2.expandDims()

增加张量的维度

```javascript
t.expandDims(axis);// 维度（可选）
```

###7.9.3.toFloat()

将张量的数据类型转换为float32

```javascript
t.toFloat();
```

##7.10.tensor1d()

创建一个一维张量（tf.tensor()可替代）

```javascript
import * as tf from '@tensorflow/tfjs';
const t = tf.tensor1d(values, dtype);
/*
values 张量的值 
dtype 数据类型（可选）'float32'|'int32'|'bool'|'complex64'|'string'
*/
```

##7.11.tensor2d()

创建一个二维张量（tf.tensor()可替代）

```javascript
import * as tf from '@tensorflow/tfjs';
const t = tf.tensor2d(values, dtype);
/*
values 张量的值 
dtype 数据类型（可选）'float32'|'int32'|'bool'|'complex64'|'string'
*/
```

##7.12.tidy()

执行传入的函数后，自动清除除返回值以外的系统分配的所有的中间张量，防止内存泄露

```javascript
import * as tf from '@tensorflow/tfjs';
const result = tf.tidy(fn);// 传入一个箭头函数
```

##7.13.train

###7.13.1adam()

Adam优化器

```javascript
import * as tf from '@tensorflow/tfjs';
optimizer = tf.train.adam(learningRate);// 学习率
```

##7.14.variable()

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

###7.14.1.assign()

给变量赋予新值

```javascript
v.assign(newValue);//newValue 新值，必须是一个tf.Tensor
```

###7.14.2.print()

输出变量的值在控制台

```javascript
v.print();
```

##7.15.zeros()

创建一个元素值全为零的张量

```javascript
import * as tf from '@tensorflow/tfjs';
const t = tf.zeros(shape, dtype);
/*
shape 张量的形状
dtype 数据类型（可选）'float32'|'int32'|'bool'|'complex64'|'string'
*/
```

#8.matplotlib

##8.1.pyplot

###8.1.1.plot()

绘制函数

```python
import matplotlib
matplotlib.pyplot.plot(x,# 自变量的值
                       y)# 因变量的值
```

###8.1.2.scatter()

绘制散点图

```python
import matplotlib
matplotlib.pyplot.scatter(x,# X轴数据
                          y)# Y轴数据
```

###8.1.3.show()

显示图像

#9.numpy

##9.1.argmax()

返回最大值的索引

```python
import numpy as np
a = [1, 2, 3]
max = np.argmax(a)# 输入可以是lists, tuples, ndarrays
```

##9.2.asarray()

将输入转化为ndarray

```python
import numpy as np
a = [[1, 2, 3]]
a = np.asarray(a,# 输入可以是lists, tuples, ndarrays
               dtype)# 数据类型，可选
```

##9.3.astype()

强制转换成新的数据类型

```python
import numpy as np
a = [1.0, 2.0]
new_a = a.astype(dtype)# 数据类型
```

##9.4.expand_dims()

增加ndarray的维度

```python
import numpy as np
a = [[1, 2], [3, 4]]
a = np.expand_dims(a,# 输入可以是lists, tuples, ndarrays
                   axis)# 维度
```

##9.5.linspace()

生成一个等差数列

```python
import numpy as np
a = np.linspace(start,# 序列的起始值 
                stop,# 序列的结束值
                num)# 生成样本数，默认50
```

##9.6.load()

从npy或者npz文件中加载数组

```python
import numpy as np
np.load(file,# 文件路径
        allow_pickle,# 使用pickle，默认False
        encoding)# 编码格式，默认ASCII
```

##9..7.mat()

从列表或者数组生成一个矩阵对象

```python
import numpy as np
a = [[1, 2, 3]]
a = np.mat(a)
```

##9.8.matmul()

矩阵乘法

```python
import numpy as np
a1 = [[1, 2, 3]]
a2 = [[1], [2], [3]]
a = np.matmul(a1, a2)
```

##9.9.mean()

按照指定的维度计算算术平均值

```python
import numpy as np
np.mean(a,# 待计算均值的列表、矩阵
				axis)# 维度
```

##9.10.transpose()

对矩阵进行转置

```python
import numpy as np
a = [[1, 2], [3, 4]]
a_t = np.transpose(a)
```

##9.11.random

###9.11.1.normal()

生成一个正态分布的数组

```python
import numpy as np
a = np.random.normal(size=[2,3])# 形状
```

##9.12.reshape()

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

##9.13.split()

将张量按某个维度拆分成多个张量

```python
import numpy as np
tensor = np.asarray([[1, 2, 5, 6], [3, 4, 7, 8]])
tensor_list = np.split(ary=tensor,# 需要拆分的张量 
                       indices_or_sections=2,# 要拆分的数量
                       axis=1)# axis按某个维度拆分
```

#10.pandas

##10.1.DataFrame()

将其他数据格式转换为DataFrame

```python
import pandas as pd
df = {'index': [0, 1, 2], 'value': [1, 2, 3]}
df = pd.DataFrame(df)
```

###10.1.1.replace()

新值替换旧值

```python
df.replace(to_replace,# 旧值
           value,# 新值
           inplace)# 布尔值，默认False，修改源文件
df.replace(dict)# 可以是字典的键为旧值，值为新值，进行替换
```

##10.2.read_csv()

读取csv文件，返回一个DataFrame对象

```python
import pandas as pd
df = pd.read_csv(filepath_or_buffer,# 文件或者缓冲区路径
                 header,# 列名，默认是0，否则是None
                 index_col)# 指定索引列 ，默认是None
```

##10.3.Series()

###10.3.1.values()

将DataFrame的值转换为ndarry

```python
import pandas as pd
df = {'index': [0, 1, 2], 'value': [1, 2, 3]}
df = pd.DataFrame(df)
array = df['index'].values
```

##10.4.to_csv()

将DataFrame生成csv文件

```python
import pandas as pd
df.to_csv(path_or_buf,# 保存的文件和路径
          header,# 列名，默认是True
          index,# 索引，默认是True
          encoding)# 编码方式，默认是‘utf-8’
```

#11.pydot

##11.1.graph_from_dot_data()

从dot字符串中加载图像

```python
import pydot
graph = pydot.graph_from_dot_data(s)# dot字符串，返回一个list
```

##11.2.graph_from_dot_file()

从dot文件中加载图像

```python
import pydot
graph = pydot.graph_from_dot_data(path)# dot文件路径，返回一个list
```

##11.3.Dot

###11.3.1.write_png()

生成png图像

```python
graph.write_png(path)# png图像的路径
```

