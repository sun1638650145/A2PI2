# <center>A²PI²-API version2.1</center>

* 2.0版本的包有些混乱，而且有很多不合理的地方，在2.1版本中进行解决
* 2.1版本中会增加代码实现的Demo的链接，便于理解
* Keras 2.3开始移除了多后端的支持，这一版的Keras移回TensorFlow的二级目录下
* 移除TensorFLow r1.x 
* Python格式规范化
  1. 类或函数功能|有无返回值
  2. 每个参数将按照意义|数据类型|默认值(枚举)|是否为可选参数
* JavaSrcipt格式规范化
  1. 类或函数功能|有无返回值
  2. //参数:意义(数据类型)|是否可省略;参数:意义(数据类型)|是否可省略;......

* 在Github上提供PDF格式的Releases，显著减小仓库的大小

# 1.catboost

| 版本   | 描述                 | 注意                |
| ------ | -------------------- | ------------------- |
| 0.23.1 | 梯度提升决策树(GBDT) | 可直接在sklearn使用 |

## 1.1.CatBoostClassifier()

实例化一个CatBoost分类器

```python
from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations,  # 迭代次数|int|500
                           learning_rate,  # 学习率|float|0.03
                           depth,  # 树的深度|int|6
                           loss_function,  # 损失函数|string or object|'Logloss'('Logloss'|'CrossEntropy')
                           task_type)  # 用于训练模型的硬件|string|None('CPU'|'GPU')
```

### 1.1.1.fit()

训练CatBoost分类器|self

```python
model.fit(X,  # 特征数据|catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series
          y,  # 标签|list or numpy.ndarray or pandas.DataFrame or pandas.Series|None|可选
          text_features,  # 特征数据的索引列|list or numpy.ndarray|None｜可选
          eval_set,  # 验证集元组列表|list of (X, y) tuple|None|可选
          verbose)  # 日志模式|bool or int|None|可选
```

### 1.1.2.feature_importances_

特征重要度|numpy.ndarray

```python
model.feature_importances_
```

### 1.1.3.feature_name_

特征名称

```python
names = model.feature_names_
```

### 1.1.4.predict()

进行预测|numpy.ndarray

```python
result = model.predict(data)  # 用于预测的数据|catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series or catboost.FeaturesData
```

# 2.cv2

| 版本     | 描述           | 注意                                                         |
| -------- | -------------- | ------------------------------------------------------------ |
| 4.2.0.34 | 图像处理软件库 | 安装时使用pip install opencv-python OpenCV的图片格式是HWC TensorFlow的是WHC |

## 2.1.imread()

加载指定路径的图片|numpy.ndarray

```python
import cv2
image = cv2.imread(filename,  # 要加载的文件的路径|str
                   flags)  # 读入的色彩方式|int or cv::ImreadModes|None
```

## 2.2.resize()

将图像调整到指定大小|numpy.ndarray

```python
import cv2
image = cv2.resize(src,  # 原始的输入图像|numpy.ndarray
                   dsize)  # 输出图像的尺寸|tuple
```

# 3.imageio

| 版本  | 描述           | 注意 |
| ----- | -------------- | ---- |
| 2.9.0 | 图像处理软件库 |      |

## 3.1.imread()

加载指定路径的图片|imageio.core.util.Array

```python
import imageio
image = imageio.imread(uri=filename)  # 要加载的文件的路径|str or pathlib.Path or bytes or file
```

# 4.lightgbm

| 版本  | 描述                 | 注意                                                      |
| ----- | -------------------- | --------------------------------------------------------- |
| 2.3.1 | 基于树的梯度提升框架 | 在macOS下安装需要先使用brew安装libomp 可直接在sklearn使用 |

## 4.1.LGBMClassifier()

实例化一个LGBMClassifier分类器

```python
from lightgbm import LGBMClassifier
model = LGBMClassifier(boosting_type,  # 集成方式|str|'gbdt'('gbdt'|'dart'|'goss'|'rf')|可选
                       max_depth,  # 基学习器的最大深度，负值表示没有限制|int|-1|可选
                       learning_rate,  # 学习率|float|0.1|可选
                       n_estimators)  # 树的数量|int|100|可选
```

### 4.1.1.fit()

训练LGBMClassifier分类器|self

```python
model.fit(X,  # 特征数据|array-like or 形状为[n_samples, n_features]的稀疏矩阵
          y,  # 标签|array-like
          eval_set)  # 验证集元组列表|list of (X, y) tuple|None|可选
```

### 4.1.2.predict()

进行预测|numpy.ndarray

```python
result = model.predict(X)# 用于预测的数据|array-like or 形状为[n_samples, n_features]的稀疏矩阵
```

# 5.matplotlib

| 版本  | 描述             | 注意 |
| ----- | ---------------- | ---- |
| 3.2.1 | Python绘图软件库 |      |

## 5.1.axes

| 版本 | 描述                                             | 注意 |
| ---- | ------------------------------------------------ | ---- |
| -    | axes是matplotlib的图形接口，提供设置坐标系的功能 |      |

### 5.1.1.annotate()

给点进行注释|matplotlib.text.Annotation

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.annotate(s='annotate',  # 注释内容|str
            xy=(0.45, 0.5),  # 注释点的坐标|(float, float)
            xytext=(0.6, 0.6),  # 注释内容的坐标｜(float, float)|None|可选
            xycoords='data',  # 注释点放置的坐标系|str|'data'｜可选
            textcoords='data',  # 注释内容放置的坐标系|str|未设置则和xycoords一致|可选
            arrowprops=dict(arrowstyle='<-'),  # 绘制箭头的样式|dict|None即不绘制箭头|可选
            size=20,  # 注释文本字号|int|10|可选
            verticalalignment='baseline',  # 垂直对齐|str|'baseline'|可选
            horizontalalignment='center',  # 水平对齐|str|'left'|可选
            bbox=dict(fc='skyblue'))  # 绘制文本框|dict|None即不绘制文本框|可选
plt.show()
```

### 5.1.2.grid()

绘制网格线

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.grid(axis='x',  # 绘制的范围|str('both'|'x'|'y')|'both'|可选
        linestyle=':')  # 网格线的样式|str('-'|'--'|'-.'|':'|'None'|' '|''|'solid'|'dashed'|'dashdot'|'dotted')|'-'|可选
plt.show()
```

### 5.1.3.legend()

放置图例

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.legend(loc='center')  # 放置的位置|str('upper right'|'upper left'|'lower left'|'lower right'|'right'|'center left'|'center right'|'lower center'|'upper center'|'center')|'best'|可选
plt.show()
```

### 5.1.4.patch

| 版本 | 描述                                  | 注意 |
| ---- | ------------------------------------- | ---- |
| -    | patches是画布颜色和边框颜色的控制接口 |      |

#### 5.1.4.1.set_alpha()

设置画布的透明度

```python
ax.patch.set_alpha(alpha)  # 透明度|float|None
```

#### 5.1.4.2.set_facecolor()

设置画布的颜色

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.patch.set_alpha(alpha=0.1)
ax.patch.set_facecolor(color='green')  # 颜色|str|None
plt.show()
```

### 5.1.5.set_xlabel()

设置x轴的内容

```python
ax.set_xlabel(xlabel='this is x label')  # 内容|str
```

### 5.1.6.set_xticks()

设置x轴的刻度

```python
ax.set_xticks(ticks=[1, 2, 3, 4])  # 刻度|list(空列表就表示不显示刻度)
```

### 5.1.7.set_yticks()

设置y轴的刻度

```python
ax.set_yticks(ticks=[])  # 刻度|list(空列表就表示不显示刻度)
```

###  5.1.8.spines

| 版本 | 描述                         | 注意 |
| ---- | ---------------------------- | ---- |
| -    | 画布的边框，包括上下左右四个 |      |

#### 5.1.8.1.set_color()

设置画布的边框的颜色

```python
ax.spines['left'].set_color(c='red')  # 颜色|str
```

### 5.1.9.text()

给点添加文本|matplotlib.text.Text

```python
ax.text(x=0.5,  # 注释点的x坐标|float|0
        y=0.5,  # 注释点的y坐标|float|0
        s='text')  # 注释的文本内容|str|''
```

## 5.2.pyplot

| 版本 | 描述                                                         | 注意 |
| ---- | ------------------------------------------------------------ | ---- |
| -    | pyplot是matplotlib的state-based接口， 主要用于简单的交互式绘图和程序化绘图 |      |

### 5.2.1.barh()

在水平方向绘制条形图

```python
import matplotlib.pyplot as plt
y = ['No.1', 'No.2', 'No.3', 'No.4']
width = [1, -0.5, 2, 6]
height = 0.8
plt.barh(y,  # 条形图的y轴坐标|array-like|
         width,  # 每个数据的值|array-like|
         height)  # 每个数据的的宽度|float|0.8(1.0为直方图)|可选
plt.show()
```

### 5.2.2.figure()

创建一个画布|matplotlib.figure.Figure

```python
import matplotlib.pyplot as plt
figure = plt.figure(figsize)  # 画布的大小|(float, float)|(6.4, 4.8)|可选
```

### 5.2.3.imread()

加载指定路径的图片|numpy.ndarray

```python
import matplotlib.pyplot as plt
image = plt.imread(fname)  # 要加载的文件的路径|str or file-like
```

### 5.2.4.imshow()

将图片数组在画布上显示|matplotlib.image.AxesImage

```python
import matplotlib.pyplot as plt
plt.imshow(X)  # 希望显示的图像数据|array-like or PIL image
```

### 5.2.5.plot()

绘制函数图像|list

```python
import matplotlib.pyplot as plt
plt.plot(*args)  # 函数的变量｜string or number且第一维度必须相同｜(x, y)
```

### 5.2.6.rcParams

实例化一个matplotlib的rc文件实例|matplotlib.RcParams

```python
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'Arial Unicode MS'  # 默认字体
```

### 5.2.7.savefig()

保存当前的画布

```python
import matplotlib.pyplot as plt
plt.savefig(fname)  # 要保存的文件的路径|str or PathLike or file-like object A path, or a Python file-like object
```

### 5.2.8.scatter()

绘制散点图|matplotlib.collections.PathCollection

```python
import matplotlib.pyplot as plt
x = [1, 2, 2, 3, 4, 5, 5.5, 6]
y = [1, 3, 2, 3, 4, 5, 5, 6]
plt.scatter(x,  # x坐标|scalar or array-like 形状必须是(n,)
            y)  # y坐标|scalar or array-like 形状必须是(n,)
plt.show()
```

### 5.2.9.show()

显示所有的画布

```python
import matplotlib.pyplot as plt
plt.show()
```

### 5.2.10.subplots()

创建一个画布和一组子图|matplotlib.figure.Figure和matplotlib.axes._subplots.AxesSubplot

```python
import matplotlib.pyplot as plt
figure, axesSubplot = plt.subplots()
```

### 5.2.11.subplots_adjust()

调整子图布局

```python
import matplotlib.pyplot as plt
plt.subplots_adjust(left=0.125,  # 子图左边框距离画布的距离|float|0.125
                    bottom,  # 子图下边框距离画布的距离|float|0.9
                    right,  # 子图右边框距离画布的距离|float|0.1
                    top,  # 子图上边框距离画布的距离|float|0.9
                    wspace,  # 两张子图之间的左右间隔|float|0.2
                    hspace)  # 两张子图之间的上下间隔|float|0.2
```

# 6.numpy

| 版本   | 描述           | 注意 |
| ------ | -------------- | ---- |
| 1.18.4 | python数值计算 |      |

## 6.1.argmax()

返回指定维度最大值的索引|numpy.int64

```python
import numpy as np
arr = [1, 2, 3]
np.argmax(a=arr,  # 输入的数组|array_like
          axis=None)  # 筛选所沿的维度|int|None|可选 
```

## 6.2.asarray()

将输入转化为一个数组|numpy.ndarray

```python
import numpy as np
arr = [1, 2, 3]
nd_arr = np.asarray(a=arr,  # 输入的数据|array-like
                    dtype=None)  # 元素的数据类型|data-type|None|可选
```

## 6.3.ceil()

逐元素进行向上取整|numpy.float64(输入是数组时numpy.ndarray)

```python
import numpy as np
arr = [5.1, 4.9]
np.ceil(arr)  # 输入数据|array_like
```

## 6.4.concatenate()

按照指定维度合并多个数组|numpy.ndarray

```python
import numpy as np
arr1 = [[1], [1], [1]]
arr2 = [[2], [2], [2]]
arr3 = [[3], [3], [3]]
np.concatenate([arr1, arr2, arr3],  # 要合并的数组|array-like
               axis=1)  # 沿指定维度合并|int|0|可选
```

## 6.5.equal()

逐个元素判断是否一致|numpy.bool_(输入是数组时numpy.ndarray)

```python
import numpy as np
arr1 = [1, 2, 3]
arr2 = [1, 2, 2]
np.equal(arr1, arr2)  # 输入的数组|array-like
```

## 6.6.exp()

逐元素计算e的幂次|numpy.float64(输入是数组时numpy.ndarray)

```python
import numpy as np
arr = [1, 2, 3]
np.exp(arr)  # 输入数据|array-like
```

## 6.7.expand_dims()

扩展数组的形状，增加维度|numpy.ndarray

```python
import numpy as np
a = [1, 2]
a = np.expand_dims(a=a,  # 输入的数组|array-like
                   axis=0)  # 添加新维度的位置|int or tuple of ints
```

## 6.8.hstack()

按照水平顺序合成一个新的数组|numpy.ndarray

```python
import numpy as np
arr1 = [[1, 2, 3, 4], [1, 2, 3, 4]]
arr2 = [[5, 6], [5, 6]]
a = np.hstack(tup=(arr1, arr2))  # 数组序列|array-like
```

## 6.9.linalg

| 版本 | 描述                  | 注意 |
| ---- | --------------------- | ---- |
| -    | numpy的线性代数函数包 |      |

### 6.9.1.norm()

计算范数|numpy.float64

```python
import numpy as np
arr = [[1, 2], [3, 4]]
np.linalg.norm(x=arr,  # 输入的矩阵或向量|array_like(维数必须是1维或2维)
               ord=1)  # 范数选项｜int or str(non-zero|int|inf|-inf|'fro'|'nuc')|None(计算2-范数)|可选
```

## 6.10.linspace()

生成指定间隔内的等差序列|numpy.ndarray

```python
import numpy as np
np.linspace(start=1,  # 序列的起始值|array_like
            stop=5,  # 序列的结束值|array_like
            num=10)  # 生成序列的样本的个数|int|50|可选
```

## 6.11.load()

从npy、npz、pickled文件加载数组或pickled对象|array or tuple or dict

```python
import numpy as np
np.load(file,  # 文件|file-like object or string or pathlib.Path
        allow_pickle,  # 允许加载npy文件中的pickle对象|bool|False|可选
        encoding)  # 读取的编码方式|str|'ASCII'|可选
```

## 6.12.log()

逐元素计算自然对数|numpy.float64(输入是数组时numpy.ndarray)

```python
import numpy as np
np.log(1)  # 输入数据|array_like
```

## 6.13.log2()

逐元素计算以2为底对数|numpy.float64(输入是数组时numpy.ndarray)

```python
import numpy as np
np.log2(1)  # 输入数据|array_like
```

## 6.14.mat()

将输入转换为一个矩阵|numpy.matrix

```python
import numpy as np
arr = [[1, 2, 3]]
matrix = np.mat(data=arr  # 输入数据|array-like
                dtype=None)  # 生成矩阵元素的数据类型|data-type|None|可选
```

## 6.15.matmul()

两个数组的矩阵乘积|numpy.ndarray

```python
import numpy as np
arr1 = [[1, 2, 3]]
arr2 = [[1], [2], [3]]
np.matmul(arr1, arr2)  # 输入的数组|array-like（不能是标量）
```

## 6.16.max()

返回最大值或者沿着某一维度最大值|numpy.ndarray or scalar

```python
import numpy as np
arr = [1., 2., 5., 3., 4.]
np.max(a=arr,  # 输入的数组|array-like
       axis=None)  # 所沿的维度|int|None|可选 
```

## 6.17.maximum()

返回数组逐个元素的最大值|numpy.ndarray

```python
import numpy as np
arr1 = [2, 3, 4]
arr2 = [1, 5, 2]
np.maximum(arr1, arr2)  # 输入的数组|array-like
```

## 6.18.mean()

沿着指定维度计算均值|numpy.float64

```python
import numpy as np
arr = [1, 2, 3]
np.mean(arr,  # 输入的数组|array-like
        axis=None)  # 所沿的维度|int or tuple of ints|None|可选 
```

## 6.19.ones()

创建一个指定为形状和类型的全一数组|numpy.ndarray

```python
import numpy as np
arr = np.ones(shape=[2, 3],  # 数组的形状|int or sequence of ints
              dtype=np.int8)  # 数组元素的数据类型|data-type|numpy.float64|可选
```

## 6.20.power()

逐个元素计算第一个元素的第二个元素次幂|scalar(输入是数组时numpy.ndarray)

```python
import numpy as np
x = np.power(2.1, 3.2)   # x1底数、x2指数|array_like
```

## 6.21.random

| 版本 | 描述                    | 注意 |
| ---- | ----------------------- | ---- |
| -    | numpy的随机数生成函数包 |      |

### 6.21.1.normal()

生成正态分布的样本|numpy.ndarray or scalar

```python
import numpy as np
arr = np.random.normal(size=[2, 3])  # 形状|int or tuple of ints|None(None则只返回一个数)|可选
```

### 6.21.2.randint()

从给定区间[low, high)生成随机整数|int or numpy.ndarray

```python
import numpy as np
np.random.randint(low=1,  # 下界|int or array-like of ints
                  high=10)  # 上界|int or array-like of ints|None(如果high为None则返回区间[0, low))|可选
```

### 6.21.3.rand()

生成一个指定形状的随机数数组|float or numpy.ndarray

```python
import numpy as np
arr = np.random.rand(2, 3)  # 数组的维度|int|(如果形状不指定，仅返回一个随机的浮点数)|可选
```

### 6.21.4.randn()

生成一个指定形状的标准正态分布的随机数数组|float or numpy.ndarray

```python
import numpy as np
arr = np.random.randn(2, 3)  # 数组的维度|int|(如果形状不指定，仅返回一个随机的浮点数)|可选
```

### 6.21.5.seed()

设置随机数生成器的随机种子

```python
import numpy as np
np.random.seed(seed)  # 随机种子|int|None|可选
```

## 6.22.reshape()

返回一个具有相同数据的新形状的数组|numpy.ndarray

```python
import numpy as np
arr = [1, 2, 3, 4]
np.reshape(a=arr,  # 要改变形状的数组|array_like
           newshape=[2, 2])  # 新的形状|int or tuple of ints
```

## 6.23.save()

将数组转换为numpy保存进二进制的npy文件

```python
import numpy as np
arr = [1, 2, 3]
np.save(file='arr.npy',  # 文件名|file or str or pathlib.Path
        arr=arr,  # 要保存的数组|array-like
        allow_pickle=True)  # 允许使用pickle对象保存数组|bool|True|可选
```

## 6.24.sort()

返回排序数组的副本|numpy.ndarray

```python
import numpy as np
arr = [1, 3, 2, 4]
new_arr = np.sort(a=arr)  # 要排序的数组|array_like
```

## 6.25.split()

将一个数组拆分为多个|list of ndarrays

```python
import numpy as np
arr = np.asarray([[1, 2, 5, 6], [3, 4, 7, 8]])
arr_list = np.split(ary=arr,  # 要拆分的数组|numpy.ndarray
                    indices_or_sections=2,  # 拆分方法|int or 1-D array(整数必须能整除)
                    axis=1)  # 沿某维度分割|int|0|可选
```

## 6.26.sqrt()

逐元素计算e的幂次|numpy.float64(输入是数组时numpy.ndarray)

```python
import numpy as np
arr = [1, 2, 3]
np.sqrt(arr)  # 输入数据|array_like
```

## 6.27.squeeze()

删除数组中维度为一的维度|numpy.ndarray

```python
import numpy as np
arr = [[1, 2, 3]]
np.squeeze(arr)  # 输入数据|array_like
```

## 6.28.std()

沿指定维度计算标准差|numpy.float64

```python
import numpy as np
arr = [1, 2, 3]
np.std(a=arr,  # 输入的数组|array-like
       axis=None)  # 所沿的维度|int or tuple of ints|None|可选
```

## 6.29.sum()

沿指定维度求和|numpy.ndarray

```python
import numpy as np
arr = [[1.2, 2.3, 3], [4, 5, 6]]
np.sum(arr,  # 输入的数组|array-like
       axis=1)  # 所沿的维度|int or tuple of ints|None|可选
```

## 6.30.transpose()

对数组进行转置|numpy.ndarray

```python
import numpy as np
arr = np.asarray([[1, 2], [3, 4]])
np.transpose(a=arr,  # 输入的数组|array-like
             axes=None)  # 轴的排列顺序|list of ints|None|可选
```

等价于上面的方法

```python
import numpy as np
arr = np.asarray([[1, 2], [3, 4]])
arr.T
```

## 6.31.var()

沿指定维度方差|numpy.ndarray

```python
import numpy as np
arr = [[1.2, 2.3, 3], [4, 5, 6]]
np.var(arr,  # 输入的数组|array-like
       axis=1)  # 所沿的维度|int or tuple of ints|None|可选
```

##6.32.zeros()

创建一个指定为形状和类型的全零数组|numpy.ndarray

```python
import numpy as np
arr = np.zeros(shape=[2, 3],  # 数组的形状|int or sequence of ints
               dtype=np.int8)  # 数组元素的数据类型|data-type|numpy.float64|可选
```

# 7.pandas

| 版本  | 描述                 | 注意 |
| ----- | -------------------- | ---- |
| 1.0.3 | 结构化数据分析软件库 |      |

## 7.1.concat()

沿指定维度合并pandas对象|pandas.core.frame.DataFrame or pandas.core.series.Series

```python
import pandas as pd
sr1 = pd.Series([1, 2, 3])
sr2 = pd.Series([1, 2, 3])
sr3 = pd.Series([1, 2, 3])
df = pd.concat([sr1, sr2, sr3],  # 待合并数据列表|DataFrame or Series
               axis=1)  # 沿行或者列合并|{0/'index', 1/'columns'}|0
```

## 7.2.DataFrame()

实例化一个DataFrame对象(二维，可变大小的，结构化数据)

```python
import pandas as pd
df_map = {'index': [0, 1, 2], 'values': [0.1, 0.2, 0.3]}
df = pd.DataFrame(data=df_map,  # 输入的数据|ndarray or Iterable or dict or DataFrame(数据必须是相同数据类型且为结构化的)
                  index=[1, 2, 3],  # 行索引|Index or array-like|None(默认0,1,...,n)
                  columns=None)  # 列索引|Index or array-like|None(默认0,1,...,n)
```

### 7.2.1.drop()

删除指定行或者列|pandas.core.frame.DataFrame

```python
import pandas as pd
df = pd.DataFrame([1, 2, 3])
df = df.drop(labels=1,  # 希望删除的行或者列|single label or list-like
             axis=0)  # 删除行或者列|{0/'index', 1/'columns'}|0
```

### 7.2.2.iloc[]

按照行号取出数据|pandas.core.frame.DataFrame or pandas.core.series.Series

```python
import pandas as pd
df = pd.DataFrame([[1, 4], [2, 5], [3, 6]])
new_df = df.iloc[0:2]  # 要提取的数据|int or array of int or slice object with ints
```

### 7.2.3.loc[]

按照行名称取出数据|pandas.core.frame.DataFrame or pandas.core.series.Series

```python
import pandas as pd
df_map = [[1, 4], [2, 5], [3, 6]]
df = pd.DataFrame(df_map, index=['a', 'b', 'c'])
new_df = df.loc['a':'b']  # 要提取的数据|label or array of label or slice object with labels(没有名称的时候就是iloc函数)
```

### 7.2.4.replace()

替换DataFrame中的值|pandas.core.frame.DataFrame or None

```python
import pandas as pd
df = pd.DataFrame([1, 2, 3, 4])
new_df = df.replace(to_replace=1,  # 被替换的值|scalar or dict or list or str or regex
                    value=2,  # 替换的值|scalar or dict or list or str or regex|None
                    inplace=False)  # 是否修改源DataFrame|bool(True没有返回值，False返回一个新的DataFrame)|False
```

### 7.2.5.reset_index()

重置DataFrame的索引为从零开始的整数索引|pandas.core.frame.DataFrame or None

```python
import pandas as pd
df_map = [[1, 4], [2, 5], [3, 6]]
df = pd.DataFrame(df_map, index=['a', 'b', 'c'])
new_df = df.reset_index(drop=True,  # 是否丢弃原来的索引|bool|False
                        inplace=False)  # 是否修改源DataFrame|bool(True没有返回值，False返回一个新的DataFrame)|False
```

### 7.2.6.sample()

随机采样指定个数的样本|pandas.core.frame.DataFrame

```python
import pandas as pd
df = pd.DataFrame([1, 2, 3, 4])
new_df = df.sample(n=None,  # 采样的个数|int|None(表示采样全部)|可选
                   frac=True)  # 是否对全部数据采样|bool|None(不可与n同时为非None的值)|可选
```

## 7.3.fillna()

填充缺失的值|pandas.core.frame.DataFrame or None

```python
import pandas as pd
df = pd.DataFrame([('a',), ('b', 2), ('c', 3)])
df.fillna(value=1,  # 缺失值|scalar, dict, Series, or DataFrame|None
          inplace=True)  # 是否修改源DataFrame|bool(True没有返回值，False返回一个新的DataFrame)|False
```

## 7.4.get_dummies()

将类别变量转换为dummy编码的变量|pandas.core.frame.DataFrame

```python
import pandas as pd
sr = pd.Series(['a', 'b', 'c', 'a'])
coding = pd.get_dummies(data=sr)  # 输入的数据|array-like, Series, or DataFrame
```

## 7.5.group_by()

使用给定列的值进行分组|pandas.core.groupby.generic.DataFrameGroupBy

```python
import pandas as pd
df = pd.DataFrame([[0, False], [1, True], [2, False]], index=['a', 'b', 'c'], columns=['c1', 'c2'])
group = df.groupby(by='c2')  # 分组依据(列名)|str(name of columns)
print(group.groups)
```

## 7.6.read_csv()

读取csv文件|DataFrame or TextParser

```python
import pandas as pd
new_df = pd.read_csv(filepath_or_buffer,  # 文件名|str or file handle|None
                     header=0,  # 列名所在的行|int or str|0|可选
                     index_col=None,  # 行名所在的列|int or str|None |可选
                     sep=',')  # 字段分隔符|str|','|可选
```

## 7.7.Series()

实例化一个Series对象(一维)|pandas.core.series.Series

```python
import pandas as pd
sr = pd.Series(data=[1, 2, 3, 4])  # 输入的数据|ndarray or Iterable or dict(数据必须是相同数据类型)
```

### 7.7.1.isin()

检查某个值是否在Series中|pandas.core.series.Series

```python
import pandas as pd
sr = pd.Series(data=[1, 2, 3, 4])
bool_list = sr.isin(values=[4])  # 检查的值|set or list-like
```

### 7.7.2.map()

使用输入的关系字典进行映射|pandas.core.series.Series

```python
import pandas as pd
df = pd.DataFrame([1, 2, 1])
map_dict = {1: 'a', 2: 'b'}
new_sr = df[0].map(map_dict)  # 映射关系|dict
```

### 7.7.3.mode()

返回数据的众数|pandas.core.series.Series

```python
import pandas as pd
sr = pd.Series([1, 2, 2, 2, 3])
m = sr.mode()
```

### 7.7.4.tolist()

返回Series值组成的列表|list

```python
import pandas as pd
sr = pd.Series([1, 2, 2, 2, 3])
l = pd.Series.tolist(sr)
```

## 7.8.to_csv()

将DataFrame保存进csv文件

```python
import pandas as pd
df = pd.DataFrame([1, 2, 3, 4])
df.to_csv(path_or_buf='./df.csv',  # 文件名|str or file handle|None
          sep=',',  # 字段分隔符|str|','|可选
          header=False,  # 列名|bool or list of str|True|可选
          index=True,  # 行名|bool|True|可选
          encoding='utf-8')  # 编码方式|str|'utf-8'|可选
```

## 7.9.unique()

返回唯一值组成的数组|numpy.ndarray

```python
import pandas as pd
sr = pd.Series([1, 2, 2, 2, 3])
arr = pd.unique(values=sr)  # 输入的数据|1d array-like
```

## 7.10.values

返回Series或者DataFrame的值组成的数组|numpy.ndarray or ndarray-like

```python
import pandas as pd
df = pd.DataFrame([1, 2, 3])
arr = df.values
```

## 7.11.value_counts()

统计非空数值的出现次数|pandas.core.series.Series

```python
import pandas as pd
sr = pd.Series([1, 2, 2, 2, 3])
c = pd.value_counts(values=sr)  # 输入的数据|1d array-like
```

# 8.PIL

| 版本  | 描述           | 注意                         |
| ----- | -------------- | ---------------------------- |
| 7.1.2 | 图像处理软件库 | 安装时使用pip install pillow |

## 8.1.Image

| 版本 | 描述          | 注意 |
| ---- | ------------- | ---- |
| -    | PIL图像修饰器 |      |

### 8.1.1.fromarray()

将一个numpy.ndarray转换成一个PIL.Image.Image|PIL.Image.Image

```python
import numpy as np
from PIL.Image import fromarray
arr = np.asarray([[0.1, 0.2], [0.3, 0.4]])
img = fromarray(obj=arr)  # 输入的数组|numpy.ndarray
```

### 8.1.2.open()

加载指定路径的图片|PIL.Image.Image

```python
from PIL.Image import open
img = open(fp)  # 要加载的文件的路径|str or pathlib.Path object or a file object
```

### 8.1.3.resize()

将图像调整到指定大小并返回副本|PIL.Image.Image

```python
from PIL.Image import open
img = open(fp)
new_img = img.resize(size=(400, 400))  # 调整后图像的尺寸|2-tuple: (width, height)
```

# 9.pydot

| 版本  | 描述                 | 注意 |
| ----- | -------------------- | ---- |
| 1.4.1 | graphviz的python接口 |      |

## 9.1.Dot

| 版本 | 描述          | 注意 |
| ---- | ------------- | ---- |
| -    | Dot语言的容器 |      |

### 9.1.1.write_png()

将图像写入文件

```python
import pydot
graph = pydot.graph_from_dot_data(s)[0]
graph.write_png(path)  # 写入文件的路径|str
```

## 9.2.graph_from_dot_data()

从dot数据中加载图像|list of pydot.Dot

```python
import pydot
graph = pydot.graph_from_dot_data(s)  # dot数据|str
```

## 9.3.graph_from_dot_file()

从dot文件中加载图像|list of pydot.Dot

```python
import pydot
graph = pydot.graph_from_dot_data(s)  # dot文件的路径|str
```

# 10.sklearn

| 版本   | 描述                           | 注意                               |
| ------ | ------------------------------ | ---------------------------------- |
| 0.23.0 | python机器学习和数据挖掘软件库 | 安装时使用pip install scikit-learn |

## 10.1.datasets

| 版本 | 描述                    | 注意                                    |
| ---- | ----------------------- | --------------------------------------- |
| -    | sklearn的官方数据集模块 | 数据的默认保存路径为~/scikit_learn_data |

### 10.1.1.load_iris()

加载并返回iris数据集|sklearn.utils.Bunch

```python
from sklearn.datasets import load_iris
dataset = load_iris()
```

## 10.2.ensemble

| 版本 | 描述                  | 注意                                                     |
| ---- | --------------------- | -------------------------------------------------------- |
| -    | sklearn的集成学习模块 | 使用scikit-learn API的其他框架也可以使用此模块的一些功能 |

### 10.2.1.AdaBoostClassifier()

实例化一个AdaBoost分类器

```python
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=50,  # 弱学习器的最大数量|int|50
                           learning_rate=1e-3)  # 学习率|float|1.0
```

### 10.2.2.RandomForestClassifier()

实例化一个随机森林分类器

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,  # 决策树的最大数量|int|100
                               criterion='gini',  # 划分方式|str('gini'或者'entropy')|'gini'
                               max_depth=None)  # 决策树的最大深度|int|None
```

### 10.2.3.VotingClassifier()

实例化一个投票分类器

```python
from sklearn.ensemble import VotingClassifier
model = VotingClassifier(estimators,  # 基学习器列表|list of (str, estimator) tuples
                         voting,  # 投票方式|str|'hard'(hard', 'soft')
                         weights)  # 基学习器的权重|array-like of shape (n_classifiers,)|None
```

#### 10.2.3.1.fit()

训练投票分类器|self

```python
model.fit(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
          y)  # 标签|array-like of shape (n_samples,)
```

#### 10.2.3.2.predict()

进行预测|numpy.ndarray

```python
result = model.predict(X)  # 用于预测的数据|{array-like, sparse matrix} of shape (n_samples, n_features)
```

#### 10.2.3.3.score()

计算验证集的平均准确率|float

```python
accuracy = model.score(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
                       y)  # 标签|array-like of shape (n_samples,)
```

## 10.3.linear_model

| 版本 | 描述                  | 注意 |
| ---- | --------------------- | ---- |
| -    | sklearn的线性模型模块 | -    |

### 10.3.1.LogisticRegression()

实例化一个逻辑回归模型

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```

#### 10.3.1.1.fit()

训练投票分类器|self

```python
model.fit(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
          y,  # 标签|array-like of shape (n_samples,)
          sample_weight)  # 类别权重|array-like of shape (n_samples,)|None
```

#### 10.3.1.2.predict()

进行预测|numpy.ndarray

```python
C = model.predict(X)  # 用于预测的数据|{array-like, sparse matrix} of shape (n_samples, n_features)
```

## 10.4.metrics

| 版本 | 描述              | 注意 |
| ---- | ----------------- | ---- |
| -    | sklearn的评估模块 | -    |

### 10.4.1.accuracy_score()

计算分类器的准确率|numpy.float64

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true,  # 真实标签|1d array-like, or label indicator array / sparse matrix
                          y_pred,  # 预测标签|1d array-like, or label indicator array / sparse matrix
                          sample_weight)  # 类别权重|array-like of shape (n_samples,)|None
```

## 10.5.model_selection

| 版本 | 描述                  | 注意 |
| ---- | --------------------- | ---- |
| -    | sklearn的数据划分模块 | -    |

### 10.5.1.cross_val_predict()

对模型的数据进行交叉验证|numpy.ndarry

```python
from sklearn.model_selection import cross_val_predict
result = cross_val_predict(estimator,  # 学习器|scikit-learn API实现的有fit和predict函数的模型
                           X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
                           y,  # 标签|array-like of shape (n_samples,)
                           cv)  # 交叉验证的划分数|int|3
```

### 10.5.2.LeaveOneOut()

实例化留一法交叉验证器

```python
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
```

#### 10.5.2.1.split()

划分数据|yield(train:numpy.ndarray, test:numpy.ndarray)

```python
loo.split(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
          y)  # 标签|array-like of shape (n_samples,)|None
```

### 10.5.3.StratifiedKFold()

实例化K折交叉验证器

```python
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits,  # 交叉验证的划分数|int|5
                        shuffle,  # 打乱数据|bool|False
                        random_state)  # 随机状态|int or RandomState instance|None
```

#### 10.5.3.1.split()

划分数据|yield(train:numpy.ndarray, test:numpy.ndarray)

```python
kfold.split(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
          y)  # 标签|array-like of shape (n_samples,)|None
```

### 10.5.4.train_test_split()

将原始数据随机划分成训练和测试子集|list(两个长度相等的arrays)

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,  # 需要划分的数据|lists, numpy arrays, scipy-sparse matrices or pandas dataframes
                                                    test_size,  # 测试数据的大小|float or int|0.25
                                                    random_state)  # 随机状态|int or RandomState instance|None
```

## 10.6.preprocessing

| 版本 | 描述                                                | 注意 |
| ---- | --------------------------------------------------- | ---- |
| -    | sklearn的数据预处理模块(缩放、居中、归一化、二值化) | -    |

### 10.6.1.MinMaxScaler()

实例化一个MinMax缩放器

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
```

#### 10.6.1.1.fit_transform()

转换数据|numpy.ndarray

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = [[1, 1], [2, 2], [3, 2], [4, 3], [5, 3]]
data = scaler.fit_transform(X=X)  # 需要转换的数据|{array-like, sparse matrix, dataframe} of shape (n_samples, n_features)
```

### 10.6.2.MultiLabelBinarizer()

实例化一个多标签二值化转换器

```python
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
```

#### 10.6.2.1.classes_

原始的标签|numpy.ndarray

```python
mlb.classes_
```

#### 10.6.2.2.fit_transform()

转换标签数据|numpy.ndarray

```python
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y = [['a', 'b'], ['a', 'c']]
label = mlb.fit_transform(y=y)  # 需要转换的标签|array-like
```

## 10.7.svm

| 版本 | 描述                    | 注意 |
| ---- | ----------------------- | ---- |
| -    | sklearn的支持向量机模块 | -    |

### 10.7.1.SVC()

实例化一个支持向量分类器

```python
from sklearn.svm import SVC
model = SVC(C,  # 正则化系数|float|1.0
            kernel,  # 核函数|str|'rbf'('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
            probability,  # 是否启用概率估计|bool|false
            class_weight)  # 类别权重|dict or 'balanced'|None
```

## 10.8.tree

| 版本 | 描述                | 注意 |
| ---- | ------------------- | ---- |
| -    | sklearn的决策树模块 | -    |

### 10.8.1.DecisionTreeClassifier()

实例化一个决策树分类器

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion,  # 划分方式|str('gini'或者'entropy')|'gini'
                               random_state)  # 随机状态|int or RandomState instance|None
```

#### 10.8.1.1.fit()

训练决策树分类器|self

```python
model.fit(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
          y)  # 标签|array-like of shape (n_samples,)
```

### 10.8.2.export_graphviz()

将决策树转换成dot字符串|str

```python
from sklearn.tree import export_graphviz
dot_str = export_graphviz(decision_tree,  # 决策树分类器|sklearn.tree._classes.DecisionTreeClassifier
                          out_file,  # 是否导出文件|file object or str|None｜可选
                          feature_names,  # 特征的名称|list of str|None|可选
                          class_names)  # 类别的名称|list of str, bool or None|None|可选
```

### 10.8.3.plot_tree()

绘制决策树

```python
from sklearn.tree import plot_tree
plot_tree(decision_tree=model)  # 决策树分类器|sklearn.tree._classes.DecisionTreeClassifier or sklearn.tree._classes.DecisionTreeRegressor
```

## 10.9.utils

| 版本 | 描述                  | 注意 |
| ---- | --------------------- | ---- |
| -    | sklearn的实用工具模块 | -    |

### 10.9.1.multiclass

#### 10.9.1.1.type_of_target()

判断数据的类型|str

```python
from sklearn.utils import multiclass
y = [1, 2, 3]
result = multiclass.type_of_target(y=y)  # 要判断数据|array-like
```

# 11.tensorflow

| 版本  | 描述         | 注意                                             |
| ----- | ------------ | ------------------------------------------------ |
| 2.3.0 | 机器学习框架 | TensorFlow 2.X的语法相同，高版本会比低版本算子多 |

## 11.1.config

### 11.1.1.experimental

#### 11.1.1.1.set_memory_growth()

设置物理设备的内存使用量

```python
import tensorflow as tf
tf.config.experimental.set_memory_growth(device,  # 物理设备|tensorflow.python.eager.context.PhysicalDevice
                                         enable)  # 是否启用内存增长|bool
```

### 11.1.2.list_physical_devices()

返回主机所有可见的物理设备|list

```python
import tensorflow as tf
devices_list = tf.config.list_physical_devices(device_type=None)  # 设备类型|str|None|可选
```

## 11.2.constant()

创建一个常张量|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.constant(value=10)  # 输入的数据|int, float or list
```

## 11.3.data

| 版本 | 描述           | 注意 |
| ---- | -------------- | ---- |
| -    | 数据输入流水线 | -    |

### 11.3.1.Dataset

#### 11.3.1.1.batch()

给数据集划分批次|tensorflow.python.data.ops.dataset_ops.BatchDataset

```python
import tensorflow as tf
dataset = tf.data.Dataset.range(6)
dataset = dataset.batch(batch_size=3)  # 批次大小|A tf.int64 scalar, int
print(list(dataset.as_numpy_iterator()))
```

#### 11.3.1.2.from_tensor_slices()

创建一个元素是张量切片的数据集|tensorflow.python.data.ops.dataset_ops.TensorSliceDataset

```python
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices(tensors=([1, 2], [3, 4], [5, 6]))  # 输入的张量|array-like(数据第一维相同)
print(list(dataset.as_numpy_iterator()))
```

#### 11.3.1.3.shuffle()

随机打乱数据集|tensorflow.python.data.ops.dataset_ops.ShuffleDataset

```python
import tensorflow as tf
dataset = tf.data.Dataset.range(5)
dataset_shuffle = dataset.shuffle(buffer_size=5)  # 采样的范围|A tf.int64 scalar, int
print(list(dataset.as_numpy_iterator()))
print(list(dataset_shuffle.as_numpy_iterator()))
```

## 11.4.einsum()

爱因斯坦求和约定|tensorflow.python.framework.ops.EagerTensor

```python
import numpy as np
import tensorflow as tf
a = np.asarray([[1], [2]])
b = np.asarray([[2, 1]])
result = tf.einsum('ij,jk->ik',  # 描述公式|str
                   a, b)  # 输入的张量|tf.Tensor or numpy.ndarray
```

## 11.5.GradientTape()

实例化一个梯度带

```python
import tensorflow as tf
tape = tf.GradientTape()
```

### 11.5.1.gradient()

计算梯度|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = 2*x
grad = tape.gradient(target=y, sources=x)  # 计算target关于sources的梯度|a list or nested structure of Tensors or Variables
```

## 11.6.image

| 版本 | 描述                 | 注意 |
| ---- | -------------------- | ---- |
| -    | 图像处理和编解码操作 | -    |

### 11.6.1.convert_image_dtype()

改变图片的数据类型|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
img = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
     [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
img = tf.image.convert_image_dtype(image=img,  # 图片|array-like
                                   dtype=tf.int8)  # 转换后的数据类型|tensorflow.python.framework.dtypes.DType
```

### 11.6.2.decode_image()

转换BMP、GIF、JPEG或PNG图像为张量|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.image.decode_image(contents,  # 图片的字节流|0-D str
                               channels,  # 转换后的色彩通道数|int|0|可选
                               dtype)  # 转换后的数据类型|tensorflow.python.framework.dtypes.DType
```

### 11.6.3.resize()

改变图片的大小|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.image.resize(images,  # 输入的图片|4-D Tensor of shape [batch, height, width, channels] or 3-D Tensorof shape [height, width, channels]
                         size)  # 改变后的大小｜int([new_height, new_width])
```

## 11.7.io

### 11.7.1.read_file()

读入文件|str

```python
import tensorflow as tf
img = tf.io.read_file(filename)  # 文件路径|str
```

## 11.8.keras

| 版本  | 描述                        | 注意                                    |
| ----- | --------------------------- | --------------------------------------- |
| 2.4.0 | TensorFlow的高阶机器学习API | Keras移除了多后端支持，推荐使用tf.keras |

### 11.8.1.applications

| 版本 | 描述                             | 注意                           |
| ---- | -------------------------------- | ------------------------------ |
| -    | 提供带有预训练权重的深度学习模型 | 默认保存路径是~/.keras/models/ |

#### 11.8.1.1.efficientnet

##### 11.8.1.1.1.EfficientNetB0()

EfficientNetB0的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.layers import Input
model = EfficientNetB0(include_top=False,  # 是否包含全连接输出层|bool|True
                       weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                       input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

##### 11.8.1.1.2.EfficientNetB4()

EfficientNetB4的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.layers import Input
model = EfficientNetB4(include_top=False,  # 是否包含全连接输出层|bool|True
                       weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                       input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

##### 11.8.1.1.3.EfficientNetB7()

EfficientNetB7的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.layers import Input
model = EfficientNetB7(include_top=False,  # 是否包含全连接输出层|bool|True
                       weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                       input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

#### 11.8.1.2.inception_resnet_v2

##### 11.8.1.2.1.InceptionResNetV2()

InceptionResNetV2的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Input
model = InceptionResNetV2(include_top=False,  # 是否包含全连接输出层|bool|True
                          weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                          input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

#### 11.8.1.3.inception_v3

##### 11.8.1.3.1.InceptionV3()

InceptionV3的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input
model = InceptionV3(include_top=False,  # 是否包含全连接输出层|bool|True
                    weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                    input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

#### 11.8.1.4.mobilenet_v2

##### 11.8.1.4.1.MobileNetV2()

MobileNetV2的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Input
model = MobileNetV2(include_top=False,  # 是否包含全连接输出层|bool|True
                    weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                    input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

#### 11.8.1.5.resnet50

##### 11.8.1.5.1.ResNet50()

```python
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input
model = ResNet50(include_top=False,  # 是否包含全连接输出层|bool|True
                 weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                 input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

#### 11.8.1.6.resnet_v2

##### 11.8.1.6.1.ResNet152V2()

ResNet152V2的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.layers import Input
model = ResNet152V2(include_top=False,  # 是否包含全连接输出层|bool|True
                    weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                    input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

#### 11.8.1.7.vgg19

##### 11.8.1.7.1.preprocess_input()

对一个批次的数据进行ImageNet格式的预处理|Preprocessed tensor or numpy.ndarray

```python
from tensorflow.keras.applications.vgg19 import preprocess_input
input = preprocess_input(x)  # 要处理的数据|Tensor or numpy.ndarray
```

##### 11.8.1.7.2.VGG19()

VGG19的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input
model = VGG19(include_top=False,  # 是否包含全连接输出层|bool|True
              weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
              input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

#### 11.8.1.8.xception

##### 11.8.1.8.1.Xception()

```python
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Input
model = Xception(include_top=False,  # 是否包含全连接输出层|bool|True
                 weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                 input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

###  11.8.2.backend

| 版本 | 描述        | 注意 |
| ---- | ----------- | ---- |
| -    | 后端函数API |      |

#### 11.8.2.1.cast()

转换张量的数据类型|tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras.backend import cast
tensor = cast(x=[1, 2, 3],  # 输入的张量|tf.Tensor or array-like
              dtype='float16')  # 转换后的数据类型|str('float16', 'float32', or 'float64')
```

#### 11.8.2.2.clip()

逐元素进行裁切到满足条件的范围|tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras.backend import clip
tensor = clip(x=[1, 2, 3, 4, 5],  # 输入的张量|tf.Tensor or array-like
              min_value=2,  # 最小值|float, integer or tensor
              max_value=4)  # 最大值|float, integer or tensor
```

#### 11.8.2.3.ctc_batch_cost()

在每个批次上计算ctc损失|tensorflow.python.framework.ops.EagerTensor(形状是(samples,1))

```python
from tensorflow.keras.backend import ctc_batch_cost
tensor = ctc_batch_cost(y_true,  # 真实的标签|tensor(samples, max_string_length)
                        y_pred,  # 预测的标签|tensor(samples, time_steps, num_categories)
                        input_length,  # 预测的长度|tensor(samples, 1)
                        label_length)  # 真实的长度|tensor(samples, 1)
```

#### 11.8.2.4.expand_dims()

扩展张量的维度|tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras.backend import expand_dims
tensor = expand_dims(x=[1, 2, 3],  # 输入的张量|tf.Tensor or array-like
                     axis=0)  # 添加新维度的位置|int
```

#### 11.8.2.5.ones_like()

创建一个和输入形状相同的全一张量|tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras.backend import ones_like
tensor = ones_like(x=[[1, 2, 3], [4, 5, 6]])  # 输入的张量|array-like
```

#### 11.8.2.6.shape()

返回张量的形状|tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras.backend import shape, ones_like
tensor = ones_like(x=[[1, 2, 3], [4, 5, 6]])
tensor_shape = shape(x=tensor)  # 输入的张量|tensor
```

#### 11.8.2.7.sigmoid()

逐元素计算sigmoid函数的值|tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras.backend import sigmoid
tensor = sigmoid(x=[1., 2., 3., 4., 5.])  # 输入的张量|tensor
```

#### 11.8.2.8.zeros_like()

创建一个和输入形状相同的全零张量|tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras.backend import zeros_like
tensor = zeros_like(x=[[1, 2, 3], [4, 5, 6]])  # 输入的张量|array-like
```

### 11.8.3.callbacks

| 版本 | 描述                                  | 注意 |
| ---- | ------------------------------------- | ---- |
| -    | 回调函数API，用于查看模型的状态和统计 |      |

#### 11.8.3.1.EarlyStopping()

实例化一个EarlyStopping，用以提前停止训练防止过拟合

```python
from tensorflow.keras.callbacks import EarlyStopping
CALLBACKS = [
    EarlyStopping(monitor='val_accuracy',  # 监控的信息|str｜'val_loss'
                  min_delta=1e-4,  # 最小变化量|float|0
                  patience=5,  # 监测容忍轮数(数据有小幅度波动可以跳过，验证频率也一定是1)|int|0
                  verbose=1)  # 日志模式|int(0, 1)|0
]
```

#### 11.8.3.2.ModelCheckpoint()

实例化一个ModelCheckpoint，用以某种频率保存模型或模型的权重

```python
from tensorflow.keras.callbacks import ModelCheckpoint
CALLBACKS = [
    ModelCheckpoint(filepath,  # 保存的路径|string or PathLike
                    monitor,  # 监控的信息|str｜'val_loss'
                    verbose,  # 日志模式|int(0, 1)|0
                    period)  # 保存的频率|int
]
```

#### 11.8.3.3.TensorBoard()

实例化一个TensorBoard，可视化训练信息

```python
from tensorflow.keras.callbacks import TensorBoard
CALLBACKS = [
    TensorBoard(log_dir,  # 保存的路径|string or PathLike
                histogram_freq,  # 绘制直方图|int(0,1)|0表示不绘制
                write_graph,  # 绘制图像|bool|True
                update_freq)  # 更新频率|str('batch' or 'epoch')|'batch'
]
```

### 11.8.4.datasets

| 版本 | 描述           | 注意                                                         |
| ---- | -------------- | ------------------------------------------------------------ |
| -    | 入门常用数据集 | 目前有boston_housing, cifar10, cifar100, fashion_mnist, imdb, mnist and reuters数据集 |

#### 11.8.4.mnist

#### 11.8.4.1.load_data()

加载mnist数据集|Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test)

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

### 11.8.5.layers

| 版本 | 描述      | 注意                                     |
| ---- | --------- | ---------------------------------------- |
| -    | 网络层API | 可以使用Functional API或者Sequential模型 |

#### 11.8.5.1.Add()

实例化一个矩阵加法层，将layer相加

```python
from tensorflow.keras.layers import Add
layer = Add(_Merge)  # 相同形状的张量(层)列表|tensorflow.python.framework.ops.Tensor
```

#### 11.8.5.2.BatchNormalization()

实例化一个批标准化层

```python
from tensorflow.keras.layers import BatchNormalization
layer = BatchNormalization()
```

#### 11.8.5.3.Bidirectional()

实例化一个循环神经网络层的双向封装器

```python
from tensorflow.keras.layers import Bidirectional, GRU
layer = GRU(units=256, return_sequences=True)
layer = Bidirectional(layer=layer)  # 网络层|keras.layers.RNN, keras.layers.LSTM or keras.layers.GRU
```

#### 11.8.5.4.Concatenate()

实例化一个合并层

```python
from tensorflow.keras.layers import Concatenate
layer = Concatenate(axis=0)(_Merge)  # 连接的维度(相同形状的张量(层)列表|tensorflow.python.framework.ops.Tensor)|int|-1
```

#### 11.8.5.5.Conv1D()

实例化一个一维卷积层

```python
from tensorflow.keras.layers import Conv1D
layer = Conv1D(filters,  # 卷积核的数量|int
               kernel_size,  # 卷积核的大小|int or tuple/list of a single integer
               strides,  # 滑动步长|int or tuple/list of a single integer|1
               padding,  # 填充方式|str('valid', 'causal' or 'same')|'valid'
               data_format,  # 数据格式|str('channels_first' or 'channels_last')|'channels_last'
               activation,  # 激活函数|str or |None
               use_bias,  # 是否使用偏置|bool|True
               kernel_initializer,  # 权重初始化|str|'glorot_uniform'
               bias_initializer)  # 偏置初始化|str|'zeros'
```

#### 11.8.5.6.Conv2D()

实例化一个二维卷积层

```python
from tensorflow.keras.layers import Conv2D, Conv1D
layer = Conv2D(filters,  # 卷积核的数量|int
               kernel_size,  # 卷积核的大小|int, tuple/list of 2 integers
               strides,  # 滑动步长|int, tuple/list of 2 integers|(1, 1)
               padding,  # 填充方式|str('valid' or 'same')|'valid'
               input_shape)  # 如果是模型的第一层，需指定输入的形状|tuple of int
```

#### 11.8.5.7.Conv2DTranspose()

实例化一个二维转置卷积层

```python
from tensorflow.keras.layers import Conv2DTranspose
layer = Conv2DTranspose(filters,  # 卷积核的数量|int
                        kernel_size,  # 卷积核的大小|int, tuple/list of 2 integers
                        strides,  # 滑动步长|int, tuple/list of 2 integers|(1, 1)
                        padding,  # 填充方式|str('valid' or 'same')|'valid'
                        use_bias)  # 是否使用偏置|bool|True
```

#### 11.8.5.8.Dense()

实例化一个全连接层

```python
from tensorflow.keras.layers import Dense
layer = Dense(units,  # 神经元的数量|int
              use_bias,  # 是否使用偏置|bool|True
              input_shape)  # 如果是模型的第一层，需指定输入的形状|tuple of int
```

#### 11.8.5.9.Dot()

实例化一个点积层

```python
from tensorflow.keras.layers import Dot
layer = Dot(axes=1)(_Merge)# 点积的维度(相同形状的张量(层)列表|tensorflow.python.framework.ops.Tensor)|int|-1
```

#### 11.8.5.10.Dropout()

实例化一个Dropout层(在训练阶段随机抑制部分神经元)

```python
from tensorflow.keras.layers import Dropout
layer = Dropout(rate=0.5)  # 丢弃比例|float
```

#### 11.8.5.11.Embedding()

实例化一个嵌入层(只能作为模型的第一层)

```python
from tensorflow.keras.layers import Embedding
layer = Embedding(input_dim,  # 输入的维度|int(最大值加一) 
                  output_dim,  # 输出的嵌入矩阵维度|int
                  embeddings_initializer,  # 嵌入矩阵初始化器|str|uniform
                  embeddings_regularizer,)  # 嵌入矩阵正则化器|str or tensorflow.keras.regularizers|None
```

#### 11.8.5.12.Flatten()

实例化一个展平层(不影响批次)

```python
from tensorflow.keras.layers import Flatten
layer = Flatten()
```

#### 11.8.5.13.GRU()

实例化一个门控循环网络层

```python
from tensorflow.keras.layers import GRU
layer = GRU(units=256,  # 神经元的数量|int
            return_sequences=True)  # 返回序列还是返回序列的最后一个输出|bool|False(返回序列的最后一个输出)
```

#### 11.8.5.14.Input()

实例化一个输入层

```python
from tensorflow.keras.layers import Input
layer = Input(shape=(224, 224, 3),  # 形状|tuple
              name='Input-Layer',  # 层名称|str|None
              dtype='int32')  # 期望的数据类型|str|None
```

#### 11.8.5.15.Lambda()

实例化一个Lambda层(将任意函数封装成网络层)

```python
from tensorflow.keras.layers import Lambda
layer = Lambda(function=lambda x: x*x,  # 要封装的函数
               output_shape=(1024,),  # 期望输出形状|tuple|None
               name='Square-Layer')  # 层名称|str|None
```

#### 11.8.5.16.LeakyReLU()

实例化一个带侧漏的RelU层

```python
from tensorflow.keras.layers import LeakyReLU
layer = LeakyReLU(alpha=0.3)  # 负斜率系数(侧漏率)|float|0.3
```

#### 11.8.5.17.LSTM()

实例化一个长短时记忆网络层

```python
from tensorflow.keras.layers import LSTM
layer = LSTM(units=256,  # 神经元的数量|int
             return_sequences=True)  # 返回序列还是返回序列的最后一个输出|bool|False(返回序列的最后一个输出)
```

#### 11.8.5.18.MaxPooling1D()

实例化一个一维最大池化层

```python
from tensorflow.keras.layers import MaxPooling1D
layer = MaxPooling1D(pool_size=2,  # 池化窗口|int|2
                     strides=None,  # 滑动步长|int or tuple/list of a single integer|None
                     padding='valid')  # 填充方式|str('valid', 'causal' or 'same')|'valid'
```

#### 11.8.5.19.Reshape()

实例化变形层(将输入的层改变成任意形状)

```python
from tensorflow.keras.layers import Reshape
layer = Reshape(target_shape)  # 目标形状|tuple
```

### 11.8.6.losses

| 版本 | 描述        | 注意 |
| ---- | ----------- | ---- |
| -    | 损失函数API |      |

#### 11.8.6.1.BinaryCrossentropy()

实例化二分类交叉熵损失函数

```python
from tensorflow.keras.losses import BinaryCrossentropy
loss = BinaryCrossentropy(from_logits=True)  # 是否将y_pred解释为张量|bool|False(True的话有更高的稳定性)
```

#### 11.8.6.2.CategoricalCrossentropy()

实例化多分类交叉熵损失函数(标签是one-hot编码)

```python
from tensorflow.keras.losses import CategoricalCrossentropy
loss = CategoricalCrossentropy(from_logits=True)  # 是否将y_pred解释为张量|bool|False(True的话有更高的稳定性)
```

#### 11.8.6.3.SparseCategoricalCrossentropy()

实例化多分类交叉熵损失函数

```python
from tensorflow.keras.losses import SparseCategoricalCrossentropy
loss = SparseCategoricalCrossentropy(from_logits=True)  # 是否将y_pred解释为张量|bool|False(True的话有更高的稳定性)
```

### 11.8.7.models

| 版本 | 描述          | 注意                                                         |
| ---- | ------------- | ------------------------------------------------------------ |
| -    | 构建Keras模型 | Keras支持两种模型Sequential和Model(Functional API)， 模型的类方法基本一致，相同的统一写在Model里 |

#### 11.8.7.1.load_model()

加载模型|Keras model

```python
from tensorflow.keras.models import load_model
model = load_model(filepath='model.h5')  # 文件路径|str or pathlib.Path
```

#### 11.8.7.2.Model()

实例化一个Model类对象(Functional API)

```python
from tensorflow.keras.models import Model
model = Model(inputs,  # 输入层|keras.Input or list of keras.Input
              outputs)  # 输出层|keras.layers
```

##### 11.8.7.2.1.build()

根据输入的形状构建模型

```python
model.build(input_shape)  # 输入的形状|tuple, TensorShape, or list of shapes
```

##### 11.8.7.2.2.compile()

配置模型训练的参数

```python
model.compile(optimizer='rmsprop',  # 优化器|str or keras.optimizers|'rmsprop'
              loss=None,  # 损失函数|str or tf.keras.losses.Loss|None
              metrics=None)  # 评估指标列表|list of metrics or keras.metrics.Metric|None 
```

##### 11.8.7.2.3.evaluate()

在测试模式下计算损失和准确率

```python
model.evaluate(x,  # 特征数据|numpy.array (or array-like), TensorFlow tensor, or a list of tensors, tf.data, generator or keras.utils.Sequence
               y=None,  # 标签|numpy.array (or array-like), TensorFlow tensor(如果x是dataset, generators,y为None)
               batch_size=32,  # 批次大小|int|32
               verbose=1)  # 日志模式|int(0, 1)|1
```

##### 11.8.7.2.4.fit()

训练模型|History.history

```python
model.fit(x,  # 特征数据|numpy.array (or array-like), TensorFlow tensor, or a list of tensors, tf.data, generator or keras.utils.Sequence
          y=None,  # 标签|numpy.array (or array-like), TensorFlow tensor(如果x是dataset, generators,y为None)
          batch_size=32,  # 批次大小|int|32
          epochs=1,  # 轮数|int|1
          verbose=1,  # 日志模式|int(0, 1, 2详细)|1
          callbacks=None,  # 回调函数|list of callbacks|None
          validation_split=0.,  # 验证数据划分|float|0.
          validation_data=None,  # 验证数据|tuple (x_val, y_val) or datasets|None
          shuffle=True,  # 是否打乱|bool|True
          steps_per_epoch=None)  # 每轮的总步数(样本数/批次大小)|int|None
```

##### 11.8.7.2.5.fit_generator()

训练模型(fit()也支持了生成器推荐使用fit())

```python
model.fit_generator(generator,  # 特征数据|generator or keras.utils.Sequence
                    steps_per_epoch=None,  # 每轮的总步数(样本数/批次大小)|int|None
                    epochs=1,  # 轮数|int|1
                    verbose=1,  # 日志模式|int(0, 1, 2详细)|1
                    callbacks=None,  # 回调函数|list of callbacks|None
                    validation_data=None,  # 验证数据|tuple (x_val, y_val) or datasets|None
                    shuffle=True)  # 是否打乱|bool|True
```

##### 11.8.7.2.6.load_weights()

加载模型的权重

```python
model.load_weights(filepath)  # 文件路径|str or pathlib.Path
```

##### 11.8.7.2.7.predict()

进行预测|numpy.ndarray

```python
model.predict(x,  # 特征数据|numpy.array (or array-like), TensorFlow tensor, tf.data, generator or keras.utils.Sequence
              batch_size=32,  # 批次大小|int|32
              verbose=0)  # 日志模式|int(0, 1, 2详细)|0
```

##### 11.8.7.2.8.output_shape

返回输出层的形状

```python
print(model.output_shape)
```

##### 11.8.7.2.9.save()

保存模型

```python
model.save(filepath,  # 文件路径|str or pathlib.Path
           save_format=None)  # 保存格式|str('tf' or 'h5')|tf
```

##### 11.8.7.2.10.summary()

输出的模型摘要

```python
model.summary()
```

#### 11.8.7.3.Sequential()

实例化一个Sequential类对象

```python
from tensorflow.keras.models import Sequential
model = Sequential()
```

##### 11.8.7.3.1.add()

添加一个layer实例到Sequential栈顶

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
model = Sequential()
model.add(layer=Input(shape=(224, 224, 3)))  # 层示例｜keras.layer
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=2, activation='sigmoid'))
```

### 11.8.8.optimizers

| 版本 | 描述      | 注意 |
| ---- | --------- | ---- |
| -    | 优化器API |      |

#### 11.8.8.1.Adam()

实例化一个Adam优化器

```python
from tensorflow.keras.optimizers import Adam
optimziers = Adam(learning_rate)  # 学习率|float|0.001
```

#### 11.8.8.2.apply_gradients()

将梯度带计算出来的值赋值给优化器

```python
from tensorflow.keras.optimizers import Adam
optimziers = Adam(learning_rate=1e-4)
Adam.apply_gradients(grads_and_vars=zip(grads, vars))  # 梯度和变量|List of (gradient, variable) pairs
```

#### 11.8.8.3.SGD()

实例化一个随机梯度下降优化器

```python
from tensorflow.keras.optimizers import SGD
optimziers = SGD(learning_rate)  # 学习率|float|0.01
```

### 11.8.9.preprocessing

| 版本 | 描述               | 注意                     |
| ---- | ------------------ | ------------------------ |
| -    | Keras数据预处理API | 可以处理序列、文本、图像 |

#### 11.8.9.1.image

##### 11.8.9.1.1.ImageDataGenerator()

实例化一个ImageDataGenerator(对图片数据进行实时数据增强，并返回generator)

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
generator = ImageDataGenerator(rotation_range=0,  # 旋转度数|int|0
                               width_shift_range=0.,  # 水平位移范围|float|0.
                               height_shift_range=0.,  # 垂直位移范围|float|0.
                               shear_range=0.,  # 裁切角度范围|float|0.
                               zoom_range=0.,  # 缩放倍数|float|0.
                               channel_shift_range=0.,  # 色彩通道移位|float|0.
                               fill_mode='nearest',  # |str{'constant', 'nearest', 'reflect' or 'wrap'}|'nearest'
                               horizontal_flip=False,  # 水平翻转|bool|False
                               vertical_flip=False)  # 垂直翻转|bool|False
```

###### 11.8.9.1.1.1.class_indices

类名称和类索引的映射字典|dict

```python
generator.flow_from_dataframe().class_indices
generator.flow_from_directory().class_indices
```

###### 11.8.9.1.1.2.flow()

给定数据和标签进行增强|yield

```python
generator.flow(x,  # 输入数据|numpy.array of rank 4 or a tuple
               y=None,  # 标签|array-like
               batch_size=32,  # 批次大小|int|32
               shuffle=True)  # 是否打乱|bool|True
```

###### 11.8.9.1.1.3.flow_from_dataframe()

给定数据和标签(从dataframe内读入)进行增强|yield

```python
generator.flow_from_dataframe(dataframe,  # 文件信息图表|pandas.DataFrame
                              directory=None,  # 文件夹|str or path|None(如果为None, 则dataframe中必须是绝对路径)
                              x_col='filename',  # 文件路径列|str|'filename'
                              y_col='class',  # 文件标签列|str|'class'
                              target_size=(256, 256),  # 生成图片的大小|tuple of int|(256, 256)
                              classes=None,  # 类名称列表|list of str|None|可选
                              class_mode='categorical',  # 标签数组类型|str{'binary', 'categorical', 'input', 'multi_output'}(None可以用作测试)|'categorical'
                              batch_size=32,  # 批次大小|int|32
                              shuffle=True,  # 是否打乱|bool|True
                              interpolation='nearest',  # 插值方式|str{'nearest', 'bilinear' and 'bicubic'}|'nearest'
                              validate_filenames=True)  # 检查文件的可靠性|bool|True
```

###### 11.8.9.1.1.4.flow_from_directory()

给定数据和标签(每一个类别是一个单独的文件夹)进行增强|yield

```python
generator.flow_from_directory(directory,  # 文件夹|str or path
                              target_size=(256, 256),  # 生成图片的大小|tuple of int|(256, 256)
                              classes=None,  # 类名称列表|list of str|None|可选
                              class_mode='categorical',  # 标签数组类型|str{'binary', 'categorical', 'input', 'multi_output'}(None可以用作测试)|'categorical'
                              batch_size=32,  # 批次大小|int|32
                              shuffle=True,  # 是否打乱|bool|True
                              interpolation='nearest')  # 插值方式|str{'nearest', 'bilinear' and 'bicubic'}|'nearest'
```

##### 11.8.9.1.2.img_to_array()

将PIL图像转换为numpy数组|numpy.ndarray

```python
from tensorflow.keras.preprocessing.image import img_to_array
array = img_to_array(img)  # 输入的图像|PIL图像
```

##### 11.8.9.1.3.load_image()

加载PIL图像|PIL图像

```python
from tensorflow.keras.preprocessing.image import load_img
img = load_img(path,  # 文件路径|str or pathlib.Path
               target_size=None)  # 读取图片的大小|tuple of int|None
```

### 11.8.10.regularizers

| 版本 | 描述        | 注意 |
| ---- | ----------- | ---- |
| -    | 正则化器API |      |

#### 11.8.10.1.l2()

实例化一个L2正则化器

```python
from tensorflow.keras.regularizers import l2
regularizer = l2(l2=0.01)  # L2正则化因子|float|0.01
```

### 11.8.11.utils

| 版本 | 描述    | 注意 |
| ---- | ------- | ---- |
| -    | 工具API |      |

#### 11.8.11.1.get_file()

从指定URL下载文件|Path to the downloaded file

```python
from tensorflow.keras.utils import get_file
file = get_file(fname,  # 文件名|str
                origin,  # 文件的URL|str
                extract)  # tar和zip文件是否解压|bool|False
```

#### 11.8.11.2.multi_gpu_model()

单机多GPU并行训练|Keras model

```python
from tensorflow.keras.utils import multi_gpu_model
model = multi_gpu_model(model,  # 要并行的模型|Keras model
                        gpus)  # 并行的GPU数量|int(要大于等于2)
```

#### 11.8.11.3.plot_model()

绘制模型的网络图

```python
from tensorflow.keras.utils import plot_model
plot_model(model,  # 模型|keras model
           to_file='model.png',  # 保存的文件名|str|'model.png'
           show_shapes=False,  # 显示每一层的形状|bool|False
           show_layer_names=True,  # 显示每一层的名称|bool|True
           rankdir='TB',  # 绘制的方向|str('TB' or 'LR')|'TB'
           dpi=96)  # dpi值|int|96
```

#### 11.8.11.4.to_categorical()

将标签的离散编码转换为one-hot编码|numpy.ndarray

```python
from tensorflow.keras.utils import to_categorical
y = [1, 2, 3, 4]
y = to_categorical(y=y,  # 输入的标签|array-like of int
                   num_classes=5)  # 类别总数|int|None
```

## 11.9.ones_like()

创建一个和输入形状相同的全一张量|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.ones_like(input=[[1, 2, 3], [4, 5, 6]])  # 输入的张量|array-like
```

## 11.10.random

### 11.10.1.normal()

生成一个标准正态分布的张量|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.random.normal(shape=[2, 3])  # 形状|1-D integer Tensor or Python array
```

## 11.11.tensordot()

计算沿指定维度的点积|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.tensordot(a=[[1], [2]],  # 张量|array-like
                      b=[[2, 1]],  # 张量|array-like
                      axes=1)  # 维度|scalar N or list or int32 Tensor of shape [2, k]
```

## 11.12.zeros_like()

创建一个和输入形状相同的全零张量|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.zeros_like(input=[[1, 2, 3], [4, 5, 6]])  # 输入的张量|array-like
```

# 12.tensorflow.js

| 版本  | 描述                                        | 注意                                                        |
| ----- | ------------------------------------------- | ----------------------------------------------------------- |
| 2.3.0 | TensorFlow.js是TensorFlow的JavaScript软件库 | TensorFlow.js现在全面使用ES6语法；如果使用node.js有轻微差异 |

## 12.1.browser

### 12.1.1.fromPixels()

从一张图片中创建tf.Tensor|tf.Tensor3D

```javascript
import * as tf from "@tensorflow/tfjs";
// pixels: 构建张量的像素(PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
let img = tf.browser.fromPixels();
```

## 12.2.div()

两个tf.Tensor逐元素相除|tf.Tensor

```javascript
import * as tf from "@tensorflow/tfjs";
let a = tf.scalar(5);
let b = tf.scalar(2);
// b: 除数(tf.Tensor|TypedArray|Array)
let c = a.div(b);
```

## 12.3.expandDims()

扩展tf.Tensor的维度|tf.Tensor

```javascript
import * as tf from "@tensorflow/tfjs";
let t = tf.tensor([1, 2, 3, 4]);
// axis: 维度(number)|可省略
let t1 = t.expandDims(0);
```

## 12.4.image

### 12.4.1.resizeBilinear()

使用线性插值法改变图像的形状|tf.Tensor3D or tf.Tensor4D

```javascript
import * as tf from '@tensorflow/tfjs';
// images: 要改变形状的图像(tf.Tensor3D|tf.Tensor4D|TypedArray|Array); size: 改变后的大小([number, number])
let resized_img = tf.image.resizeBilinear(img, [224, 224]);
```

## 12.5.LayersModel

### 12.5.1.predict()

进行预测|tf.Tensor|tf.Tensor[]

```javascript
import * as tf from "@tensorflow/tfjs";
// x: 用于预测的数据(tf.Tensor|tf.Tensor[])
let result = model.predict();
```

### 12.5.2.summary()

输出的模型摘要

```javascript
import * as tf from "@tensorflow/tfjs";
model.summary();
```

## 12.6.loadLayersModel()

加载一个tf.LayersModel(使用Keras训练的非Functional API训练的模型)|Promise<tf.LayersModel>

```javascript
import * as tf from "@tensorflow/tfjs";
// pathOrIOHandler: 模型的路径(file://仅限tfjs-node; http://或者https://可以是绝对或者相对路径)
let model = tf.loadLayersModel();
```

## 12.7.print()

输出信息的控制台

```javascript
import * as tf from "@tensorflow/tfjs";
// x: 要输出的张量(tf.Tensor)
tf.print();
```

## 12.8.scalar()

创建一个tf.Tensor(scalar)的标量

```javascript
import * as tf from "@tensorflow/tfjs";
// value: 标量的数值; dtype: 数字的数据类型|可省略
let s = tf.scalar(10, "float32");
tf.print(s);
```

## 12.9.sub()

两个tf.Tensor逐元素相减|tf.Tensor

```javascript
import * as tf from "@tensorflow/tfjs";
let a = tf.scalar(1);
let b = tf.scalar(2);
// b: 减数(tf.Tensor|TypedArray|Array)
let c = a.sub(b);
```

## 12.10.tensor()

创建一个tf.Tensor的张量

```javascript
import * as tf from "@tensorflow/tfjs";
// value: 张量的数值
let t = tf.tensor(10);
tf.print(t);
```

### 12.10.1.data()

异步获取tf.Tensor的值|Promise<DataTypeMap[NumericDataType]>

```javascript
import * as tf from "@tensorflow/tfjs";
let t = tf.tensor(10);
let value = t.data();
```

## 12.11.tidy()

执行传入的函数后，自动清除除返回值以外的系统分配的所有的中间张量，防止内存泄露|void ,number,string,TypedArray,tf.Tensor,tf.Tensor[],{[key: string]:tf.Tensor,number,string}

```javascript
import * as tf from "@tensorflow/tfjs";
// fn: 传入的函数
let result = tf.tidy(fn);
```

# 13.tensorflow_hub

| 版本  | 描述                  | 注意                                                         |
| ----- | --------------------- | ------------------------------------------------------------ |
| 0.8.0 | TensoFlow的官方模型库 | 暂不清楚模型的默认保存路径，最好使用os.environ['TFHUB_CACHE_DIR']手动指定一个位置；SavedModel模型不可以转换成hdf5格式 |

## 13.1.KerasLayer()

将SavedModel或者Hub.Module修饰成一个tf.keras.layers.Layer实例

```python
from tensorflow_hub import KerasLayer
layer = KerasLayer(handle,  # 模型的路径|str
                   trainable,  # 能否训练|bool|(hub.Modules不可训练)|可选
                   output_shape,  # 输出的形状|tuple|None(模型本身有形状就不能设置)|可选
                   input_shape,  # 期望的输入的形状|tuple|可选
                   dtype)  # 期望的数据类型|tensorflow.python.framework.dtypes.DType|可选
```

## 13.2.load()

加载一个SavedModel|tensorflow.python.saved_model

```python
from tensorflow_hub import load
model = load(handle)  # 模型的路径|str
```

# 14.xgboost

| 版本  | 描述       | 注意                |
| ----- | ---------- | ------------------- |
| 1.1.1 | 梯度提升树 | 可直接在sklearn使用 |

## 14.1.XGBClassifier()

实例化一个XGBoost分类器

```python
from xgboost import XGBClassifier
model = XGBClassifier(max_depth,  # 基学习器(梯度提升树)的最大深度|int|None|可选
                      learning_rate,  # 学习率|float|None|可选
                      n_estimators,  # 梯度提升树的数量(相当于学习轮数)|int|100
                      subsample,  # 随机采样率|float|None|可选
                      colsample_bytree)   # 构造每棵树，属性随机采样率|float|None|可选
```

### 14.1.1.fit()

训练XGBoost分类器|self

```python
model.fit(X,  # 特征数据|array-like
          y,  # 标签|array-like
          eval_set,  # 验证集元组列表|list of (X, y) tuple|None|可选
          eval_metric,  # 验证使用的评估指标|str or list of str or callable|None|可选 
          verbose)  # 日志模式|bool|True
```

### 14.1.2.predict()

进行预测|numpy.ndarray

```python
result = model.predict(data)  
# 用于预测的数据|array_like
```