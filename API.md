# <center>A²PI²-API version2.1.1</center>

* 2.1.1版本增加CPP库
* Python格式规范化
  1. 类或函数功能|有无返回值
  2. 每个参数将按照意义|数据类型{枚举, ...}|默认值|是否为可选参数
* JavaSrcipt格式规范化
  1. 类或函数功能|有无返回值
  2. //参数:意义(数据类型)|是否可省略;参数:意义(数据类型)|是否可省略;......
* CC格式规范化 
  1. 类或函数功能
  2. // 返回值，函数的相关输入输出
* 在Github上提供PDF格式的Releases，显著减小仓库的大小

# 1.catboost

| 版本   | 描述                 | 注意                |
| ------ | -------------------- | ------------------- |
| 0.23.1 | 梯度提升决策树(GBDT) | 可直接在sklearn使用 |

## 1.1.CatBoostClassifier()

实例化一个CatBoost分类器

```python
from catboost import CatBoostClassifier
model = CatBoostClassifier(iterations=100,  # 迭代次数|int|500
                           learning_rate=1e-1,  # 学习率|float|0.03
                           depth=10,  # 树的深度|int|6
                           l2_leaf_reg=3,  # 损失函数使用L2正则化|float|3.0
                           loss_function='Logloss',  # 损失函数|{'Logloss', 'CrossEntropy'} or object|'Logloss'
                           od_type='Iter',  # 过拟合检查|{'IncToDec', 'Iter'}|'IncToDec'
                           random_seed=16,  # 随机种子|int|None
                           cat_features=None)  # 分类的索引列|list or numpy.ndarray|None 
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
model.feature_names_
```

### 1.1.4.predict()

进行预测|numpy.ndarray

```python
result = model.predict(data)  # 用于预测的数据|catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series or catboost.FeaturesData
```

## 1.2.CatBoostRegressor()

实例化一个CatBoost回归器

```python
from catboost import CatBoostRegressor
model = CatBoostRegressor(learning_rate=0.02,  # 学习率|float|0.03
                          max_depth=5,  # 树的深度|int|6
                          l2_leaf_reg=10,  # 损失函数使用L2正则化|float|3.0
                          od_type='Iter',  # 过拟合检查|{'IncToDec', 'Iter'}|'IncToDec'
                          od_wait=100,  # 在最佳的迭代次数之后继续训练的轮数|int|20
                          colsample_bylevel=0.8,  # 随机子空间法,随机选择特征时,每次拆分选择使用的特征的百分比|float(0, 1]|1
                          bagging_temperature=0.2,  # 使用贝叶斯为分配初始化的权重|float|1.0
                          bootstrap_type=None,  # 启动类型|str|{'Bayesian', 'Bernoulli', 'MVS', 'Poisson'(仅限GPU), None}
                          random_state=11,  # 随机种子|int|None
                          allow_writing_files=False)  # 允许导出缓存文件|bool|True
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

# 3.Eigen

| 版本  | 描述              | 注意               |
| ----- | ----------------- | ------------------ |
| 3.3.7 | CPP线性代数模版库 | 可直接使用brew安装 |

## 3.1.ArrayXd

实例化一个动态一维数组(双精度)

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::ArrayXd arr(4);
    arr << 1, 2, 3, 4;
    std::cout << arr << std::endl;
    return 0;
}
```

## 3.2.ArrayXXd

实例化一个动态二维数组(双精度)

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::ArrayXXd arr(2, 2);
    arr << 1, 2, 3, 4;
    std::cout << arr << std::endl;
    return 0;
}
```

### 3.2.1.exp()

逐元素计算e的幂次

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    Eigen::ArrayXXd arr = mat.array();
    std::cout << arr.exp() << std::endl;
    return 0;
}
```

### 3.2.2.log()

逐元素计算自然对数

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 2.718281, 7.38906, 20.0855, 54.5982;
    Eigen::ArrayXXd arr = mat.array();
    std::cout << arr.log() << std::endl;
    return 0;
}
```

### 3.2.3.pow()

逐元素计算指定的幂次

```python
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    Eigen::ArrayXXd arr = mat.array();
    std::cout << arr.pow(2) << std::endl;
    return 0;
}
```

### 3.2.4.tanh()

逐元素计算双曲正切

```python
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    Eigen::ArrayXXd arr = mat.array();
    std::cout << arr.tanh() << std::endl;
    return 0;
}
```

## 3.3.BDCSVD<>

对角分治奇异值分解.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    
    Eigen::BDCSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeFullU|Eigen::ComputeFullV);
    
    return 0;
}
```

### 3.3.1.matrixU()

获取m阶酉矩阵.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    
    Eigen::BDCSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeFullU|Eigen::ComputeFullV);
    std::cout << svd.matrixU() << std::endl;
    
    return 0;
}
```

### 3.3.2.matrixV()

获取n阶酉矩阵.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    
    Eigen::BDCSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeFullU|Eigen::ComputeFullV);
    std::cout << svd.matrixV() << std::endl;
    
    return 0;
}
```

### 3.3.3.singularValues()

获取奇异值(按照降序排序).

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    
    Eigen::BDCSVD<Eigen::MatrixXd> svd(mat, Eigen::ComputeFullU|Eigen::ComputeFullV);
    std::cout << svd.singularValues() << std::endl;
    
    return 0;
}
```

## 3.4.Map<>

映射到现有矩阵或者向量

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::Matrix<double, 2, 2> mat0;
    // 映射矩阵或者向量的首地址的指针, map的行数和列数
    Eigen::Map<Eigen::MatrixXd> map(mat0.data(), 1, 4);
    
    mat0 << 1, 2, 3, 4;
    
    std::cout << mat0 << std::endl;
    std::cout << map << std::endl;
    return 0;
}
```

## 3.5.Matrix<>

实例化一个已知矩阵

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
  	// 标量类型, 矩阵的行数和列数
    Eigen::Matrix<int, 2, 2> mat;
    mat << 1, 2, 3, 4;
    std::cout << mat << std::endl;
    return 0;
}
```

## 3.6.MatrixXd

实例化一个动态矩阵(双精度)

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    // 用法一
    Eigen::Matrix<double, 2, 2> mat0;
    mat0 << 1, 2, 3, 4;
    Eigen::MatrixXd mat1 = mat0;
    std::cout << mat1 << std::endl;
      
    // 用法二
    Eigen::MatrixXd mat2(2, 2);
    mat2 << 5, 6, 7, 8;
    std::cout << mat2 << std::endl;
    return 0;
}
```

### 3.6.1.adjoint()

获取矩阵的伴随(共轭转置)矩阵

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    std::cout << mat.adjoint() << std::endl;
    return 0;
}
```

### 3.6.2.array()

将矩阵修饰成数组，便于执行元素的操作

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    std::cout << mat.array() << std::endl;
    return 0;
}
```

### 3.6.3.cols()

获取矩阵的列数

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::Matrix<double, 2, 3> mat0;
    mat0 << 1, 2, 3, 4, 5, 6;
    Eigen::MatrixXd mat1 = mat0;
    std::cout << mat1.cols() << std::endl;
    return 0;
}
```

### 3.6.4.data()

返回矩阵或向量的首地址的指针

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::Matrix<double, 2, 2> mat0;
    mat0 << 1, 2, 3, 4;
    Eigen::MatrixXd mat1 = mat0;
    std::cout << mat1.data() << std::endl;
    return 0;
}
```

### 3.6.5.inverse()

计算矩阵的逆

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::Matrix<double, 2, 2> mat0;
    mat0 << 1, 2, 3, 4;
    Eigen::MatrixXd mat1 = mat0;
    std::cout << mat1.inverse() << std::endl;
    return 0;
}
```

### 3.6.6.maxCoeff()

返回矩阵的最大值和位置

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::Matrix<double, 2, 3> mat0;
    mat0 << 1, 2, 3, 4, 5, 6;
  
    // 用法一
    std::cout << mat0.maxCoeff() << std::endl;
    // 用法二
    int row, col;
    std::cout << mat0.maxCoeff(&row, &col) << std::endl;
    std::cout << "row:" << row << " col:" << col << std::endl;
    return 0;
}
```

### 3.6.7.row()

访问矩阵的指定行元素

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::Matrix<double, 2, 2> mat0;
    mat0 << 1, 2, 3, 4;
    Eigen::MatrixXd mat1 = mat0;
    // 从零开始的整数
    std::cout << mat1.row(1) << std::endl;
    return 0;
}
```

### 3.6.8.rows()

获取矩阵的行数

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::Matrix<double, 2, 3> mat0;
    mat0 << 1, 2, 3, 4, 5, 6;
    Eigen::MatrixXd mat1 = mat0;
    std::cout << mat1.rows() << std::endl;
    return 0;
}
```

### 3.6.9.rowwise()

对矩阵逐行进行操作

```python
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::Matrix<double, 2, 3> mat0;
    mat0 << 1, 2, 3, 4, 5, 6;
    Eigen::MatrixXd mat1 = mat0;
    std::cout << mat1.rowwise().sum() << std::endl;
    return 0;
}
```

### 3.6.10.size()

获取矩阵的元素总数

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::Matrix<double, 2, 3> mat0;
    mat0 << 1, 2, 3, 4, 5, 6;
    Eigen::MatrixXd mat1 = mat0;
    std::cout << mat1.size() << std::endl;
    return 0;
}
```

### 3.6.11.sum()

计算矩阵元素的和

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat0(2, 2);
    mat0 << 5, 6, 7, 8;
    std::cout << mat0.sum() << std::endl;
    return 0;
}
```

### 3.6.12.transpose()

对矩阵进行转置

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::Matrix<double, 2, 3> mat0;
    mat0 << 1, 2, 3, 4, 5, 6;
    Eigen::MatrixXd mat1 = mat0;
    std::cout << mat1.transpose() << std::endl;
    return 0;
}
```

## 3.7.RowVectorXd

实例化一个行向量(双精度)

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::RowVectorXd vec(4);
    vec << 1, 2, 3, 4;
    std::cout << vec << std::endl;
    return 0;
}
```

### 3.7.1.size()

获取行向量的元素总数

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::RowVectorXd vec(4);
    vec << 1, 2, 3, 4;
    std::cout << vec.size() << std::endl;
    return 0;
}
```

## 3.8.VectorXd

实例化一个列向量(双精度)

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::VectorXd vec(4);
    vec << 1, 2, 3, 4;
    std::cout << vec << std::endl;
    return 0;
}
```

### 3.8.1.asDiagonal()

将特征向量转换对角阵.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::VectorXd vec(2);
    vec << 1, 2;
    Eigen::MatrixXd mat = vec.asDiagonal();
    
    std::cout << mat << std::endl;
    return 0;
}
```

# 4.h5py

| 版本   | 描述             | 注意 |
| ------ | ---------------- | ---- |
| 2.10.0 | HDF5的python接口 |      |

## 4.1.File()

创建一个文件对象|h5py._hl.files.File

```python
import h5py
fp = h5py.File(name='./file.h5',  # 硬盘上的文件名|str or file-like object
               mode='w')  # 模式|{'r', 'r+', 'w', 'w- or x', 'a'}
```

### 4.1.1.attrs[]

添加到文件对象的属性

```python
import h5py
fp = h5py.File(name='./file.h5', mode='w')
fp.attrs['a'] = 1
print(fp.attrs['a'])
```

### 4.1.2.close()

关闭文件对象

```python
import h5py
fp = h5py.File('./file.h5', 'w')
fp.close()
```

### 4.1.3.create_dataset()

创建一个新的HDF5数据集|h5py._hl.dataset.Dataset

```python
import h5py
fp = h5py.File(name='./file.h5', mode='w')
dataset = fp.create_dataset(name='dataset',  # 数据集的名称|str
                            dtype=float)  # 元素的类型|numpy.dtype or str
```

### 4.1.4.create_group()

创建一个新的HDF5组|h5py._hl.group.Group

```python
import h5py
fp = h5py.File(name='./file.h5', mode='w')
group = fp.create_group(name='group')  # 组的名称|str
```

# 5.imageio

| 版本  | 描述           | 注意 |
| ----- | -------------- | ---- |
| 2.9.0 | 图像处理软件库 |      |

## 5.1.imread()

加载指定路径的图片|imageio.core.util.Array

```python
import imageio
image = imageio.imread(uri=filename)  # 要加载的文件的路径|str or pathlib.Path or bytes or file
```

# 6.lightgbm

| 版本  | 描述                 | 注意                                                      |
| ----- | -------------------- | --------------------------------------------------------- |
| 2.3.1 | 基于树的梯度提升框架 | 在macOS下安装需要先使用brew安装libomp 可直接在sklearn使用 |

## 6.1.LGBMClassifier()

实例化一个LGBMClassifier分类器

```python
from lightgbm import LGBMClassifier
model = LGBMClassifier(boosting_type,  # 集成方式|str|'gbdt'('gbdt'|'dart'|'goss'|'rf')|可选
                       max_depth,  # 基学习器的最大深度，负值表示没有限制|int|-1|可选
                       learning_rate,  # 学习率|float|0.1|可选
                       n_estimators)  # 树的数量|int|100|可选
```

### 6.1.1.fit()

训练LGBMClassifier分类器|self

```python
model.fit(X,  # 特征数据|array-like or 形状为[n_samples, n_features]的稀疏矩阵
          y,  # 标签|array-like
          eval_set)  # 验证集元组列表|list of (X, y) tuple|None|可选
```

### 6.1.2.predict()

进行预测|numpy.ndarray

```python
result = model.predict(X)# 用于预测的数据|array-like or 形状为[n_samples, n_features]的稀疏矩阵
```

# 7.matplotlib

| 版本  | 描述             | 注意 |
| ----- | ---------------- | ---- |
| 3.2.1 | Python绘图软件库 |      |

## 7.1.axes

| 版本 | 描述                                             | 注意 |
| ---- | ------------------------------------------------ | ---- |
| -    | axes是matplotlib的图形接口，提供设置坐标系的功能 |      |

### 7.1.1.annotate()

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

### 7.1.2.axis()

坐标轴的设置选项

```python
import matplotlib.pyplot as plt
ax = plt.subplot()
ax.axis('off')
```

### 7.1.3.clabel()

在等高线上显示高度

```python
import numpy as np
import matplotlib.pyplot as plt
ax = plt.subplot()
x = np.linspace(1, 9, 10)
y = np.linspace(1, 9, 10)
X, Y = np.meshgrid(x, y)
cs = ax.contour(X, Y, X + Y, colors='orange', linewidths=1)
ax.clabel(cs)
plt.show()
```

### 7.1.4.contour()

绘制等高线|matplotlib.contour.QuadContourSet

```python
import numpy as np
import matplotlib.pyplot as plt
ax = plt.subplot()
x = np.linspace(1, 9, 10)
y = np.linspace(1, 9, 10)
X, Y = np.meshgrid(x, y)
cs = ax.contour(X,  # 横坐标|array-like
                Y,  # 纵坐标|array-like
                X + Y,  # 横纵坐标的关系公式|array-like(必须是2D的)
                colors='orange',  # 等高线的颜色|str
                linewidths=1)  # 等高线的宽度|int
plt.show()
```

### 7.1.5.grid()

绘制网格线

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.grid(axis='x',  # 绘制的范围|str('both'|'x'|'y')|'both'|可选
        linestyle=':')  # 网格线的样式|str('-'|'--'|'-.'|':'|'None'|' '|''|'solid'|'dashed'|'dashdot'|'dotted')|'-'|可选
plt.show()
```

### 7.1.6.legend()

放置图例

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.legend(loc='center')  # 放置的位置|str('upper right'|'upper left'|'lower left'|'lower right'|'right'|'center left'|'center right'|'lower center'|'upper center'|'center')|'best'|可选
plt.show()
```

### 7.1.7.patch

| 版本 | 描述                                  | 注意 |
| ---- | ------------------------------------- | ---- |
| -    | patches是画布颜色和边框颜色的控制接口 |      |

#### 7.1.7.1.set_alpha()

设置画布的透明度

```python
ax.patch.set_alpha(alpha)  # 透明度|float|None
```

#### 7.1.7.2.set_facecolor()

设置画布的颜色

```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.patch.set_alpha(alpha=0.1)
ax.patch.set_facecolor(color='green')  # 颜色|str|None
plt.show()
```

### 7.1.8.set_title()

设置标题

```python
import matplotlib.pyplot as plt
ax = plt.subplot()
ax.set_title('this is title')
plt.show()
```

### 7.1.9.set_xlabel()

设置x轴的内容

```python
ax.set_xlabel(xlabel='this is x label')  # 内容|str
```

### 7.1.10.set_xticks()

设置x轴的刻度

```python
ax.set_xticks(ticks=[1, 2, 3, 4])  # 刻度|list(空列表就表示不显示刻度)
```

### 7.1.11.set_yticks()

设置y轴的刻度

```python
ax.set_yticks(ticks=[])  # 刻度|list(空列表就表示不显示刻度)
```

###  7.1.12.spines

| 版本 | 描述                         | 注意 |
| ---- | ---------------------------- | ---- |
| -    | 画布的边框，包括上下左右四个 |      |

#### 7.1.12.1.set_color()

设置画布的边框的颜色

```python
ax.spines['left'].set_color(c='red')  # 颜色|str
```

### 7.1.13.text()

给点添加文本|matplotlib.text.Text

```python
ax.text(x=0.5,  # 注释点的x坐标|float|0
        y=0.5,  # 注释点的y坐标|float|0
        s='text')  # 注释的文本内容|str|''
```

## 7.2.pyplot

| 版本 | 描述                                                         | 注意 |
| ---- | ------------------------------------------------------------ | ---- |
| -    | pyplot是matplotlib的state-based接口， 主要用于简单的交互式绘图和程序化绘图 | -    |

### 7.2.1.axis()

坐标轴的设置选项

```python
import matplotlib.pyplot as plt
plt.axis([xmin, xmax, ymin, ymax])
```

### 7.2.2.barh()

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

### 7.2.3.clabel()

在等高线上显示高度

```python
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(1, 9, 10)
y = np.linspace(1, 9, 10)
X, Y = np.meshgrid(x, y)
cs = plt.contour(X, Y, X + Y, colors='orange', linewidths=1)
plt.clabel(cs)
plt.show()
```

### 7.2.4.colorbar()

获取色彩条.|matplotlib.colorbar.Colorbar

```python
import matplotlib.pyplot as plt

arr = [[1, 2], [3, 4]]
plt.matshow(A=arr)
plt.colorbar()
plt.show()
```

### 7.2.5.figure()

创建一个画布|matplotlib.figure.Figure

```python
import matplotlib.pyplot as plt
figure = plt.figure(figsize)  # 画布的大小|(float, float)|(6.4, 4.8)|可选
```

### 7.2.6.imread()

加载指定路径的图片|numpy.ndarray

```python
import matplotlib.pyplot as plt
image = plt.imread(fname)  # 要加载的文件的路径|str or file-like
```

### 7.2.7.imshow()

将图片数组在画布上显示|matplotlib.image.AxesImage

```python
import matplotlib.pyplot as plt
plt.imshow(X,  # 希望显示的图像数据|array-like or PIL image
           cmap)  # 显示的色彩|str 
```

### 7.2.8.matshow()

将矩阵绘制成图像.

```python
import matplotlib.pyplot as plt

arr = [[1, 2], [3, 4]]
plt.matshow(A=arr)
plt.show()
```

### 7.2.9.pcolormesh()

使用非规则的矩形创建网格背景图

```python
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(1, 9, 10)
y = np.linspace(1, 9, 10)
X, Y = np.meshgrid(x, y)
plt.pcolormesh(X,  # 横坐标|array-like
               Y,  # 纵坐标|array-like
               X + Y,  # 横纵坐标的关系公式|array-like(必须是2D的)
               alpha=0.75,  # 透明度|float|None
               cmap='GnBu',  # 配色方案|str|None
               shading='nearest')  # 阴影|{'flat', 'nearest', 'gouraud', 'auto'}|'flat'|可选
plt.show()
```

### 7.2.10.plot()

绘制函数图像|list

```python
import matplotlib.pyplot as plt
plt.plot(*args)  # 函数的变量｜string or number且第一维度必须相同｜(x, y)
```

### 7.2.11.rcParams

实例化一个matplotlib的rc文件实例|matplotlib.RcParams

```python
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'Arial Unicode MS'  # 默认字体
```

### 7.2.12.savefig()

保存当前的画布

```python
import matplotlib.pyplot as plt
plt.savefig(fname)  # 要保存的文件的路径|str or PathLike or file-like object A path, or a Python file-like object
```

### 7.2.13.scatter()

绘制散点图|matplotlib.collections.PathCollection

```python
import matplotlib.pyplot as plt
x = [1, 2, 2, 3, 4, 5, 5.5, 6]
y = [1, 3, 2, 3, 4, 5, 5, 6]
plt.scatter(x,  # x坐标|scalar or array-like 形状必须是(n,)
            y,  # y坐标|scalar or array-like 形状必须是(n,)
            s=150,  # 点的大小|int
            c='red',  # 点的颜色|str
            marker='o',  # 点的标记的形状|str
            edgecolors='green')  # 标记的颜色|str
plt.show()
```

### 7.2.14.show()

显示所有的画布

```python
import matplotlib.pyplot as plt
plt.show()
```

### 7.2.15.subplot()

在当前画布上创建一个子图|matplotlib.figure.Figure和matplotlib.axes._subplots.AxesSubplot

```python
import matplotlib.pyplot as plt
ax = plt.subplot()
```

### 7.2.16.subplots()

创建一个画布和一组子图|matplotlib.figure.Figure和matplotlib.axes._subplots.AxesSubplot

```python
import matplotlib.pyplot as plt
figure, axesSubplot = plt.subplots(nrows=4,  # 列子图数量|int|1
                                   ncols=4,  # 行子图数量|int|1
                                   figsize=(10, 5))  # 画布的大小|tuple of int
```

### 7.2.17.subplots_adjust()

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

### 7.2.18.tight_layout()

自动调整子图

```python
import matplotlib.pyplot as plt
plt.tight_layout()
```

### 7.2.19.title()

设置标题

```python
import matplotlib.pyplot as plt
plt.title(label='this is title')  # 标题|str
plt.show()
```

### 7.2.20.xlabel()

设置x轴的内容

```python
import matplotlib.pyplot as plt
plt.xlabel(xlabel='x')
```

### 7.2.21.xlim()

设置x轴显示范围

```python
import matplotlib.pyplot as plt

plt.xlim([1, 2])  # (int, int)|[左界, 右界].
plt.show()
```

### 7.2.22.ylabel()

设置y轴的内容

```python
import matplotlib.pyplot as plt
plt.ylabel(ylabel='y')
```

# 8.numpy

| 版本   | 描述           | 注意 |
| ------ | -------------- | ---- |
| 1.18.4 | python数值计算 |      |

## 8.1.abs()

逐元素计算绝对值|numpy.ndarray

```python
import numpy as np
arr = [1, 2, -1, 3]
x = np.abs(arr)  # 输入的数组|array-like
```

## 8.2.any()

判断数组是否存在某个元素为True，如果有返回True，否则False|numpy.bool_

```python
import numpy as np
arr = [1, 1, 1, 1]
x = np.any(a=arr)  # 输入的数组|array-like
```

## 8.3.arange()

返回指定范围的数组|numpy.ndarray

```python
import numpy as np
arr = np.arange(start=2,  # 开始的值|number
                stop=10)  # 结束的值|number
```

## 8.4.argmax()

返回指定维度最大值的索引|numpy.int64

```python
import numpy as np
arr = [1, 2, 3]
np.argmax(a=arr,  # 输入的数组|array_like
          axis=None)  # 筛选所沿的维度|int|None|可选 
```

## 8.5.around()

逐元素四舍五入取整|numpy.ndarray

```python
import numpy as np
arr = [1.4, 1.6]
x = np.around(a=arr)  # 输入的数组|array-like
```

## 8.6.asarray()

将输入转换为一个数组|numpy.ndarray

```python
import numpy as np
arr = [1, 2, 3]
nd_arr = np.asarray(a=arr,  # 输入的数据|array-like
                    dtype=None)  # 元素的数据类型|data-type|None|可选
```

## 8.7.asmatrix()

将输入转换为一个矩阵|numpy.matrix

```python
import numpy as np
mat = np.asmatrix(data=[1, 2, 3, 4])  # 输入的数据|array-like
```

## 8.8.ceil()

逐元素进行向上取整|numpy.float64(输入是数组时numpy.ndarray)

```python
import numpy as np
arr = [5.1, 4.9]
np.ceil(arr)  # 输入数据|array_like
```

## 8.9.concatenate()

按照指定维度合并多个数组|numpy.ndarray

```python
import numpy as np
arr1 = [[1], [1], [1]]
arr2 = [[2], [2], [2]]
arr3 = [[3], [3], [3]]
np.concatenate([arr1, arr2, arr3],  # 要合并的数组|array-like
               axis=1)  # 沿指定维度合并|int|0|可选
```

## 8.10.c_[]

将第二个数组沿水平方向连接|numpy.ndarray

```python
import numpy as np
arr1 = [[1, 2], [1, 2], [1, 2]]
arr2 = [[3], [3], [3]]
arr = np.c_[arr1, arr2]
```

## 8.11.diag()

提取对角线的值, 或者构建对角阵.|numpy.ndarray

```python
import numpy as np
arr = [1, 2, 3]
x = np.diag(v=arr)  # array-like|输入的数组.
```

## 8.12.dot()

计算两个数组的点乘|numpy.ndarray

```python
import numpy as np
arr1 = [[1, 2, 3]]
arr2 = [[1], [2], [3]]
np.dot(a=arr1, b=arr2)  # 输入的数组|array-like
```

## 8.13.equal()

逐个元素判断是否一致|numpy.bool_(输入是数组时numpy.ndarray)

```python
import numpy as np
arr1 = [1, 2, 3]
arr2 = [1, 2, 2]
np.equal(arr1, arr2)  # 输入的数组|array-like
```

## 8.14.exp()

逐元素计算e的幂次|numpy.float64(输入是数组时numpy.ndarray)

```python
import numpy as np
arr = [1, 2, 3]
np.exp(arr)  # 输入数据|array-like
```

## 8.15.expm1()

逐元素计算e的幂次并减一|numpy.float64(输入是数组时numpy.ndarray)

```python
import numpy as np
arr = [1, 2, 3]
np.expm1(arr)  # 输入数据|array-like
```

## 8.16.expand_dims()

扩展数组的形状，增加维度|numpy.ndarray

```python
import numpy as np
a = [1, 2]
a = np.expand_dims(a=a,  # 输入的数组|array-like
                   axis=0)  # 添加新维度的位置|int or tuple of ints
```

## 8.17.eye()

生成一个单位阵|numpy.ndarray

```python
import numpy as np
matrix = np.eye(N=3)  # 矩阵的行数|int
```

## 8.18.hstack()

按照水平顺序合成一个新的数组|numpy.ndarray

```python
import numpy as np
arr1 = [[1, 2, 3, 4], [1, 2, 3, 4]]
arr2 = [[5, 6], [5, 6]]
a = np.hstack(tup=(arr1, arr2))  # 数组序列|array-like
```

## 8.19.linalg

| 版本 | 描述                  | 注意 |
| ---- | --------------------- | ---- |
| -    | numpy的线性代数函数包 |      |

### 8.19.1.inv()

计算矩阵的逆|numpy.ndarray

```python
import numpy as np
arr = [[1, 2],
       [3, 4]]
matrix = np.linalg.inv(a=arr)  # 输入的矩阵|array_like
```

### 8.19.2.norm()

计算范数|numpy.float64

```python
import numpy as np
arr = [[1, 2], [3, 4]]
np.linalg.norm(x=arr,  # 输入的矩阵或向量|array_like(维数必须是1维或2维)
               ord=1)  # 范数选项｜int or str(non-zero|int|inf|-inf|'fro'|'nuc')|None(计算2-范数)|可选
```

### 8.19.3.svd()

奇异值分解.|numpy.ndarray

```python
import numpy as np
arr = [[1, 2], [3, 4]]
u, s, vh = np.linalg.svd(a=arr)  # array_like|输入的矩阵.
```

## 8.20.linspace()

生成指定间隔内的等差序列|numpy.ndarray

```python
import numpy as np
np.linspace(start=1,  # 序列的起始值|array_like
            stop=5,  # 序列的结束值|array_like
            num=10)  # 生成序列的样本的个数|int|50|可选
```

## 8.21.load()

从npy、npz、pickled文件加载数组或pickled对象|array or tuple or dict

```python
import numpy as np
np.load(file,  # 文件|file-like object or string or pathlib.Path
        allow_pickle,  # 允许加载npy文件中的pickle对象|bool|False|可选
        encoding)  # 读取的编码方式|str|'ASCII'|可选
```

## 8.22.log()

逐元素计算自然对数|numpy.float64(输入是数组时numpy.ndarray)

```python
import numpy as np
np.log(1)  # 输入数据|array_like
```

## 8.23.log1p()

逐元素计算本身加一的自然对数|numpy.float64(输入是数组时numpy.ndarray)

```python
import numpy as np
np.log1p(1)  # 输入数据|array_like
```

## 8.24.log2()

逐元素计算以2为底对数|numpy.float64(输入是数组时numpy.ndarray)

```python
import numpy as np
np.log2(1)  # 输入数据|array_like
```

## 8.25.mat()

将输入转换为一个矩阵|numpy.matrix

```python
import numpy as np
arr = [[1, 2, 3]]
matrix = np.mat(data=arr  # 输入数据|array-like
                dtype=None)  # 生成矩阵元素的数据类型|data-type|None|可选
```

## 8.26.matmul()

两个数组的矩阵乘积|numpy.ndarray

```python
import numpy as np
arr1 = [[1, 2, 3]]
arr2 = [[1], [2], [3]]
np.matmul(arr1, arr2)  # 输入的数组|array-like（不能是标量）
```

## 8.27.max()

返回最大值或者沿着某一维度最大值|numpy.ndarray or scalar

```python
import numpy as np
arr = [1., 2., 5., 3., 4.]
np.max(a=arr,  # 输入的数组|array-like
       axis=None)  # 所沿的维度|int|None|可选 
```

## 8.28.maximum()

返回数组逐个元素的最大值|numpy.ndarray

```python
import numpy as np
arr1 = [2, 3, 4]
arr2 = [1, 5, 2]
np.maximum(arr1, arr2)  # 输入的数组|array-like
```

## 8.29.mean()

沿着指定维度计算均值|numpy.float64

```python
import numpy as np
arr = [1, 2, 3]
np.mean(arr,  # 输入的数组|array-like
        axis=None)  # 所沿的维度|int or tuple of ints|None|可选 
```

## 8.30.meshgrid()

生成一个坐标矩阵|list of numpy.ndarray

```python
import numpy as np
x_crood = np.linspace(0, 4, 5)
y_crood = np.linspace(0, 4, 5)
vector_matrix = np.meshgrid(x_crood, y_crood)  # 坐标向量|array_like
```

## 8.31.nonzero()

返回非零元素的索引|tuple

```python
import numpy as np
arr = np.asarray([1, 2, 3, 4, 0, 0, 5])
np.nonzero(a=arr)  # 输入的数组|array-like
```

## 8.32.ones()

创建一个指定为形状和类型的全一数组|numpy.ndarray

```python
import numpy as np
arr = np.ones(shape=[2, 3],  # 数组的形状|int or sequence of ints
              dtype=np.int8)  # 数组元素的数据类型|data-type|numpy.float64|可选
```

## 8.33.power()

逐个元素计算第一个元素的第二个元素次幂|scalar(输入是数组时numpy.ndarray)

```python
import numpy as np
x = np.power(2.1, 3.2)   # x1底数、x2指数|array_like
```

## 8.34.random

| 版本 | 描述                    | 注意 |
| ---- | ----------------------- | ---- |
| -    | numpy的随机数生成函数包 |      |

### 8.34.1.normal()

生成正态分布的样本|numpy.ndarray or scalar

```python
import numpy as np
arr = np.random.normal(size=[2, 3])  # 形状|int or tuple of ints|None(None则只返回一个数)|可选
```

### 8.34.2.permutation()

随机置换序列|numpy.ndarray

```python
import numpy as np
arr = [1, 2, 3, 4]
arr = np.random.permutation(arr)  # 输入的数组|array-like
```

### 8.34.3.randint()

从给定区间[low, high)生成随机整数|int or numpy.ndarray

```python
import numpy as np
np.random.randint(low=1,  # 下界|int or array-like of ints
                  high=10)  # 上界|int or array-like of ints|None(如果high为None则返回区间[0, low))|可选
```

### 8.34.4.rand()

生成一个指定形状的随机数数组|float or numpy.ndarray

```python
import numpy as np
arr = np.random.rand(2, 3)  # 数组的维度|int|(如果形状不指定，仅返回一个随机的浮点数)|可选
```

### 8.34.5.randn()

生成一个指定形状的标准正态分布的随机数数组|float or numpy.ndarray

```python
import numpy as np
arr = np.random.randn(2, 3)  # 数组的维度|int|(如果形状不指定，仅返回一个随机的浮点数)|可选
```

### 8.34.6.RandomState()

实例化一个伪随机数生成器|RandomState(MT19937)

```python
import numpy as np
rs = np.random.RandomState(seed=2020)  # 随机种子|int|None|可选
```

#### 8.34.6.1.shuffle()

随机打乱数据

```python
import numpy as np
rs = np.random.RandomState(seed=2020)
arr = [1, 2, 3, 4]
rs.shuffle(arr)
print(arr)
```

### 8.34.7.seed()

设置随机数生成器的随机种子

```python
import numpy as np
np.random.seed(seed)  # 随机种子|int|None|可选
```

## 8.35.ravel()

展平一个数组|numpy.ndarray

```python
import numpy as np
arr = np.asarray([[1, 2], [3, 4]])
np.ravel(a=arr)  # 输入的数组|array-like
```

## 8.36.reshape()

返回一个具有相同数据的新形状的数组|numpy.ndarray

```python
import numpy as np
arr = [1, 2, 3, 4]
np.reshape(a=arr,  # 要改变形状的数组|array_like
           newshape=[2, 2])  # 新的形状|int or tuple of ints
```

## 8.37.save()

将数组转换为numpy保存进二进制的npy文件

```python
import numpy as np
arr = [1, 2, 3]
np.save(file='arr.npy',  # 文件名|file or str or pathlib.Path
        arr=arr,  # 要保存的数组|array-like
        allow_pickle=True)  # 允许使用pickle对象保存数组|bool|True|可选
```

## 8.38.sort()

返回排序数组的副本|numpy.ndarray

```python
import numpy as np
arr = [1, 3, 2, 4]
new_arr = np.sort(a=arr)  # 要排序的数组|array_like
```

## 8.39.split()

将一个数组拆分为多个|list of ndarrays

```python
import numpy as np
arr = np.asarray([[1, 2, 5, 6], [3, 4, 7, 8]])
arr_list = np.split(ary=arr,  # 要拆分的数组|numpy.ndarray
                    indices_or_sections=2,  # 拆分方法|int or 1-D array(整数必须能整除)
                    axis=1)  # 沿某维度分割|int|0|可选
```

## 8.40.sqrt()

逐元素计算e的幂次|numpy.float64(输入是数组时numpy.ndarray)

```python
import numpy as np
arr = [1, 2, 3]
np.sqrt(arr)  # 输入数据|array_like
```

## 8.41.squeeze()

删除数组中维度为一的维度|numpy.ndarray

```python
import numpy as np
arr = [[1, 2, 3]]
np.squeeze(arr)  # 输入数据|array_like
```

## 8.42.std()

沿指定维度计算标准差|numpy.float64

```python
import numpy as np
arr = [1, 2, 3]
np.std(a=arr,  # 输入的数组|array-like
       axis=None)  # 所沿的维度|int or tuple of ints|None|可选
```

## 8.43.sum()

沿指定维度求和|numpy.ndarray

```python
import numpy as np
arr = [[1.2, 2.3, 3], [4, 5, 6]]
np.sum(arr,  # 输入的数组|array-like
       axis=1)  # 所沿的维度|int or tuple of ints|None|可选
```

## 8.44.transpose()

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

## 8.45.var()

沿指定维度方差|numpy.ndarray

```python
import numpy as np
arr = [[1.2, 2.3, 3], [4, 5, 6]]
np.var(arr,  # 输入的数组|array-like
       axis=1)  # 所沿的维度|int or tuple of ints|None|可选
```

## 8.46.void()

创建一个numpy.void类型的对象

```python
import numpy as np
o = np.void(b'abc')  # 输入的数据|bytes
```

## 8.47.zeros()

创建一个指定为形状和类型的全零数组|numpy.ndarray

```python
import numpy as np
arr = np.zeros(shape=[2, 3],  # 数组的形状|int or sequence of ints
               dtype=np.int8)  # 数组元素的数据类型|data-type|numpy.float64|可选
```

# 9.pandas

| 版本  | 描述                 | 注意 |
| ----- | -------------------- | ---- |
| 1.0.3 | 结构化数据分析软件库 |      |

## 9.1.concat()

沿指定维度合并pandas对象|pandas.core.frame.DataFrame or pandas.core.series.Series

```python
import pandas as pd
sr1 = pd.Series([1, 2, 3])
sr2 = pd.Series([1, 2, 3])
sr3 = pd.Series([1, 2, 3])
df = pd.concat([sr1, sr2, sr3],  # 待合并数据列表|DataFrame or Series
               axis=1)  # 沿行或者列合并|{0/'index', 1/'columns'}|0
```

## 9.2.DataFrame()

实例化一个DataFrame对象(二维，可变大小的，结构化数据)

```python
import pandas as pd
df_map = {'index': [0, 1, 2], 'values': [0.1, 0.2, 0.3]}
df = pd.DataFrame(data=df_map,  # 输入的数据|ndarray or Iterable or dict or DataFrame(数据必须是相同数据类型且为结构化的)
                  index=[1, 2, 3],  # 行索引|Index or array-like|None(默认0,1,...,n)
                  columns=None)  # 列索引|Index or array-like|None(默认0,1,...,n)
```

### 9.2.1.columns

返回dataframe的行标签|pandas.core.indexes.base.Index

```python
import pandas as pd
df_map = {'index': [0, 1, 2], 'values': [0.1, 0.2, 0.3]}
df = pd.DataFrame(data=df_map)
print(df.columns)
```

### 9.2.2.corr()

计算列的成对相关度.|pandas.core.frame.DataFrame

```python
import pandas as pd

df = pd.DataFrame(data={'index': [0, 1, 2], 'values': [0.1, 0.2, 0.3]})
correlation = df.corr()
```

### 9.2.3.drop()

删除指定行或者列|pandas.core.frame.DataFrame

```python
import pandas as pd
df = pd.DataFrame([1, 2, 3])
df = df.drop(labels=1,  # 希望删除的行或者列|single label or list-like
             axis=0)  # 删除行或者列|{0/'index', 1/'columns'}|0
```

### 9.2.4.drop_duplicates()

 删除重复的行|pandas.core.frame.DataFrame

```python
import pandas as pd
df = pd.DataFrame({'key': [0, 1, 2, 1], 'values': ['a', 'b', 'a', 'b']})
df.drop_duplicates(subset=None,  # 仅选子列进行删除|None
                   keep='first',  # 保留重复项的位置|{'first', 'last', False}('first'保留第一次出现的, 'last'保留最后一次出现的, False全部删除)|'first'
                   inplace=True)  # 是否修改源DataFrame|bool(True没有返回值，False返回一个新的DataFrame)|False
```

### 9.2.5.fillna()

填充缺失值|pandas.core.frame.DataFrame or None

```python
import pandas as pd
df = pd.DataFrame({'key': ['a', 'b', 'c', 'd'], 'value': [1, 2, None, 4]})
df.fillna(value=10,  # 填充进的值
          inplace=True)  # 是否修改源DataFrame|bool(True没有返回值，False返回一个新的DataFrame)|False
```

### 9.2.6.head()

返回前n行数据|pandas.core.frame.DataFrame

```python
import pandas as pd
df = pd.DataFrame([1, 2, 3])
print(df.head(n=1))  # 选择的行数|int|5
```

### 9.2.7.iloc[]

按照行号取出数据|pandas.core.frame.DataFrame or pandas.core.series.Series

```python
import pandas as pd
df = pd.DataFrame([[1, 4], [2, 5], [3, 6]])
new_df = df.iloc[0:2]  # 要提取的数据|int or array of int or slice object with ints
```

### 9.2.8.info()

显示摘要信息(包括索引、非Non值计数、数据类型和内存占用)

```python
import pandas as pd
df_map = {'index': [0, 1, 2], 'values': [0.1, 0.2, 0.3]}
df = pd.DataFrame(data=df_map,
                  index=[1, 2, 3],
                  columns=None)
df.info()
```

### 9.2.9.loc[]

按照行名称取出数据|pandas.core.frame.DataFrame or pandas.core.series.Series

```python
import pandas as pd
df_map = [[1, 4], [2, 5], [3, 6]]
df = pd.DataFrame(df_map, index=['a', 'b', 'c'])
new_df = df.loc['a':'b']  # 要提取的数据|label or array of label or slice object with labels(没有名称的时候就是iloc函数)
```

### 9.2.10.median()

获取中位数|pandas.core.series.Series

```python
import pandas as pd
df = pd.DataFrame({'key': ['a', 'b', 'c', 'd'], 'value': [1, 2, 3, 4]})
df.median()
```

### 9.2.11.merge()

将两列进行合并|pandas.core.frame.DataFrame

```python
import pandas as pd
df1 = pd.DataFrame({'index': [0, 1, 2], 'values': ['a', 'b', 'a']})
df2 = pd.DataFrame({'values': ['a', 'b'], 'numbers': [1, 2]})
df = pd.merge(left=df1,  # 参与合并左侧的数据|DataFrame
              right=df2,  # 参与合并右侧的数据|DataFrame
              how='inner',  # 合并方式|{'inner', 'outer', 'left' 'right'}(交集, 并集, 左连接, 右连接)|'inner'
              left_on='values',  # 左侧数据的参考项|label or list, or array-like
              right_on='values',  # 右侧数据的参考项|label or list, or array-like
              sort=True)  # 是否排序|bool|True
```

### 9.2.12.replace()

替换DataFrame中的值|pandas.core.frame.DataFrame or None

```python
import pandas as pd
df = pd.DataFrame([1, 2, 3, 4])
new_df = df.replace(to_replace=1,  # 被替换的值|scalar or dict or list or str or regex
                    value=2,  # 替换的值|scalar or dict or list or str or regex|None
                    inplace=False)  # 是否修改源DataFrame|bool(True没有返回值，False返回一个新的DataFrame)|False
```

### 9.2.13.reset_index()

重置DataFrame的索引为从零开始的整数索引|pandas.core.frame.DataFrame or None

```python
import pandas as pd
df_map = [[1, 4], [2, 5], [3, 6]]
df = pd.DataFrame(df_map, index=['a', 'b', 'c'])
new_df = df.reset_index(drop=True,  # 是否丢弃原来的索引|bool|False
                        inplace=False)  # 是否修改源DataFrame|bool(True没有返回值，False返回一个新的DataFrame)|False
```

### 9.2.14.sample()

随机采样指定个数的样本|pandas.core.frame.DataFrame

```python
import pandas as pd
df = pd.DataFrame([1, 2, 3, 4])
new_df = df.sample(n=None,  # 采样的个数|int|None(表示采样全部)|可选
                   frac=True)  # 是否对全部数据采样|bool|None(不可与n同时为非None的值)|可选
```

## 9.3.date_range()

生成一个固定时间频率的索引|pandas.core.indexes.datetimes.DatetimeIndex

```python
import pandas as pd
datetime_index = pd.date_range(start='2014/06/10',  # 生成时间开始的界线|str or datetime-like
                               periods=5,  # 生成的数量|int|可选
                               freq='M')  # 生成的频率|str or DateOffset|'D'
```

## 9.4.fillna()

填充缺失的值|pandas.core.frame.DataFrame or None

```python
import pandas as pd
df = pd.DataFrame([('a',), ('b', 2), ('c', 3)])
df.fillna(value=1,  # 缺失值|scalar, dict, Series, or DataFrame|None
          inplace=True)  # 是否修改源DataFrame|bool(True没有返回值，False返回一个新的DataFrame)|False
```

## 9.5.get_dummies()

将类别变量转换为dummy编码的变量|pandas.core.frame.DataFrame

```python
import pandas as pd
sr = pd.Series(['a', 'b', 'c', 'a'])
coding = pd.get_dummies(data=sr)  # 输入的数据|array-like, Series, or DataFrame
```

## 9.6.group_by()

使用给定列的值进行分组|pandas.core.groupby.generic.DataFrameGroupBy

```python
import pandas as pd
df = pd.DataFrame([[0, False], [1, True], [2, False]], index=['a', 'b', 'c'], columns=['c1', 'c2'])
group = df.groupby(by='c2')  # 分组依据(列名)|str(name of columns)
print(group.groups)
```

## 9.7.isnull()

查找缺失值|pandas.core.series.Series

```python
import pandas as pd
sr = pd.Series([1, 2, None, 4])
sr.isnull()
```

## 9.8.notnull()

查找非缺失值|pandas.core.series.Series

```python
import pandas as pd
sr = pd.Series([1, 2, None, 4])
sr.notnull()
```

## 9.9.read_csv()

读取csv文件|DataFrame or TextParser

```python
import pandas as pd
new_df = pd.read_csv(filepath_or_buffer,  # 文件名|str or file handle|None
                     sep=',',  # 字段分隔符|str|','|可选
                     header=0,  # 列名所在的行|int or str|0|可选
                     index_col=None,  # 行名所在的列|int or str|None|可选
                     encoding=None)  # 编码方式|str|None|可选
```

## 9.10.Series()

实例化一个Series对象(一维)|pandas.core.series.Series

```python
import pandas as pd
sr = pd.Series(data=[1, 2, 3, 4])  # 输入的数据|ndarray or Iterable or dict(数据必须是相同数据类型)
```

### 9.10.1.dt

#### 9.10.1.1.day

提取时间中日期|pandas.core.series.Series

```python
import pandas as pd
sr = pd.Series(data=['2020/12/25', '1998/10/26'])
sr = pd.to_datetime(sr)
day_sr = sr.dt.day
```

#### 9.10.1.2.dayofweek

将时间转换为周几|pandas.core.series.Series

```python
import pandas as pd
sr = pd.Series(data=['2020/11/26', '1998/10/26'])
sr = pd.to_datetime(sr)
dayofweek_sr = sr.dt.dayofweek
```

#### 9.10.1.3.hour

提取时间中小时|pandas.core.series.Series

```python
import pandas as pd
sr = pd.Series(data=['2020/12/25 10:30:00', '1998/10/26 23:00:01'])
sr = pd.to_datetime(sr)
hour_sr = sr.dt.hour
```

#### 9.10.1.4.month

提取时间中月份|pandas.core.series.Series 

```python
import pandas as pd
sr = pd.Series(data=['2020/12/25', '1998/10/26'])
sr = pd.to_datetime(sr)
month_sr = sr.dt.month
```

#### 9.10.1.5.weekday

将时间转换为周几(和dayofweek功能相同)|pandas.core.series.Series

```python
import pandas as pd
sr = pd.Series(data=['2020/11/26', '1998/10/26'])
sr = pd.to_datetime(sr)
dayofweek_sr = sr.dt.weekday
```

### 9.10.2.isin()

检查某个值是否在Series中|pandas.core.series.Series

```python
import pandas as pd
sr = pd.Series(data=[1, 2, 3, 4])
bool_list = sr.isin(values=[4])  # 检查的值|set or list-like
```

### 9.10.3.map()

使用输入的关系字典进行映射|pandas.core.series.Series

```python
import pandas as pd
df = pd.DataFrame([1, 2, 1])
map_dict = {1: 'a', 2: 'b'}
new_sr = df[0].map(map_dict)  # 映射关系|dict
```

### 9.10.4.mode()

返回数据的众数|pandas.core.series.Series

```python
import pandas as pd
sr = pd.Series([1, 2, 2, 2, 3])
m = sr.mode()
```

### 9.10.5.plot()

绘制图像

```python
import pandas as pd
import matplotlib.pyplot as plt

sr = pd.Series([0, 1, 2])
sr.plot()
plt.show()
```

### 9.10.6.tolist()

返回Series值组成的列表|list

```python
import pandas as pd
sr = pd.Series([1, 2, 2, 2, 3])
l = pd.Series.tolist(sr)
```

## 9.11.to_csv()

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

## 9.12.to_datetime()

将输入数据转换为时间|pandas.core.series.Series

```python
import pandas as pd
sr = pd.Series(data=['2020/11/26', '1998/10/26'])
sr = pd.to_datetime(sr)
```

## 9.13.unique()

返回唯一值组成的数组|numpy.ndarray

```python
import pandas as pd
sr = pd.Series([1, 2, 2, 2, 3])
arr = pd.unique(values=sr)  # 输入的数据|1d array-like
```

## 9.14.values

返回Series或者DataFrame的值组成的数组|numpy.ndarray or ndarray-like

```python
import pandas as pd
df = pd.DataFrame([1, 2, 3])
arr = df.values
```

## 9.15.value_counts()

统计非空数值的出现次数|pandas.core.series.Series

```python
import pandas as pd
sr = pd.Series([1, 2, 2, 2, 3])
c = pd.value_counts(values=sr)  # 输入的数据|1d array-like
```

# 10.PIL

| 版本  | 描述           | 注意                         |
| ----- | -------------- | ---------------------------- |
| 7.1.2 | 图像处理软件库 | 安装时使用pip install pillow |

## 10.1.Image

| 版本 | 描述          | 注意 |
| ---- | ------------- | ---- |
| -    | PIL图像修饰器 |      |

### 10.1.1.fromarray()

将一个numpy.ndarray转换成一个PIL.Image.Image|PIL.Image.Image

```python
import numpy as np
from PIL.Image import fromarray
arr = np.asarray([[0.1, 0.2], [0.3, 0.4]])
img = fromarray(obj=arr)  # 输入的数组|numpy.ndarray
```

### 10.1.2.open()

加载指定路径的图片|PIL.Image.Image

```python
from PIL.Image import open
img = open(fp)  # 要加载的文件的路径|str or pathlib.Path object or a file object
```

### 10.1.3.resize()

将图像调整到指定大小并返回副本|PIL.Image.Image

```python
from PIL.Image import open
img = open(fp)
new_img = img.resize(size=(400, 400))  # 调整后图像的尺寸|2-tuple: (width, height)
```

## 10.2.ImageOps

| 版本 | 描述            | 注意 |
| ---- | --------------- | ---- |
| -    | PIL标准图像操作 | -    |

### 10.2.1.autocontrast

最大化(标准化)图片对比度|PIL.Image.Image

```python
from PIL.Image import open
from PIL.ImageOps import autocontrast

image_path = './image.jpg'

image = open(image_path)
processed_image = autocontrast(image)
```

# 11.pybind11

| 版本  | 描述                | 注意                                                         |
| ----- | ------------------- | ------------------------------------------------------------ |
| 2.6.0 | CPP和Python操作接口 | 需要安装python软件包pytest pybind11，Linux还需要python3-dev，不要使用brew安装 |

## 11.1.第一个例子

1. example.cc代码

```c++
#include "pybind11/pybind11.h"
namespace py = pybind11;

int add(int i, int j) {
  	return i + j;
}

// example就是python软件包名
PYBIND11_MODULE(example, m) {
    // python下就是.__doc__
		m.doc() = "模块的描述信息";
  	// python下函数名
  	m.def("add",
          // 对应cpp函数的引用
          &add,
          // python下就是add.__doc__ 使用 R"pbdoc()pbdoc"将保存注释的格式
          "A function which adds two numbers",
          // python函数的参数名称
          py::arg("i"), py::arg("j"));
    // 设置python软件包内的变量
  	m.attr("__version__") = "1.0";
}
```

2. 使用c++编译，并生成so文件

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` example.cc -o example`python3-config --extension-suffix`
```

## 11.2.绑定Eigen

1. example.cc代码

```c++
#include "pybind11/pybind11.h"
// 添加eigen.h的头文件
#include "pybind11/eigen.h"

#include "Eigen/LU"
namespace py = pybind11;

Eigen::MatrixXd inv(const Eigen::MatrixXd &xs) {
  	return xs.inverse();
}

PYBIND11_MODULE(example, m) {
  	m.doc() = "模块的描述信息";
  	m.def("inv", &inv);
}
```

2. 使用c++编译，并生成so文件

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
 -I /path/to/eigen/3.3.7/include/eigen3 \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

## 11.3.实现重载

1. sample.cc代码

```c++
#include "pybind11/pybind11.h"

int add(int x, int y) {
    return x + y;
}

double add(double x, double y) {
    return x + y;
}

PYBIND11_MODULE(example, m) {
  	// 实现重载.
    m.def("add", pybind11::overload_cast<int, int>(&add));
    m.def("add", pybind11::overload_cast<double, double>(&add));
}
```

2. 使用c++编译，并生成so文件

```shell
c++ -O3 -Wall -shared -std=c++14 -undefined dynamic_lookup `python3 -m pybind11 --includes` example.cc -o example`python3-config --extension-suffix`
```

## 11.4.实现自定义的异常

pybind11中仅提供了有限的CPP异常自动转换为Python异常，只有注册后Python解释器才能捕获

1. sample.cc代码

```c++
#include "pybind11/pybind11.h"

class CustomException : public std::exception {
public:
    const char * what() const noexcept override {
        return "自定义异常";
    }
};

// 测试函数直接抛出异常.
void test() {
    throw CustomException();
}

PYBIND11_MODULE(example, m) {
    // 注册定义异常(最后一个参数可将异常在Python中继承Python具体的异常, 使之可被Python解释器以具体的Python异常捕获)
    pybind11::register_exception<CustomException>(m, "PyCustomException", PyExc_BaseException);

    m.def("test", &test);
}
```

2. 使用c++编译，并生成so文件

```shell
 c++ -O3 -Wall -shared -std=c++14 -undefined dynamic_lookup \
  -I /usr/local/Cellar/eigen/3.3.7/include/eigen3 \
  `python3 -m pybind11 --includes` \
  example.cc -o example`python3-config --extension-suffix`
```

3. test.py 测试

```python
import example

try:
    example.test()
except BaseException:
    print('成功捕获异常')
```

## 11.5.实现类

这个例子将实现CPP类转换为Python类，访问函数和变量

1. example.cc

```c++
#include <iostream>
#include "pybind11/pybind11.h"

class Animal {
  public:
    Animal() {
        this->name = "animal";
    }

    void call() {
        std::cout << "Ah!" << std::endl;
    }

public:
    std::string name; // 私有变量不能转换到python下，只能通过设定def_readonly设置权限
};

PYBIND11_MODULE(example, m) {
    pybind11::class_<Animal>(m, "Animal")
        .def(pybind11::init())
        .def("call", &Animal::call)
        .def_readonly("name", &Animal::name);
}
```

2. 使用c++编译，并生成so文件

```shell
 c++ -O3 -Wall -shared -std=c++14 -undefined dynamic_lookup \
 -I /usr/local/Cellar/eigen/3.3.7/include/eigen3 \
 `python3 -m pybind11 --includes` \
example.cc -o example`python3-config --extension-suffix`
```

## 11.6.实现继承

1. example.cc

```c++
#include <iostream>
#include <utility>
#include "pybind11/pybind11.h"

class Animal {
  public:
    Animal() {
        this->name = "animal";
    }

    void call() {
        std::cout << "Ah!" << std::endl;
    }

  public:
    std::string name;
};

class Cat: public Animal {
  public:
    Cat() {
        this->name = "cat";
    }

    explicit Cat(std::string name) {
        this->name = std::move(name);
    }

    void call() {
        std::cout << "Meow~" << std::endl;
    }
};

class Dog: public Animal {
  public:
    Dog() {
        this->name = "dog";
    }

    void call() {
        std::cout << "Wang~" << std::endl;
    }
};

PYBIND11_MODULE(example, m) {
    pybind11::class_<Animal>(m, "Animal")
        .def(pybind11::init())
        .def("call", &Animal::call)
        .def_readonly("name", &Animal::name);

    pybind11::class_<Cat, Animal>(m, "Cat")
        .def(pybind11::init())
        .def(pybind11::init<std::string>()) // pybind11不能自动实现重载函数, 必须显式声明出来.
        .def("call", &Cat::call)
        .def_readonly("name", &Cat::name);

    pybind11::class_<Dog>(m, "Dog") // 不声明继承父类, 在Python中将被认定为直接继承的object.
        .def(pybind11::init())
        .def("call", &Dog::call)
        .def_readonly("name", &Dog::name);
}
```

2. 使用c++编译，并生成so文件

```shell
 c++ -O3 -Wall -shared -std=c++14 -undefined dynamic_lookup \
 -I /usr/local/Cellar/eigen/3.3.7/include/eigen3 \
 `python3 -m pybind11 --includes` \
example.cc -o example`python3-config --extension-suffix`
```

3. test.py 测试

```python
import example

animal = example.Animal()
print(animal.name)
animal.call()

cat = example.Cat("Garfield")
print(cat.name)
cat.call()
print(isinstance(cat, example.Animal))

dog = example.Dog()
print(isinstance(dog, example.Animal))
```

## 11.7.设置默认参数

1. example.cc代码

```c++
#include "pybind11/pybind11.h"

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(example, m) {
    m.def("add", &add, pybind11::arg("i")=1, pybind11::arg("j")=1);  // 在pybind11::arg上直接添加默认参数.
}
```

2. 使用c++编译，并生成so文件

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` example.cc -o example`python3-config --extension-suffix`
```

3. test.py 测试

```python
import example
ans = example.add()
print(ans)
```

## 11.8.使用Python的print函数

1. example.cc代码

```c++
#include "pybind11/pybind11.h"

void my_print(std::string text) {
    pybind11::print(text);
}

PYBIND11_MODULE(example, m) {
    m.def("my_print", &my_print);
}
```

2. 使用c++编译，并生成so文件

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` example.cc -o example`python3-config --extension-suffix`
```

3. test.py 测试

```python
import example
example.my_print('Hello World!')
```

## 11.9.在Python侧使用alias

1. example.cc代码

```c++
#include "pybind11/pybind11.h"

void my_print(std::string text) {
    pybind11::print(text);
}

PYBIND11_MODULE(example, m) {
    m.def("my_print", &my_print);

    m.attr("m_print") = m.attr("my_print");
}
```

2. 使用c++编译，并生成so文件

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup `python3 -m pybind11 --includes` example.cc -o example`python3-config --extension-suffix`
```

3. test.py 测试

```python
import example

example.my_print('Hello World!')
example.m_print('Hello World!')

print(example.m_print)
print(example.my_print)
```

# 12.pybind11

| 版本  | 描述                              | 注意                    |
| ----- | --------------------------------- | ----------------------- |
| 2.6.0 | CPP和Python操作接口的python软件库 | 自带pybind11的CPP软件包 |

## 12.1.setup_helpers

| 版本 | 描述                          | 注意                 |
| ---- | ----------------------------- | -------------------- |
| -    | 为pybind11的CPP软件包提供帮助 | 主要在setup.py中使用 |

### 12.1.1.build_ext

实例化一个build_ext(在编译时自动寻找支持的最高版本的c++编译器)

```python
from setuptools import setup
from pybind11.setup_helpers import build_ext

setup(
    cmdclass={'build_ext': build_ext},
)
```

### 12.1.2.Pybind11Extension

实例化一个Pybind11Extension(自动构建C++11+ Extension模块，即自动添加动态链接库)

```python
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

extension_modules = [
    Pybind11Extension(
        'path/to/xxx.so',  # 生成的链接库的位置|str
        'path/to/source.cc',  # 扩展源码的位置|str
        include_dirs='path/to/include/xxx',  # 依赖的包的位置|str
        language='c++',  # 扩展使用的编程语言|str|c++
    )
]

setup(
    ext_modules=extension_modules,
)
```

# 13.pydot

| 版本  | 描述                 | 注意 |
| ----- | -------------------- | ---- |
| 1.4.1 | graphviz的python接口 | -    |

## 13.1.Dot

| 版本 | 描述          | 注意 |
| ---- | ------------- | ---- |
| -    | Dot语言的容器 |      |

### 13.1.1.write_png()

将图像写入文件

```python
import pydot
graph = pydot.graph_from_dot_data(s)[0]
graph.write_png(path)  # 写入文件的路径|str
```

## 13.2.graph_from_dot_data()

从dot数据中加载图像|list of pydot.Dot

```python
import pydot
graph = pydot.graph_from_dot_data(s)  # dot数据|str
```

## 13.3.graph_from_dot_file()

从dot文件中加载图像|list of pydot.Dot

```python
import pydot
graph = pydot.graph_from_dot_data(s)  # dot文件的路径|str
```

# 14.scipy

| 版本  | 描述                 | 注意 |
| ----- | -------------------- | ---- |
| 1.4.1 | python科学计算软件库 | -    |

## 14.1.stats

| 版本 | 描述              | 注意 |
| ---- | ----------------- | ---- |
| -    | scipy的统计功能库 | -    |

### 14.1.1.boxcox()

进行Box_Cox幂变换|numpy.ndarray和numpy.float64

```python
import numpy as np
from scipy.stats import boxcox
y = np.asarray([1, 2, 3, 4, 5])
y_trans, lmbda = boxcox(x=y)  # 输入的数组|numpy.ndarray(必须是一维的)
```

## 14.2.special

### 14.2.1.inv_boxcox()

进行Box_Cox幂变换的逆变换|numpy.ndarray

```python
import numpy as np
from scipy.special import inv_boxcox
y_trans = np.asarray([0., 0.88891532, 1.64391667, 2.32328259, 2.95143046])
lmbda = 0.690296586
y = inv_boxcox(y_trans,  # 输入的数组|numpy.ndarray(必须是一维的)
               lmbda)  # 公式中的lambda|float
```

# 15.sklearn

| 版本   | 描述                           | 注意                               |
| ------ | ------------------------------ | ---------------------------------- |
| 0.23.0 | python机器学习和数据挖掘软件库 | 安装时使用pip install scikit-learn |

## 15.1.datasets

| 版本 | 描述                    | 注意                                    |
| ---- | ----------------------- | --------------------------------------- |
| -    | sklearn的官方数据集模块 | 数据的默认保存路径为~/scikit_learn_data |

### 15.1.1.load_iris()

加载并返回iris数据集|sklearn.utils.Bunch

```python
from sklearn.datasets import load_iris
dataset = load_iris()
```

## 15.2.ensemble

| 版本 | 描述                  | 注意                                                     |
| ---- | --------------------- | -------------------------------------------------------- |
| -    | sklearn的集成学习模块 | 使用scikit-learn API的其他框架也可以使用此模块的一些功能 |

### 15.2.1.AdaBoostClassifier()

实例化一个AdaBoost分类器

```python
from sklearn.ensemble import AdaBoostClassifier
model = AdaBoostClassifier(n_estimators=50,  # 弱学习器的最大数量|int|50
                           learning_rate=1e-3)  # 学习率|float|1.0
```

### 15.2.2.GradientBoostingClassifier()

实例化一个梯度提升分类器

```python
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(learning_rate=0.1,  # 学习率|float|0.1
                                   n_estimators=100)  # 弱学习器的最大数量|int|100
```

### 15.2.3.RandomForestClassifier()

实例化一个随机森林分类器

```python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100,  # 决策树的最大数量|int|100
                               criterion='gini',  # 划分方式|str('gini'或者'entropy')|'gini'
                               max_depth=None)  # 决策树的最大深度|int|None
```

### 15.2.4.RandomForestRegressor()

实例化一个随机森林回归

```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100,  # 决策树的最大数量|int|100
                              n_jobs=4,  # 并行数量|int|None|可选
                              verbose=1)  # # 日志模式|int|0|可选
```

### 15.2.5.StackingClassifier()

实例化一个Stacking分类器

```python
from sklearn.ensemble import StackingClassifier
model = StackingClassifier(estimators,  # 基学习器列表|list of (str, estimator) tuples
                           final_estimator=None)  # 二级学习器|None(LogisticRegression())
```

### 15.2.6.VotingClassifier()

实例化一个投票分类器

```python
from sklearn.ensemble import VotingClassifier
model = VotingClassifier(estimators,  # 基学习器列表|list of (str, estimator) tuples
                         voting,  # 投票方式|str|'hard'(hard', 'soft')
                         weights)  # 基学习器的权重|array-like of shape (n_classifiers,)|None
```

#### 15.2.6.1.fit()

训练投票分类器|self

```python
model.fit(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
          y)  # 标签|array-like of shape (n_samples,)
```

#### 15.2.6.2.predict()

进行预测|numpy.ndarray

```python
result = model.predict(X)  # 用于预测的数据|{array-like, sparse matrix} of shape (n_samples, n_features)
```

#### 15.2.6.3.score()

计算验证集的平均准确率|float

```python
accuracy = model.score(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
                       y)  # 标签|array-like of shape (n_samples,)
```

## 15.3.linear_model

| 版本 | 描述                  | 注意 |
| ---- | --------------------- | ---- |
| -    | sklearn的线性模型模块 | -    |

### 15.3.1.LinearRegression()

实例化一个线性回归模型

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

#### 15.3.1.1.fit()

训练线性回归模型|self

```python
model.fit(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
          y,  # 标签|array-like of shape (n_samples,)
          sample_weight)  # 类别权重|array-like of shape (n_samples,)|None
```

#### 15.3.1.2.predict()

进行预测|numpy.ndarray

```python
C = model.predict(X)  # 用于预测的数据|{array-like, sparse matrix} of shape (n_samples, n_features)
```

#### 15.3.1.3.score()

计算验证集的平均准确率|float

```python
accuracy = model.score(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
                       y)  # 标签|array-like of shape (n_samples,)
```

### 15.3.2.LogisticRegression()

实例化一个逻辑回归模型

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
```

#### 15.3.2.1.fit()

训练逻辑回归模型|self

```python
model.fit(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
          y,  # 标签|array-like of shape (n_samples,)
          sample_weight)  # 类别权重|array-like of shape (n_samples,)|None
```

#### 15.3.2.2.predict()

进行预测|numpy.ndarray

```python
C = model.predict(X)  # 用于预测的数据|{array-like, sparse matrix} of shape (n_samples, n_features)
```

## 15.4.metrics

| 版本 | 描述              | 注意 |
| ---- | ----------------- | ---- |
| -    | sklearn的评估模块 | -    |

### 15.4.1.accuracy_score()

计算分类器的准确率|numpy.float64

```python
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true,  # 真实标签|1d array-like, or label indicator array / sparse matrix
                          y_pred,  # 预测标签|1d array-like, or label indicator array / sparse matrix
                          sample_weight)  # 类别权重|array-like of shape (n_samples,)|None
```

## 15.5.model_selection

| 版本 | 描述                  | 注意 |
| ---- | --------------------- | ---- |
| -    | sklearn的数据划分模块 | -    |

### 15.5.1.cross_val_predict()

对模型的数据逐个进行交叉验证|numpy.ndarry

```python
from sklearn.model_selection import cross_val_predict
result = cross_val_predict(estimator,  # 学习器|scikit-learn API实现的有fit和predict函数的模型
                           X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
                           y,  # 标签|array-like of shape (n_samples,)
                           cv)  # 交叉验证的划分数|int|3
```

### 15.5.2.cross_val_score()

对模型进行交叉验证|numpy.ndarry

```python
from sklearn.model_selection import cross_val_predict
result = cross_val_predict(estimator,  # 学习器|scikit-learn API实现的有fit和predict函数的模型
                           X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
                           y,  # 标签|array-like of shape (n_samples,)
                           scoring,  # 评分函数|str
                           cv)  # 交叉验证的划分数|int|3
```

### 15.5.3.GridSearchCV()

实例化网格搜索器

```python
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(estimator,  # 学习器|scikit-learn API实现的有score函数的模型
                  param_grid,  # 参数网格|dict or list of dictionaries|
                  scoring,  # 评分方式|str|None
                  n_jobs,  # 并行数量|int|None|可选
                  cv,  # 交叉验证的划分数|int|None
                  verbose)  # 日志模式|int|0
```

#### 15.5.3.1.fit()

组合所有参数训练

```python
gs.fit(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
       y)  # 标签|array-like of shape (n_samples,)
```

#### 15.5.3.2.best_params_

最佳参数

```python
gs.best_params_
```

#### 15.5.3.3.best_score_

最佳分数

```python
gs.best_score_
```

### 15.5.4.LeaveOneOut()

实例化留一法交叉验证器

```python
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
```

#### 15.5.4.1.split()

划分数据|yield(train:numpy.ndarray, test:numpy.ndarray)

```python
loo.split(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
          y)  # 标签|array-like of shape (n_samples,)|None
```

### 15.5.5.StratifiedKFold()

实例化K折交叉验证器

```python
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits,  # 交叉验证的划分数|int|5
                        shuffle,  # 打乱数据|bool|False
                        random_state)  # 随机状态|int or RandomState instance|None
```

#### 15.5.5.1.n_splits

交叉验证的划分数|int

```python
kflod.n_splits
```

#### 15.5.5.2.split()

划分数据|yield(train:numpy.ndarray, test:numpy.ndarray)

```python
kfold.split(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
            y)  # 标签|array-like of shape (n_samples,)|None
```

### 15.5.6.train_test_split()

将原始数据随机划分成训练和测试子集|list(两个长度相等的arrays)

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,  # 需要划分的数据|lists, numpy arrays, scipy-sparse matrices or pandas dataframes
                                                    test_size,  # 测试数据的大小|float or int|0.25
                                                    random_state)  # 随机状态|int or RandomState instance|None
```

## 15.6.preprocessing

| 版本 | 描述                                                | 注意 |
| ---- | --------------------------------------------------- | ---- |
| -    | sklearn的数据预处理模块(缩放、居中、归一化、二值化) | -    |

### 15.6.1.LabelEncoder()

实例化一个标签编码器

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
```

#### 15.6.1.1.fit_transform()

转换标签数据|array-like

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = ['a', 'a', 'b', 'c']
y = le.fit_transform(y=y)  # 需要转换的标签|array-like
```

### 15.6.2.MinMaxScaler()

实例化一个MinMax缩放器

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
```

#### 15.6.2.1.fit_transform()

转换数据|numpy.ndarray

```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = [[1, 1], [2, 2], [3, 2], [4, 3], [5, 3]]
data = scaler.fit_transform(X=X)  # 需要转换的数据|{array-like, sparse matrix, dataframe} of shape (n_samples, n_features)
```

### 15.6.3.MultiLabelBinarizer()

实例化一个多标签二值化转换器

```python
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
```

#### 15.6.3.1.classes_

原始的标签|numpy.ndarray

```python
mlb.classes_
```

#### 15.6.2.2.fit_transform()

转换标签数据|numpy.ndarray

```python
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y = [['a', 'b'], ['a', 'c']]
label = mlb.fit_transform(y=y)  # 需要转换的标签|array-like
```

## 15.7.svm

| 版本 | 描述                    | 注意 |
| ---- | ----------------------- | ---- |
| -    | sklearn的支持向量机模块 | -    |

### 15.7.1.SVC()

实例化一个支持向量分类器

```python
from sklearn.svm import SVC
model = SVC(C,  # 正则化系数|float|1.0
            kernel,  # 核函数|str|'rbf'('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
            probability,  # 是否启用概率估计|bool|false
            class_weight)  # 类别权重|dict or 'balanced'|None
```

### 15.7.2.SVR()

实例化一个支持向量回归

```python
from sklearn.svm import SVR
model = SVR(C,  # 正则化系数|float|1.0
            kernel)  # 核函数|str|'rbf'('linear', 'poly', 'rbf', 'sigmoid', 'precomputed')
```

## 15.8.tree

| 版本 | 描述                | 注意 |
| ---- | ------------------- | ---- |
| -    | sklearn的决策树模块 | -    |

### 15.8.1.DecisionTreeClassifier()

实例化一个决策树分类器

```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion,  # 划分方式|str('gini'或者'entropy')|'gini'
                               random_state)  # 随机状态|int or RandomState instance|None
```

#### 15.8.1.1.fit()

训练决策树分类器|self

```python
model.fit(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
          y)  # 标签|array-like of shape (n_samples,)
```

### 15.8.2.export_graphviz()

将决策树转换成dot字符串|str

```python
from sklearn.tree import export_graphviz
dot_str = export_graphviz(decision_tree,  # 决策树分类器|sklearn.tree._classes.DecisionTreeClassifier
                          out_file,  # 是否导出文件|file object or str|None｜可选
                          feature_names,  # 特征的名称|list of str|None|可选
                          class_names)  # 类别的名称|list of str, bool or None|None|可选
```

### 15.8.3.plot_tree()

绘制决策树

```python
from sklearn.tree import plot_tree
plot_tree(decision_tree=model)  # 决策树分类器|sklearn.tree._classes.DecisionTreeClassifier or sklearn.tree._classes.DecisionTreeRegressor
```

## 15.9.utils

| 版本 | 描述                  | 注意 |
| ---- | --------------------- | ---- |
| -    | sklearn的实用工具模块 | -    |

### 15.9.1.multiclass

#### 15.9.1.1.type_of_target()

判断数据的类型|str

```python
from sklearn.utils import multiclass
y = [1, 2, 3]
result = multiclass.type_of_target(y=y)  # 要判断数据|array-like
```

# 16.tensorflow

| 版本  | 描述         | 注意                                             |
| ----- | ------------ | ------------------------------------------------ |
| 2.3.0 | 机器学习框架 | TensorFlow 2.X的语法相同，高版本会比低版本算子多 |

## 16.1.config

### 16.1.1.experimental

#### 16.1.1.1.set_memory_growth()

设置物理设备的内存使用量

```python
import tensorflow as tf
tf.config.experimental.set_memory_growth(device,  # 物理设备|tensorflow.python.eager.context.PhysicalDevice
                                         enable)  # 是否启用内存增长|bool
```

### 16.1.2.experimental_connect_to_cluster()

连接到指定的集群

```python
import tensorflow as tf
tf.config.experimental_connect_to_cluster(cluster_spec_or_resolver)  # 一个集群|
```

### 16.1.3.list_physical_devices()

返回主机所有可见的物理设备|list

```python
import tensorflow as tf
devices_list = tf.config.list_physical_devices(device_type=None)  # 设备类型|str|None|可选
```

## 16.2.constant()

创建一个常张量|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.constant(value=10)  # 输入的数据|int, float or list
```

## 16.3.data

| 版本 | 描述           | 注意 |
| ---- | -------------- | ---- |
| -    | 数据输入流水线 | -    |

### 16.3.1.Dataset

#### 16.3.1.1.batch()

给数据集划分批次|tensorflow.python.data.ops.dataset_ops.BatchDataset

```python
import tensorflow as tf
dataset = tf.data.Dataset.range(6)
dataset = dataset.batch(batch_size=3)  # 批次大小|A tf.int64 scalar, int
print(list(dataset.as_numpy_iterator()))
```

#### 16.3.1.2.experimental

##### 16.3.1.2.1.AUTOTUNE

CPU自动调整常数

```python
import tensorflow as tf
tf.data.experimental.AUTOTUNE
```

#### 16.3.1.3.from_tensor_slices()

创建一个元素是张量切片的数据集|tensorflow.python.data.ops.dataset_ops.TensorSliceDataset

```python
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices(tensors=([1, 2], [3, 4], [5, 6]))  # 输入的张量|array-like(数据第一维相同)
print(list(dataset.as_numpy_iterator()))
```

#### 16.3.1.4.map()

对数据集的每一个元素应用map_func进行处理，并返回一个新的数据集

|tensorflow.python.data.ops.dataset_ops.MapDataset or tensorflow.python.data.ops.dataset_ops.ParallelMapDataset

```python
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices(tensors=([1, 2, 3], [4, 5, 6]))
def map_func(x, y): return x+1, y-2
dataset = dataset.map(map_func=map_func,  # 处理函数|function or lambda
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 并行处理的数量|int|None|可选
print(list(dataset.as_numpy_iterator()))
```

#### 16.3.1.5.prefetch()

对数据集的读取进行预加载|tensorflow.python.data.ops.dataset_ops.PrefetchDataset

```python
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices(tensors=([1, 2, 3], [4, 5, 6]))
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)  # 缓冲区大小|int
print(list(dataset.as_numpy_iterator()))
```

#### 16.3.1.6.shuffle()

随机打乱数据集|tensorflow.python.data.ops.dataset_ops.ShuffleDataset

```python
import tensorflow as tf
dataset = tf.data.Dataset.range(5)
dataset_shuffle = dataset.shuffle(buffer_size=5)  # 采样的范围|A tf.int64 scalar, int
print(list(dataset.as_numpy_iterator()))
print(list(dataset_shuffle.as_numpy_iterator()))
```

#### 16.3.1.7.take()

从dataset中取出指定个数的数据创建新的数据集|tensorflow.python.data.ops.dataset_ops.TakeDataset

```python
import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices(tensors=([1, 2, 3], [4, 5, 6]))
sample = dataset.take(count=1)  # 取出的个数|int
```

### 16.3.2.experimental

#### 16.3.2.1.make_csv_dataset()

读取CSV文件并转换成数据集.|tensorflow.python.data.ops.dataset_ops.PrefetchDataset

```python
import tensorflow as tf

dataset = tf.data.experimental.make_csv_dataset(file_pattern='./dataset/train.csv',  # str|CSV文件的路径.
                                                batch_size=128,  # int|批次大小.
                                                column_names=['survived', 'sex', 'age', 'n_siblings_spouses', 																									'parch', 'fare', 'class', 'deck', 'embark_town', 'alone'],  # 																									list of str(可选)|None｜列名.
                                                label_name='survived',  # str(可选)|None|标签列名.
                                                num_epochs=1)  # int|None|数据集重复的次数.
```

## 16.4.distribute

| 版本 | 描述           | 注意                                 |
| ---- | -------------- | ------------------------------------ |
| -    | 用于分布式训练 | tf.keras.utils.multi_gpu_model被移除 |

### 16.4.1.cluster_resolver

#### 16.4.1.1.TPUClusterResolver()

实例化一个TPU集群解释器

```python
import tesnorflow as tf
resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
```

### 16.4.2.MirroredStrategy()

实例化一个镜像策略(用于在单台主机上使用多个GPU设备进行训练)

```python
import tensorflow as tf
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # 模型构建代码
```

### 16.4.3.TPUStrategy()

实例化一个TPU或TPU Pods的策略

```python
import tensorflow as tf
strategy = tf.distribute.TPUStrategy(tpu_cluster_resolver)  # TPU集群信息|tf.distribute.cluster_resolver.TPUClusterResolver|None
with strategy.scope():
    # 模型构建代码
```

## 16.5.einsum()

爱因斯坦求和约定|tensorflow.python.framework.ops.EagerTensor

```python
import numpy as np
import tensorflow as tf
a = np.asarray([[1], [2]])
b = np.asarray([[2, 1]])
result = tf.einsum('ij,jk->ik',  # 描述公式|str
                   a, b)  # 输入的张量|tf.Tensor or numpy.ndarray
```

## 16.6.feature_column

### 16.6.1.categorical_column_with_vocabulary_list()

实例化一个分类列.|tensorflow.python.feature_column.feature_column_v2.VocabularyListCategoricalColumn

```python
import tensorflow as tf

categorical_column = tf.feature_column.categorical_column_with_vocabulary_list(key='sex',  # str|特征名称.
                                                                               vocabulary_list=['male', 'female'])  # list of str|属性名称.
```

### 16.6.2.indicator_column()

对分类列进行独热表示.|tensorflow.python.feature_column.feature_column_v2.IndicatorColumn

```python
import tensorflow as tf

categorical_column = tf.feature_column.categorical_column_with_vocabulary_list('sex',
                                                                               ['male', 'female'])
categorical_onehot = tf.feature_column.indicator_column(categorical_column)  # CategoricalColumn|一个分类列.
```

### 16.6.3.numeric_column()

实例化一个数值列.|tensorflow.python.feature_column.feature_column_v2.NumericColumn

```python
import tensorflow as tf

numeric_column = tf.feature_column.numeric_column(key='age')  # str|特征名称.
```

## 16.7.GradientTape()

实例化一个梯度带

```python
import tensorflow as tf
tape = tf.GradientTape()
```

### 16.7.1.gradient()

计算梯度|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = 2*x
grad = tape.gradient(target=y, sources=x)  # 计算target关于sources的梯度|a list or nested structure of Tensors or Variables
```

## 16.8.image

| 版本 | 描述                 | 注意 |
| ---- | -------------------- | ---- |
| -    | 图像处理和编解码操作 | -    |

### 16.8.1.convert_image_dtype()

改变图片的数据类型|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
img = [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
     [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]]
img = tf.image.convert_image_dtype(image=img,  # 图片|array-like
                                   dtype=tf.int8)  # 转换后的数据类型|tensorflow.python.framework.dtypes.DType
```

### 16.8.2.decode_image()

转换BMP、GIF、JPEG或PNG图像为张量|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.image.decode_image(contents,  # 图片的字节流|0-D str
                               channels,  # 转换后的色彩通道数|int|0|可选
                               dtype)  # 转换后的数据类型|tensorflow.python.framework.dtypes.DType
```

### 16.8.3.decode_jpeg()

转换JPEG图像为张量|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.image.decode_jpeg(contents,  # 图片的字节流|0-D str
                              channels)  # 转换后的色彩通道数|int|0|可选
```

### 16.8.4.decode_png()

转换PNG图像为张量|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.image.decode_png(contents,  # 图片的字节流|0-D str
                             channels)  # 转换后的色彩通道数|int|0|可选
```

### 16.8.3.resize()

改变图片的大小|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.image.resize(images,  # 输入的图片|4-D Tensor of shape [batch, height, width, channels] or 3-D Tensorof shape [height, width, channels]
                         size)  # 改变后的大小｜int([new_height, new_width])
```

## 16.9.io

### 16.9.1.read_file()

读入文件|str

```python
import tensorflow as tf
img = tf.io.read_file(filename)  # 文件路径|str
```

## 16.10.keras

| 版本  | 描述                        | 注意                                    |
| ----- | --------------------------- | --------------------------------------- |
| 2.4.0 | TensorFlow的高阶机器学习API | Keras移除了多后端支持，推荐使用tf.keras |

### 16.10.1.applications

| 版本 | 描述                             | 注意                           |
| ---- | -------------------------------- | ------------------------------ |
| -    | 提供带有预训练权重的深度学习模型 | 默认保存路径是~/.keras/models/ |

#### 16.10.1.1.efficientnet

##### 16.10.1.1.1.EfficientNetB0()

EfficientNetB0的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.layers import Input
model = EfficientNetB0(include_top=False,  # 是否包含全连接输出层|bool|True
                       weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                       input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

##### 16.10.1.1.2.EfficientNetB3()

EfficientNetB4的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.layers import Input
model = EfficientNetB3(include_top=False,  # 是否包含全连接输出层|bool|True
                       weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                       input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

##### 16.10.1.1.3.EfficientNetB4()

EfficientNetB4的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.efficientnet import EfficientNetB4
from tensorflow.keras.layers import Input
model = EfficientNetB4(include_top=False,  # 是否包含全连接输出层|bool|True
                       weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                       input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

##### 16.10.1.1.4.EfficientNetB7()

EfficientNetB7的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.efficientnet import EfficientNetB7
from tensorflow.keras.layers import Input
model = EfficientNetB7(include_top=False,  # 是否包含全连接输出层|bool|True
                       weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                       input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

##### 16.10.1.1.5.preprocess_input()

对一个批次的数据进行ImageNet格式的预处理|Preprocessed tensor or numpy.ndarray

```python
from tensorflow.keras.applications.efficientnet import preprocess_input
input = preprocess_input(x)  # 要处理的数据|Tensor or numpy.ndarray
```

#### 16.10.1.2.imagenet_utils

##### 16.10.1.2.1.preprocess_input()

对一个批次的数据进行ImageNet格式的预处理|Preprocessed tensor or numpy.ndarray

```python
from tensorflow.keras.applications.vgg19 import preprocess_input
input = preprocess_input(x,  # 要处理的数据|Tensor or numpy.ndarray
                         mode)  # 转换的模式|str|'caffe' 
```

#### 16.10.1.3.inception_resnet_v2

##### 16.10.1.3.1.InceptionResNetV2()

InceptionResNetV2的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.layers import Input
model = InceptionResNetV2(include_top=False,  # 是否包含全连接输出层|bool|True
                          weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                          input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

#### 16.10.1.4.inception_v3

##### 16.10.1.4.1.InceptionV3()

InceptionV3的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input
model = InceptionV3(include_top=False,  # 是否包含全连接输出层|bool|True
                    weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                    input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

#### 16.10.1.5.mobilenet_v2

##### 16.10.1.5.1.MobileNetV2()

MobileNetV2的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Input
model = MobileNetV2(include_top=False,  # 是否包含全连接输出层|bool|True
                    weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                    input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

#### 16.10.1.6.resnet50

##### 16.10.1.6.1.ResNet50()

```python
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input
model = ResNet50(include_top=False,  # 是否包含全连接输出层|bool|True
                 weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                 input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

#### 16.10.1.7.resnet_v2

##### 16.10.1.7.1.ResNet50V2()

ResNet50V2的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.layers import Input
model = ResNet50V2(include_top=False,  # 是否包含全连接输出层|bool|True
                   weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                   input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

##### 16.10.1.7.2.ResNet152V2()

ResNet152V2的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.layers import Input
model = ResNet152V2(include_top=False,  # 是否包含全连接输出层|bool|True
                    weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                    input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

#### 16.10.1.8.vgg19

##### 16.10.1.8.1.VGG19()

VGG19的预训练模型|tensorflow.python.keras.engine.functional.Functional

```python
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input
model = VGG19(include_top=False,  # 是否包含全连接输出层|bool|True
              weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
              input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

#### 16.10.1.9.xception

##### 16.10.1.9.1.Xception()

```python
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Input
model = Xception(include_top=False,  # 是否包含全连接输出层|bool|True
                 weights='imagenet',  # 初始化权重|'imagenet' or None or path|'imagenet'
                 input_tensor=Input(shape=[224, 224, 3]))  # 输入层|tensorflow.python.framework.ops.Tensor(layers.Input())
```

###  16.10.2.backend

| 版本 | 描述        | 注意 |
| ---- | ----------- | ---- |
| -    | 后端函数API |      |

#### 16.10.2.1.cast()

转换张量的数据类型|tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras.backend import cast
tensor = cast(x=[1, 2, 3],  # 输入的张量|tf.Tensor or array-like
              dtype='float16')  # 转换后的数据类型|str('float16', 'float32', or 'float64')
```

#### 16.10.2.2.clear_session()

销毁当前的计算图并创建一个新的计算图

```python
from tensorflow.keras.backend import clear_session
clear_session()
```

#### 16.10.2.3.clip()

逐元素进行裁切到满足条件的范围|tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras.backend import clip
tensor = clip(x=[1, 2, 3, 4, 5],  # 输入的张量|tf.Tensor or array-like
              min_value=2,  # 最小值|float, integer or tensor
              max_value=4)  # 最大值|float, integer or tensor
```

#### 16.10.2.4.ctc_batch_cost()

在每个批次上计算ctc损失|tensorflow.python.framework.ops.EagerTensor(形状是(samples,1))

```python
from tensorflow.keras.backend import ctc_batch_cost
tensor = ctc_batch_cost(y_true,  # 真实的标签|tensor(samples, max_string_length)
                        y_pred,  # 预测的标签|tensor(samples, time_steps, num_categories)
                        input_length,  # 预测的长度|tensor(samples, 1)
                        label_length)  # 真实的长度|tensor(samples, 1)
```

#### 16.10.2.5.ctc_decode()

解码softmax的输出|tuple of tensorflow.python.framework.ops.EagerTensor解码元素列表和解码序列的对数概率

```python
from tensorflow.keras.backend import ctc_decode
t = ctc_decode(y_pred,  # 模型的预测值|tensor(samples, time_steps, num_categories)
               input_length,  # 样本序列的长度|tensor(samples,) 每个样本值是字典总数
               greedy)  # 执行更快的搜索路径|bool|True
```

#### 16.10.2.6.expand_dims()

扩展张量的维度|tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras.backend import expand_dims
tensor = expand_dims(x=[1, 2, 3],  # 输入的张量|tf.Tensor or array-like
                     axis=0)  # 添加新维度的位置|int
```

#### 16.10.2.7.get_value()

返回一个变量的值|值所对应的数据类型

```python
from tensorflow.keras.models import Model
from tensorflow.keras.backend import get_value
model = Model()
model.compile(optimizer='adam')
value = get_value(x=model.optimizer)
```

#### 16.10.2.8.ones_like()

创建一个和输入形状相同的全一张量|tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras.backend import ones_like
tensor = ones_like(x=[[1, 2, 3], [4, 5, 6]])  # 输入的张量|array-like
```

#### 16.10.2.9.set_value()

设置一个变量的值(只能设置数值)

```python
from tensorflow.keras.backend import set_value
set_value(x,  # 需要设置新值的变量
          value)  # 设置的新值|numpy.ndarray(必须和原来形状一致)
```

#### 16.10.2.10.shape()

返回张量的形状|tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras.backend import shape, ones_like
tensor = ones_like(x=[[1, 2, 3], [4, 5, 6]])
tensor_shape = shape(x=tensor)  # 输入的张量|tensor
```

#### 16.10.2.11.sigmoid()

逐元素计算sigmoid函数的值|tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras.backend import sigmoid
tensor = sigmoid(x=[1., 2., 3., 4., 5.])  # 输入的张量|tensor
```

#### 16.10.2.12.zeros_like()

创建一个和输入形状相同的全零张量|tensorflow.python.framework.ops.EagerTensor

```python
from tensorflow.keras.backend import zeros_like
tensor = zeros_like(x=[[1, 2, 3], [4, 5, 6]])  # 输入的张量|array-like
```

### 16.10.3.callbacks

| 版本 | 描述                                  | 注意 |
| ---- | ------------------------------------- | ---- |
| -    | 回调函数API，用于查看模型的状态和统计 |      |

#### 16.10.3.1.EarlyStopping()

实例化一个EarlyStopping，用以提前停止训练防止过拟合

```python
from tensorflow.keras.callbacks import EarlyStopping
CALLBACKS = [
    EarlyStopping(monitor='val_accuracy',  # 监控的信息|str｜'val_loss'
                  min_delta=1e-4,  # 最小变化量|float|0
                  patience=5,  # 监测容忍轮数(数据有小幅度波动可以跳过，验证频率也一定是1)|int|0
                  verbose=1,  # 日志模式|int(0, 1)|0
                  restore_best_weights=True)  # 恢复最佳状态的权重保存|bool|False(保存最后一步)
]
```

#### 16.10.3.2.LearningRateScheduler()

实例化一个LearningRateScheduler，对学习率进行定时改变

```python
from tensorflow.keras.callbacks import LearningRateScheduler
CALLBACKS = [
    LearningRateScheduler(schedule,  # 定时器函数|function(epoch轮数作为输入, 学习率作为输出.)|
                          verbose=1)  # 日志模式|int(0, 1)|0
]
```

#### 16.10.3.3.ModelCheckpoint()

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

#### 16.10.3.4.ReduceLROnPlateau()

实例化一个ReduceLROnPlateau，当评估停止变化的时候，降低学习率

```python
from tensorflow.keras.callbacks import ReduceLROnPlateau
CALLBACKS = [
    ReduceLROnPlateau(monitor='val_loss',  # 监控的信息|str｜'val_loss'
                      factor=0.1,  # 学习率衰减因子|float|0.1(new_learning_rate = factor * learning_rate)
                      patience=5,  # 监测容忍轮数(数据有小幅度波动可以跳过，验证频率也一定是1)|int|0
                      verbose=1,  # 日志模式|int(0, 1)|0
                      min_delta=1e-3,  # 最小变化量|float|0.0001
                      min_lr=0)  # 最小学习率|float|0
]
```

#### 16.10.3.5.TensorBoard()

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

### 16.10.4.datasets

| 版本 | 描述           | 注意                                                         |
| ---- | -------------- | ------------------------------------------------------------ |
| -    | 入门常用数据集 | 目前有boston_housing, cifar10, cifar100, fashion_mnist, imdb, mnist and reuters数据集 |

#### 16.10.4.mnist

#### 16.10.4.1.load_data()

加载mnist数据集|Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test)

```python
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

### 16.10.5.layers

| 版本 | 描述      | 注意                                     |
| ---- | --------- | ---------------------------------------- |
| -    | 网络层API | 可以使用Functional API或者Sequential模型 |

#### 16.10.5.1.Activation()

实例化一个激活层

```python
from tensorflow.keras.layers import Activation
layer = Activation(activation)  # 要使用的激活函数|str or tensorflow.keras.activations中的函数
```

#### 16.10.5.2.Add()

实例化一个矩阵加法层，将layer相加

```python
from tensorflow.keras.layers import Add
layer = Add(_Merge)  # 相同形状的张量(层)列表|tensorflow.python.framework.ops.Tensor
```

#### 16.10.5.3.AdditiveAttention()

实例化一个Bahdanau注意力层

```python
from tensorflow.keras.layers import AdditiveAttention
layer = AdditiveAttention()
```

#### 16.10.5.4.BatchNormalization()

实例化一个批标准化层

```python
from tensorflow.keras.layers import BatchNormalization
layer = BatchNormalization()
```

#### 16.10.5.5.Bidirectional()

实例化一个循环神经网络层的双向封装器

```python
from tensorflow.keras.layers import Bidirectional, GRU
layer = GRU(units=256, return_sequences=True)
layer = Bidirectional(layer=layer)  # 网络层|keras.layers.RNN, keras.layers.LSTM or keras.layers.GRU
```

#### 16.10.5.6.Concatenate()

实例化一个合并层

```python
from tensorflow.keras.layers import Concatenate
layer = Concatenate(axis=0)(_Merge)  # 连接的维度(相同形状的张量(层)列表|tensorflow.python.framework.ops.Tensor)|int|-1
```

#### 16.10.5.7.Conv1D()

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

#### 16.10.5.8.Conv2D()

实例化一个二维卷积层

```python
from tensorflow.keras.layers import Conv2D
layer = Conv2D(filters,  # 卷积核的数量|int
               kernel_size,  # 卷积核的大小|int, tuple/list of 2 integers
               strides,  # 滑动步长|int, tuple/list of 2 integers|(1, 1)
               padding,  # 填充方式|str('valid' or 'same')|'valid'
               input_shape)  # 如果是模型的第一层，需指定输入的形状|tuple of int
```

#### 16.10.5.9.Conv2DTranspose()

实例化一个二维转置卷积层

```python
from tensorflow.keras.layers import Conv2DTranspose
layer = Conv2DTranspose(filters,  # 卷积核的数量|int
                        kernel_size,  # 卷积核的大小|int, tuple/list of 2 integers
                        strides,  # 滑动步长|int, tuple/list of 2 integers|(1, 1)
                        padding,  # 填充方式|str('valid' or 'same')|'valid'
                        use_bias)  # 是否使用偏置|bool|True
```

#### 16.10.5.10.Dense()

实例化一个全连接层

```python
from tensorflow.keras.layers import Dense
layer = Dense(units,  # 神经元的数量|int
              use_bias,  # 是否使用偏置|bool|True
              input_shape)  # 如果是模型的第一层，需指定输入的形状|tuple of int
```

#### 16.10.5.11.DenseFeatures()

实例化DenseFeatures层

```python
from tensorflow.keras.layers import DenseFeatures

layer = DenseFeatures(feature_columns)  # list of tensorflow.python.feature_column|特征列.
```

#### 16.10.5.12.Dot()

实例化一个点积层

```python
from tensorflow.keras.layers import Dot
layer = Dot(axes=1)(_Merge)# 点积的维度(相同形状的张量(层)列表|tensorflow.python.framework.ops.Tensor)|int|-1
```

#### 16.10.5.13.Dropout()

实例化一个Dropout层(在训练阶段随机抑制部分神经元)

```python
from tensorflow.keras.layers import Dropout
layer = Dropout(rate=0.5)  # 丢弃比例|float
```

#### 16.10.5.14.Embedding()

实例化一个嵌入层(只能作为模型的第一层)

```python
from tensorflow.keras.layers import Embedding
layer = Embedding(input_dim,  # 输入的维度|int(最大值加一) 
                  output_dim,  # 输出的嵌入矩阵维度|int
                  embeddings_initializer,  # 嵌入矩阵初始化器|str|uniform
                  embeddings_regularizer,)  # 嵌入矩阵正则化器|str or tensorflow.keras.regularizers|None
```

#### 16.10.5.15.experimental

##### 16.10.5.15.1.preprocessing

###### 16.10.5.15.1.1.get_vocabulary()

获取词汇表|list

```python
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
char_to_num = StringLookup(mask_token=None,
                           num_oov_indices=0,
                           vocabulary=['a', 'b', 'c', 'd'],
                           invert=False)
vocab = char_to_num.get_vocabulary()
```

###### 16.10.5.15.1.1.StringLookup()

实例化一个StringLookup(将词汇表映射到整数索引)

```python
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
vocab = ['a', 'b', 'c', 'd', 'a']
char_to_num = StringLookup(mask_token=None,  # 词汇表的最大大小|int|None(表示没有限制)
                           num_oov_indices=0,  # 超出词汇表的数量的标记|int|1
                           vocabulary=['a', 'b', 'c', 'd'],  # 词汇表|list
                           invert=False)  # 反转|bool|False(如果是True将整数映射回词汇表)
tensor = char_to_num(vocab)
```

#### 16.10.5.16.Flatten()

实例化一个展平层(不影响批次)

```python
from tensorflow.keras.layers import Flatten
layer = Flatten()
```

#### 16.10.5.17.GlobalAveragePooling1D()

实例化一个全局一维平均池化层

```python
from tensorflow.keras.layers import GlobalAveragePooling1D
layer = GlobalAveragePooling1D()
```

#### 16.10.5.18.GlobalMaxPooling1D()

实例化一个全局一维最大池化层

```python
from tensorflow.keras.layers import GlobalMaxPooling1D
layer = GlobalMaxPooling1D()
```

#### 16.10.5.19.GRU()

实例化一个门控循环网络层

```python
from tensorflow.keras.layers import GRU
layer = GRU(units=256,  # 神经元的数量|int
            return_sequences=True)  # 返回序列还是返回序列的最后一个输出|bool|False(返回序列的最后一个输出)
```

#### 16.10.5.20.Input()

实例化一个输入层

```python
from tensorflow.keras.layers import Input
layer = Input(shape=(224, 224, 3),  # 形状|tuple
              name='Input-Layer',  # 层名称|str|None
              dtype='int32')  # 期望的数据类型|str|None
```

#### 16.10.5.21.Lambda()

实例化一个Lambda层(将任意函数封装成网络层)

```python
from tensorflow.keras.layers import Lambda
layer = Lambda(function=lambda x: x*x,  # 要封装的函数
               output_shape=(1024,),  # 期望输出形状|tuple|None
               name='Square-Layer')  # 层名称|str|None
```

#### 16.10.5.22.Layer()

Keras所有的层都继承于此(实现必要方法就可以自定义层)

```python
from tensorflow.keras.layers import Layer
class MyLayer(Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        # 初始化代码

    def call(self, inputs):
      	# 处理代码 
        return outputs
```

#### 16.10.5.23.LeakyReLU()

实例化一个带侧漏的RelU层

```python
from tensorflow.keras.layers import LeakyReLU
layer = LeakyReLU(alpha=0.3)  # 负斜率系数(侧漏率)|float|0.3
```

#### 16.10.5.24.LSTM()

实例化一个长短时记忆网络层

```python
from tensorflow.keras.layers import LSTM
layer = LSTM(units=256,  # 神经元的数量|int
             return_sequences=True,  # 返回序列还是返回序列的最后一个输出|bool|False(返回序列的最后一个输出)
             dropout=0.1)  # 随机丢弃率|float|0.
```

#### 16.10.5.25.MaxPooling1D()

实例化一个一维最大池化层

```python
from tensorflow.keras.layers import MaxPooling1D
layer = MaxPooling1D(pool_size=2,  # 池化窗口|int|2
                     strides=None,  # 滑动步长|int or tuple/list of a single integer|None
                     padding='valid')  # 填充方式|str('valid', 'causal' or 'same')|'valid'
```

#### 16.10.5.26.MaxPooling2D()

实例化一个二维最大池化层

```python
from tensorflow.keras.layers import MaxPooling2D
layer = MaxPooling1D(pool_size=2,  # 池化窗口|int or tuple of 2 int|(2,2)
                     strides=None,  # 滑动步长|int or tuple of 2 int|None
                     padding='valid')  # 填充方式|str('valid', 'causal' or 'same')|'valid'
```

#### 16.10.5.27.Reshape()

实例化变形层(将输入的层改变成任意形状)

```python
from tensorflow.keras.layers import Reshape
layer = Reshape(target_shape)  # 目标形状|tuple
```

#### 16.10.5.28.SeparableConv2D()

实例化深度方向的可分离二维卷积

```python
from tensorflow.keras.layers import SeparableConv2D
layer = SeparableConv2D(filters,  # 卷积核的数量|int
                        kernel_size,  # 卷积核的大小|int, tuple/list of 2 integers
                        strides,  # 滑动步长|int, tuple/list of 2 integers|(1, 1)
                        padding)  # 填充方式|str('valid' or 'same')|'valid'
```

#### 16.10.5.29.TimeDistributed()

实例化一个时间片封装器

```python
from tensorflow.keras.layers import Dense, TimeDistributed

layer = Dense(32, activation='relu')
layer = TimeDistributed(layer)
```

#### 16.10.5.30.UpSampling2D()

实例化二维上采样层

```python
from tensorflow.keras.layers import UpSampling2D
layer = UpSampling2D(size)  # 上采样因子|int or tuple of 2 integers|2
```

#### 16.10.5.31.ZeroPadding2D()

实例化一个二维输入的零填充层

```python
from tensorflow.keras.layers import ZeroPadding2D

layer = ZeroPadding2D(padding=(2, 2))  # int or tuple of int|填充数.
```

### 16.10.6.losses

| 版本 | 描述        | 注意 |
| ---- | ----------- | ---- |
| -    | 损失函数API |      |

#### 16.10.6.1.BinaryCrossentropy()

实例化二分类交叉熵损失函数

```python
from tensorflow.keras.losses import BinaryCrossentropy
loss = BinaryCrossentropy(from_logits=True)  # 是否将y_pred解释为张量|bool|False(True的话有更高的稳定性)
```

#### 16.10.6.2.CategoricalCrossentropy()

实例化多分类交叉熵损失函数(标签是one-hot编码)

```python
from tensorflow.keras.losses import CategoricalCrossentropy
loss = CategoricalCrossentropy(from_logits=True)  # 是否将y_pred解释为张量|bool|False(True的话有更高的稳定性)
```

#### 16.10.6.3.MeanAbsoluteError()

实例化平均绝对损失函数

```python
from tensorflow.keras.losses import MeanAbsoluteError
loss = MeanAbsoluteError()
```

#### 16.10.6.4.SparseCategoricalCrossentropy()

实例化多分类交叉熵损失函数

```python
from tensorflow.keras.losses import SparseCategoricalCrossentropy
loss = SparseCategoricalCrossentropy(from_logits=True)  # 是否将y_pred解释为张量|bool|False(True的话有更高的稳定性)
```

### 16.10.7.metrics

| 版本 | 描述        | 注意 |
| ---- | ----------- | ---- |
| -    | 评估函数API |      |

#### 16.10.7.1.MAE()

平均绝对误差评估函数|mean_absolute_error

```python
from tensorflow.keras.metrics import MAE
mae = MAE
```

### 16.10.8.models

| 版本 | 描述          | 注意                                                         |
| ---- | ------------- | ------------------------------------------------------------ |
| -    | 构建Keras模型 | Keras支持两种模型Sequential和Model(Functional API)， 模型的类方法基本一致，相同的统一写在Model里 |

#### 16.10.8.1.load_model()

加载模型|Keras model

```python
from tensorflow.keras.models import load_model
model = load_model(filepath='model.h5')  # 文件路径|str or pathlib.Path
```

#### 16.10.8.2.Model()

实例化一个Model类对象(Functional API)

```python
from tensorflow.keras.models import Model
model = Model(inputs,  # 输入层|keras.Input or list of keras.Input
              outputs)  # 输出层|keras.layers
```

##### 16.10.8.2.1.build()

根据输入的形状构建模型

```python
model.build(input_shape)  # 输入的形状|tuple, TensorShape, or list of shapes
```

##### 16.10.8.2.2.compile()

配置模型训练的参数

```python
model.compile(optimizer='rmsprop',  # 优化器|str or keras.optimizers|'rmsprop'
              loss=None,  # 损失函数|str or tf.keras.losses.Loss|None
              metrics=None)  # 评估指标列表|list of metrics or keras.metrics.Metric|None 
```

##### 16.10.8.2.3.evaluate()

在测试模式下计算损失和准确率

```python
model.evaluate(x,  # 特征数据|numpy.array (or array-like), TensorFlow tensor, or a list of tensors, tf.data, generator or keras.utils.Sequence
               y=None,  # 标签|numpy.array (or array-like), TensorFlow tensor(如果x是dataset, generators,y为None)
               batch_size=32,  # 批次大小|int|32
               verbose=1)  # 日志模式|int(0, 1)|1
```

##### 16.10.8.2.4.fit()

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
          initial_epoch=0,  # 初始化训练轮数(多用于续训)|int|0
          shuffle=True,  # 是否打乱|bool|True
          class_weight=None,  # 类别的权重字典(只在训练时有效)|dict|None|可选
          steps_per_epoch=None,  # 每轮的总步数(样本数/批次大小)|int|None
          workers=1,  # 使用的线程数(仅用于tf.keras.utils.Sequence)|int|1
          use_multiprocessing=False)  # 是否使用多线程(仅用于tf.keras.utils.Sequence)|bool|False
```

##### 16.10.8.2.5.fit_generator()

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

##### 16.10.8.2.6.load_weights()

加载模型的权重

```python
model.load_weights(filepath)  # 文件路径|str or pathlib.Path
```

##### 16.10.8.2.7.predict()

进行预测|numpy.ndarray

```python
model.predict(x,  # 特征数据|numpy.array (or array-like), TensorFlow tensor, tf.data, generator or keras.utils.Sequence
              batch_size=32,  # 批次大小|int|32
              verbose=0)  # 日志模式|int(0, 1, 2详细)|0
```

##### 16.10.8.2.8.output_shape

返回输出层的形状

```python
print(model.output_shape)
```

##### 16.10.8.2.9.save()

保存模型

```python
model.save(filepath,  # 文件路径|str or pathlib.Path
           save_format=None)  # 保存格式|str('tf' or 'h5')|tf
```

##### 16.10.8.2.10.summary()

输出的模型摘要

```python
model.summary()
```

#### 16.10.8.3.Sequential()

实例化一个Sequential类对象

```python
from tensorflow.keras.models import Sequential
model = Sequential()
```

##### 16.10.8.3.1.add()

添加一个layer实例到Sequential栈顶

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
model = Sequential()
model.add(layer=Input(shape=(224, 224, 3)))  # 层示例｜keras.layer
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=2, activation='sigmoid'))
```

### 16.10.9.optimizers

| 版本 | 描述      | 注意 |
| ---- | --------- | ---- |
| -    | 优化器API |      |

#### 16.10.9.1.Adam()

实例化一个Adam优化器

```python
from tensorflow.keras.optimizers import Adam
optimziers = Adam(learning_rate)  # 学习率|float|0.001
```

#### 16.10.9.2.apply_gradients()

将梯度带计算出来的值赋值给优化器

```python
from tensorflow.keras.optimizers import Adam
optimziers = Adam(learning_rate=1e-4)
Adam.apply_gradients(grads_and_vars=zip(grads, vars))  # 梯度和变量|List of (gradient, variable) pairs
```

#### 16.10.9.3.SGD()

实例化一个随机梯度下降优化器

```python
from tensorflow.keras.optimizers import SGD
optimziers = SGD(learning_rate)  # 学习率|float|0.01
```

### 16.10.10.preprocessing

| 版本 | 描述               | 注意                     |
| ---- | ------------------ | ------------------------ |
| -    | Keras数据预处理API | 可以处理序列、文本、图像 |

#### 16.10.10.1.image

##### 16.10.10.1.1.array_to_img()

将numpy数组转换为PIL图像|numpy.ndarray

```python
from tensorflow.keras.preprocessing.image import array_to_img
image = array_to_img(x)  # 输入的数组|numpy.ndarray
```

##### 16.10.10.1.2.ImageDataGenerator()

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

###### 16.10.10.1.2.1.class_indices

类名称和类索引的映射字典|dict

```python
generator.flow_from_dataframe().class_indices
generator.flow_from_directory().class_indices
```

###### 16.10.10.1.2.2.flow()

给定数据和标签进行增强|yield

```python
generator.flow(x,  # 输入数据|numpy.array of rank 4 or a tuple
               y=None,  # 标签|array-like
               batch_size=32,  # 批次大小|int|32
               shuffle=True)  # 是否打乱|bool|True
```

###### 16.10.10.1.2.3.flow_from_dataframe()

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

###### 16.10.10.1.2.4.flow_from_directory()

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

##### 16.10.10.1.4.img_to_array()

将PIL图像转换为numpy数组|numpy.ndarray

```python
from tensorflow.keras.preprocessing.image import img_to_array
array = img_to_array(img)  # 输入的图像|PIL图像
```

##### 16.10.10.1.3.load_image()

加载PIL图像|PIL图像

```python
from tensorflow.keras.preprocessing.image import load_img
img = load_img(path,  # 文件路径|str or pathlib.Path
               target_size=None)  # 读取图片的大小|tuple of int|None
```

#### 16.10.10.2.timeseries_dataset_from_array()

从数组中创建滑动窗口的时间序列数据集.|tensorflow.python.data.ops.dataset_ops.BatchDataset

```python
from tensorflow.keras.preprocessing import timeseries_dataset_from_array

dataset = timeseries_dataset_from_array(data,  # numpy.ndarray or eager tensor|输入数据.
                                        targets,  # numpy.ndarray or eager tensor|标签.
                                        sequence_length,  # int|输出的序列长度.
                                        sampling_rate=1,  # int|1|连续时间步之间的时间间隔.
                                        batch_size=128)  # int|128|批次大小.
```

### 16.10.11.regularizers

| 版本 | 描述        | 注意 |
| ---- | ----------- | ---- |
| -    | 正则化器API |      |

#### 16.10.11.1.l2()

实例化一个L2正则化器

```python
from tensorflow.keras.regularizers import l2
regularizer = l2(l2=0.01)  # L2正则化因子|float|0.01
```

### 16.10.12.utils

| 版本 | 描述    | 注意 |
| ---- | ------- | ---- |
| -    | 工具API |      |

#### 16.10.12.1.get_file()

从指定URL下载文件|Path to the downloaded file

```python
from tensorflow.keras.utils import get_file
file = get_file(fname,  # 文件名|str
                origin,  # 文件的URL|str
                extract)  # tar和zip文件是否解压|bool|False
```

#### 16.10.12.2.plot_model()

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

#### 16.10.12.3.Sequence()

实现数据序列(必须实现\_\_getitem\_\_, \_\_len\_\_)

```python
from tensorflow.keras.utils import Sequence
class DataSequence(Sequence):
    def __init__(self, **kwargs):
        super(DataSequence, self).__init__(**kwargs)
        self.on_epoch_end()
        
    def __getitem__(self, item):
        """获取一个批次数据."""

    def __len__(self):
        """批次总数量."""

    def on_epoch_end(self):
        """每轮结束后对数据集进行某种操作."""
```

#### 16.10.12.4.to_categorical()

将标签的离散编码转换为one-hot编码|numpy.ndarray

```python
from tensorflow.keras.utils import to_categorical
y = [1, 2, 3, 4]
y = to_categorical(y=y,  # 输入的标签|array-like of int
                   num_classes=5)  # 类别总数|int|None
```

## 16.11.ones()

创建一个全一的张量|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.ones(shape=(3, 2), # 输入的张量|array-like
                 dtype='int64') # 元素的数据类型|str|dtypes.float32
```

## 16.12.ones_like()

创建一个和输入形状相同的全一张量|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.ones_like(input=[[1, 2, 3], [4, 5, 6]])  # 输入的张量|array-like
```

## 16.13.random

### 16.13.1.normal()

生成一个标准正态分布的张量|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.random.normal(shape=[2, 3])  # 形状|1-D integer Tensor or Python array
```

## 16.14.strings

| 版本 | 描述           | 注意 |
| ---- | -------------- | ---- |
| -    | 字符串操作模块 | -    |

### 16.14.1.reduce_join()

将所有的字符连接成一个字符串|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
input = ['1', 'a', '2', 'b']
x = tf.strings.reduce_join(inputs=input)  # 输入的字符|array-like
```

### 16.14.2.unicode_split()

将输入的字符串的每一个字符转换成Unicode编码的bytes|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
string = 'cat'
tensor = tf.strings.unicode_split(input=string,  # 输入字符串|str
                                  input_encoding='UTF-8')  # 输入字符串的编码|str
```

## 16.15.tensordot()

计算沿指定维度的点积|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.tensordot(a=[[1], [2]],  # 张量|array-like
                      b=[[2, 1]],  # 张量|array-like
                      axes=1)  # 维度|scalar N or list or int32 Tensor of shape [2, k]
```

## 16.16.tpu

### 16.16.1.experimental

#### 16.16.1.1.initialize_tpu_system()

初始化TPU设备

```python
import tensorflow as tf
tf.tpu.experimental.initialize_tpu_system(cluster_resolver=tpu)  # TPU集群信息|tf.distribute.cluster_resolver.TPUClusterResolver|None
```

## 16.17.transpose()

对张量进行转置|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = [[[1, 2, 3], [4, 5, 6]]]
tensor = tf.transpose(a=tensor,  # 输入的数组|array-like
                      perm=[1, 0, 2])  # 轴的排列顺序|list of ints|None|可选
```

## 16.18.zeros_like()

创建一个和输入形状相同的全零张量|tensorflow.python.framework.ops.EagerTensor

```python
import tensorflow as tf
tensor = tf.zeros_like(input=[[1, 2, 3], [4, 5, 6]])  # 输入的张量|array-like
```

# 17.tensorflow.js

| 版本  | 描述                                        | 注意                                                        |
| ----- | ------------------------------------------- | ----------------------------------------------------------- |
| 2.3.0 | TensorFlow.js是TensorFlow的JavaScript软件库 | TensorFlow.js现在全面使用ES6语法；如果使用node.js有轻微差异 |

## 17.1.browser

### 17.1.1.fromPixels()

从一张图片中创建tf.Tensor|tf.Tensor3D

```javascript
import * as tf from "@tensorflow/tfjs";
// pixels: 构建张量的像素(PixelData|ImageData|HTMLImageElement|HTMLCanvasElement|HTMLVideoElement)
let img = tf.browser.fromPixels();
```

## 17.2.div()

两个tf.Tensor逐元素相除|tf.Tensor

```javascript
import * as tf from "@tensorflow/tfjs";
let a = tf.scalar(5);
let b = tf.scalar(2);
// b: 除数(tf.Tensor|TypedArray|Array)
let c = a.div(b);
```

## 17.3.expandDims()

扩展tf.Tensor的维度|tf.Tensor

```javascript
import * as tf from "@tensorflow/tfjs";
let t = tf.tensor([1, 2, 3, 4]);
// axis: 维度(number)|可省略
let t1 = t.expandDims(0);
```

## 17.4.image

### 17.4.1.resizeBilinear()

使用线性插值法改变图像的形状|tf.Tensor3D or tf.Tensor4D

```javascript
import * as tf from '@tensorflow/tfjs';
// images: 要改变形状的图像(tf.Tensor3D|tf.Tensor4D|TypedArray|Array); size: 改变后的大小([number, number])
let resized_img = tf.image.resizeBilinear(img, [224, 224]);
```

## 17.5.LayersModel

### 17.5.1.predict()

进行预测|tf.Tensor|tf.Tensor[]

```javascript
import * as tf from "@tensorflow/tfjs";
// x: 用于预测的数据(tf.Tensor|tf.Tensor[])
let result = model.predict();
```

### 17.5.2.summary()

输出的模型摘要

```javascript
import * as tf from "@tensorflow/tfjs";
model.summary();
```

## 17.6.loadLayersModel()

加载一个tf.LayersModel(使用Keras训练的非Functional API训练的模型)|Promise<tf.LayersModel>

```javascript
import * as tf from "@tensorflow/tfjs";
// pathOrIOHandler: 模型的路径(file://仅限tfjs-node; http://或者https://可以是绝对或者相对路径)
let model = tf.loadLayersModel();
```

## 17.7.print()

输出信息的控制台

```javascript
import * as tf from "@tensorflow/tfjs";
// x: 要输出的张量(tf.Tensor)
tf.print();
```

## 17.8.scalar()

创建一个tf.Tensor(scalar)的标量

```javascript
import * as tf from "@tensorflow/tfjs";
// value: 标量的数值; dtype: 数字的数据类型|可省略
let s = tf.scalar(10, "float32");
tf.print(s);
```

## 17.9.sub()

两个tf.Tensor逐元素相减|tf.Tensor

```javascript
import * as tf from "@tensorflow/tfjs";
let a = tf.scalar(1);
let b = tf.scalar(2);
// b: 减数(tf.Tensor|TypedArray|Array)
let c = a.sub(b);
```

## 17.10.tensor()

创建一个tf.Tensor的张量

```javascript
import * as tf from "@tensorflow/tfjs";
// value: 张量的数值
let t = tf.tensor(10);
tf.print(t);
```

### 17.10.1.data()

异步获取tf.Tensor的值|Promise<DataTypeMap[NumericDataType]>

```javascript
import * as tf from "@tensorflow/tfjs";
let t = tf.tensor(10);
let value = t.data();
```

## 17.11.tidy()

执行传入的函数后，自动清除除返回值以外的系统分配的所有的中间张量，防止内存泄露|void ,nmber,string,TypedArray,tf.Tensor,tf.Tensor[],{[key: string]:tf.Tensor,number,string}

```javascript
import * as tf from "@tensorflow/tfjs";
// fn: 传入的函数
let result = tf.tidy(fn);
```

# 18.tensorflow_datasets

| 版本  | 描述                  | 注意         |
| ----- | --------------------- | ------------ |
| 4.2.0 | TensoFlow的官方数据集 | 需要使用代理 |

## 18.1.load()

加载数据集.|dict of tf.data.Datasets

```python
import tensorflow_datasets as tfds

ds_train, ds_test = tfds.load(name='mnist',  # str|DatasetBuilder的注册名称.
                              split=['train', 'test'],  # ['train', 'test'], train[80%:](可选)|None|是否拆分数据.
                              shuffle_files=True,  # bool|False|是否打散数据.
                              as_supervised=True)  # bool|False|是否监督(是返回带有标签的tf.data.Dataset, 否返回字典形式的tf.data.Dataset).
```

# 19.tensorflow_hub

| 版本  | 描述                  | 注意                                                         |
| ----- | --------------------- | ------------------------------------------------------------ |
| 0.8.0 | TensoFlow的官方模型库 | 暂不清楚模型的默认保存路径，最好使用os.environ['TFHUB_CACHE_DIR']手动指定一个位置；SavedModel模型不可以转换成hdf5格式 |

## 19.1.KerasLayer()

将SavedModel或者Hub.Module修饰成一个tf.keras.layers.Layer实例

```python
from tensorflow_hub import KerasLayer
layer = KerasLayer(handle,  # 模型的路径|str
                   trainable,  # 能否训练|bool|(hub.Modules不可训练)|可选
                   output_shape,  # 输出的形状|tuple|None(模型本身有形状就不能设置)|可选
                   input_shape,  # 期望的输入的形状|tuple|可选
                   dtype)  # 期望的数据类型|tensorflow.python.framework.dtypes.DType|可选
```

## 19.2.load()

加载一个SavedModel|tensorflow.python.saved_model

```python
from tensorflow_hub import load
model = load(handle)  # 模型的路径|str
```

# 20.tokenizers

| 版本  | 描述           | 注意 |
| ----- | -------------- | ---- |
| 0.9.2 | 自定义的标记器 |      |

## 20.1.ByteLevelBPETokenizer()

实例化一个字节级的BPE标记器

```python
from tokenizers import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer(vocab='./vocab-roberta-base.json',  # 词汇表|str|None
                                  merges='./merges-roberta-base.txt',  # token表|str|None
                                  add_prefix_space=True,  # 是否使用特殊标记器|bool|True|可选
                                  lowercase=True)  # 转换为小写字母|bool|可选
```

### 20.1.1.decode()

解码给定的id列表|str

```python
from tokenizers import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer(vocab='./roberta/vocab-roberta-base.json',
                                  merges='./roberta/merges-roberta-base.txt',
                                  add_prefix_space=True,
                                  lowercase=True)
raw_text = 'Hello Transformers!'
encoder = tokenizer.encode(sequence=raw_text)
text = tokenizer.decode(ids=encoder.ids)  # 要解码的id列表|list
print(text)
```

### 20.1.2.encode()

编码给定的序列和序列对|tokenizers.Encoding

```python
from tokenizers import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer(vocab='./vocab-roberta-base.json',
                                  merges='./merges-roberta-base.txt',
                                  add_prefix_space=True,
                                  lowercase=True)
raw_text = 'Hello Transformers!'
encoder = tokenizer.encode(sequence=raw_text)  # 输入序列|str
```

#### 20.1.2.1.ids

编码后的id列表|list

```python
print(encoder.ids)
```

# 21.transformers

| 版本  | 描述                                                | 注意                                      |
| ----- | --------------------------------------------------- | ----------------------------------------- |
| 3.4.0 | 基于Pytorch或者TensorFlow 2上最先进的自然语言处理库 | 默认保存路径为~/.cache/torch/transformers |

## 21.1.AlbertTokenizer

### 21.1.1.\_\_call\_\_()

为模型标记一个或者多个序列序列数据|input_ids, (token_type_ids, attention_mask)需要设置返回为真

```python
from transformers import AlbertTokenizer
tokenizer = AlbertTokenizer.from_pretrained(pretrained_model_name_or_path='albert-base-v1',
                                            do_lower_case=True)
encoder = tokenizer(text=x,  # 需要标记的文本|str or list of str or list of list
                    add_special_tokens=True,  # 是否使用特殊标记器|bool|True|可选
                    padding=True,  # 是否填充到最大长度|bool|False|可选
                    truncation=True,  # 是否截断到最大长度|bool|False|可选
                    max_length=128,  # 填充和截断的最大长度(XLNet没有最大长度, 此参数将被禁用)|int|可选
                    return_tensors='tf',  # 返回张量|{'tf', 'pt', 'np'}|list of int|可选
                    return_token_type_ids=True,  # 是否返回令牌ID|bool|可选
                    return_attention_mask=True)  # 是否返回注意力掩码|bool|可选
```

### 21.1.2.from_pretrained()

实例化一个Albert的预训练标记器|transformers.tokenization_albert.AlbertTokenizer

```python
from transformers import AlbertTokenizer
tokenizer = AlbertTokenizer.from_pretrained(pretrained_model_name_or_path='albert-base-v1',  # 预训练的名称或位置|str
                                            do_lower_case=True)  # 转换为小写字母|bool|可选
```

## 21.2.BertTokenizer

### 21.2.1.\_\_call\_\_()

为模型标记一个或者多个序列序列数据|input_ids, (token_type_ids, attention_mask)需要设置返回为真

```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-uncased',
                                          do_lower_case=True)
encoder = tokenizer(text=x,  # 需要标记的文本|str or list of str or list of list
                    add_special_tokens=True,  # 是否使用特殊标记器|bool|True|可选
                    padding=True,  # 是否填充到最大长度|bool|False|可选
                    truncation=True,  # 是否截断到最大长度|bool|False|可选
                    max_length=128,  # 填充和截断的最大长度(XLNet没有最大长度, 此参数将被禁用)|int|可选
                    return_tensors='tf',  # 返回张量|{'tf', 'pt', 'np'}|list of int|可选
                    return_token_type_ids=True,  # 是否返回令牌ID|bool|可选
                    return_attention_mask=True)  # 是否返回注意力掩码|bool|可选
```

### 21.2.2.from_pretrained()

实例化一个Bert的预训练标记器|transformers.tokenization_bert.BertTokenizer

```python
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-uncased',  # 预训练的名称或位置|str
                                          do_lower_case=True,  # 转换为小写字母|bool|可选
                                          cache_dir=None)  # 缓存的目录|str|可选 
```

## 21.3.RobertaConfig

### 21.3.1.from_pretrained()

从预训练模型配置中实例化PretrainedConfig|transformers.configuration_roberta.RobertaConfig

```python
from transformers import RobertaConfig
config = RobertaConfig.from_pretrained(pretrained_model_name_or_path='roberta-base')  # 预训练的名称或位置|str
```

## 21.4.TFAlbertModel

### 21.4.1.from_pretrained()

从预训练模型配置中实例化TF2的模型|transformers.modeling_tf_albert.TFAlbertModel

```python
from transformers import TFAlbertModel
model = TFAlbertModel.from_pretrained(pretrained_model_name_or_path='albert-base-v1',  # 预训练的名称或位置|str
                                      trainable=True)  # 能否训练|bool|可选
```

## 21.5.TFBertModel

### 21.5.1.from_pretrained()

从预训练模型配置中实例化TF2的模型|transformers.modeling_tf_bert.TFBertModel

```python
from transformers import TFBertModel
model = TFBertModel.from_pretrained(pretrained_model_name_or_path='bert-base-uncased',  # 预训练的名称或位置|str
                                    trainable=True,  # 能否训练|bool|可选
                                    cache_dir=None)  # 缓存的目录|str|可选 
```

## 21.6.TFRobertaModel

### 21.6.1.from_pretrained()

从预训练模型配置中实例化TF2的模型|transformers.modeling_tf_roberta.TFRobertaModel

```python
from transformers import TFRobertaModel
model = TFRobertaModel.from_pretrained(pretrained_model_name_or_path='roberta-base',  # 预训练的名称或位置|str
                                       config=RobertaConfig.from_pretrained('roberta-base'))  # 模型的配置类|transformers.PretrainedConfig
```

# 22.xgboost

| 版本  | 描述       | 注意                |
| ----- | ---------- | ------------------- |
| 1.1.1 | 梯度提升树 | 可直接在sklearn使用 |

## 22.1.XGBClassifier()

实例化一个XGBoost分类器

```python
from xgboost import XGBClassifier
model = XGBClassifier(max_depth,  # 基学习器(梯度提升树)的最大深度|int|None|可选
                      learning_rate,  # 学习率|float|None|可选
                      n_estimators,  # 梯度提升树的数量(相当于学习轮数)|int|100
                      objective,  # 使用的损失函数|str|'reg:squarederror'
                      booster,  # 使用的基学习器|str('gbtree', 'gblinear', 'dart')|None|可选
                      n_jobs,  # 并行数量|int|None|可选
                      subsample,  # 随机采样率|float|None|可选
                      colsample_bytree,  # 构造每棵树，属性随机采样率|float|None|可选
                      random_state)  # 随机状态|int|None|可选
```

### 22.1.1.fit()

训练XGBoost分类器|self

```python
model.fit(X,  # 特征数据|array-like
          y,  # 标签|array-like
          eval_set,  # 验证集元组列表|list of (X, y) tuple|None|可选
          eval_metric,  # 验证使用的评估指标|str or list of str or callable|None|可选 
          verbose)  # 日志模式|bool|True
```

### 22.1.2.predict()

进行预测|numpy.ndarray

```python
result = model.predict(data)  
# 用于预测的数据|array_like
```

## 22.2.XGBRegressor()

实例化一个XGBoost回归器

```python
from xgboost import XGBRegressor
model = XGBRegressor(max_depth,  # 基学习器(梯度提升树)的最大深度|int|None|可选
                     learning_rate,  # 学习率|float|None|可选
                     n_estimators,  # 梯度提升树的数量(相当于学习轮数)|int|100
                     objective,  # 使用的损失函数|str|'reg:squarederror'
                     n_jobs,  # 并行数量|int|None|可选
                     subsample,  # 随机采样率|float|None|可选
                     colsample_bytree,  # 构造每棵树，属性随机采样率|float|None|可选
                     random_state)  # 随机状态|int|None|可选
```

### 22.2.1.fit()

训练XGBoost回归器|self

```python
model.fit(X,  # 特征数据|array-like
          y,  # 标签|array-like
          eval_set,  # 验证集元组列表|list of (X, y) tuple|None|可选
          eval_metric,  # 验证使用的评估指标|str or list of str or callable|None|可选
          early_stopping_rounds,  # 早停的轮数|int|None
          verbose)  # 日志模式|bool|True
```

### 22.2.2.predict()

进行预测|numpy.ndarray

```python
result = model.predict(data)  
# 用于预测的数据|array_like
```

### 22.2.3.score()

计算验证集的平均准确率|float

```python
accuracy = model.score(X,  # 特征数据|{array-like, sparse matrix} of shape (n_samples, n_features)
                       y)  # 标签|array-like of shape (n_samples,)
```
