# <center>A²PI²-API version2.2</center>

* 2.2版本添加了关于M1芯片的适配情况, 使用Rosetta 2实现的运行的也将标记为否.
* 将TensorFlow生态的全部包单独存放在一个文件.
* 简化重复的API信息

# 1.catboost

| 版本 | 描述                  | 注意                                                         | 适配M1 |
| ---- | --------------------- | ------------------------------------------------------------ | ------ |
| 0.26 | 梯度提升决策树(GBDT). | 1. 可直接在sklearn使用.                                                                                                                              2. 模型的类方法基本没有差异, 具体参见`CatBoostClassifier`的类方法. | 否     |

## 1.1.CatBoostClassifier()

实例化CatBoost分类器.

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(iterations=None,  # int|500|最大的决策树数量.
                           learning_rate=None,  # float|0.03|学习率.
                           depth=None,  # int|6|决策树的最大深度.
                           l2_leaf_reg=None,  # float|3.0|损失函数的L2正则化系数.
                           loss_function=None,  # str or object|'Logloss'|损失函数.
                           od_wait=None,  # int|None|得到最佳结果后迭代的次数.
                           od_type=None,  # {'IncToDec', 'Iter'}|None|过拟合检测器的类型.
                           random_seed=None,  # int|None|随机种子.
                           bagging_temperature=None,  # float|None|使用贝叶斯分配初始化权重.
                           bootstrap_type=None,  # {'Bayesian', 'Bernoulli', 'Poisson', 'MVS'}|启动类型.
                           colsample_bylevel=None,  # float|随机特征选择时, 每次拆分选择使用的特征百分比.
                           allow_writing_files=None,  # bool|True|是否导出缓存文件.
                           cat_features=None)  # list or numpy.ndarray|None|类别的索引列.
```

### 1.1.1.fit()

训练CatBoost分类器.|`self`

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier(iterations=None,
                           learning_rate=None,
                           depth=None,
                           l2_leaf_reg=None,
                           loss_function=None,
                           od_wait=None,
                           od_type=None,
                           random_seed=None,
                           bagging_temperature=None,
                           bootstrap_type=None,
                           colsample_bylevel=None,
                           allow_writing_files=None,
                           cat_features=None)
model.fit(X,  # catboost.Pool or list or numpy.ndarray or pandas.DataFrame or pandas.Series|特征数据.
          y=None,  # list or numpy.ndarray or pandas.DataFrame or pandas.Series(可选)|None|标签.
          text_features=None,  # list or numpy.ndarray(可选)|None|特征数据的索引列.
          eval_set=None,  # catboost.Pool or list(可选)|None|验证集元组列表.
          verbose=None)  # bool or int|日志显示模式.
```

### 1.1.2.feature_importances_

特征的重要度.|`numpy.ndarray`

```python
model.feature_importances_
```

### 1.1.3.feature_names_

特征的名称.|`list of str`

```python
model.feature_names_
```

### 1.1.4.predict()

使用CatBoost分类器进行预测.|`numpy.ndarray`

```python
from catboost import CatBoostClassifier

model = CatBoostClassifier()
y_preds = model.predict(data)  # catboost.Pool or list of features or list of lists or numpy.ndarray or pandas.DataFrame or pandas.Series or catboost.FeaturesData|特征数据.
```

## 1.2.CatBoostRegressor()

实例化CatBoost回归器.

```python
from catboost import CatBoostRegressor

model = CatBoostRegressor(iterations=None,  # int|500|最大的决策树数量.
                          learning_rate=None,  # float|0.03|学习率.
                          depth=None,  # int|6|决策树的最大深度.
                          l2_leaf_reg=None,  # float|3.0|损失函数的L2正则化系数.
                          loss_function=None,
                          # {'RMSE', 'MAE', 'Quantile:alpha=value', 'LogLinQuantile:alpha=value', 'Poisson', 'MAPE',
                          # 'Lq:q=value', 'SurvivalAft:dist=value;scale=value'}|'RMSE'|损失函数.
                          od_wait=None,  # int|None|得到最佳结果后迭代的次数.
                          od_type=None,  # {'IncToDec', 'Iter'}|None|过拟合检测器的类型.
                          random_seed=None,  # int|None|随机种子.
                          bagging_temperature=None,  # float|None|使用贝叶斯分配初始化权重.
                          bootstrap_type=None,  # {'Bayesian', 'Bernoulli', 'Poisson', 'MVS'}|启动类型.
                          colsample_bylevel=None,  # float|随机特征选择时, 每次拆分选择使用的特征百分比.
                          allow_writing_files=None,  # bool|True|是否导出缓存文件.
                          cat_features=None)  # list or numpy.ndarray|None|类别的索引列.
```

# 2.cv2

| 版本  | 描述            | 注意                                                         | 适配M1 |
| ----- | --------------- | ------------------------------------------------------------ | ------ |
| 4.5.0 | 图像处理软件库. | 1. M1目前需要使用conda安装, 使用conda install opencv                                                                                   2. OpenCV的图片格式是HWC, TensorFlow的是WHC.                                                                                          3. Intel-based Mac安装时使用pip install opencv-python | 是     |

## 2.1.imread()

加载指定路径的图片.|`numpy.ndarray`

```python
import cv2

image = cv2.imread(filename='./image.png',  # str|要加载的文件的路径.
                   flags=2)  # int|None|读入的色彩方式.
```

## 2.2.resize()

将图像调整到指定大小.|`numpy.ndarray`

```python
import cv2

image = cv2.imread('./image.png')
image = cv2.resize(src=image,  # numpy.ndarray|输入的图像.
                   dsize=(2000, 1000))  # tuple|修改后图像的尺寸.
```

# 3.Eigen

| 版本  | 描述               | 注意                                                         | 适配M1 |
| ----- | ------------------ | ------------------------------------------------------------ | ------ |
| 3.4.0 | C++线性代数模版库. | 1. 在ubuntu 16.04下请使用源码安装, apt的最高版本是3.3beta0, 存在BUG. | 是     |

## 3.1.ArrayXd

实例化一维(双精度)动态数组.

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

实例化二维(双精度)动态数组.

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

逐元素计算e的幂次.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::ArrayXXd arr(2, 2);
    arr << 1, 2, 3, 4;
    
    std::cout << arr.exp() << std::endl;
    
    return 0;
}
```

### 3.2.2.log()

逐元素计算自然对数.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::ArrayXXd arr(2, 2);
    arr << 2.718281, 7.38906, 20.0855, 54.5982;

    std::cout << arr.log() << std::endl;
    
    return 0;
}
```

### 3.2.3.pow()

逐元素计算指定的幂次.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::ArrayXXd arr(2, 2);
    arr << 1, 2, 3, 4;

    std::cout << arr.pow(2) << std::endl;
    
    return 0;
}
```

### 3.2.4.tanh()

逐元素计算双曲正切.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::ArrayXXd arr(2, 2);
    arr << 1, 2, 3, 4;

    std::cout << arr.tanh() << std::endl;
    
    return 0;
}
```

## 3.3.BDCSVD<>

实例化对角分治SVD.

```c++
#include "Eigen/Dense"

int main() {
    // 用法一(仅初始化).
    Eigen::BDCSVD<Eigen::MatrixXd> svd;
  	// 用法二(初始化并传入要分解的矩阵).
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    Eigen::BDCSVD<Eigen::MatrixXd> bdc_svd(mat);
    
    return 0;
}
```

### 3.3.1.compute()

对要分解的矩阵进行计算.

```c++
#include "Eigen/Dense"

int main() {
    Eigen::BDCSVD<Eigen::MatrixXd> svd;
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    svd.compute(mat);
    
    return 0;
}
```

### 3.3.2.matrixU()

获取U矩阵(左奇异向量).

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

### 3.3.3.matrixV()

获取V矩阵(右奇异向量).

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

### 3.3.4.singularValues()

获取奇异值向量(按照降序排序).

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

映射到现有的矩阵或向量.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::Matrix<double, 2, 2> mat;
    // 映射矩阵或者向量的首地址的指针, 映射后的行数和列数
    Eigen::Map<Eigen::MatrixXd> map(mat.data(), 1, 4);
    
    mat << 1, 2, 3, 4;
    
    std::cout << mat << std::endl;
    std::cout << map << std::endl;
    
    return 0;
}
```

## 3.5.Matrix<>

实例化矩阵.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    // 元素类型, 矩阵的行数和列数
    Eigen::Matrix<int, 2, 2> mat;
    mat << 1, 2, 3, 4;
    
    std::cout << mat << std::endl;
    
    return 0;
}
```

## 3.6.MatrixXd

实例化(双精度)动态矩阵.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    
    std::cout << mat << std::endl;
    
    return 0;
}
```

### 3.6.1.adjoint()

获取矩阵的伴随(共轭转置)矩阵.

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

将矩阵修饰成数组, 便于执行逐元素的操作.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    
    // eigen3不能跟numpy直接对元素直接操作,
    // 使用array方法后将被修饰成一个ArrayWrapper对象,
    // 才能使用例如log, exp等方法.
    
    // 这样是无法编译的.
    /* std::cout << mat.log() << std::endl; */
  
    // 正确做法.
    std::cout << mat.array().log() << std::endl;
    
    return 0;
}
```

### 3.6.3.cols()

获取矩阵的列数.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3, 4, 5, 6;
    
    std::cout << mat.cols() << std::endl;
    
    return 0;
}
```

### 3.6.4.data()

获取矩阵首地址的指针.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    
    std::cout << mat.data() << std::endl;
    
    return 0;
}
```

### 3.6.5.inverse()

获取矩阵的逆矩阵.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    
    std::cout << mat.inverse() << std::endl;
    
    return 0;
}
```

### 3.6.6.maxCoeff()

获取矩阵元素的最大值和对应的下标.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    int row, col;
    Eigen::MatrixXd mat(2, 3);
    mat << 1, 2, 3, 4, 5, 6;
  
    // 用法一(仅返回最大值).
    std::cout << "max value:" << mat.maxCoeff() << std::endl;
    // 用法二(返回最大值和对应下标).
    std::cout << "max value:" << mat.maxCoeff(&row, &col) << " row:" << row << " col:" << col << std::endl;
    
    return 0;
}

```

### 3.6.7.Random

实例化(双精度)均匀分布的随机矩阵.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd rnd_mat = Eigen::MatrixXd::Random(2, 2);
    
    std::cout << rnd_mat << std::endl;

    return 0;
}
```

### 3.6.8.reshaped()

改变矩阵的形状.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;

    std::cout << mat.reshaped(1, 4) << std::endl;

    return 0;
}
```

### 3.6.9.resize()

原地改变矩阵的形状.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;

    mat.resize(1, 4);
    
    std::cout << mat << std::endl;

    return 0;
}
```

### 3.6.10.row()

获取矩阵的某行.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;

    std::cout << mat.row(1) << std::endl;
    
    return 0;
}
```

### 3.6.11.rows()

获取矩阵的行数.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(3, 2);
    mat << 1, 2, 3, 4, 5, 6;
    
    std::cout << mat.rows() << std::endl;
    
    return 0;
}
```

### 3.6.12.rowwise()

将矩阵修饰成多个向量, 对矩阵进行逐行操作.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(3, 2);
    mat << 1, 2, 3, 4, 5, 6;
    
    std::cout << mat.rowwise().sum() << std::endl;
    
    return 0;
}
```

### 3.6.13.size()

获取矩阵的元素总数.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(3, 2);
    mat << 1, 2, 3, 4, 5, 6;
    
    std::cout << mat.size() << std::endl;
    
    return 0;
}
```

### 3.6.14.sum()

计算矩阵的元素和.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 5, 6, 7, 8;
    
    std::cout << mat.sum() << std::endl;
    
    return 0;
}
```

### 3.6.15.transpose()

获取矩阵的转置矩阵.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(2, 2);
    mat << 1, 2, 3, 4;
    
    std::cout << mat.transpose() << std::endl;
    
    return 0;
}
```

### 3.6.16.Zero()

实例化一个全零矩阵.

```c++
#include <iostream>
#include "Eigen/Core"

int main() {
    Eigen::MatrixXd zero_mat = Eigen::MatrixXd::Zero(2, 3);
  
    std::cout << zero_mat << std::endl;
    
    return 0;
}
```

#### 3.6.16.1.unaryExpr()

接收一个函数对矩阵进行逐元素的操作.

```c++
#include <iostream>
#include "Eigen/Core"

double i = 1.0;

int main() {
    Eigen::MatrixXd mat = Eigen::MatrixXd::Zero(2, 3).unaryExpr(
        [] (double element) {
            return i ++;
        }
    );

    std::cout << mat << std::endl;
  
    return 0;
}
```

## 3.7.RowVectorXd

实例化(双精度)动态行向量.

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

获取行向量的元素总数.

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

## 3.8.RowVectorXi

实例化一个行向量(整型).

```c++
#include <iostream>
#include "Eigen/Core"

int main() {
    Eigen::RowVectorXi vec(4);
    vec << 1, 2, 3, 4;
  
    std::cout << vec << std::endl;
  
    return 0;
}
```

### 3.8.1.size()

获取行向量的元素总数.

```c++
#include <iostream>
#include "Eigen/Core"

int main() {
    Eigen::RowVectorXi vec(4);
    vec << 1, 2, 3, 4;
  
    std::cout << vec.size() << std::endl;
  
    return 0;
}
```

## 3.9.VectorXd

实例化(双精度)列态行向量.

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

### 3.9.1.asDiagonal()

将向量转换对角阵.

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::VectorXd vec(4);
    vec << 1, 2, 3, 4;
    Eigen::MatrixXd diag_mat = vec.asDiagonal();
    
    std::cout << diag_mat << std::endl;
    
    return 0;
}
```

## 3.10. 对矩阵进行切片

```c++
#include <iostream>
#include "Eigen/Dense"

int main() {
    Eigen::MatrixXd mat(3, 3);
    mat << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    std::cout << mat(Eigen::all, Eigen::all) << std::endl;
    std::cout << "对任意行进行切片:\n" << mat({0, 2}, Eigen::all) << std::endl;
    std::cout << "对任意列进行切片:\n" << mat(Eigen::all, {1, 2}) << std::endl;
    std::cout << "对任意行列进行切片:\n" << mat({0, 2}, 2) << std::endl;
    return 0;
}
```

# 4.h5py

| 版本  | 描述                  | 注意                        | 适配M1 |
| ----- | --------------------- | --------------------------- | ------ |
| 3.1.0 | HDF5的Python操作接口. | 1. M1目前需要使用conda安装. | 是     |

## 4.1.File()

创建一个文件对象.|`h5py._hl.files.File`

```python
import h5py

fp = h5py.File(name='./file.h5',  # str or file-like object|硬盘上的文件名.
               mode='w')  # {'r', 'r+', 'w', 'w- or x', 'a'}|模式.
```

### 4.1.1.attrs[]

添加到文件对象的属性.

```python
import h5py

fp = h5py.File('./file.h5', 'w')
fp.attrs['a'] = 1

print(fp.attrs['a'])
```

### 4.1.2.close()

关闭文件对象.

```python
import h5py

fp = h5py.File('./file.h5', 'w')
fp.close()
```

### 4.1.3.create_dataset()

创建新的HDF5数据集.|`h5py._hl.dataset.Dataset`

```python
import h5py

fp = h5py.File('./file.h5', 'w')
dataset = fp.create_dataset(name='dataset',  # str|数据集的名称.
                            dtype=float)  # Numpy Dtype or str|数据集元素的类型.
```

### 4.1.4.create_group()

创建新的HDF5组.|`h5py._hl.group.Group`

```python
import h5py

fp = h5py.File(name='./file.h5', mode='w')
group = fp.create_group(name='group')  # str|组的名称.
```

# 5.imageio

| 版本  | 描述            | 注意 | 适配M1 |
| ----- | --------------- | ---- | ------ |
| 2.9.0 | 图像处理软件库. | -    | 是     |

## 5.1.imread()

加载指定路径的图片.|`imageio.core.util.Array`

```python
import imageio

image = imageio.imread(uri='./image.jpg')  # str or pathlib.Path or bytes or file|要加载的文件的路径.
```

# 6.lightgbm

| 版本  | 描述                  | 注意                                                         | 适配M1 |
| ----- | --------------------- | ------------------------------------------------------------ | ------ |
| 3.1.1 | 基于树的梯度提升框架. | 1. M1目前需要使用conda安装.                                                                                                                           2. 可直接在sklearn使用.                                                                                                                                       3. Intel-based Mac需要先使用brew安装libomp | 是     |

## 6.1.LGBMClassifier()

实例化LGBM分类器.

```python
from lightgbm import LGBMClassifier

model = LGBMClassifier(boosting_type='gbdt',  # {'gbdt', 'dart', 'goss', 'rf'}(可选)|'gbdt'|集成方式.
                       max_depth=-1,  # int(可选)|-1|基学习器的最大深度，负值表示没有限制.
                       learning_rate=0.1,  # float(可选)|0.1|学习率.
                       n_estimators=100)  # int(可选)|100|树的数量.
```

### 6.1.1.fit()

训练LGBM分类器.|`self`

```python
model.fit(X,  # array-like or 形状为[n_samples, n_features]的稀疏矩阵|特征数据.
          y,  # array-like|标签.
          eval_set)  # list of (X, y) tuple(可选)|None|验证集元组列表.
```

### 6.1.2.predict()

使用LGBM分类器进行预测.|`numpy.ndarray`

```python
y_preds = model.predict(X)  # array-like or 形状为[n_samples, n_features]的稀疏矩阵|用于预测的数据.
```

# 7.matplotlib

| 版本  | 描述          | 注意 | 适配M1 |
| ----- | ------------- | ---- | ------ |
| 3.3.2 | Python绘图库. | -    | 是     |

## 7.1.axes

| 版本 | 描述                                              | 注意 |
| ---- | ------------------------------------------------- | ---- |
| -    | axes是matplotlib的图形接口, 提供设置坐标系的功能. | -    |

### 7.1.1.add_patch()

添加元素.

```python
import matplotlib.pyplot as plt

axes = plt.subplot()
axes.add_patch(p=plt.Rectangle((0, 0), width=0.5, height=0.5))

plt.show()
```

### 7.1.2.annotate()

为坐标点进行注释.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.annotate(text='annotate',  # str|注释的内容.
            xy=(0.45, 0.5),  # (float, float)|注释点的坐标.
            xytext=(0.6, 0.6),  # (float, float)(可选)|None|注释内容的坐标.
            xycoords='data',  # str(可选)|'data'|注释点放置的坐标系.
            textcoords='data',  # str(未设置则和xycoords一致, 可选)|注释内容放置的坐标系.
            arrowprops=dict(arrowstyle='<-'),  # dict(可选)|None(即不绘制箭头)|绘制箭头的样式.
            size=20,  # int(可选)|10|注释文本字号.
            verticalalignment='baseline',  # str(可选)|'baseline'|垂直对齐.
            horizontalalignment='center',  # str(可选)|'left'|水平对齐.
            bbox=dict(fc='skyblue'))  # dict(可选)|None(即不绘制文本框)|绘制文本框.

plt.show()
```

### 7.1.3.axis()

坐标轴的设置选项.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.axis('off')
```

### 7.1.4.clabel()

在等高线上显示高度.

```python
import numpy as np
import matplotlib.pyplot as plt

ax = plt.subplot()
x = np.linspace(0, 9, 10)
y = np.linspace(0, 9, 10)
X, Y = np.meshgrid(x, y)
cs = ax.contour(X, Y, X + Y, colors='orange', linewidths=1)
ax.clabel(cs)

plt.show()
```

### 7.1.5.contour()

绘制等高线.|`matplotlib.contour.QuadContourSet`

```python
import numpy as np
import matplotlib.pyplot as plt

ax = plt.subplot()
x = np.linspace(0, 9, 10)
y = np.linspace(0, 9, 10)
X, Y = np.meshgrid(x, y)
cs = ax.contour(X,  # array-like|横坐标.
                Y,  # array-like|纵坐标.
                X + Y,  # array-like(必须是2D的)|横纵坐标的关系公式.
                colors='orange',  # str|等高线的颜色.
                linewidths=1.0,  # float|等高线的宽度.
                linestyles='dashed')  # str|等高线的形状.

plt.show()
```

### 7.1.6.grid()

绘制网格线.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.grid(axis='x',  # {'both', 'x', 'y'}(可选)|'both'|绘制的范围.
        linestyle=':')  # {'-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'}(可选)|'-'|网格线的样式.

plt.show()
```

### 7.1.7.legend()

放置图例.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.legend(loc='center')  # {'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'}(可选)|'best'|放置的位置.

plt.show()
```

### 7.1.8.patch

| 版本 | 描述                                   | 注意 |
| ---- | -------------------------------------- | ---- |
| -    | patches是画布颜色和边框颜色的控制接口. | -    |

#### 7.1.8.1.set_alpha()

设置画布的透明度.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.patch.set_alpha(alpha=0.1)  # float|透明度.
ax.patch.set_facecolor('green')

plt.show()
```

#### 7.1.8.2.set_facecolor()

设置画布的颜色.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.patch.set_alpha(0.1)
ax.patch.set_facecolor(color='green')  # str|颜色.

plt.show()
```

### 7.1.9.set_title()

设置标题.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.set_title('this is title')

plt.show()
```

### 7.1.10.set_xlabel()

设置x轴的内容.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.set_xlabel(xlabel='this is x label')  # str|文本内容.

plt.show()
```

### 7.1.11.set_xticks()

设置x轴的刻度.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.set_xticks(ticks=[1, 2, 3, 4])  # list(空列表就表示不显示刻度)|刻度.

plt.show()
```

### 7.1.12.set_yticks()

设置y轴的刻度.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.set_yticks(ticks=[1, 2, 3, 4])  # list(空列表就表示不显示刻度)|刻度.

plt.show()
```

###  7.1.13.spines

| 版本 | 描述                          | 注意 |
| ---- | ----------------------------- | ---- |
| -    | 画布的边框, 包括上下左右四个. | -    |

#### 7.1.13.1.set_color()

设置画布边框的颜色.

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.spines['left'].set_color(c='red')  # str|颜色.
ax.spines['right'].set_color(c='yellow')
ax.spines['top'].set_color(c='blue')
ax.spines['bottom'].set_color(c='green')

plt.show()
```

### 7.1.14.text()

给点添加文本.|`matplotlib.text.Text`

```python
import matplotlib.pyplot as plt

ax = plt.subplot()
ax.text(x=0.5,  # float|注释点的x坐标.
        y=0.5,  # float|注释点的y坐标.
        s='this is text')  # str|注释的文本内容.

plt.show()
```

## 7.2.pyplot

| 版本 | 描述                                                         | 注意 |
| ---- | ------------------------------------------------------------ | ---- |
| -    | pyplot是matplotlib的state-based接口, 主要用于简单的交互式绘图和程序化绘图. | -    |

### 7.2.1.axis()

设置坐标轴.

```python
import matplotlib.pyplot as plt

xmin, ymin = 0, 0
xmax, ymax = 10, 10
plt.axis([xmin, xmax, ymin, ymax])

plt.show()
```

### 7.2.2.barh()

绘制水平方向的条形图.

```python
import matplotlib.pyplot as plt

plt.barh(y=['No.1', 'No.2', 'No.3', 'No.4'],  # float或者array-like|条形图的y轴坐标.
         width=[1, -0.5, 2, 6],  # float或者array-like|每个数据的值.
         height=0.8)  # float(可选)|0.8|每个数据条的的宽度.

plt.show()
```

### 7.2.3.clabel()

在等高线上显示高度.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 9, 10)
y = np.linspace(1, 9, 10)
X, Y = np.meshgrid(x, y)
cs = plt.contour(X, Y, X + Y, colors='orange', linewidths=1)
plt.clabel(cs)  # matplotlib.contour.QuadContourSet|等高线标签.

plt.show()
```

### 7.2.4.colorbar()

显示色彩条.|`matplotlib.colorbar.Colorbar`

```python
import matplotlib.pyplot as plt

x = [[1], [2], [3], [4]]
plt.matshow(A=x)
plt.colorbar()

plt.show()
```

### 7.2.5.figure()

创建画布.|`matplotlib.figure.Figure`

```python
import matplotlib.pyplot as plt

figure = plt.figure(figsize=(5, 5))  # (float, float)(可选)|(6.4, 4.8)|画布的尺寸.
```

### 7.2.6.imread()

加载指定路径的图片.|`numpy.ndarray`

```python
import matplotlib.pyplot as plt

image = plt.imread(fname='./image.jpg')  # str or file-like|要加载的文件的路径.
```

### 7.2.7.imshow()

将图片数组在画布上显示.|`matplotlib.image.AxesImage`

```python
import matplotlib.pyplot as plt

image = plt.imread('./image.jpg')
plt.imshow(X=image,  # array-like or PIL image|希望显示的图像数据.
           cmap=None)  # str or matplotlib.colors.Colormap|None|色图.

plt.show()
```

### 7.2.8.matshow()

将矩阵绘制成图像.

```python
import matplotlib.pyplot as plt

mat = [[1], [2], [3], [4]]
plt.matshow(A=mat)  # array-like(M, N)|要绘制的矩阵.

plt.show()
```

### 7.2.9.pcolormesh()

使用非规则的矩形创建网格背景图.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 9, 10)
y = np.linspace(1, 9, 10)
X, Y = np.meshgrid(x, y)
plt.pcolormesh(X,  # array-like|横坐标.
               Y,  # array-like|纵坐标.
               X + Y,  # array-like(必须是2D的)|横纵坐标的关系公式.
               alpha=0.75,  # float|None|透明度.
               cmap='GnBu',  # str|None|色图.
               shading='nearest')  # {'flat', 'nearest', 'gouraud', 'auto'}(可选)|'flat'|阴影.

plt.show()
```

### 7.2.10.plot()

绘制函数图像.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 9, 10)
y = np.linspace(1, 9, 10)
plt.plot(x, y)  # 1D array-like|函数的变量.

plt.show()
```

### 7.2.11.Rectangle()

实例化矩阵.

```python
import matplotlib.pyplot as plt

axes = plt.subplot()
axes.add_patch(plt.Rectangle(xy=(0, 0),  # (float, float)|锚点.
                             width=0.5,  # float|矩行的宽度.
                             height=0.5))  # float|矩行的长度.

plt.show()
```

### 7.2.12.rcParams[]

实例化配置文件实例.

```python
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = 'Arial Unicode MS'  # 默认字体
```

### 7.2.13.savefig()

保存当前的画布.

```python
import matplotlib.pyplot as plt

plt.savefig(fname)  # str or path-like or file-like A path|要加载的文件的路径.
```

### 7.2.14.semilogx()

在x轴上绘制对数缩放的图.

```python
import matplotlib.pyplot as plt

plt.semilogx(range(1, 6), range(1, 6))

plt.show()
```

### 7.2.15.scatter()

绘制散点图.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10, num=10)
y = np.linspace(0, 5, num=10)
plt.scatter(x=x,  # float or array-like 形状必须是(n, )|x坐标.
            y=y,  # float or array-like 形状必须是(n, )|y坐标.
            s=150,  # float or array-like, 形状必须是(n, )(可选)|点的大小.
            c='blue',  # str|点的颜色.
            marker='o',  # str|点的标记的形状.
            edgecolors='green')  # str|标记的颜色.

plt.show()
```

### 7.2.16.show()

显示画布.

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(1, 9, 10)
y = np.linspace(1, 9, 10)
plt.plot(x, y)

plt.show()
```

### 7.2.17.subplot()

在当前画布上创建子图.|`matplotlib.axes._subplots.AxesSubplot`

```python
import matplotlib.pyplot as plt

axes = plt.subplot()
```

### 7.2.18.subplots()

同时创建画布和一组子图.|`matplotlib.figure.Figure`和`array of matplotlib.axes._subplots.AxesSubplot`

```python
import matplotlib.pyplot as plt

figure, axes = plt.subplots(nrows=4,  # int|1|子图数量的行数.
                            ncols=4,  # int|1|子图数量的列数.
                            figsize=(10, 10))  # tuple(list) of int|画布的大小.
```

### 7.2.19.subplots_adjust()

调整子图布局.

```python
import matplotlib.pyplot as plt

plt.subplots_adjust(left=0.125,  # float(可选)|None|子图左边框距离画布的距离.
                    bottom=0.9,  # float(可选)|None|子图底边框距离画布的距离.
                    right=0.1,  # float(可选)|None|子图右边框距离画布的距离.
                    top=0.9,  # float(可选)|None|子图顶边框距离画布的距离.
                    wspace=0.2,  # float(可选)|None|两张子图之间的左右间隔.
                    hspace=0.2)  # float(可选)|None|两张子图之间的上下间隔.
```

### 7.2.20.tight_layout()

自动调整子图布局.

```python
import matplotlib.pyplot as plt

plt.tight_layout()
```

### 7.2.21.title()

设置标题.

```python
import matplotlib.pyplot as plt

plt.title(label='this is title')  # str|文本内容.

plt.show()
```

### 7.2.22.xlabel()

设置x轴的内容.

```python
import matplotlib.pyplot as plt

plt.xlabel(xlabel='x')  # str|文本内容.

plt.show()
```

### 7.2.23.xlim()

设置x轴的显示范围.

```python
import matplotlib.pyplot as plt

plt.xlim([1, 2])  # [left, right]|[左界, 右界].

plt.show()
```

### 7.2.24.ylabel()

设置y轴的内容.

```python
import matplotlib.pyplot as plt

plt.ylabel(ylabel='y')  # str|文本内容.

plt.show()
```

# 8.numpy

| 版本   | 描述            | 注意                        | 适配M1 |
| ------ | --------------- | --------------------------- | ------ |
| 1.19.4 | Python数值计算. | 1. M1目前需要使用conda安装. | 是     |

## 8.1.abs()

逐元素计算绝对值.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 2, -1, 3]
x = np.abs(arr)  # array_like|输入的数据.
```

## 8.2.any()

判断数组是否存在某个元素为True, 如果有返回True, 否则返回False.|`numpy.bool_`

```python
import numpy as np

arr = [1, 0, 1, 1]
x = np.any(a=arr)  # array_like|输入的数据.
```

## 8.3.arange()

返回指定范围的整数数组.|`numpy.ndarray`

```python
import numpy as np

arr = np.arange(start=1,  # number(可选)|None|起始值.
                stop=10,  # number|结束值.
                step=2)  # number(可选)|None|步长.
```

## 8.4.argmax()

返回指定维度最大值的索引.|`numpy.int64`

```python
import numpy as np

arr = [1, 2, 3]
max_value = np.argmax(a=arr,  # array_like|输入的数据.
                			axis=None)  # int(可选)|None|筛选所沿的维度.
```

## 8.5.argsort()

对数组索引进行升序排序.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 3, 2, 5, 4]
arr = np.argsort(a=arr)  # array_like|输入的数据.
```

## 8.6.around()

逐元素进行四舍五入取整.|`numpy.ndarray`

```python
import numpy as np

arr = [1.4, 1.6]
x = np.around(a=arr)  # array_like|输入的数据.
```

## 8.7.asarray()

将输入转换为ndarray数组.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 2, 3]
nd_arr = np.asarray(a=arr,  # array_like|输入的数据.
                    dtype=None)  # data-type(可选)|None|元素的数据类型.
```

## 8.8.asmatrix()

将输入转换为矩阵.|`numpy.matrix`

```python
import numpy as np

arr = [1, 2, 3]
mat = np.asmatrix(data=arr)  # array_like|输入的数据.
```

## 8.9.ceil()

逐元素进行向上取整.|`numpy.ndarray`

```python
import numpy as np

arr = [5.1, 4.9]
x = np.ceil(arr)  # array_like|输入的数据.
```

## 8.10.clip()

逐元素裁切张量.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 2, 3]
new_arr = np.clip(a=arr,  # array_like|输入的数据.
                  a_min=2,  # scalar or array_like or None|最小值.
                  a_max=2)  # scalar or array_like or None|最大值.
```

## 8.11.concatenate()

按照指定维度合并多个数组.|`numpy.ndarray`

```python
import numpy as np

arr0 = [[1], [1], [1]]
arr1 = [[2], [2], [2]]
arr2 = [[3], [3], [3]]
x = np.concatenate([arr0, arr1, arr2],  # array_like|要合并的数组.
                   axis=1)  # int(可选)|0|沿指定维度合并.
```

## 8.12.cos()

逐元素计算余弦值.|`numpy.ndarray`

```python
import numpy as np

arr = [5.1, 4.9]
x = np.cos(arr)  # array_like|输入的数据.
```

## 8.13.c_[]

将第二个数组沿水平方向与第一个数组连接.|`numpy.ndarray`

```python
import numpy as np

arr0 = [[1, 2], [1, 2], [1, 2]]
arr1 = [[3], [3], [3]]
x = np.c_[arr0, arr1]
```

## 8.14.diag()

提取对角线的值, 或构建对角阵.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 2, 3]
x = np.diag(v=arr)  # array_like|输入的数据.
```

## 8.15.dot()

计算两个数组的点乘.|`numpy.ndarray`

```python
import numpy as np

arr0 = [[1, 2, 3]]
arr1 = [[1], [2], [3]]
x = np.dot(a=arr0,  # array_like|第一个元素.
           b=arr1)  # array_like|第二个元素.
```

## 8.16.equal()

逐元素判断元素值是否一致.|`numpy.ndarray`

```python
import numpy as np

arr1 = [1, 2, 3]
arr2 = [1, 2, 2]
x = np.equal(arr1, arr2)  # array_like|输入的数据.
```

## 8.17.exp()

逐元素计算e的幂次.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 2, 3]
x = np.exp(arr)  # array_like|输入的数据.
```

## 8.18.expm1()

逐元素计算e的幂次并减1.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 2, 3]
x = np.expm1(arr)  # array_like|输入的数据.
```

## 8.19.expand_dims()

增加数组的维度.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 2, 3]
x = np.expand_dims(a=arr,  # array_like|输入的数组.
                   axis=0)  # int or tuple of ints|添加新维度的位置.
```

## 8.20.eye()

生成单位阵.|`numpy.ndarray`

```python
import numpy as np

mat = np.eye(N=3)  # int|矩阵的行数.
```

## 8.21.hstack()

按照行合并数组.|`numpy.ndarray`

```python
import numpy as np

arr0 = [[1, 2], [1, 2]]
arr1 = [[3], [3]]
x = np.hstack(tup=[arr0, arr1])  # array-like|数组序列.
```

## 8.22.lexsort()

根据指定键(列)进行排序.|`numpy.ndarray`

```python
import numpy as np

arr = np.asarray([[1, 0.4],
                  [3, 0.2],
                  [2, 0.1],
                  [4, 0.3]])
indices = np.lexsort(keys=[arr[:, 0], ])

x = arr[indices]
```

## 8.23.linalg

| 版本 | 描述               | 注意 |
| ---- | ------------------ | ---- |
| -    | numpy的线性代数库. | -    |

### 8.23.1.inv()

获取矩阵的逆矩阵.|`numpy.ndarray`

```python
import numpy as np

mat = [[1, 2],
       [3, 4]]
x = np.linalg.inv(a=mat)  # array_like|输入的矩阵.
```

### 8.23.2.norm()

计算矩阵或向量范数.|`numpy.float64`

```python
import numpy as np

arr = [[1, 2],
       [3, 4]]
x = np.linalg.norm(x=arr,  # array_like|输入的数据.
                   ord=2)  # {non-zero int, inf, -inf, 'fro', 'nuc'}(可选)|None|范数选项.
```

### 8.23.3.svd()

奇异值分解.|`tuple of numpy.ndarray`

```python
import numpy as np

arr = [[1, 2],
       [3, 4]]
u, s, vh = np.linalg.svd(a=arr)  # array_like|输入的数据.
```

## 8.24.linspace()

返回指定间隔内的等差数列.|`numpy.ndarray`

```python
import numpy as np

x = np.linspace(start=1,  # array_like|起始值.
                stop=10,  # array_like|结束值.
                num=10)  # int(可选)|50|生成序列的样本的总数.
```

## 8.25.load()

从npy、npz或者序列化文件加载数组或序列化的对象.|`numpy.ndarray`

```python
import numpy as np

arr = np.load(file='./arr.npy',  # file-like object, string, or pathlib.Path|读取的文件路径.
              allow_pickle=True,  # bool(可选)|False|允许加载序列化的数组.
              encoding='ASCII')  # str(可选)|'ASCII'|解码方式.
```

## 8.26.log()

逐元素计算自然对数.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 2, 3]
x = np.log(arr)  # array_like|输入的数据.
```

## 8.27.log1p()

逐元素计算本身加1的自然对数.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 2, 3]
x = np.log1p(arr)  # array_like|输入的数据.
```

## 8.28.log2()

逐元素计算以2为底对数.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 2, 3]
x = np.log2(arr)  # array_like|输入的数据.
```

## 8.29.mat()

将输入转换为矩阵.|`numpy.matrix`

```python
import numpy as np

arr = [[1, 2, 3]]
mat = np.mat(data=arr,  # array_like|输入的数据.
             dtype=None)  # data-type|None|矩阵元素的数据类型.
```

## 8.30.matmul()

两个数组的矩阵乘积|`numpy.ndarray`

```python
import numpy as np

arr0 = [[1, 2, 3]]
arr1 = [[1], [2], [3]]
x = np.matmul(arr0,  # array_like|第一个元素.
              arr1)  # array_like|第二个元素.
```

## 8.31.max()

返回沿指定维度的最大值.|`numpy.float64`

```python
import numpy as np

arr = [1, 2, 5, 3, 4]
max_value = np.max(a=arr,  # array_like|输入的数据.
                   axis=None)  # int(可选)|None|所沿的维度.
```

## 8.32.maximum()

返回数组逐元素的最大值.|`numpy.ndarray`

```python
import numpy as np

arr0 = [2, 3, 4]
arr1 = [1, 5, 2]
x = np.maximum(arr0, arr1)  # array_like|输入的数据.
```

## 8.33.mean()

沿指定维度计算均值.|`numpy.float64`

```python
import numpy as np

arr = [1, 2, 3]
x = np.mean(a=arr,  # array_like|输入的数据.
            axis=None)  # int(可选)|None|所沿的维度.
```

## 8.34.meshgrid()

生成坐标矩阵.|`list of numpy.ndarray`

```python
import numpy as np

x_coord = np.linspace(0, 4, 5)
y_coord = np.linspace(0, 4, 5)
vec_mat = np.meshgrid(x_coord, y_coord)  # array_like|坐标向量.
```

## 8.35.nonzero()

返回非零元素索引.|`tuple_of_arrays`

```python
import numpy as np

arr = np.asarray([1, 2, 3, 4, 0, 0, 5])
x = np.nonzero(a=arr)  # array_like|输入的数据.
```

## 8.36.ones()

生成全一数组.|`numpy.ndarray`

```python
import numpy as np

x = np.ones(shape=[2, 3],  # int or sequence of ints|数组的形状.
            dtype=np.int8)  # data-type(可选)|numpy.float64|矩阵元素的数据类型.
```

## 8.37.power()

逐元素计算指定幂次.|`numpy.ndarray`

```python
import numpy as np

x = np.power([1, 2], [1, 3])   # array_like|底数和指数.
```

## 8.38.random

| 版本 | 描述                     | 注意 |
| ---- | ------------------------ | ---- |
| -    | numpy的随机数生成函数库. | -    |

### 8.38.1.choice()

从给定的1D数组中随机采样.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 2, 3]
num = np.random.choice(a=arr,  # 1-D array-like or int|输入的数组.
                       size=1)  # int or tuple of ints(可选)|None|采样结果形状.
```

### 8.38.2.multinomial()

从多项分布中抽取样本.|`numpy.ndarray`

```python
import numpy as np

x = np.random.multinomial(n=1,  # int|实验次数.
                          pvals=[1/2, 1/3, 1/6],  # sequence of floats|每个部分的概率(概率和为1).
                          size=1)  # int or tuple of ints(可选)|None|数组的形状.
```

### 8.38.3.normal()

生成正态分布样本.|`numpy.ndarray`

```python
import numpy as np

x = np.random.normal(size=[2, 3])  # int or tuple of ints(可选)|None|数组的形状.
```

### 8.38.4.permutation()

随机打乱序列.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 2, 3, 4]
x = np.random.permutation(arr)  # array_like|输入的数据.
```

### 8.38.5.rand()

生成随机数组.|`numpy.ndarray`

```python
import numpy as np

x = np.random.rand(2, 3)  # int(可选)|None|数组的形状.
```

### 8.38.6.randint()

返回指定区间[low, high)随机整数.|`int`

```python
import numpy as np

x = np.random.randint(low=1,  # int or array-like of ints|左边界.
                      high=10)  # int or array-like of ints(可选)|None|右边界.
```

### 8.38.7.randn()

生成正态分布随机数组.|`numpy.ndarray`

```python
import numpy as np

x = np.random.randn(2, 3)  # int(可选)|None|数组的形状.
```

### 8.38.8.RandomState()

实例化伪随机数生成器.|`numpy.random.mtrand.RandomState`

```python
import numpy as np

rs = np.random.RandomState(seed=2021)  # None|随机种子.
```

#### 8.38.8.1.shuffle()

打乱数据.|`numpy.ndarray`

```python
import numpy as np

rs = np.random.RandomState(seed=2021)
x = np.asarray([1, 2, 3, 4])
rs.shuffle(x)
```

### 8.38.9.seed()

设置随机种子.

```python
import numpy as np

np.random.seed(seed=2021)  # None|随机种子.
```

## 8.39.ravel()

展平数组.|`numpy.ndarray`

```python
import numpy as np

x = np.asarray([[1, 2], [3, 4]])
x = np.ravel(a=x,  # array_like|输入的数据.
             order='f')  # {'C','F', 'A', 'K'}(可选)|'C'|索引读取顺序.
```

## 8.40.reshape()

改变数组的形状.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 2, 3, 4]
x = np.reshape(a=arr,  # array_like|要改变形状的数组.
               newshape=[2, 2])  # int or tuple of ints|新的形状.
```

## 8.41.save()

将数组保存进二进制的npy文件.

```python
import numpy as np

arr = [1, 2, 3]
np.save(file='arr.npy',  # file, str, or pathlib.Path|文件保存的路径.
        arr=arr,  # array_like|要保存的数据.
        allow_pickle=True)  # |bool(可选)|True|允许使用序列化保存数组.
```

## 8.42.sort()

返回排序(升序)后的数组.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 3, 2, 4]
x = np.sort(a=arr)  # array_like|要排序的数组.
```

## 8.43.split()

拆分数组.|`list of ndarrays`

```python
import numpy as np

arr = np.asarray([[1, 2, 5, 6], [3, 4, 7, 8]])
arr_list = np.split(ary=arr,  # numpy.ndarray|要拆分的数组.
                    indices_or_sections=2,  # int or 1-D array|拆分方式.
                    axis=1)  # int(可选)|0|所沿的维度.
```

## 8.44.sqrt()

逐元素计算平方根.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 2, 3]
x = np.sqrt(arr)  # array_like|输入的数据.
```

## 8.45.squeeze()

删除维度为一的维度.|`numpy.ndarray`

```python
import numpy as np

arr = [[1, 2, 3]]
x = np.squeeze(arr)  # array_like|输入的数据.
```

## 8.46.std()

沿指定维度计算标准差.|`numpy.float64`

```python
import numpy as np

arr = [1, 2, 5, 3, 4]
std_value = np.std(a=arr,  # array_like|输入的数据.
                   axis=None)  # None or int or tuple of ints(可选)|None|所沿的维度.
```

## 8.47.sum()

沿指定维度求和.|`numpy.ndarray`

```python
import numpy as np

arr = [1, 2, 5, 3, 4]
sum_value = np.sum(a=arr,  # array_like|输入的数据.
                   axis=None)  # None or int or tuple of ints(可选)|None|所沿的维度.
```

## 8.48.transpose()

转置数组.|`numpy.ndarray`

```python
import numpy as np

arr = np.asarray([[1, 2], [3, 4]])
# 方法一
x0 = np.transpose(a=arr,  # 输入的数组|array-like
                  axes=None)  # tuple or list of ints(可选)|None|轴的排列顺序.

# 方法二
x1 = arr.T
```

## 8.49.var()

沿指定维度计算方差.|`numpy.float64`

```python
import numpy as np

arr = [1, 2, 5, 3, 4]
var_value = np.var(a=arr,  # array_like|输入的数据.
                   axis=None)  # None or int or tuple of ints(可选)|None|所沿的维度.
```

## 8.50.void()

实例化`numpy.void`对象.

```python
import numpy as np

x = np.void(b'abc')  # bytes|输入的数据.
```

## 8.51.vstack()

按照列合并数组.|`numpy.ndarray`

```python
import numpy as np

arr0 = [[1, 2]]
arr1 = [[3, 4]]
x = np.vstack(tup=[arr0, arr1])  # array-like|数组序列.
```

## 8.52.where()

根据判断条件, 真值返回`x`, 假值返回`y`.|`numpy.ndarray`

```python
import numpy as np

a = 1
b = 2
arr = np.where(a > b,  # array_like, bool|判断条件.
               True,  # array_like|None|情况为真的返回值.
               False)  # array_like|None|情况为假的返回值.
```

## 8.53.zeros()

生成全零数组.|`numpy.ndarray`

```python
import numpy as np

x = np.zeros(shape=[2, 3],  # int or sequence of ints|数组的形状.
             dtype=np.int8)  # data-type(可选)|numpy.float64|矩阵元素的数据类型.
```

# 9.pandas

| 版本  | 描述                  | 注意                        | 适配M1 |
| ----- | --------------------- | --------------------------- | ------ |
| 1.1.4 | 结构化数据分析软件库. | 1. M1目前需要使用conda安装. | 是     |

## 9.1.concat()

沿特定轴连接pandas对象.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

sr0 = pd.Series([1, 4, 7])
sr1 = pd.Series([2, 5, 8])
sr2 = pd.Series([3, 6, 9])
df = pd.concat([sr0, sr1, sr2],  # Series or DataFrame|要连接的pandas对象.
               axis=1)  # {0/'index', 1/'columns'}|0|所沿的维度.
```

## 9.2.DataFrame()

实例化DataFrame对象.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

df_map = {'key': [0, 1, 2], 'values': [0.1, 0.5, 1.0]}
df = pd.DataFrame(data=df_map,  # ndarray (structured or homogeneous), Iterable, dict, or DataFrame|输入的数据.
                  index=['一', '二', '三'],  # Index or array-like|None(0, 1, 2, ..., n)|索引名.
                  columns=None)  # Index or array-like|None(0, 1, 2, ..., n)|列名.
```

### 9.2.1.columns

DataFrame的列标签.|`pandas.core.indexes.base.Index`

```python
import pandas as pd

df_map = {'key': [0, 1, 2], 'values': [0.1, 0.5, 1.0]}
df = pd.DataFrame(df_map)

print(df.columns)
```

### 9.2.2.convert_dtypes()

将数据自动转换成最佳数据类型.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

df_map = {'key': [0, 1, 2], 'values': [0.1, 0.5, 1.0]}
df = pd.DataFrame(df_map)
df = df.convert_dtypes()
```

### 9.2.3.corr()

计算列成对相关度.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

df_map = {'key': [0, 1, 2], 'values': [0.1, 0.5, 1.0]}
df = pd.DataFrame(df_map)
correlation_value = df.corr()
```

### 9.2.4.drop()

根据指定的标签删除行或者列.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

df_map = {'key': [0, 1, 2], 'values': [0.1, 0.5, 1.0]}
df = pd.DataFrame(df_map)
new_df = df.drop(labels=1,  # single label or list-like|None|要删除的行或者列的标签.
             		 axis=0)  # {0/'index', 1/'columns'}|0|所沿的维度.
```

### 9.2.5.drop_duplicates()

返回删除重复行的DataFrame.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

df_map = {'key': [0, 1, 1, 2], 'values': [0.1, 0.5, 0.5, 1.0]}
df = pd.DataFrame(df_map)
new_df = df.drop_duplicates(subset=None,  # column label or sequence of labels(可选)|None|仅选子列进行删除.
                        		keep='first',  # {'first', 'last', False}|'first'|保留重复项的位置.
                        		inplace=False)  # bool|False|是否修改源DataFrame.
```

### 9.2.6.fillna()

填充缺失值.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

df_map = {'key': [0, 1, 2, 3], 'values': [0.1, 0.5, None, 1.0]}
df = pd.DataFrame(df_map)
new_df = df.fillna(value=10,  # scalar, dict, Series, or DataFrame|填充缺失的值.
               		 inplace=False)  # bool|False|是否修改源DataFrame.
```

### 9.2.7.head()

返回前n行数据.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

df_map = {'key': [0, 1, 2, 3], 'values': [0.1, 0.5, None, 1.0]}
df = pd.DataFrame(df_map)
head_value = df.head(n=1)  # int|5|行数.
```

### 9.2.8.iloc[]

按照行号取出数据.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

df_map = {'key': ['a', 'b', 'c'], 'values': [1, 2, 3]}
df = pd.DataFrame(df_map)
new_df = df.iloc[0:2]
```

### 9.2.9.info()

在终端打印摘要信息.

```python
import pandas as pd

df_map = {'key': ['a', 'b', 'c'], 'values': [1, 2, 3]}
df = pd.DataFrame(df_map)
df.info()
```

### 9.2.10.loc[]

按照行名称取出数据.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

df_map = {'key': ['a', 'b', 'c'], 'values': [1, 2, 3]}
df = pd.DataFrame(df_map, index=df_map['key'])
new_df = df.loc['a':'b']
```

### 9.2.11.map()

根据输入的对应关系映射Series.|`pandas.core.series.Series`

```python
import pandas as pd

df = pd.DataFrame({'key': ['a', 'b', 'c'], 'values': [1, 2, 3]})
map_dict = {'a': 3, 'b': 2, 'c': 1}
sr = df['key'].map(map_dict)
```

### 9.2.12.median()

获取DataFrame的中位数.|`pandas.core.series.Series`

```python
import pandas as pd

df_map = {'key': ['a', 'b', 'c'], 'values': [1, 2, 3]}
df = pd.DataFrame(df_map)
median_value = df.median()
```

### 9.2.13.replace()

替换DataFrame中的值.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

df = pd.DataFrame({'key': ['a', 'b', 'c'], 'values': [1, 2, 3]})
new_df = df.replace(to_replace=2,  # str, regex, list, dict, Series, int, float, or None|None|被替换的值.
                    value=5,  # scalar, dict, list, str, regex|None|替换的值.
                    inplace=False)  # bool|False|是否修改源DataFrame.
```

### 9.2.14.reset_index()

重置DataFrame中的索引.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

df = pd.DataFrame({'key': ['a', 'b', 'c'], 'values': [1, 2, 3]}, index=[1, 2, 3])
new_df = df.reset_index(drop=True,  # bool|False|是否丢弃原来的索引.
                        inplace=False)  # bool|False|是否修改源DataFrame.
```

### 9.2.15.sample()

返回随机采样的DataFrame样本.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

df = pd.DataFrame({'key': ['a', 'b', 'c', 'd'], 'values': [1, 2, 3, 4]})
new_df = df.sample(n=None,  # int(可选)|None|采样数.
                   frac=0.75)  # float(可选)|None|采样的比例.
```

### 9.2.16.select_dtypes()

返回指定元素类型的列组成的新DataFrame.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

df = pd.DataFrame({'key': ['a', 'b', 'c', 'd'], 'values': [1, 2, 3, 4]})
df = df.select_dtypes(include='int')  # scalar or list-like|None|指定的数据类型.
```

### 9.2.17.to_csv()

写入csv文件.

```python
import pandas as pd

df = pd.DataFrame({'key': [0, 1, 2], 'values': [0.1, 0.5, 1.0]})
df.to_csv(path_or_buf='./table.csv',  # str or file handle|None|写入的文件路径.
          sep=',',  # str|','|使用的分隔符.
          header=True,  # bool or list of str|True|列名.
          index=True,  # bool|True|行名.
          encoding=None)  # str(可选)|'utf-8'|编码方式.
```

### 9.2.18.values

返回DataFrame的值数据.|`numpy.ndarray`

```python
import pandas as pd

df = pd.DataFrame(data={'key': [0, 1, 2], 'values': [0.1, 0.5, 1.0]})
values = df.values
```

## 9.3.date_sample()

生成固定频率的时间索引.|`pandas.core.indexes.datetimes.DatetimeIndex`

```python
import pandas as pd

datetime_index = pd.date_range(start='2021/05/18',  # str or datetime-like(可选)|生成时间的开始界线.
                               periods=5,  # int(可选)|生成的周期.
                               freq='M')  # str or DateOffset|'D'|生成的频率.
```

## 9.4.get_dummies()

将类别变量转换成Dummy编码的变量.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

sr = pd.Series(['a', 'b', 'c', 'a'])
coding = pd.get_dummies(data=sr)  # array-like, Series, or DataFrame|输入的数据.
```

## 9.5.groupby()

按照列对DataFrame进行分组.|`pandas.core.groupby.generic.DataFrameGroupBy`

```python
import pandas as pd

df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['c1', 'c2'])
group = df.groupby(by='c2')
```

## 9.6.isnull()

检测逐个元素是否是缺失值.|`pandas.core.series.Series`

```python
import pandas as pd

sr = pd.Series([1, 2, None])
value = pd.isnull(sr)
```

## 9.7.merge()

将两个DataFrame按照列键值进行合并.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

df0 = pd.DataFrame({'key': ['a', 'b', 'c'], 'values': [1, 2, 3]})
df1 = pd.DataFrame({'values': [1, 2, 3], 'tag': [0, 0, 1]})
df = pd.merge(left=df0,  # DataFrame|要合并的DataFrame.
              right=df1,  # DataFrame|要合并的DataFrame.
              how='inner',  # {'left', 'right', 'outer', 'inner'}|'inner'|合并的方式.
              left_on='values',  # label or list, or array-like|None|左侧参考项.
              right_on='values',  # label or list, or array-like|None|右侧参考项.
              sort=True)  # bool|False|是否进行排序.
```

## 9.8.notnull()

检测逐个元素是否是非缺失值.|`pandas.core.series.Series`

```python
import pandas as pd

sr = pd.Series([1, 2, None])
value = pd.notnull(sr)
```

## 9.9.read_csv()

读取csv文件.|`pandas.core.frame.DataFrame`

```python
import pandas as pd

df = pd.read_csv(filepath_or_buffer='./table.csv',  # str, path object or file-like object|读取的文件路径.
                 sep=',',  # str|','|使用的分隔符.
                 header=0,  # int, list of int|'infer'|列名.
                 index_col=None,  # int, str, sequence of int / str, or False|None|行名.
                 encoding=None)  # str(可选)|None|编码方式.
```

## 9.10.Series()

实例化Series对象.|`pandas.core.series.Series`

```python
import pandas as pd

sr = pd.Series(data=[0, 1, 2],  # array-like, Iterable, dict, or scalar value|输入的数据.
               index=[1, 2, 3])  # array-like or Index (1d)|None|索引名.
```

### 9.10.1.dt

#### 9.10.1.1day

提取时间中的日期信息.|`pandas.core.series.Series`

```python
import pandas as pd

sr = pd.Series(['2016/05/20', '2021/05/20'])
sr = pd.to_datetime(sr)
day = sr.dt.day
```

#### 9.10.1.2.dayofweek

提取时间中的周几信息.|`pandas.core.series.Series`

```python
import pandas as pd

sr = pd.Series(['2016/05/20', '2021/05/20'])
sr = pd.to_datetime(sr)
dayofweek = sr.dt.dayofweek
```

#### 9.10.1.3.hour

提取时间中的小时信息.|`pandas.core.series.Series`

```python
import pandas as pd

sr = pd.Series(['2016/05/20 8:55:56', '2021/05/20 20:55:56'])
sr = pd.to_datetime(sr)
hour = sr.dt.hour
```

#### 9.10.1.4.month

提取时间中的月份信息.|`pandas.core.series.Series`

```python
import pandas as pd

sr = pd.Series(['2016/05/20', '2021/05/20'])
sr = pd.to_datetime(sr)
month = sr.dt.month
```

#### 9.10.1.5.weekday

提取时间中的周几信息.|`pandas.core.series.Series`

```python
import pandas as pd

sr = pd.Series(['2016/05/20', '2021/05/20'])
sr = pd.to_datetime(sr)
weekday = sr.dt.weekday
```

### 9.10.2.isin()

逐元素判断Series是否包含指定值.|`pandas.core.series.Series`

```python
import pandas as pd

sr = pd.Series([0, 1, 2])
sr_isin = sr.isin(values=[2])  # set or list-like|指定值. 
```

### 9.10.3.mode()

返回数据集的众数.|`pandas.core.series.Series`

```python
import pandas as pd

sr = pd.Series([0, 1, 2, 2])
value = sr.mode()
```

### 9.10.4.plot()

绘制图像.

```python
import matplotlib.pyplot as plt
import pandas as pd

sr = pd.Series([0, 1, 2])
sr.plot()
```

### 9.10.5.sort_index()

通过索引值为Series排序.|`pandas.core.series.Series`

```python
import pandas as pd

sr = pd.Series([0, 1, 2], index=[3, 2, 1])
sr = sr.sort_index()
```

### 9.10.6.tolist()

返回值列表.|`list`

```python
import pandas as pd

sr = pd.Series([0, 1, 2], index=[3, 2, 1])
value = sr.tolist()
```

## 9.11.to_datetime()

将数据转换为日期类型.|`pandas.core.series.Series`

```python
import pandas as pd

sr = pd.Series(['2016/05/20', '2021/05/20'])
sr = pd.to_datetime(arg=sr)  # int, float, str, datetime, list, tuple, 1-d array, Series, DataFrame/dict-like|输入数据.
```

## 9.12.unique()

返回唯一值组成的数组.|`numpy.ndarray`

```python
import pandas as pd

sr = pd.Series([0, 1, 1, 1, 2])
arr = pd.unique(values=sr)  # 1d array-like|输入的数据.
```

## 9.13.value_count()

计算非空值出现的次数.|`pandas.core.series.Series`

```python
import pandas as pd

sr = pd.Series([1, 2, 2, 3, None])
values = pd.value_counts(values=sr)  # ndarray (1-d)|输入的数据.
```

# 10.PIL

| 版本  | 描述               | 注意                            | 适配M1 |
| ----- | ------------------ | ------------------------------- | ------ |
| 8.4.0 | Python 图像处理库. | 1. 安装时使用pip install pillow | 是     |

## 10.1.Image

| 版本 | 描述             | 注意 |
| ---- | ---------------- | ---- |
| -    | PIL图像类装饰器. | -    |

### 10.1.1.convert()

转换图像的色彩空间.|`PIL.Image.Image`

```python
from PIL.Image import open

im = open('img.jpeg')
im = im.convert(mode='CMYK')  # {'L', 'RGB', 'RGBA', 'CMYK'}|None|模式.
```

### 10.1.2.fromarray()

将ndarray转换为图像.|`PIL.Image.Image`

```python
import numpy as np
from PIL.Image import fromarray

arr = np.asarray([[0.1, 0.2], [0.3, 0.4]])
im = fromarray(obj=arr)  # numpy.ndarray|输入的数组.
```

### 10.1.3.open()

打开图像文件.|`PIL.Image.Image`

```python
from PIL.Image import open

im = open(fp='img.jpeg')  # A filename (string), pathlib.Path object or a file object|加载的图像路径.
```

### 10.1.4.resize()

调整图像的大小.|`PIL.Image.Image`

```python
from PIL.Image import open

im = open('img.jpeg')
new_im = im.resize(size=[100, 100])
```

### 10.1.5.save()

保存图像文件.

```python
from PIL.Image import open

im = open('img.jpeg')
im.save(fp='new_image.jpg')  # A filename (string), pathlib.Path object or a file object|保存图像路径.
```

## 10.2.ImageOps

| 版本 | 描述             | 注意 |
| ---- | ---------------- | ---- |
| -    | PIL标准图像操作. | -    |

### 10.2.1.autocontrast

最大化(标准化)图像的对比度.|`PIL.Image.Image`

```python
from PIL.Image import open
from PIL.ImageOps import autocontrast

im = open('img.jpeg')
processed_im = autocontrast(image=im)  # PIL.Image.Image|输入的图像.
```

# 11.pybind11

| 版本  | 描述                          | 注意                                                         | 适配M1 |
| ----- | ----------------------------- | ------------------------------------------------------------ | ------ |
| 2.6.2 | C++11 和 Python 混编操作接口. | 1. 在 Linux 下需要使用 python3-dev                                                                                                                       2. pybind11 的 Python 软件包能自动安装对应的C++库文件.                                                                      3. 需要同时安装 pybind11-global 才能让 cmake 正确的在虚拟环境中找到 pybind11. | 是     |

## 11.1.第一个例子

1. example.cc代码

```c++
#include "pybind11/pybind11.h"

int add(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(example, m) {  // example为Python模块名.
    m.doc() = "这是一个测试模块.";  // __doc__ Python模块的说明信息

    m.def("add", &add,  // Python侧的函数名和对应的C++函数引用.
          R"pbdoc(整数加法函数.)pbdoc",  // add.__doc__ 函数的说明信息.
          pybind11::arg("a"),  // Python侧函数的参数名称.
          pybind11::arg("b"));

    m.attr("__version__") = "0.1a0";  // 设置其他模块内变量.
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

或者, 使用CMakeLists.txt编译(手动`cmake ..`, 以clion为例IDE找不到正确的环境).

```cmake
cmake_minimum_required(VERSION 3.19)
project(example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

# 设置O3级别优化.
set(CMAKE_CXX_FLAGS "-O3")

# 配置pybind11
find_package(pybind11)

# 设置编译动态库路径.
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

pybind11_add_module(example SHARED example.cc)
```

3. test.py进行测试.

```python
import example

ans = example.add(a=1, b=1)
```

## 11.2.类与继承

### 11.2.1.实现类

1. example.cc代码

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
        std::string name;  // 私用变量不能转换到Python侧, 只能设置def_readonly的访问权限.
};

PYBIND11_MODULE(example, m) {
    pybind11::class_<Animal>(m, "Animal")
        .def(pybind11::init())
        .def("call", &Animal::call)
        .def_readonly("name", &Animal::name);
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import example

animal = example.Animal()
print(animal.name)
animal.call()
```

### 11.2.2.实现继承

1. example.cc代码

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

        explicit Dog(std::string name) {
            this->name = std::move(name);
        }

        void call() {
            std::cout << "Woof!" << std::endl;
        }
};

PYBIND11_MODULE(example, m) {
    pybind11::class_<Animal>(m, "Animal")
        .def(pybind11::init())
        .def("call", &Animal::call)
        .def_readonly("name", &Animal::name);

    pybind11::class_<Cat, Animal>(m, "Cat")
        .def(pybind11::init())
        .def(pybind11::init<std::string>())  // pybind11不能自动重载, 需要显式声明.
        .def("call", &Cat::call)
        .def_readonly("name", &Cat::name);

    pybind11::class_<Dog>(m, "Dog")  // Python侧和C++侧的继承关系并不一致, 不显示声明将继承object.
        .def(pybind11::init())
        .def(pybind11::init<std::string>())
        .def("call", &Dog::call)
        .def_readonly("name", &Dog::name);
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import example

cat = example.Cat("Garfield")
print(cat.name)
cat.call()
print(isinstance(cat, example.Animal))

dog = example.Dog()
dog.call()
print(isinstance(dog, example.Animal))
```

### 11.2.3.允许Python侧动态获取新属性

1. example.cc代码

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
    std::string name;  // 私有变量不能转换到Python侧.
};

PYBIND11_MODULE(example, m) {
    pybind11::class_<Animal>(m, "Animal", pybind11::dynamic_attr())  // 允许动态获取新属性.
        .def(pybind11::init())
        .def("call", &Animal::call)
        .def_readonly("name", &Animal::name);
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import example

animal = example.Animal()
animal.age = 3
```

### 11.2.4.属性的访问权限

1. example.cc代码

```c++
#include <iostream>

#include "pybind11/pybind11.h"

class Animal {
    public:
        Animal(std::string name, int age) {
            this->name = std::move(name);
            this->age = age;
        }

        void call() {
            std::cout << "Ah!" << std::endl;
        }

    public:
        std::string name;
        int age;
};

PYBIND11_MODULE(example, m) {
    pybind11::class_<Animal>(m, "Animal")
        .def(pybind11::init<std::string, int>())
        .def("call", &Animal::call)
        .def_readwrite("name", &Animal::name)  // 设置属性权限可读可写.
        .def_readonly("age", &Animal::age);  // 设置属性权限只读.
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import example

animal = example.Animal("Garfield", 3)
print("name: %s, age: %d." % (animal.name, animal.age))

animal.name = "Cat"
try:
    animal.age = 4
except AttributeError:
    print("年龄是只读参数, 不可修改")
print("name: %s, age: %d." % (animal.name, animal.age))
```

### 11.2.5.在C++侧使用Python对象的属性

1. example.cc代码

```c++
#include <iostream>

#include "pybind11/pybind11.h"

void get_person_name(const pybind11::object &person) {
    std::string name = getattr(person, "name").cast<std::string>();
    std::cout << "Person's name:" << name << std::endl;
}

PYBIND11_MODULE(example, m) {
    m.def("get_person_name", &get_person_name);
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import example


class Person(object):
    def __init__(self, money, name):
        super(Person, self).__init__()

        self.money = money
        self.name = name


example.get_person_name(Person(money=10, name='Tom'))
```

## 11.3.异常处理

### 11.3.1.实现自定义异常

0. pybind11只提供了有限的可被Python解释器捕获的异常, 因此只有手动注册没有提供的异常.
1. example.cc代码

```c++
#include "pybind11/pybind11.h"

class CustomException: public std::exception {
    public:
        const char * what() const noexcept override {
            return "自定义异常.";
        }
};

void test() {
    throw CustomException();
}

PYBIND11_MODULE(example, m) {
	  // 在pybind11中注册自定义异常(将自定义异常继承一个Python的异常).
    pybind11::register_exception<CustomException>(m, "PyCustomException", PyExc_BaseException); 
    m.def("test", &test);
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import example

try:
    example.test()
except BaseException:
    print('成功捕获异常.')
```

## 11.4.设置参数相关

### 11.4.1.常规情况

1. example.cc代码

```c++
#include "pybind11/pybind11.h"

int add(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(example, m) {
    m.def("add", &add, pybind11::arg("a")=1, pybind11::arg("b")=1);  // 直接在参数上添加默认信息.
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import example

ans = example.add()
```

### 11.4.2.默认参数为None

1. example.cc代码

```c++
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

int add(int a, std::optional<int> b) {
    if (!b.has_value()) {
        return a;
    } else {
        return a + b.value();
    }
}

PYBIND11_MODULE(example, m) {
    m.def("add", &add,
          pybind11::arg("a")=1,
          pybind11::arg("b")=pybind11::none());
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++17 -undefined dynamic_lookup \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import example

ans0 = example.add(1)
ans1 = example.add(1, 2)
```

## 11.5. Python的语法糖

### 11.5.1.在Python侧使用alias

1. example.cc代码

```c++
#include "pybind11/pybind11.h"

void my_print(const std::string& text) {
    pybind11::print(text);
}

PYBIND11_MODULE(example, m) {
    m.def("my_print", &my_print);
    
    m.attr("m_print") = m.attr("my_print");
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import example

example.my_print("Hello world")
```

### 11.5.2.接受任意数量的参数

1. example.cc代码

```c++
#include "pybind11/pybind11.h"

int add(const pybind11::args &args, const pybind11::kwargs &kwargs) {
    int sum = 0;

    for (auto item: args) {
        sum += pybind11::cast<int>(item);
    }
    for (auto item: kwargs) {
        sum += pybind11::cast<int>(item.second);
    }

    return sum;
}

PYBIND11_MODULE(example, m) {
    m.def("add", &add);  // 不要在此注册参数列表.
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import example

ans = example.add(1, 2, a=3, b=4)
```

### 11.5.3.使用Python侧的print函数

1. example.cc代码

```c++
#include "pybind11/pybind11.h"

void my_print(const std::string& text) {
    pybind11::print(text);
}

PYBIND11_MODULE(example, m) {
    m.def("my_print", &my_print);
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import example

example.my_print("Hello world")
```

## 11.6.绑定NumPy

### 11.6.1.直接访问

1. example.cc代码

```c++
#include <iostream>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

void range(const pybind11::array &array) {
    if (array.ndim() == 2 and array.dtype().kind() == 'i') {  // 判断是否是一个2D数组和元素的数据类型.
        for (int i = 0; i < array.shape(0); i ++) {
            for (int j = 0; j < array.shape(1); j ++) {
                std::cout << *(int*)array.data(i, j) << " ";
            }
            std::cout << std::endl;
        }
    }
}

PYBIND11_MODULE(example, m) {
    m.def("range", &range, pybind11::arg("array"));
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import numpy as np

import example

example.range(np.asarray([[1, 2], [3, 4]]))
```

### 11.6.2.提供的方法

1. example.cc代码

```c++
#include <iostream>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

void show_methods(pybind11::array &array) {
    std::cout << "ndim:" << array.ndim() << std::endl;  // 数组的维度.
    std::cout << "data:" << array.data() << std::endl;  // 数组索引处的指针.
    std::cout << "mutable_data:" << array.mutable_data() << std::endl;  // 数组索引处的可变指针.

    std::cout << "shape:"; // 数组每个维度的大小.
    auto shape = array.request().shape;
    for (long i : shape) {
        std::cout << " " << i;
    }
    std::cout << std::endl;

    std::cout << "size:" << array.size() << std::endl;  // 数组元素的总数.
    std::cout << "dtype:" << (std::string)pybind11::str(array.dtype()) << std::endl;  // 数组元素的数据类型.
    std::cout << "kind:" << array.dtype().kind() << std::endl;  // 数组元素的数据类型字符代码('biufcmMOSUV').
}

PYBIND11_MODULE(example, m) {
    m.def("show_methods", &show_methods, pybind11::arg("array"));
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import numpy as np

import example

example.show_methods(np.asarray([[1, 2], [3, 4]]))
```

## 11.7.绑定CUDA

### 11.7.1.调用GPU计算

1. example.cu代码

```c++
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

__global__ void kernel(const double scalar, double *vector, const int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        vector[i] *= scalar;
    }
}

void run_kernel(const double scalar, double *vector, const int n) {
    dim3 threadsPerBlock(256, 1, 1);
    dim3 numberOfBlock(32, 1, 1);

    kernel<<<numberOfBlock, threadsPerBlock>>>(scalar, vector, n);
}

void multiply_with_scalar(const double &scalar, pybind11::array &array) {
    double *gpu_array;
    int length = (int)array.request().shape[0];
    size_t size = length * sizeof(double);

    cudaMalloc(&gpu_array, size);

    cudaMemcpy(gpu_array, array.data(), size, cudaMemcpyHostToDevice);  // 将CPU上的数据拷贝到GPU上.
    run_kernel(scalar, gpu_array, length);  // 调用核函数.
    cudaMemcpy(array.mutable_data(), gpu_array, size, cudaMemcpyDeviceToHost);  // 将GPU上计算后的数据拷贝回CPU.

    cudaFree(gpu_array);
}

PYBIND11_MODULE(example, m) {
    m.def("multiply_with_scalar", &multiply_with_scalar,
          pybind11::arg("scalar"), pybind11::arg("array"));
}
```

2. 使用nvcc编译, 并产生对应的动态链接库.

```shell
nvcc -O3 -shared -Xcompiler -fPIC -std=c++11 \
 `python3 -m pybind11 --includes` \
 example.cu -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import numpy as np

import example

arr = np.asarray([1., 2., 3., 4., 5.])
example.multiply_with_scalar(3, arr)
print(arr)
```

### 11.7.2.调用cuBLAS

1. example.cu代码

```c++
#include "cublas_v2.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

void kernel(pybind11::array_t<float> &a,
            pybind11::array_t<float> &b,
            pybind11::array_t<float> &c) {
    cublasStatus_t status; // 创建cuBLAS状态信息.
    cublasHandle_t handle; // 创建cuBLAS句柄.
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS句柄创建失败.");
    }

    float *gpu_a, *gpu_b, *gpu_c;
    int m = (int)a.request().shape[0];
    int k = (int)a.request().shape[1];
    int n = (int)b.request().shape[1];
    size_t a_size = m * k * sizeof(float);
    size_t b_size = k * n * sizeof(float);
    size_t c_size = m * n * sizeof(float);

    // 分配显存.
    cudaMalloc(&gpu_a, a_size);
    cudaMalloc(&gpu_b, b_size);
    cudaMalloc(&gpu_c, c_size);

    // 将CPU上数据拷贝到GPU上.
    cudaMemcpy(gpu_a, a.data(), a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b, b.data(), b_size, cudaMemcpyHostToDevice);

    // 使用cuBLAS的矩阵乘法函数.
    float alpha = 1.0, beta = 0.0;
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m,
                k,
                &alpha,
                gpu_b, n,
                gpu_a, k,
                &beta,
                gpu_c, n);

    // 将GPU上计算后的数据拷贝回CPU.
    cudaMemcpy(c.mutable_data(), gpu_c, c_size, cudaMemcpyDeviceToHost);

    // 释放显存.
    cudaFree(gpu_a);
    cudaFree(gpu_b);
    cudaFree(gpu_c);

    // 销毁cuBLAS句柄.
    cublasDestroy(handle);
}

pybind11::array matmul_on_gpu(pybind11::array_t<float> &a,
                              pybind11::array_t<float> &b) {
    // 初始化全零矩阵.
    auto c = pybind11::array_t<float>(pybind11::array::ShapeContainer({a.request().shape[0],
                                                                       b.request().shape[1]}));

    // 调用计算函数.
    kernel(a, b, c);

    return c;
}

PYBIND11_MODULE(example, m) {
    m.def("matmul_on_gpu", &matmul_on_gpu,
          pybind11::arg("a"), pybind11::arg("b"));
}
```

2. 使用nvcc编译, 并产生对应的动态链接库.

```shell
nvcc -O3 -lcublas -shared -Xcompiler -fPIC -std=c++11 \
 `python3 -m pybind11 --includes` \
 example.cu -o example`python3-config --extension-suffix`
```

或者, 使用CMakeLists.txt编译.

```cmake
cmake_minimum_required(VERSION 3.17)
project(example LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

# 设置O3级别优化.
set(CMAKE_CUDA_FLAGS "-O3")

# 配置pybind11.
find_package(pybind11 REQUIRED)

# 设置编译动态库路径.
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

pybind11_add_module(example SHARED example.cu)

# 链接cuBLAS库.
target_link_libraries(example PRIVATE -lcublas)

# 可分离汇编.
set_target_properties(example PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
```

3. test.py进行测试.

```python
import numpy as np

import example

mat_a = np.asarray([[1, 2, 3],
                    [4, 5, 6]], dtype=np.float32)
mat_b = np.asarray([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12]], dtype=np.float32)

mat_c = example.matmul_on_gpu(mat_a, mat_b)
print(mat_c)
```

## 11.8.其他

### 11.8.1.绑定Eigen

1. example.cc代码

```c++
#include "pybind11/eigen.h"  // pybind11的eigen转换头文件.
#include "pybind11/pybind11.h"
#include "Eigen/Dense"

Eigen::MatrixXd transpose(const Eigen::MatrixXd &mat) {
    return mat.transpose();
}

PYBIND11_MODULE(example, m) {
    m.def("transpose", &transpose, pybind11::arg("mat"));
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup \
 -I /opt/homebrew/Cellar/eigen/3.3.9/include/eigen3 \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import example

trans_mat = example.transpose([[1, 2], [3, 4]])
```

### 11.8.2.实现重载

1. example.cc代码

```c++
#include "pybind11/pybind11.h"

int add(int a, int b) {
    return a + b;
}

double add(double a, double b) {
    return a + b;
}

PYBIND11_MODULE(example, m) {
    m.def("add", pybind11::overload_cast<int, int>(&add));  // pybind11的重载接口.
    m.def("add", pybind11::overload_cast<double, double>(&add));
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++14 -undefined dynamic_lookup \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import example

int_ans = example.add(1, 1)
float_ans = example.add(0.9, 1.1)
```

### 11.8.3.区分32位和64位的数据类型

0. 受限于 Python 3.x 没有区分32位和64位的类型, 这里需要依赖于NumPy, 同时pybind11没有对于NumPy标量处理的模板, 因此需要借助`pybind11::buffer`实现.

1. example.cc代码

```c++
#include "pybind11/eigen.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

typedef std::int32_t int32;
typedef std::int64_t int64;

std::variant<Eigen::MatrixXf, Eigen::MatrixXd> create_matrix(const pybind11::buffer &rows,
                                                             const pybind11::buffer &cols) {
    auto type_code = rows.request().format;

    if (type_code == "i") {
        Eigen::MatrixXf matrix = Eigen::MatrixXf::Zero(pybind11::cast<int32>(rows), pybind11::cast<int32>(cols));

        return matrix;
    } else if (type_code == "l") {
        Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(pybind11::cast<int64>(rows), pybind11::cast<int64>(cols));

        return matrix;
    }

    return std::variant<Eigen::MatrixXf, Eigen::MatrixXd>();
}

PYBIND11_MODULE(example, m) {
    m.def("create_matrix", &create_matrix);
}
```

2. 使用c++编译, 并产生对应的动态链接库.

```shell
c++ -O3 -Wall -shared -std=c++17 -undefined dynamic_lookup \
 -I /opt/homebrew/Cellar/eigen/3.3.9/include/eigen3 \
 `python3 -m pybind11 --includes` \
 example.cc -o example`python3-config --extension-suffix`
```

3. test.py进行测试.

```python
import numpy as np

import example

n32 = np.int32(2)
n64 = np.int64(2)

ans32 = example.create_matrix(n32, n32)
print(ans32.dtype)
ans64 = example.create_matrix(n64, n64)
print(ans64.dtype)
```

# 12.pybind11

| 版本  | 描述                          | 注意                                                         | 适配M1 |
| ----- | ----------------------------- | ------------------------------------------------------------ | ------ |
| 2.6.2 | C++11 和 Python 混编操作接口. | 1. 在 Linux 下需要使用 python3-dev                                                                                                                       2. pybind11 的 Python 软件包能自动安装对应的C++库文件.                                                                      3. 需要同时安装 pybind11-global 才能让 cmake 正确的在虚拟环境中找到 pybind11. | 是     |

## 12.1.setup_helpers

### 12.1.1.build_ext

实例化一个build_ext将在编译的过程中自动寻找系统所拥有的最高版本的c++

```python
from setuptools import setup

from pybind11.setup_helpers import build_ext


setup(
    cmdclass={'build_ext': build_ext},
)
```

### 12.1.2.Pybind11Extension

Pybind11Extension用于自动构建c++的动态链接库.

```python
from setuptools import setup

from pybind11.setup_helpers import Pybind11Extension


extension_modules = {
    Pybind11Extension(
        'path/to/xxx.so',  # str|生成动态链接库的路径.
        'path/to/source.cc',  # str|扩展源码的路径.
        include_dirs='path/to/include/xxx',  # str|依赖包的路径.
        language='c++',  # str|扩展使用的编程语言.
    )
}

setup(
    ext_modules=extension_modules,
)
```

# 13.pydot

| 版本  | 描述                  | 注意                                                         | 适配M1 |
| ----- | --------------------- | ------------------------------------------------------------ | ------ |
| 1.4.2 | graphviz的Python接口. | 1. 需要同时安装pydot和graphviz                                                                                                     2.M1目前需要使用conda安装, 使用pip安装不能正常识别. | 是     |

## 13.1.Dot

### 13.1.1.write_png()

将图写入png图片.

```python
import pydot

graph = pydot.graph_from_dot_data(s='digraph{a; b; a->b;}')
graph[0].write_png(path='./1.png')  # str|保存的文件位置.
```

## 13.2.graph_from_dot_data()

从dot脚本中加载Dot列表.|`list`

```python
import pydot

graph = pydot.graph_from_dot_data(s='digraph{a; b; a->b;}')  # str|dot脚本.
graph[0].write_png('./1.png')
```

## 13.3.graph_from_dot_file()

从dot文件中加载Dot列表.|`list`

```python
import pydot

graph = pydot.graph_from_dot_file(path='./test.dot')  # str|dot脚本文件.
graph[0].write_png('./1.png')
```

# 14.scipy

| 版本  | 描述            | 注意                        | 适配M1 |
| ----- | --------------- | --------------------------- | ------ |
| 1.6.3 | Python科学计算. | 1. M1目前需要使用conda安装. | 是     |

## 14.1.stats

### 14.1.1.boxcox()

返回Box-Cox幂变换变换后的数据集.|`numpy.ndarray`和`float`(可选)

```python
from scipy.stats import boxcox

y_trans, lmbda = boxcox(x=[1, 2, 3, 4, 5])  # numpy.ndarray|输入的数据.
```

### 14.1.2.f

#### 14.1.2.1.cdf()

计算F分布的累积分布函数.|`numpy.ndarray`

```python
from scipy.stats import f

value = f.cdf(x=range(0, 10),  # array-like|输入的数据.
              dfn=1,  # float|第一自由度.
              dfd=1)  # float|第二自由度.
```

### 14.1.3.ttest_rel()

计算两个样本的t检验.|`scipy.stats.stats.Ttest_relResult`

```python
from scipy.stats import ttest_rel

res = ttest_rel(a=[1, 2, 3],  # array_like｜输入的样本a.
                b=[2, 4, 6])  # array_like｜输入的样本b.
```

## 14.2.special

### 14.2.1.inv_boxcox()

返回Box-Cox幂变换变换前的数据集.|`numpy.ndarray`

```python
from scipy.stats import boxcox
from scipy.special import inv_boxcox

y_trans, lmbda = boxcox(x=[1, 2, 3, 4, 5])
y = inv_boxcox(y_trans,  # numpy.ndarray|变换后的数据.
               lmbda)  # float|变换参数.
```

# 15.sklearn

| 版本   | 描述                          | 注意                                                         | 适配M1 |
| ------ | ----------------------------- | ------------------------------------------------------------ | ------ |
| 0.24.2 | Python机器学习和数据挖掘模块. | 1. M1目前需要使用conda安装.                                                                                         2. 安装的包名是scikit-learn. | 是     |

## 15.1.datasets

| 版本 | 描述                 | 注意                                                         |
| ---- | -------------------- | ------------------------------------------------------------ |
| -    | sklearn的内置数据集. | 数据集保存的位置 /path/to/lib/python3/site-packages/sklearn/datasets/data |

### 15.1.1.load_iris()

加载返回iris数据集.|`sklearn.utils.Bunch`

```python
from sklearn.datasets import load_iris

dataset = load_iris()
```

## 15.2.ensemble

| 版本 | 描述                   | 注意                                                         |
| ---- | ---------------------- | ------------------------------------------------------------ |
| -    | sklearn的集成学习模块. | 1. 基于sklearn API的其他框架可以使用此模块的一些功能.                                                                                    2. 模型的类方法基本没有差异, 具体参见`LinearRegression`的类方法. |

### 15.2.1.AdaBoostClassifier()

实例化AdaBoost分类器.

```python
from sklearn.ensemble import AdaBoostClassifier

model = AdaBoostClassifier(n_estimators=50,  # int|50|基学习器的数量.
                           learning_rate=1.)  # float|1.0|学习率.
```

### 15.2.2.GradientBoostingClassifier()

实例化梯度提升分类器.

```python
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(learning_rate=0.1,  # float|0.1|学习率.
                                   n_estimators=100)  # int|100|基学习器的数量.

```

### 15.2.3.RandomForestClassifier()

实例化随机森林分类器.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100,  # int|100|基学习器的数量.
                               criterion='gini',  # {'gini', 'entropy'}|'gini'|划分方式.
                               max_depth=None,  # int|None|决策树的最大深度.
                               n_jobs=None,  # int|None|并行运行数量.
                               verbose=0)  # int|0|日志显示模式.
```

### 15.2.4.RandomForestRegressor()

实例化随机森林回归器.

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100,  # int|100|基学习器的数量.
                              criterion='mse',  # {'mse', 'mae'}|'mse'|划分方式.
                              max_depth=None,  # int|None|决策树的最大深度.
                              n_jobs=None,  # int|None|并行运行数量.
                              verbose=0)  # int|0|日志显示模式.
```

### 15.2.5.StackingClassifier()

实例化Stacking分类器.

```python
from sklearn.ensemble import StackingClassifier

model = StackingClassifier(estimators,  # list of (str, estimator)|基学习器的列表.
                           final_estimator=None)  # estimator|sklearn.linear_model.LogisticRegression|二级学习器.
```

### 15.2.6.VotingClassifier()

实例化投票分类器.

```python
from sklearn.ensemble import VotingClassifier

model = VotingClassifier(estimators,  # list of (str, estimator)|基学习器的列表.
                         voting='hard',  # {'hard', 'soft'}|'hard'|投票方式.
                         weights=None)  # array-like of shape (n_classifiers,)|None|基学习器的权重.
```

## 15.3.linear_model

| 版本 | 描述                   | 注意 |
| ---- | ---------------------- | ---- |
| -    | sklearn的线性模型模块. | -    |

### 15.3.1.LinearRegression()

实例化线性回归器.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
```

#### 15.3.1.1.fit()

训练线性回归器.|`self`

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X,  # {array-like, sparse matrix} of shape (n_samples, n_features)|特征数据.
          y,  # array-like of shape (n_samples,) or (n_samples, n_targets)|标签.
          sample_weight=None)  # array-like of shape (n_samples,)|None|样本权重.
```

#### 15.3.1.2.predict()

使用线性回归器进行预测.|`numpy.ndarray`

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
y_preds = model.predict(X)  # {array-like, sparse matrix} of shape (n_samples, n_features)|特征数据.
```

#### 15.3.1.3.score()

计算验证集的平均准确率.|`float`

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
accuracy = model.score(X,  # {array-like, sparse matrix} of shape (n_samples, n_features)|特征数据.
                       y,  # array-like of shape (n_samples,) or (n_samples, n_targets)|标签.
                       sample_weight=None)  # array-like of shape (n_samples,)|None|样本权重.
```

### 15.3.2.LogisticRegression()

实例化逻辑回归器.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
```

## 15.4.metrics

| 版本 | 描述               | 注意 |
| ---- | ------------------ | ---- |
| -    | sklearn的评估模块. | -    |

### 15.4.1.accuracy_score()

计算分类器的准确率.|`numpy.float64`

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_true,  # 1d array-like, or label indicator array / sparse matrix|真实标签.
                          y_pred,  # 1d array-like, or label indicator array / sparse matrix|预测标签.
                          sample_weight=None)  # array-like of shape (n_samples,)|None|样本权重.
```

### 15.4.2.confusion_matrix()

计算分类器的混淆矩阵.|`numpy.ndarray`

```python
from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_true,  # array-like of shape (n_samples,)|真实标签.
                          y_pred,  # array-like of shape (n_samples,)|预测标签.
                          sample_weight=None)  # array-like of shape (n_samples,)|None|样本权重.
```

### 15.4.3.r2_score()

计算R2决定系数.|`numpy.float64`

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true,  # array-like of shape (n_samples,) or (n_samples, n_outputs)|真实标签.
              y_pred,  # array-like of shape (n_samples,) or (n_samples, n_outputs)|预测标签.
              sample_weight=None)  # array-like of shape (n_samples,)|None|样本权重.
```

## 15.5.model_selection

| 版本 | 描述                   | 注意 |
| ---- | ---------------------- | ---- |
| -    | sklearn的模型选择模块. | -    |

### 15.5.1.cross_val_predict()

对每个样本进行交叉验证.|`numpy.ndarray`

```python
from sklearn.model_selection import cross_val_predict

res = cross_val_predict(estimator,  # estimator object|基学习器.
                        X,  # array-like of shape (n_samples, n_features)|特征数据.
                        y=None,  # array-like of shape (n_samples,) or (n_samples, n_outputs)|None|标签.
                        cv=None)  # int|5|交叉验证的划分数.
```

### 15.5.2.cross_val_score()

进行交叉验证.|`numpy.ndarray`

```python
from sklearn.model_selection import cross_val_score

res = cross_val_score(estimator,  # estimator object|基学习器.
                      X,  # array-like of shape (n_samples, n_features)|特征数据.
                      y=None,  # array-like of shape (n_samples,) or (n_samples, n_outputs)|None|标签.
                      scoring=None,  # str or callable|None|评分函数.
                      cv=None)  # int|5|交叉验证的划分数.
```

### 15.5.3.GridSearchCV()

实例化网格搜索器.

```python
from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(estimator,  # estimator object|基学习器.
                  param_grid,  # dict or list of dictionaries|参数网格.
                  scoring=None,  # str, callable, list, tuple or dict|None|评分函数.
                  n_jobs=None,  # int|None|并行运行数量.
                  cv=None,  # int|5|交叉验证的划分数.
                  verbose=0)  # int|0|日志显示模式.
```

#### 15.5.3.1.fit()

组合所有参数网格进行训练.|`self`

```python
from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(estimator,
                  param_grid,
                  scoring=None,
                  n_jobs=None,
                  cv=None,
                  verbose=0)
gs.fit(X,  # array-like of shape (n_samples, n_features)|特征数据.
       y=None)  # array-like of shape (n_samples, n_output) or (n_samples,)|None|标签.
```

#### 15.5.3.2.best_params_

最佳参数.|`dict`

```python
gs.best_params_
```

#### 15.5.3.3.best_score_

最佳平均交叉验证分数.|`float`

```python
gs.best_score_
```

### 15.5.4.LeaveOneOut()

实例化留一法交叉验证器.

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
```

#### 15.5.4.1.split()

划分数据.|`yield`

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
train_set, test_set = loo.split(X,  # array-like of shape (n_samples, n_features)|特征数据.
                                y=None)  # array-like of shape (n_samples,)|标签.
```

### 15.5.5.StratifiedKFold()

实例化分层K折交叉验证器.

```python
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5,  # int|5|交叉验证的划分数.
                        shuffle=False,  # bool|False|是否打乱数据.
                        random_state=None)  # int, RandomState instance or None|None|随机状态.
```

#### 15.5.5.1.n_splits

交叉验证的划分数.|`int`

```python
kfold.n_splits
```

#### 15.5.5.2.split()

划分数据.|`yield`

```python
from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits=5,
                        shuffle=False,
                        random_state=None)
train_set, test_set = kfold.split(X,  # array-like of shape (n_samples, n_features)|特征数据.
                                  y=None)  # array-like of shape (n_samples,)|标签.
```

### 15.5.6.train_test_split()

将数据集拆分成训练和测试集.|`list`

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y,  # lists, numpy arrays, scipy-sparse matrices or pandas dataframes|要拆分的数据.
                                                    test_size=None,  # float or int|None|测试集的大小.
                                                    random_state=None,  # int, RandomState instance or None|None|随机状态.
                                                    shuffle=True)  # bool|True|是否打乱数据.
```

## 15.6.preprocessing

| 版本 | 描述                     | 注意                                                         |
| ---- | ------------------------ | ------------------------------------------------------------ |
| -    | sklearn的数据预处理模块. | 1. 预处理器的类方法基本没有差异, 具体参见`LabelEncoder`的类方法. |

### 15.6.1.LabelEncoder()

实例化标签编码器.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
```

#### 15.6.1.1.fit_transform()

预处理数据.|`numpy.ndarray`

```python
from sklearn.preprocessing import LabelEncoder

raw_y = ['a', 'a', 'b', 'c']
le = LabelEncoder()
y = le.fit_transform(y=raw_y)  # array-like of shape (n_samples,)|要处理的数据.
```

### 15.6.2.MinMaxScaler()

实例化MinMax缩放器.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
```

### 15.6.3.MultiLabelBinarizer()

实例化多标签二值化缩放器.

```python
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
```

#### 15.6.3.1.classes_

原始的标签.|`numpy.ndarray`

```python
mlb.classes_
```

### 15.6.4.StandardScaler()

实例化标准化器.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

## 15.7.svm

| 版本 | 描述                     | 注意                                                         |
| ---- | ------------------------ | ------------------------------------------------------------ |
| -    | sklearn的支持向量机模块. | 1. 模型的类方法基本没有差异, 具体参见`LinearRegression`的类方法. |

### 15.7.1.SVC()

实例化支持向量分类器.

```python
from sklearn.svm import SVC

model = SVC(C=1.0,  # float|1.0|正则化系数.
            kernel='rbf',  # {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}|'rbf'|核函数.
            probability=False,  # bool|False|是否启用概率估计.
            class_weight=None)  # dict or 'balanced'|None|类别权重.
```

### 15.7.2.SVR()

实例化支持向量回归器.

```python
from sklearn.svm import SVR

model = SVR(kernel='rbf',  # {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}|'rbf'|核函数.
            C=1.0)  # float|1.0|正则化系数.
```

## 15.8.tree

| 版本 | 描述                 | 注意                                                         |
| ---- | -------------------- | ------------------------------------------------------------ |
| -    | sklearn的决策树模块. | 1. 模型的类方法基本没有差异, 具体参见`LinearRegression`的类方法. |

### 15.8.1.DecisionTreeClassifier()

实例化决策树分类器.

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(criterion='gini',  # {'gini', 'entropy'}|'gini'|划分方式.
                               random_state=None)  # int, RandomState instance or None|None|随机状态.
```

### 15.8.2.export_graphviz()

导出决策树结构为Dot语言.|`str`

```python
from sklearn.tree import DecisionTreeClassifier, export_graphviz

model = DecisionTreeClassifier(criterion='gini',
                               random_state=None)
dot_str = export_graphviz(decision_tree=model,  # decision tree regressor or classifier|要绘制的决策树.
                          out_file=None,  # object or str|None|是否导出文件.
                          feature_names=None,  # list of str|None|特征的名称.
                          class_names=None)  # list of str or bool|None|类别的名称.
```

### 15.8.3.plot_tree()

绘制决策树.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree

model = DecisionTreeClassifier(criterion='gini',
                               random_state=None)
plot_tree(decision_tree=model)  # decision tree regressor or classifier|要绘制的决策树.
```

## 15.9.utils

| 版本 | 描述               | 注意 |
| ---- | ------------------ | ---- |
| -    | sklearn的工具模块. | -    |

### 15.9.1.multiclass

#### 15.9.1.1.type_of_target()

判断数据的类型.|`str`

```python
from sklearn.utils.multiclass import type_of_target

y = ['a', 'b', 'c']
res = type_of_target(y=y)  # array-like|输入的数据.
```

### 15.9.2.resample()

对数组进行重采样.|`list`

```python
from sklearn.utils import resample

arr = [1, 2, 3, 4, 5]
new_arr = resample(arr,  # array-like|输入的数据, (可以输入多个).
                   random_state=2022)  # int|None|随机状态.
```

# 16.tokenizers

| 版本   | 描述                  | 注意                        | 适配M1 |
| ------ | --------------------- | --------------------------- | ------ |
| 0.10.1 | 快速和自定义的分词器. | 1. M1目前需要使用conda安装. | 是     |

## 16.1.ByteLevelBPETokenizer()

实例化字符级BPE分词器.

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer(vocab='./vocab.json',  # str(可选)|None|词汇表.
                                  merges='./merges.txt',  # str(可选)|None|分词表.
                                  add_prefix_space=False,  # bool|False|是否田间前缀空间.
                                  lowercase=False)  # bool|False|是否全部转换为小写字母.
```

### 16.1.1.decode()

解码给定的ID列表.|`str`

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer(vocab='./vocab.json',
                                  merges='./merges.txt',
                                  add_prefix_space=False,
                                  lowercase=False)

encoding_list = [31414, 34379, 328]
raw_text = tokenizer.decode(ids=encoding_list)  # list|要解码的ID列表.
```

### 16.1.2.encode()

编码给定的序列(对).|`tokenizers.Encoding`

```python
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer(vocab='./vocab.json',
                                  merges='./merges.txt',
                                  add_prefix_space=False,
                                  lowercase=False)

raw_text = 'Hello Transformers!'
encoding = tokenizer.encode(sequence=raw_text)  # str|要编码的序列(对).
```

#### 16.1.2.1.ids

编码后的ID列表.|`list`

```python
encoding.ids
```

# 17.transformers

| 版本  | 描述                | 注意                                                         | 适配M1 |
| ----- | ------------------- | ------------------------------------------------------------ | ------ |
| 4.6.1 | SOTA自然语言处理库. | 1. 默认的缓存路径是~/.cache/huggingface/transformers                                                               2. 部分功能需要依赖sentencepiece模块. | 是     |

## 17.1.AlbertTokenizer

### 17.1.1.\__call__()

为Albert分词(预处理)一个或者多个数据.|{`input_ids`, (`token_type_ids`), (`attention_mask`)}

```python
from transformers import AlbertTokenizer

tokenizer = AlbertTokenizer.from_pretrained(pretrained_model_name_or_path='albert-base-v2',
                                            do_lower_case=True)
x = 'Hello Transformer!'
encoder = tokenizer(text=x,  # list of str|需要预处理的文本.
                    add_special_tokens=True,  # bool(可选)|True|是否使用特殊标记器.
                    padding=False,  # bool(可选)|False|是否填充到最大长度.
                    truncation=False,  # bool(可选)|False|是否截断到最大长度.
                    max_length=128,  # int(可选)|None|填充和截断的最大长度.
                    return_tensors='tf',  # {'tf', 'pt', 'np'}(可选)|None|返回张量的类型.
                    return_token_type_ids=False,  # bool(可选)|False|是否返回令牌ID.
                    return_attention_mask=False)  # bool(可选)|False|是否返回注意力掩码.
```

### 17.1.2.from_pretrained()

实例化Albert预训练分词器.|`transformers.models.albert.tokenization_albert.AlbertTokenizer`

```python
from transformers import AlbertTokenizer

tokenizer = AlbertTokenizer.from_pretrained(pretrained_model_name_or_path='albert-base-v2',  # str|预训练分词器的名称或者路径.
                                            do_lower_case=True)  # bool(可选)|True|是否全部转换为小写字母.
```

## 17.2.BertTokenizer()

### 17.2.1.\__call__()

为Bert分词(预处理)一个或者多个数据.|{`input_ids`, (`token_type_ids`), (`attention_mask`)}

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-uncased',
                                            do_lower_case=True)
x = 'Hello Transformer!'
encoder = tokenizer(text=x,  # list of str|需要预处理的文本.
                    add_special_tokens=True,  # bool(可选)|True|是否使用特殊标记器.
                    padding=False,  # bool(可选)|False|是否填充到最大长度.
                    truncation=False,  # bool(可选)|False|是否截断到最大长度.
                    max_length=128,  # int(可选)|None|填充和截断的最大长度.
                    return_tensors='tf',  # {'tf', 'pt', 'np'}(可选)|None|返回张量的类型.
                    return_token_type_ids=False,  # bool(可选)|False|是否返回令牌ID.
                    return_attention_mask=False)  # bool(可选)|False|是否返回注意力掩码.
```

### 17.2.2.from_pretrained()

实例化Bert预训练分词器.|`transformers.models.bert.tokenization_bert.BertTokenizer`

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-uncased',  # str|预训练分词器的名称或者路径.
                                          do_lower_case=True)  # bool(可选)|True|是否全部转换为小写字母.
```

## 17.3.RobertaConfig

### 17.3.1.from_pretrained()

获取Roberta的预训练配置信息.|`transformers.models.roberta.configuration_roberta.RobertaConfig`

```python
from transformers import RobertaConfig

config = RobertaConfig.from_pretrained(pretrained_model_name_or_path='roberta-base')  # str|预训练的配置信息名称或者路径.
```

## 17.4.TFAlbertModel

### 17.4.1.from_pretrained()

实例化预训练的Albert模型.|`transformers.models.albert.modeling_tf_albert.TFAlbertModel`

```python
from transformers import TFAlbertModel

model = TFAlbertModel.from_pretrained(pretrained_model_name_or_path='albert-base-v2',  # str|预训练模型的名称或者路径.
                                      trainable=True)  # bool|True|参数是否可以训练.
```

## 17.5.TFBertModel

### 17.5.1.from_pretrained()

实例化预训练的Bert模型.|`transformers.models.bert.modeling_tf_bert.TFBertModel`

```python
from transformers import TFBertModel

model = TFBertModel.from_pretrained(pretrained_model_name_or_path='bert-base-uncased',  # str|预训练模型的名称或者路径.
                                    trainable=True)  # bool|True|参数是否可以训练.
```

## 17.6.TFRobertaModel

### 17.6.1.from_pretrained()

实例化预训练的Roberta模型.|`transformers.models.roberta.modeling_tf_roberta.TFRobertaModel`

```python
from transformers import TFRobertaModel

model = TFRobertaModel.from_pretrained(pretrained_model_name_or_path='roberta-base',  # str|预训练模型的名称或者路径.
                                       trainable=True)  # bool|True|参数是否可以训练.
```

# 18.xgboost

| 版本  | 描述                  | 注意                                                         | 适配M1 |
| ----- | --------------------- | ------------------------------------------------------------ | ------ |
| 1.4.2 | 梯度提升决策树(GBDT). | 1. 可直接在sklearn使用.                                                                                                                              2. 模型的类方法基本没有差异, 具体参见`XGBClassifier`的类方法. | 是     |

## 18.1.XGBClassifier()

实例化XGBoost分类器.

```python
from xgboost import XGBClassifier

model = XGBClassifier(max_depth=None,  # int|None|基学习器的最大深度.
                      learning_rate=None,  # float|None|学习率.
                      n_estimators=100,  # int|100|基学习器的数量.
                      objective=None,  # str or callable|None|损失函数.
                      booster=None,  # {'gbtree', 'gblinear', 'dart'}|None|基学习器的类型.
                      n_jobs=None,  # int|None|并行运行数量.
                      subsample=None,  # float|None|随机采样率.
                      colsample_bytree=None,  # float|None|每棵树的属性随机采样率.
                      random_state=None)  # int|None|随机状态.
```

### 18.1.1.fit()

训练XGBoost分类器.|`self`

```python
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X,  # array_like|特征数据.
          y,  # array_like|标签.
          eval_set=None,  # list(可选)|验证集元组列表.
          eval_metric=None,  # str, list of str, or callable(可选)|验证时使用的评估指标.
          early_stopping_rounds=None,  # int|早停的轮数.
          verbose=True)  # bool|日志显示模式.
```

### 18.1.2.predict()

使用XGBoost分类器进行预测.|`numpy.ndarray`

```python
from xgboost import XGBClassifier

model = XGBClassifier()
y_preds = model.predict(X)  # array_like|特征数据.
```

### 18.1.3.score()

计算验证集的平均准确率.|`float`

```python
from xgboost import XGBClassifier

model = XGBClassifier()
accuracy = model.score(X,  # array-like of shape (n_samples, n_features)|特征数据.
                       y)  # array-like of shape (n_samples,) or (n_samples, n_outputs)|标签.
```

## 18.2.XGBRegressor()

实例化XGBoost回归器.

```python
from xgboost import XGBRegressor

model = XGBRegressor(max_depth=None,  # int|None|基学习器的最大深度.
                     learning_rate=None,  # float|None|学习率.
                     n_estimators=100,  # int|100|基学习器的数量.
                     objective=None,  # str or callable|None|损失函数.
                     booster=None,  # {'gbtree', 'gblinear', 'dart'}|None|基学习器的类型.
                     n_jobs=None,  # int|None|并行运行数量.
                     subsample=None,  # float|None|随机采样率.
                     colsample_bytree=None,  # float|None|每棵树的属性随机采样率.
                     random_state=None)  # int|None|随机状态.
```

