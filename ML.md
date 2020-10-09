# <center>A²PI²-ML version2.1</center>

# 第一章 绪论

## 1.1.引言

* 计算机科学是研究关于算法的学问，机器学习是研究关于学习算法的学问

## 1.2.基本术语

* 数据集(data set) 是一组记录的集合
* 示例(instance)样本(sample)  是对一个事件或对象的描述
* 属性(attribute)特征(feature) 是反映事件或对象在某方面的表现或性质的事项
* 属性值(attribute value) 属性上的取值
* 属性空间(attribute space)样本空间(sample space) 属性张成的空间
* 特征向量(feature vector) 每个示例都可以在样本空间里找到对应的坐标位置，由于空间的每个点也对应一个坐标向量，一个实例也可以叫做一个特征向量
* 维数(dimensionality) 属性的个数
* 学习(learning)训练(training) 从数据中学得模型的过程
* 训练数据(training data) 训练过程中使用的数据
* 训练样本(training sample) 训练数据中的具体一个样本
* 训练集(training set) 训练样本组成的集合
* 假设(hypothesis) 学得模型对应了关于数据的某种潜在规律
* 真相、真实(ground-truth) 潜在规律本身
* 学习器(learner) 具体学习算法在给定数据和参数空间上的实例化
* 预测(prediction) 模型基于训练样本的得到的某种结果
* 标记(label) 关于样本的结果信息
* 样例(example) 拥有标记的样本
* 标记空间、输出空间(label space)  所有标记的集合
* 有监督任务(supervised learning) 有标记信息的数据集
   1. 分类(classification) 预测结果是离散值
   2. 回归(regression) 预测结果是连续值
* 无监督任务(unsupervised learning) 没有标记信息的数据集
   1. 聚类(clustering) 将训练集分组，每组为一个簇(cluster)，簇可能存在潜在的概念划分
* 测试(testing) 学习模型后使用模型预测的过程
* 测试样本(testing sample) 被预测的样本
* 泛化(generalization) 学得模型适用于新样本的能力

## 1.3.假设空间

* 假设空间(hypothesis space) 所有假设组成的空间
* 版本空间(version space) 与训练集一致的假设集合
   1. 假设空间按照删除与正例不一致（与反例相同）的假设的搜索方式在某个训练集上得到的结果就是版本空间

## 1.4.归纳偏好

* 归纳偏好(inductive bias) 机器学习算法在学习过程中对某种类型假设的偏好

   1. 例如，在回归学习中，一般认为相似的样本具有相似的输出，我们应该让算法偏向于归纳这种假设

* 没有免费的午餐定理(No Free Lunch Theorem) 所有算法的期望性能和随机胡猜一样
  
  $$
  \sum\limits_{f} E_{ote}(\mathcal{E}_a|X,f)=\sum\limits_{f} E_{ote}(\mathcal{E}_b|X,f)
  $$
  
   > [NFL定理证明](https://www.jianshu.com/p/e1705306f6a3)


# 第二章 模型评估与选择

## 2.1.经验误差与过拟合

* 错误率(error rate) 分类错误的样本占样本总数的比例
* 精度(accuracy) 1-错误率 
* 误差(error) 学习器在训练集的实际预测输出与样本的真实输出之间的差异
* 训练误差(training error)经验误差(empirical error) 学习器在训练集上的误差
* 泛化误差(generalization error) 学习器在新样本上的误差
   1. 我们希望得到一个泛化误差小的学习器，然而，我们根据训练样本只能得到一个经验误差很小的学习器
* 过拟合(overfitting) 把训练样本自身的一些特点当作了所有潜在样本具有的某种一般性质，导致泛化性能下降
* 欠拟合(underfitting) 对训练样本的一般性质尚未学习好
   1. 一般情况下，学习能力高低分别会导致过拟合和欠拟合，欠拟合可以通过增加扩展分支（决策树）、训练轮数（神经网络），过拟合则是机器学习的主要障碍，过拟合是无法避免的

## 2.2.评估方法

* 测试集(testing set) 测试学习器对新样本的判别能力
* 测试误差(testing error) 用作泛化误差的近似
   1. 我们一般假设测试样本是从真实样本分布中独立同分布采样而得。测试集应该和训练集互斥

### 2.2.1.留出法(hold-out)

* 将数据集D划分为互斥的两个集合，其中一个为训练集S，另一个为测试集T，即$D=S∪T$,$S∩T=\varnothing$,基于S训练出的模型，用T来评估测试误差，作为泛化误差的估计

   1. 按照采样(sampling)的角度看待数据集的划分过程，则保留类别比例的采样方式通常为分层采样(stratified sampling)
   
   2. 训练集S过大会导致接近数据集D、测试集T较小，评估结果不够准确；S过小会导致基于S和D训练出模型的差别过大，降低评估结果的保真性(fidelity)
   
   3. 通常取2/3~4/5的样本作为训练集


### 2.2.2.交叉验证法(cross validation)

* 将数据集D划分为k个大小相同的互斥子集，即 $ D = D_1 \cup D_2 \cup ... \cup D_k,D_i \cup D_j= \varnothing (i \neq j) $ 。每个子集$D_i$ 都保持数据分布的一致性，即按照分层采样得到。每次取k -1个作为训练集，其余作为测试集，进行k次训练和测试，返回k个结果的均值

   1. 交叉验证法评估结果依赖于k的取值，因此也叫做“k折交叉验证”(k-fold cross validation)
   
   2. k通常取值为10
   
   3. 若数据集D包含m个样本，令k=m，得到交叉验证法的一个特例：留一法(Leave-One-Out)


### 2.2.3.自助法(bootstrapping)

* 自助法以自助采样法(bootstrap sampling)为基础，在一个含有m个样本的数据集D中，每次随机抽取一个样本拷贝入$ D' $ ($D'$ 中有重复的样本)，重复执行m次，使得$$D'$$ 中也有m个样本 

   1. 样本在m次采样始终采样不到的概率为$(1 - \frac{1}{m})^m $，极限是 $ \lim\limits_{m \to \infty}(1 - \frac{1}{m})^m = \frac{1}{e} \approx 0.368 $
   
   2. 使用始终没有被采样的部分作为测试集
   
   3. 这样的测试结果，叫包外估计(out-of-bagestimate)
   
   4. 自助法适用于较小和难以划分的数据集，对集成学习有很大好处


### 2.2.4.调参与最终模型

* 参数的取值范围是在实数范围，参数的选择一般是按照取值范围和步长
   1. 超参数:数目通常在10以内，采用人工设定
   2. 模型的参数:比如神经网络中的参数，采用学习（比如训练轮数）
* 包含m个样本的数据集D，在模型评估选择时，选用了部分数据做训练，另一部分做评估，在学习算法和参数配置选定以后，需要重新将所有的样本，即数据集D进行训练，得到的模型交给用户
* 模型在实际使用的过程中遇到的数据称为测试数据
* 模型评估与选择中用于评估测试的数据叫做验证集(validation set)，比如，研究算法泛化性能时，我们测试集估计实际使用的泛化能力，而把训练数据划分为训练集和验证集，基于验证集进行模型选择和调餐参数

## 2.3.性能度量(performance measure)

* 性能度量(performance measure) 衡量模型泛化性能的评价标准
  
  ---
  
   1. 回归任务的性能度量是均方误差(mean squared error) 
      $$
      E(f;D)=\frac{1}{m} \sum\limits_{{i=1}}^m (f(\textbf{x_i})-y_i)^2
      $$
  
   2. 对于数据分布$\mathcal{D}$和概率密度函数$p(·)$，均方误差的描述为 
      $$
      E(f;D)=\int{_{\textbf{x}\sim{}\mathcal{D}}(f(\textbf{x})-y)^2p(\textbf{x})d\textbf{x}}
      $$

  ---

### 2.3.1.错误率与精度

* 错误率是分类错误的样本数占样本总数的比例

* 精度是分类正确的样本数占样本总数的比例

  ---

   1. 样例集$$D$$分类错误率定义为
      $$
      E(f;D)=\frac{1}{m} \sum\limits_{{i=1}}^m \mathbb I(f(\textbf{x_i}) \neq y_i)
      $$
      精度定义为
      $$
      acc(f;D) = \frac{1}{m} \sum\limits_{{i=1}}^m \mathbb I(f(\textbf{x_i}) = y_i) = 1 - E(f;D)
      $$
     
   2. 对于数据分布$\mathcal{D}$和概率密度函数$p(·)$，错误率定义为
      $$
      E(f;D)=\int{_{\textbf{x}\sim{}\mathcal{D}} \mathbb I(f(\textbf{x}) \neq y)p(\textbf{x})d\textbf{x}}
      $$
      精度定义为
      $$
      acc(f;D) = \int{_{\textbf{x}\sim{}\mathcal{D}} \mathbb I(f(\textbf{x}) = y)p(\textbf{x})d\textbf{x}} = 1 - E(f;D)
      $$

  ---

### 2.3.2.查准度(precision)、查全率(recall)与F1

* 样例根据其真实类别和学习器预测类别的组合为真正例(true positive)、假正例(false positive)、真反例(true negative)、假反例(false negative)

  ---

   1. 查准率定义为
      $$
      P=\frac{TP}{TP+FP}
      $$

   2. 查全率定义为
      $$
      R=\frac{TP}{TP+FN}
      $$

  ---

* 查准率$P$和查全率$R$是一对矛盾的度量

* 以查准率为纵轴、以查全率为横轴作图，得到查准率-查全率曲线，简称“$P$-$R$曲线”，显示曲线的的图称为“$P$-$R$图”

* 一个学习器$P$-$R$曲线完全“包住”另一个的，则说明前者性能好与后者性能

* 当查准率$P$=查全率$R$时的取值，叫做平衡点(Break-Even Point)

* 由于BEP过于简化，更常用的是$F1$度量

  ---

   1. $F1$度量定义为
      $$
      F1=\frac{2 \times P \times R}{P + R}=\frac{2 \times TP}{样例总数+TP-TN}
      $$

   2. $F1$是基于查准率和查全率的平均调和平均定义的

      $$
      \frac{1}{F1}=\frac{1}{2}·(\frac{1}{P}+\frac{1}{R})
      $$

   3. F1的一般形式是$F_\beta$，$F_\beta$定义为

      $$
      F_\beta=\frac{(1+ \beta^2) \times P \times R}{(\beta^2 \times P) + R}
      $$

   4. 当$\beta = 1$时，退化为$F1$；$\beta > 1$时，查全率有更大的影响；$\beta < 1$时，查准率有更大的影响

  ---

* 在n个二分类混淆矩阵上宏查准率、宏查全率和宏F1定义为

  ---

   1. 宏查准率
      $$
      macro-P=\frac{1}{n} \sum\limits_{{i=1}}^nP_i
      $$
      
   2. 宏查全率
      $$
      macro-R=\frac{1}{n} \sum\limits_{{i=1}}^nR_i
      $$
      
   3. 宏F1

      $$
      macro-F1=\frac{2 \times macro-P \times macro-R}{macro-P + macro-R}
      $$

  ---

* 在n个二分类混淆矩阵上将$TP$、$FP$、$TN$、$FN$进行平均$\overline{TP}$、$\overline{FP}$、$\overline{TN}$、$\overline {FN}$进而得到微查准率、微查全率和微F1

  ---

    1. 微查准率
       $$
       micro-P=\frac{\overline {TP}}{\overline{TP}+\overline{FP}}
       $$
  
    2. 微查全率
       $$
       micro-R=\frac{\overline {TP}}{\overline{TP}+\overline{FR}}
       $$
  
    3. 微F1
       $$
       micro-F1=\frac{2 \times micro-P \times micro-R}{micro-P + micro-R}
       $$
  
  ---

### 2.3.3.ROC与AUC

* ROC曲线是衡量学习器泛化性能的有力工具，其中ROC曲线的纵轴是真正例率$TPR=\frac {TP}{TP + FN}$、横轴是假正例率$FPR = \frac{FP}{TN+FP}$

   ---

   1. AUC(Area Under ROC Curve)面积可以反映学习器性能
   $$
   \begin{equation}
          \begin{aligned}
      		AUC&=\frac{1}{2}\sum\limits_{i=1}^{m-1}(x_{i+1} - x_i)·(y_i+y_{i+1})\\
    			   &=1-\ell_{rank}
          \end{aligned}
      \end{equation}
   $$
   
   2. $m^+$个正例、$m^-$个反例，$D^+$为正例集合、$D^-$为反例集合排序的损失，AUC和Mann-Whitney U检验等价
      $$
      \ell_{rank}=\frac{1}{m^+m^-}\sum\limits_{{m^+\in D^+}}\sum\limits_{{m^-\in D^-}}\Big(\mathbb I{(f(x^+)<f(x^-)}+\frac{1}{2}\mathbb I{(f(x^+)=f(x^-))}\Big)
      $$
  
  ---
  

  > [ROC曲线画法](https://blog.csdn.net/u013385925/article/details/80385873)

### 2.3.4.代价敏感错误率与代价曲线

* 为了权衡不同类型错误所造成的不同损失，可为错误赋予非均等代价(unequal cost)

* 在考虑非均等代价下，我们希望最小化总体代价(total cost)，以二分类问题为例，

  此时错误率为

  $$
E(f;D;cost)=\frac{1}{m} \Big (\sum\limits_{{x_i\in D^+}} \mathbb I(f(\textbf{x_i}) \neq y_i) \times cost_{01} + \sum\limits_{{x_i\in D^-}} \mathbb I(f(\textbf{x_i}) \neq y_i) \times cost_{10}\Big )
  $$
  
* 在非均等代价下，ROC不能直接反映学习器的期望总体代价，而需要通过代价曲线(cost curve)
  
  ---
  
   1. 代价曲线横轴是取值[0,1]的正例概率代价
      $$
      P(+)cost=\frac{p \times cost_{01}}{p \times cost_{01} + (1- p) \times cost_{10}}
      $$
  
   2. 纵轴是取值为[0,1]归一化代价
      $$
      cost_{norm}=\frac{FNR \times p \times cost_{01}+ FPR \times (1 - p) \times cost_{10}}{p \times cost_{01} + (1- p) \times cost_{10}}
      $$
  
  ---
  
* 每个ROC曲线上的一点转化为代价平面上的一条线段，取所有线段的下界围成的面积为期望的总体代价

## 2.4.比较检验

* 使用统计假设检验(hypotheis test)进行学习器性能的比较

### 2.4.1.假设检验

---

  1. 在包含m个样本的测试集上，泛化错误率为$\epsilon$的学习器被测得测试错误率$ \hat\epsilon $的概率是
     $$
     P(\hat\epsilon;\epsilon)=\dbinom{m}{\hat\epsilon \times m}\epsilon^{\hat\epsilon \times m}(1 -\epsilon)^{m - \hat\epsilon \times m}
     $$

  2. 考虑$\epsilon \leq \epsilon_0$，则在$1-\alpha$的概率内所观测到最大错误率
     $$
     \bar{\epsilon} = max\epsilon\ \  s.t.\ \ \sum^m\limits_{i=\epsilon_0 \times m+1} \dbinom{m}{i} \epsilon^i(1-\epsilon)^{m - i}< \alpha
     $$

  3. $k$个测试错误率的平测试错误率$\mu$和方差$\sigma^2$分别为
     $$
     \mu= \frac{1}{k}\sum\limits_{i=1}^k\hat\epsilon_i\qquad\sigma^2=\frac{1}{k-1}\sum\limits_{i=1}^k(\hat\epsilon_i-\mu)^2
     $$

  4. $k$个测试错误率看作泛化错误率$\epsilon_0$的独立采样，变量$\tau_t=\frac{\sqrt{k}(\mu-\epsilon_0)}{\sigma}$服从自由度为$k-1$的$t$分布

---

### 2.4.2.交叉验证t检验

* 用于比较不同学习器的性能
* 学习器A和学习器B使用k折交叉验证法得到测试错误率分别为$\epsilon^A_1,\epsilon^A_2,......,\epsilon^A_k$和$\epsilon^B_1,\epsilon^B_2,......,\epsilon^B_k$，可以使用k折交叉验证“成对t检验”(paired t-tests)进行比较检验，基本思想是如果两个学习器性能相同，则使用相同数据集测试错误率应该相同，即$\epsilon^A_i=\epsilon^B_i$分别对每对测试错误率求差$\Delta_i = \epsilon^A_i-\epsilon^B_i$，（如果两个学习器性能相同它们的差值应该为零），计算均值$\mu$和方差$\sigma^2$，在显著度$\alpha$下，变量$\tau_t=\Big|\frac{\sqrt{k}\mu}{\sigma}\Big|$小于临界值$t_{\alpha/2,\ k-1}$则假设不能被拒绝，说明学习器A和学习器B的性能相当；如果不一样，则认为平均错误率小的性能更好
* 以上需要满足测试错误率均为泛化错误率的独立采样，但是通常样本有限，训练集会有重叠，即测试错误率不独立，会过高估计假设成立的概率，因此，可以采用“5$\times$2交叉验证”法5$\times$2交叉验证是作5次2折交叉验证，每次2折交叉验证之前随机将数据打乱，使得5次交叉验证数据划分不重复，对学习器A和B的第i次的两个测试错误率求差，计算平均值$\mu=0.5(\Delta_1^1+\Delta_1^2)$，对每次2折实验的结果都计算出其方差$\sigma^2_i=(\Delta^1_i-\frac{\Delta^1_i+\Delta^2_i}{2})^2+(\Delta^2_i-\frac{\Delta^1_i+\Delta^2_i}{2})^2$，变量$\tau_t=\mu/\sqrt{0.2\sum\limits_{i=1}^5\sigma^2_i}$

### 2.4.3.McNemar检验

* 对于二分类问题，使用留出法可以得到分类器分类结果的差别，如果两个学习器性能相同，则$e_{01}=e_{10}$，那么变量$|e_{01}-e_{10}|$应当服从正态分布。McNemar检验考虑变量
$\tau_{\chi^2}=\frac{(|e_{01}-e_{10}|-1)^2}{e_{01}+e_{10}}$在给定显著度$\alpha$，如果变量小于临界值$\chi^2_{\alpha}$则假设不能被拒绝，说明学习器A和学习器B的性能相当；如果不一样，则认为平均错误率小的性能更好

### 2.4.4.Friedman检验与Nemenyi后续检验

* 在一组数据集上对多个算法性能比较，既可以在每个数据集上分别列出不同算法两两比较，也可以使用基于算法排序的Friedman检验就可以满足我们的要求

  ---

   1. 使用留出法或者交叉验证法得到每个算法在每个数据集上的测试结果，然后根据测试的性能进行排序，并计算平均序值

   2. 使用Friedman检验算法性能假设N个数据集，k个算法，令$r_i$表示第i个算法的平均序值，变量
      $$
      \begin{equation}
          \begin{aligned}
          \tau_{\chi^2}&=\frac{k - 1}{k}·\frac{12N}{k^2-1}\sum\limits_{i = 1}^k\Big(r_i-\frac{k+1}{2}\Big)^2\\
          &=\frac{12N}{k(k+1)}\Big(\sum\limits_{i = 1}^kr_i^2-\frac{k(k+1)^2}{4}\Big)
          \end{aligned}
      \end{equation}
      $$
      
      但是“原始Friedman检验”过于保守，通常使用变量
      $$
      \tau_{F}=\frac{(N-1)\tau_{\chi^2}}{N(k-1)-\tau_{\chi^2}}
      $$
  
   3. 若性能显著不同，此时需要进行后续检验(post-hoc test)进行进一步检验，这里我们使用Nemenyi后续检验，Nemenyi检验计算出平均序值差别的临界值域为
      $$
      CD=q_\alpha\sqrt{\frac{k(k+1)}{6N}}
      $$
  
   4. 如果两个算法的平均序值之差超出临界值域$CD$，则认为两个算法的性能有显著差异
  
  ---

## 2.5.偏差与方差

* 偏差-方差分解(bias-variance decomposition)是解释学习算法泛化性能的一种工具

  ---

   1. 回归任务的算法期望为
      $$
      \bar f(x) = \mathbb E_D[f(x;D)]
      $$

   2. 使用样本相同的不同算法产生的方差为 
      $$
      var(x)=\mathbb E_D\Big[\big(f(x;D)-\bar f(x)\big)^2\Big]
      $$

   3. 噪声为
      $$
      \varepsilon^2 = \mathbb E_D\Big[(y_D-y)^2\Big]
      $$

   4. 期望输出和真实标记输出的偏差
      $$
      bias^2(x) = (\bar f(x)-y)^2
      $$

   5. 算法的期望泛化误差（假定噪声期望为0）
      $$
      E(f;D)= \mathbb E_D\big[(f(x;D)-y_D)^2\big] =bias^2(x) + var(x) + \varepsilon^2
      $$

  ---

* 偏差度量了学习算法期望预测与真实结果的偏离程度，即刻画了学习算法本身的拟合能力

* 方差度量了同样大小训练集的变动导致的学习性能的变化，即刻画数据扰动所造成的影响

* 噪声则表达了当前任务学习算法所能达到的期望泛化误差的下界，即刻画问题本身的难度

* 偏差和方差是有冲突的，称为偏差-方差窘境(bias-variance dilemma)

# 第三章 线性模型

## 3.1.基本形式

* 给定由d个属性描述的示例$x=(x_i;x_2;...;x_d)$，其中$x_i$是$x$在第$i$个属性的取值，线性模型(linear model)是试图学得一个通过属性的线性组合来预测的函数，即

  $$
  f(x)=w_1x_1+w_2x_2+...+w_dx_d+b
  $$

* 一般也可用向量形式写成，即
  $$
  f(x)=w^Tx+b
  $$
  其中$w=(w_1;w_2;...;w_d)$，当w和b学得以后就可以确定模型了

* 功能强大的非线性模型(nonlinear model)可以在线性模型的基础上通过引入层级结构或高维映射而得

* $w$直观表达了各属性在预测中的重要性(权重)，因此线性模型具有很好的可解释性(comprehensibility)

## 3.2.线性回归

* 给定数据集$D = \left\{ (x_1,y_1),(x_2,y_2),...,(x_m,y_m) \right\}$，其中$x_i = (x_{i1};x_{i2};...;x_{id}), y_i \in \mathbb R$。线性回归(linear regression)试图学习一个线性模型尽可能准确的预测实值输出标记

* 对于离散属性，若属性值间存在“序”(oeder)关系，可以通过连续化将其转化为连续值，例如，二值属性身高的取值高矮可以转化为{1.0,0.0}

* 若属性之间没有序关系，假定有k个属性值，可以转化为k维向量，例如，瓜类的取值有西瓜、黄瓜和南瓜可以转化为(0,0,1),(0,1,0),(1,0,0)

* 线性回归试图学得
  $$
  f(x_i) = wx_i+b,\;{\large使得}f(x_i)\simeq y_i
  $$

* 均方误差是回归任务最常用的性能度量

  ---

   1. 令均方误差最小化，即
      $$
      \begin{equation}
             \begin{aligned}
             (w^*,b^*)&=\mathop{\arg\min} \limits_{(w,b)}\sum\limits_{i =  1}^m(f(x_i) -y_i)^2\\
                      &=\mathop{\arg\min} \limits_{(w,b)}\sum\limits_{i = 1}^m(w_ix_i + b -y_i)^2\\
                      &=\mathop{\arg\min} \limits_{(w,b)}\sum\limits_{i = 1}^m(y_i - w_ix_i - b)^2
             \end{aligned}
         \end{equation}
      $$

   2. 均方误差对应了欧式距离(Euclidean distance)，因此拥有非常好的几何意义

   3. 基于均方误差最小化的方法叫做最小二乘法(least square method)，最小二乘法就是试图找到一条直线，使得所有样本到直线上的欧式距离之和最小

   4. 求解$E_{(w,b)}= \sum^m_{i=1}(y_i-wx_i-b)^2$最小化的过程，称为线性回归模型的最小二乘“参数估计”(parameter estimation)，可以对w和b分别求偏导，得到
      $$
      \begin{equation}
             \begin{aligned}
             \frac{\partial E_{(w,b)}} {\partial w}&=2\Big(w\sum \limits_{i = 1}^mx_i^2 - \sum \limits_{i = 1}^m(y_i -b)x_i\Big)\\
                      \frac{\partial E_{(w,b)}} {\partial b}&=2\Big(mb - \sum \limits_{i = 1}^m(y_i-wx_i))
             \end{aligned}
         \end{equation}
      $$
令上式为0，得到w和b的最优解的闭式解
      $$
      \begin{equation}
             \begin{aligned}
             w&=\frac{\sum \limits_{i = 1}^m y_i(x_i - \bar x)}{\sum \limits_{i = 1}^mx_i^2-\frac{1}{m}\Big(\sum \limits_{i = 1}^mx_i\Big)^2},\ {\large其中}\bar x = \frac{1}{m}\sum \limits_{i = 1}^mx_i\\
             b&=\frac{1}{m}\sum \limits_{i = 1}^m(y_i - wx_i)
             \end{aligned}
         \end{equation}
      $$
      

  ---

* 更一般的情况是在数据集$D$中，样本由$d$个属性描述，此时多元线性回归(multivariate linear regression)试图学得
  $$
  f(x_i) = w^Tx_i+b,\;{\large使得}f(x_i)\simeq y_i
  $$
  
* 对于多元线性回归，也可利用最小二乘法对w和b进行估计

  ---

   1. 将w和b吸收入向量形式$\hat w = (w,b)$，相应的，把数据集D表示一个$m \times (d+1)$大小的矩阵$\textbf X$，其中的每行对应一个示例，该行的前d个元素对应于示例的d个属性值，最后一个元素恒置为1，即
      $$
      \mathbf X = \left(\begin{matrix}
      x_{11} & x_{12} & \cdots & x_{1d} & 1\\
      x_{21} & x_{22} & \cdots & x_{2d} & 1\\
      \vdots & \vdots & \ddots & \vdots & \vdots\\
      x_{m1} & x_{m2} & \cdots & x_{md} & 1\\
      \end{matrix}\right) = \left(\begin{matrix}x^T_1 & 1\\ x^T_2 &  1\\ \vdots  & \vdots \\ x^T_m & 1\\\end{matrix}\right)
      $$
      将标记也写成向量形式$y=(y_1;y_2;...;y_m)$，于是有
      $$
      \hat w^* = \mathop{\arg\min} \limits_{\hat w}(y-\textbf X\hat w)^T(y-\textbf X\hat w)
      $$
      令$E_\hat w = (y-\textbf X\hat w)^T(y-\textbf X\hat w)$，对$\hat w$求导得到
      $$
      \frac{\partial E\hat w}{\partial\hat w}=2\textbf X^T(\textbf X\hat w-y)
      $$
   2. 令上式为0，得到$\hat w$的最优解的闭式解，当$\textbf X^T\textbf X$为满秩矩阵或者正定矩阵时
     
      $$
      \hat w^* = (\textbf X^T\textbf X)^{-1}\textbf X^Ty
      $$
      令$\hat x_i= (x_i;1)$，此时学得的线性回归模型是
      $$
      f(\hat x_i)=\hat x_i^T(\textbf X^T\textbf X)^{-1}\textbf X^Ty
  $$
      当矩阵不是满秩矩阵的时候，可解出多个解，此时作为输出的将由学习算法的归纳偏好决定，常见的做法是引入正则化(regularization)项
      

  ---

* 当我们认为示例对应的输出标记是在指数尺度上的变化，那就可以将输出标记的对数作为线性模型逼近的目标，即

  $$
  \ln y=w^Tx+b
  $$
  我们称为对数线性回归(log linear regression)，虽然它实际上是$e^{w^Tx+b}$去逼近$y$，但在上式形式上还是线性回归，但实质上已经是求输入到输出的非线性映射了

* 更一般的则是广义线性模型(generalized linear model)
  $$
  y=g^{-1}(w^Tx+b)
  $$
  

  其中，$g(·)$是单调可微的联系函数(link function)，显然，对数线性回归就是广义线性回归在$g(·)=ln(·)$的特例


## 3.3.对数几率回归

* 当需要进行分类任务的时候，需要找到一个单调可微函数将分类任务的真实标记$y$与线性回归模型的预测值联系起来

* 例如二分类任务，我们需要的输出标记$y\in\left\{0,1\right\}$，而线性回归模型产生的预测值$z=w^Tx+b$是个实值，因此我们需要将实值$z$转换为$0/1$值，最理想的是单位阶跃函数(unit-step function)或称Heaviside函数
  $$
  \begin{equation}y=\left\{\begin{aligned}
  0,z<0; \\
  0.5,z=0; \\
  1,z>0;\end{aligned}\right.\end{equation}
  $$
  
  即若预测值$z$大于0就判别为正例，小于0就判别为反例，预测值为临界值0就任意判别
  
* 但是由于单位阶跃函数不连续，一次不能直接作为广义线性模型中的联系函数$g^-(·)$，于是需要找到一个一定程度上近似单位阶跃函数的替代函数(surrogate function)，并且这个函数单调可微

  ---
  
   1. 对数几率函数(logistic function)为常用的替代函数
      $$
      y = \frac{1}{1 + e^{-z}}
      $$
      将对数几率函数作为$g^{-}(·)$代入$y=g^{-1}(w^Tx+b)$得到
      $$
      y = \frac{1}{1 + e^{-(w^Tx+b)}}
      $$
      继而可变化为
      $$
      \ln \frac{y}{1 - y} = w^Tx+b
      $$
      将其中$y$作为正例的可能性$1-y$作为反例的可能性，两者的比值称为几率(odds)，反映了x作为正例的相对可能性，对几率取对数则得到对数几率(log odds,logit)
  
  ---
  


* 由此可以发现，利用线性回归模型的预测结果去逼近真实标记的对数几率，因此模型称为对数几率回归(logistic regression)或者逻辑回归(logistic regression)

* 我们将$y$视为类后验概率估计$p(y=1|x)$，则$\ln \frac{y}{1 - y} = w^Tx+b$可以重写为$\ln \frac{p(y=1|x)}{p(y=0|x)} = w^Tx+b$显然有$p(y=1|x)=\frac{e^{w^Tx+b}}{1+e^{w^T+b}}$和$p(y=0|x)= \frac{1}{1+e^{w^T+b}}$于是，使用极大似然法(maximum likelihood method)进行估计$w$和$b$

  ---

   1. 给定数据集$\left\{(x_i,y_i)\right\}^m_{i=1}$对数回归模型最大化对数似然(log-likelihood)
      $$
      \ell(w,b) = \sum \limits^m_{i=1}\ln p(y_i|x_i;w,b)
      $$
  其中似然函数是
      $$
      P(w,b)=\prod\limits^m_{i=1}p(y_i|x_i;w,b)
      $$
      即令每个样本属于真实值的概率越大越好
      
      令$\beta = (w;b)$,$\hat x = (x;1)$，则$w^Tx+b$可以简写为
      $$
      \beta^T\hat x
      $$

      令$p_1(\hat x;\beta) = p(y=1|\hat x;\beta)$,$p_0(\hat x;\beta) = p(y=0|\hat x;\beta)=1-p_1(\hat x;\beta)$，则似然项可以重写为

      $$
      p(y_i|x_i;w,b)=y_ip_1(\hat x_i;\beta)+(1-y_i)p_0(\hat x_i;\beta)
      $$
  最大化对数似然，得
      $$
      \ell(\beta) = \sum \limits^m_{i=1}\big(y_i\beta^T\hat x_i-\ln\big(1+e^{\beta^T\hat x_i}\big)\big)
      $$
      上式等价于最小化$-\ell(\beta)$，即
      $$
      \ell(\beta) = \sum \limits^m_{i=1}\big(-y_i\beta^T\hat x_i+\ln\big(1+e^{\beta^T\hat x_i}\big)\big)
      $$
      这是一个关于$\beta$的高阶可导连续凸函数，根据凸优化理论、梯度下降法(gradient descent method)、牛顿法(Newton method)都可以求最优解
      $$
      \beta^*= \mathop{\arg\min} \limits_{\beta}\ell(\beta)
      $$
      
  ---
  
  > [对数几率回归推导](https://blog.csdn.net/hongbin_xu/article/details/78270526)

* 牛顿法求最优解

  ---

   1. 牛顿法第$t+1$轮迭代公式是
      $$
      \beta^{t+1}=\beta^t-\Big(\frac{\partial^2\ell(\beta)}{\partial\beta\partial\beta^T}\Big)^{-1}\frac{\partial\ell(\beta)}{\partial\beta}
      $$
  
   2. 其中关于$\beta$的一阶导数是
      $$
      \frac{\partial\ell(\beta)}{\partial\beta}= -\sum \limits ^m_{i=1}\hat x_i(y_i-p_1(\hat x;\beta))
      $$
  
   3. 关于$\beta$的二阶导数是
      $$
      \frac{\partial^2\ell(\beta)}{\partial\beta\partial\beta^T}= \sum \limits ^m_{i=1}\hat x_i\hat x_i^Tp_1(\hat x_i;\beta)(1-p_1(\hat x;\beta))
      $$
  ---

## 3.4.线性判别分析

* 线性判别分析(Linear Discriminant Analysis)是一种经典的线性学习方法，也称为Fisher判别分析

* LDA的思想：给定训练样例集，设法将样例投影在一条直线上，使得同类样本的投影点尽可能接近、异类样本的投影点尽可能远离；当出现新的样本时，将其投影在这条直线上，根据投影点的位置确定类别

  ---

   1. 数据集$D = \left\{(x_i,y_i)\right\}^m_{i=1},y_i\in\left\{0,1\right\}$，令$X_i$、$\mu_i$、$\sum_i$分别表示第$i\in\left\{0,1\right\}$类示例的集合、均值向量、协方差矩阵
  
   2. 将数据投影到直线$w$上，则两类样本的中心在直线上的投影分别是$w^T\mu_0$和$w^T\mu_1$；若将所有的样本都投影到直线上，两类样本的协方差是$w^T\sum_0w$和$w^T\sum_1w$
  
   3. 我们希望同类样例的投影点接近，可以让同类样例的协方差尽可能小，即$w^T\sum_0w+w^T\sum_1w$尽可能小
  
   4. 我们还希望异类样例的投影点尽可能远离，我们可以使它们的类中心之间的距离尽可能大，即$||w^T\mu_0-w^T\mu1||^2_2=w^T(\mu_0-\mu_1)(\mu_0-\mu_1)^Tw$尽可能大
  
   5. 我们同时考虑同类样例协方差小和异类样例类中心距大的目标，即可最大化
      $$
      \begin{equation}
             \begin{aligned}
             J&=\frac{||w^T\mu_0-w^T\mu_1||^2_2}{w^T\sum_0w+w^T\sum_1w}\\
              &=\frac{w^T(\mu_0-\mu_1)(\mu_0-\mu_1)^Tw}{w^T\sum_0w+w^T\sum_1w}
             \end{aligned}
      \end{equation}
      $$
  
   6. 定义类内散度矩阵(within-class scatter matrix)
      $$
      \begin{equation}
             \begin{aligned}
             S_w&=\sum_0 + \sum_1\\
              &=\sum\limits_{x \in X_0}(x-\mu_0)(x-\mu_0)^T+\sum\limits_{x \in X_1}(x-\mu_1)(x-\mu_1)^T
             \end{aligned}
      \end{equation}
      $$
      定义类间散度矩阵(between-class scatter matrix)
      $$
      S_b=(\mu_0-\mu_1)(\mu_0-\mu_1)^T
      $$
      那么
      $$
      J = \frac{w^TS_bw}{w^TS_ww}
      $$
      这就是LDA最大化的目标，$S_b$和$S_w$的广义瑞利商(generalized Rayleigh quotient)
  
   7. 因为$J$的分子和分母是关于$w$的二次项，可以将$w^TS_ww=1$，那么$J$等价于
      $$
      \begin{equation}
             \begin{aligned}
             &\max\limits_w w^TS_bw\\
             &s.t.\ w^TS_ww=1
             \end{aligned}
      \end{equation}
      $$
      即
      $$
      \begin{equation}
             \begin{aligned}
             &\min\limits_w -w^TS_bw\\
             &s.t.\ w^TS_ww=1
             \end{aligned}
      \end{equation}
      $$
      使用拉格朗日乘子法得
      $$
      S_bw=\lambda S_ww
      $$
      其中$\lambda$是拉格朗日乘子。注意到$S_bw$的方向恒为$\mu_0-\mu_1$，令$S_bw= \lambda(\mu_0 - \mu_1)$
      $$
      w = S^{-1}_w(\mu_0-\mu_1)
      $$
  
   8. 可以对$S_w$进行奇异值分解$S_w=U\sum V^T$再由$S^{-1}_w=V\sum^{-1}U^T$得到$S_w^{-1}$
  
  ---
  
  > [LDA证明](https://blog.csdn.net/dhaiuda/article/details/84325203)

* LDA可以推广到多分类问题

  ---
  
   1. 定义全局散度矩阵
      $$
      \begin{equation}
             \begin{aligned}
             S_t&=S_b+S_w\\
             &=\sum\limits^m_{i=1}(x_i- \mu)(x_i -\mu)^T
             \end{aligned}
      \end{equation}
      $$
      其中$\mu$是所有示例的均值
  
   2. 将类内散度矩阵$S_w$重新定义为每个类别的散度矩阵之和
      $$
      S_w=\sum\limits^N_{i=1}S_{w_i}
      $$
      其中
      $$
      S_{w_i}=\sum\limits_{x \in X_i}(x-\mu_i)(x-\mu_i)^T
      $$
      由此可得
      $$
      \begin{equation}
             \begin{aligned}
             S_b&=S_t-S_w\\
             &=\sum\limits_{i=1}^Nm_i(\mu_i-\mu)(\mu_i-\mu)^T
             \end{aligned}
      \end{equation}
      $$
      优化目标是
      $$
      \max\limits_W\frac{tr(W^TS_bW)}{tr(W^TS_wW)}
      $$
      其中$W\in \mathbb R^{d\times(N-1)}$，$tr(·)$是矩阵的迹
  
   3. 通过广义特征值求解得
      $$
      S_bW=\lambda S_wW
      $$
  
   4. $W$的闭式解是$S^{-1}_wS_b$的$d'$个最大非零广义特征值所对应的特征向量组成的矩阵，$d'\leq N-1$

  ---

* 将$W$视为一个投影矩阵，则多分类LDA将样本投影到$d'$维空间，通常$d'$小于数据原有的属性数$d$。因此，LDA也是一种监督降维技术

## 3.5.多分类学习

* 现实中，很多多分类任务可以直接使用二分类的推广，例如LDA的推广

* 考虑N个类别$C_1,C_2,...,C_N$，多分类任务的基本思路是拆解法，即将多分类任务分为多个二分类任务求解

* 分类学习器也称为分类器(classifier)

* 多分类问题的关键是对多分类任务进行拆分和对多个分类器进行集成

* 对于给定数据集$D=\left\{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\right\},y_i \in \left\{C_1,C_2,...,C_N\right\}$，拆分策略有三种

  ---

   1. 一对一(One vs. One，简称OvO)

      将$N$个类别两两配对，从而产生$N(N-1)/2$个二分类任务，例如为区分$C_i$和$C_j$训练一个分类器。在测试阶段，将样本同时提交给所有的份分类器，因此得到$N(N-1)/2$个分类结果。最终结果可以投票产生：即预测最多的类别为最终分类结果

  2. 一对其余(One vs. Rest，简称OvR，OvR也叫One vs. All)

     将一个类的样例作为正例，其余类的样例作为反例来训练$N$个分类器。在预测阶段，如果只有一个分类器预测为正类，则对应类别标记为最终结果；如果多个分类器预测为正类，则通常考虑预测置信度最大的类别标记为分类结果

  3. 多对多(Many vs. Many，简称MvM)

     每次将若干个类作为正类，若干个其他类作为反类。显然上述的OvO和OvR都是MvM的特例。MvM的正反类构造需有特殊的设计，例如，纠错输出码(Error Correcting Output Codes，简称ECOC)

     

  OvR需要训练N个分类器，OvO则需要训练$N(N-1)/2$个分类器；OvO的存储开销和测试时间较大于OvR；但是训练的时候OvR的每个分类器需要使用全部样本，OvO只需要使用两个类别的样本，在类别很多的时候OvO比OvR的时间开销小

  ---

* ECOC的工作过程

  ---

   1. 编码：对N个类别做M次划分，每次划分一部分类别做为正类，一部分划分为反类，从而设计一个二分类训练集；这样一共产生M个训练集，训练M个分类器

   2. 解码：使用M个分类器分别对测试样本进行预测，这些预测会组成一个编码。将这个编码和每个类别各自的编码进行比较，其中距离最小的类别为预测类别

   3. 类别划分通过编码矩阵(coding matrix)指定。编码矩阵主要有二元码和三元码。二元码将分别为每个类别指定为正类和反类，三元码还将指定除正反类以外的停用类

   4. ECOC便编码的纠错能力来源于如果某个分类器出错，根据距离计算决定结果的方式仍能正确判断。一般来说，同一个学习任务，ECOC编码越长，纠错能力越强。但是，ECOC编码越长意味着分类器的数目增加，计算和存储的开销增大，而且有限的类别，可能的组合数是有限的，码长超过一定范围是没有意义的

   5. 同等长度的编码，任意两个类别的编码距离越远，纠错能力越强

  ---

## 3.6.分类不平衡问题

* 之前介绍的分类学习算法都是假设，不同类别的训练样本的数量是相同的。类别不平衡(class-imbalance)就是指分类任务中不同类别的训练样例数目差别很大的情况

* 对于OvR和MvM，由于对每个类进行了相同的处理，拆分出的二分类问题的类别不平很问题会相互抵消

* 这里以线性分类器进行讨论

  ---

   1. 对于$y=w^Tx+b$分类新样本$x$的时候，实际上是用预测出的$y$值与一个阈值进行比较，例如通常认为$y>0.5$时判别为正例，反之，则为反例
  
   2. $y$实际上表达了正例的可能性，几率$\frac{y}{y-1}$则反映了正例可能性和反例可能性的比值，当阈值为0.5时，表明分类器的正、反例可能性相同，即分类器决策规则是
      $$
      {\large若}\frac{y}{1-y}>1{\large则\ \ 预测为正例}
      $$
      然而当训练集的正反例数目不同的时候，令$m^+$表示正例的数目，$m^-$表示反例的数目，则观测几率是$\frac{m^+}{m^-}$，由于我们假设训练集是真实样本总体的无偏采样，因此观测几率和真实几率是等价的。此时，分类器的决策规则是
      $$
      {\large若}\frac{y}{1-y}>\frac{m^+}{m^-}{\large则\ \ 预测为正例}
      $$
      我们需要将分类器的决策模式变更为基于$\frac{y}{1-y}>1$的，因此需令
      $$
      \frac{y'}{1-y'}=\frac{y}{1-y}\times\frac{m^-}{m^+}
      $$
      这就是类别不平衡学习的一个策略----再缩放(rescaling)再平衡(rebalance)
  
  ---
  
* 在实际操作中再缩放却并不平凡，因为“训练集是真实样本总体的无偏采样”是不成立的，因此不能使用观测几率来代替真实几率

* 现有三种技术的做法分别是（我们假定正类样例较少，反类样例较多）

  ---

   1. 直接对训练集的反类样例进行欠采样(undersampling)下采样(downsampling)，即去除一部分反例使得正反样例数目接近

   2. 对训练集里的正类样例进行过采样(oversampling)上采样(upsampling)，即增加一些正例使得正、反例数目接近

   3. 直接基于原始训练集进行学习，在用训练好的分类器进行预测时，嵌入$\frac{y'}{1-y'}=\frac{y}{1-y}\times\frac{m^-}{m^+}$到决策过程中，称为阈值移动(threshold-moving)

      

  欠采样法的时间开销小于过采样法，因为前者丢弃了很多反例，使实际使用的训练集小于初始训练集，而过采样法则增加了许多正例，其训练集大于初始训练集

  过采样法不能简单地对初始正例样本进行重复采样，否则会导致严重的过拟合

  * 使用SMOTE算法对训练集里的正例进行插值来产生额外的正例

  欠采样法若随机的丢弃反例，可能的导致重要信息的丢失

  * 使用EasyEnsemble算法，利用集成学习的机制，将反例划分为若干个集合供不同学习器使用

  ---

* 再缩放也是代价敏感学习(cost-sensitive learning)的基础。将$\frac{y'}{1-y'}=\frac{y}{1-y}\times\frac{m^-}{m^+}$转化为$\frac{y'}{1-y'}=\frac{y}{1-y}\times\frac{cost^+}{cost^-}$，其中$cost^+$、$cost^-$分别是将正例误分为反例的代价和将反例误分为正例的代价

# 第四章 决策树

## 4.1.基本流程

* 决策树(decision tree)是一类常见的机器学习方法。以二分类任务为例，我们希望从给定训练数据集学得一个模型用以对新示例进行分类，把这个样本分类任务，可以看作“当前样本属于正类吗？”这个问题的“决策”过程

* 决策树是基于树结构来进行决策的。决策过程中提出的每一个判定问题都是对某个属性的测试；每个测试的结果或是导出最终结论，或是导出进一步的判定问题，其考虑范围是在上次决策结果的限定范围之内

* 一般的，一颗决策树包含一个根结点、若干个内部结点和若干个叶结点；叶结点对应于决策结果，其他每个结点则对应于一个属性测试

* 决策树学习的目的是为了产生一颗泛化能力强，即处理未见示例能力强的决策树，其基本流程遵循简单且直观的分而治之(divide-and-conquer)策略

* 决策树学习的基本算法

  ---

  输入：训练集$D=\{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\};$

  ​			属性集$A=\{a_1,a_2,...,a_d\}$.

  过程：函数$TreeGenerate(D,A)$

  1:生成结点node；

  2:if $D$中的样本全属于同一类别$C$  then

  3:	将node标记为$C$类叶结点；return

  4:end if

  5:if $A = \varnothing$  $OR$   $D$中的样本在$A$上取值相同 then

  6:	将node标记为叶结点，其类别标记为$D$中的样本数最多的类；return

  7:end if

  8:从$A$中选择最优划分属性$a_*$；

  9:for $a_*$的每一个值$a^v_*$do

  10:	为node生成一个分支；令$D_v$表示$D$中在$a_*$上取值为$a^v_*$的样本子集；

  11:	if $D_v$为空 then

  12:		将分支结点标记为叶结点，其类型标记为$D$中样本的最对的分类；return

  13:	else

  14:		以$TreeGenerate(D_v,A\backslash\{a_*\})$为分支结点

  15:	end if

  16:end for

  输出：以node为根结点的一棵决策树

  ---


* 决策树的基本算法递归返回的情形

  ---

   1. 当前结点包含的样本全属于同一类别，无需划分，递归返回

   2. 当前属性集为空，或是所有样本在所有属性上的取值相同，无法划分，递归返回

      * 我们把当前结点标记为叶结点，并将其类别设为该结点所含样本最多的类别

   3. 当前结点包含的样本集合为空，不能划分，递归返回

      * 把当前结点标记为叶结点，但将其类别设为其父结点所含样本最多的类别

  ---

## 4.2.划分选择

* 决策树学习的关键是如何选择最优划分属性

  1. 一般而言，随着划分过程不断进行，我们希望决策树的分支结点所包含的样本尽可能属于一个类别，即结点的纯度(purity)越来越高

### 4.2.1.信息增益(information gain)

* 信息熵(information entropy)是度量样本集合纯度最常用的指标

  ---

   1. 假定当前样本集合$D$中第$k$类样本所占比例为$p_k(k=1,2,...,|\gamma|)$ ，则$D$的信息熵定义为
      $$
      Ent(D)=-\sum\limits^{|\gamma|}_{k=1}p_k\log_2p_k
      $$

   2. 若$p=0$，则$plog_2p=0$

   3. $Ent(D)$的最小值为0，最大值为$\log_2|\gamma|$

   4. $Ent(D)$的值越小，则$D$的纯度越高

  ---

* 假定离散属性$a$有$V$个可能的取值${a^1,a^2,...,a^V}$，若使用$a$来对样本集$D$进行划分，则会产生$V$个分支结点，其中第$v$个分支结点包含了$D$中所有在属性$a$上的取值为$a^v$的样本，记为$D^v$。由此，可以计算$D^v$的信息熵，由于不同分支结点所包含的样本数不同，给分支结点赋予权重$|D^v|/|D|$，于是可以计算用属性$a$对样本集$D$进行划分所获的信息增益(information gain)

  ---

   1. $$
      Gain(D,a)=Ent(D)-\sum\limits^V_{v=1}\frac{|D^v|}{|D|}Ent(D^v)
      $$
  
   2. 信息增益越大，使用该属性所获得的纯度提升就越大
  
   3. 选择属性$a_*=\mathop{\arg\max}\limits_{a\in A}Gain(D,a)$ 
  
  ---

### 4.2.2.增益率(gain ratio)

* 信息增益准则对可取数值数目较多的属性有偏好

* 减少这种偏好的不利影响，引入了增益率(gain ratio)

  ---

   1. 增益率定义为
      $$
      Gain\_ratio(D,a)=\frac{Gain(D,a)}{IV(a)}
      $$

   2. 其中，$IV(a) = -\sum\limits^V_{v=1}\frac{|D^v|}{|D|}log_2\frac{|D^v|}{|D|}$，称为属性$a$的固有值(intrinsic value)，属性$a$的可能取值数目越多($V$越大)，则$IV(a)$的值越大

  ---

* 增益率准则对可能取值数目较少的属性有偏好

* C4.5算法使用了启发式增益率，从候选划分属性中找出信息增益高于平均水平的属性，在从中选择增益率最高的

### 4.2.3.基尼指数

* CART(Classification and Regression Tree)决策树使用基尼指数(GIni index)来选择划分属性

  ---

   1. 基尼指数可以度量数据集$D$的纯度
      $$
      \begin{equation}
             \begin{aligned}
             Gini(D)&=\sum\limits_{k=1}^{|\gamma|}\sum\limits_{k'≠k} p_kp'_k\\
             &=1-\sum\limits_{k=1}^{|\gamma|}p_k^2
             \end{aligned}
      \end{equation}
      $$

  ---

* 基尼指数反映了从数据集$D$中随机抽取两个样本，其类别标记不一致的概率。因此，基尼指数越小，数据集$D$的纯度越高

  ---

   1. 属性$a$的基尼指数定义
      $$
      Gini\_index(D,a)= \sum \limits^V_{v=1}\frac{|D^v|}{D}Gini(D^v)
      $$
  
   2. 因此，我们候选属性集合$A$中，选择那个使得划分后基尼指数最小的属性作为最优划分，即
      $$
      a_*=\mathop{\arg\min}\limits_{a\in A}Gini\_index(D,a)
      $$
  
  ---

## 4.3.剪枝处理

* 剪枝(pruning)是决策树学习算法应对过拟合的主要手段
* 决策树学习中，为了尽可能正确分类训练样本，结点划分过程不断重复，会造成决策树分支过多，这时就可能导致因训练样本学得“太好”，以致于把训练集自身的一些特点当作所有数据都具有的一般性质而导致过拟合
* 决策树剪枝的基本策略有预剪枝(prepruning)和后剪枝(postpruning)

### 4.3.1.预剪枝(prepruning)

* 预剪枝是在决策树生成过程中，对每个结点在划分前进行估计，如果当前结点的划分不能带来决策树的泛化性能提升，则停止划分并将当前的结点标记为叶结点
* 只有一层划分的决策树，也叫做决策树桩
* 预剪枝显著减少了决策树的训练时间开销和测试时间开销
* 虽然有些分支当前划分不能提高泛化性能，但在其基础上进行后续划分有可能导致性能显著提高
* 预剪枝决策树有欠拟合的风险

### 4.3.2.后剪枝(postpruning)

* 后剪枝则是先从训练集生成一棵完整的决策树，然后自底向上地对非叶结点进行考察，若将该结点对应的子树替换为叶结点能带来决策树泛化性能提升，则将该子树替换为叶结点
* 后剪枝决策树的欠拟合风险较小，泛化性能优于预剪枝决策树
* 由于后剪枝过程是在生成完全决策树之后进行的，并且要自底向上地对树中的所有非叶结点进行逐一考察，因此训练时间开销要远大于未剪枝决策树和预剪枝决策树

## 4.4.连续与缺失值

### 4.4.1.连续值处理

* 对于连续属性，由于可取数目不再有限，因此，不能直接根据连续属性的可取值来对结点进行划分

* 需要使用连续属性离散化技术

* 最简单的策略是二分法(bi-partition)对连续属性进行处理，这是$C4.5$决策树算法采用的机制

  ---

   1. 在给定样本集$D$和连续属性$a$，假定$a$在$D$上出现了$n$个不同的取值，将这些值从小到大进行排序，记为$\{a^1,a^2,...,a^n\}$。基于划分点$t$可将$D$分为子集$D_t^-$和$D_t^+$，其中$D_t^-$包含那些在属性$a$上取值不大于$t$的样本，而$D_t^+$则包含那些在属性$a$上取值大于$t$的样本。显然，对相邻的属性取值$a^i$与$a^{i+1}$来说，$t$在区间$[a^i,a^{i+1})$中取任意值所产生的划分结果相同
  
   2. 因此，对连续属性$a$，我们可考察包含$n-1$个元素的候选划分点集合为$T_a=\Big\{\frac{a^i+a^{i+1}}{2}|1\leq i\leq n-1\Big\}$，把区间$[a^i,a^{i+1})$的中位点$\frac{a^i+a^{i+1}}{2}$作为候选划分点
  
   3. 根据信息增益选取最优划分点，通过最优划分点进行样本集合的划分
      $$
      \begin{equation}
             \begin{aligned}
             Gain(D,a)&=\max\limits_{t \in {T_a}} Gain(D,a,t)\\
             &=\max\limits_{t \in {T_a}}Ent(D)- \sum\limits_{\lambda\in\{-,+\}}\frac{|D^\lambda_t|}{|D|}Ent(D^\lambda_t)
             \end{aligned}
      \end{equation}
      $$
  
  ---
  
* 与离散属性不同，当前结点划分属性为连续属性，该属性还可作为其后代结点的划分属性

### 4.4.2.缺失值处理

* 现实任务中常会遇到不完整样本，即样本的某些属性值缺失

  ---

   1. 给定训练集$D$和属性$a$，令$\tilde{D}$表示$D$中在属性$a$上没有缺失值的样本子集，我们可以仅根据$\tilde{D}$来判定属性$a$的优劣。假定属性$a$有$V$个可取值$\{a^1,a^2,...,a^V\}$，令$\tilde{D^v}$表示$\tilde{D}$在属性$a$上的取值为$a^v$的样本子集，$\tilde{D^k}$表示$\tilde{D}$中属性第$k$类$(k=1,2,...,|\gamma|)$的样本子集，则显然有$\tilde{D}=\bigcup^{|\gamma|}_{k=1}\tilde{D_k}$，$\tilde{D}=\bigcup^{V}_{v=1}\tilde{D^v}$。假定我们为每一个样本$x$赋予一个权重$w_x$，并定义

      $\rho$无缺失值样本所占比例
      $$
      \rho=\frac{\sum_{x\in\tilde{D}}w_x}{\sum_{x\in D}w_x}
      $$
      $\tilde{p_k}$无缺失值样本中第$k$类所占比例
      $$
      \tilde{p}_k=\frac{\sum_{x\in\tilde{D_k}}w_x}{\sum_{x\in\tilde{D}}w_x}\ \ (a\leq k\leq|\gamma|)
      $$
      $\tilde{r_v}$无缺失值样本中在属性$a$中取值$a^v$的样本所占比例
      $$
      \tilde{r}_v=\frac{\sum_{x\in\tilde{D^v}}w_x}{\sum_{x\in\tilde{D}}w_x} \ \ (1\leq v\leq V)
      $$
      显然
      $$
      \sum^{|\gamma|}_{k=1}\tilde{p_k}=1\\
      \sum^V_{v=1}\tilde{r_v}=1
      $$

   2. 将信息增益的计算式推广为
      $$
      \begin{equation}
             \begin{aligned}
             Gain(D,a)&=\rho\times Gain(\tilde{D},a)\\
             &=\rho\times\Big(Ent(\tilde{D})-\sum\limits^V_{v=1}\tilde{r_v}Ent(\tilde{D^v})\Big)
             \end{aligned}
      \end{equation}
      $$

   3. 信息熵推广为
      $$
      Ent(\tilde{D})=-\sum\limits^{|\gamma|}_{k=1}\tilde{p_k}\log_2\tilde{p_k}
      $$

  ---

* 若样本$x$在划分属性$a$上取值已知，则将$x$划分入其取值对应的子结点，且样本权值在子结点中保持为$w_x$

* 若样本$x$在划分属性$a$上的取值未知，则将$x$划分入所有的子结点，且样本权值在与属性值$a^v$对应的子结点中调整为$\tilde{r}_v·w_x$；直观地看，这就是让同一个样本以不同的概率划入不同的子结点中去

## 4.5.多变量决策树(multivariable decision tree)

* 若把每个属性视为坐标空间的一个坐标轴，则$d$个属性描述的样本就对应了$d$维空间中的一个数据点，对样本的分类意味着在坐标空间中寻找不同类样本之间的分类边界。

* 决策树所形成的分类边界有一个明显的特点：轴平行(axis-paraellel)，即它的分类边界由若干个与坐标轴平行的分段组成

* 分类边界的每一段都是与坐标轴平行的，这样的分类边界使得学习结果有较好的可解释性。

* 但是，学习任务的真实分类边界比较复杂时，必须使用很多段划分才能获得较好的近似，此时决策树会相当复杂，由于需要进行大量的属性测试，时间开销比较大

* 如果可以使用斜的划分边界，则决策树模型将大为简化

* 多变量决策树(multivariable decision tree)就是能实现斜划分甚至是更复杂的决策树

  ---

   1. 多变量决策树中，非叶结点不再是仅对某个属性，而是对属性的线性组合进行测试
   2. 每个非叶结点是一个形如$\sum^d_{i=1}w_ia_i=t$的线性分类器，其中$w_i$是属性$a_i$的权重，$w_i$和$t$可在该结点所含的样本集和属性集上学得
   3. 多变量决策树与传统的单变量决策树(univariable decision tree)不同，在多变量决策树中的学习过程是建立一个合适的线性分类器
  
  ---

# 第五章 神经网络

## 5.1.神经元模型

* 神经网络(neural networks)是由具有适应性的简单单元组成的广泛并行互连的网络，它的组织能够模拟生物神经系统对真实世界物体所作出的交互反应

* 神经网络中最基本的成分是神经元(neuron)模型

  1. 在生物神经网络中，每个神经元与其他神经元相连，当它兴奋时，就会向相连的神经元发送化学物质，从而改变这些神经元内的电位；如果某神经元的电位超过了一个阈值(threshold)，那么它就会被激活，即兴奋起来，向其他神经元发送化学物质

  2. 上述的情景可以抽象为M-P神经元模型，神经元接收到来自$n$个其他神经元传递过来的输入信号，这些输入信号通过带权重的连接(connection)进行传递，神经元接收到的总输入值将于神经元的阈值进行比较，然后通过激活函数(activation function)处理以产生神经元的输出
     $$
     y=f\Bigg(\sum\limits_{i=1}^nw_ix_i-\theta\Bigg)
     $$
     其中$x_i$是第$i$个神经元的输入，$w_i$是对应神经元的连接权重，$\theta$是阈值，$y$是输出

  3. 理想的激活函数是阶跃函数，但是阶跃函数具有不连续、不光滑的性质，因此实际使用的激活函数是Sigmoid函数

* 许多个神经元按一定层次结构连接起来，就得到了神经网络

* 神经网络可以视为包含了许多参数的数学模型，这个模型是若干个函数，例如$y=f(\sum_iw_ix_i-\theta_j)$相互嵌套代入而得

* 有效的神经网络学习算法大多数以数学证明为支撑

## 5.2.感知机与多层网络

* 感知机(Perceptron)由两层神经元组层，输入层接受外界输入信号后传递给输出层，输出层是M-P神经元，亦称阈值逻辑单元(threshold logic unit)

* 感知机能实现与、或、非逻辑运算，假定$f$是阶跃函数

  1. 与$(x_1\and x_2)$:令$w_1=w_2=1，\theta=2，$则$y=f(1·x_1+1·x_2-2)$，仅在$x_1=x_2=1$时，$y=1$
  2. 或$(x_1\or x_2)$:令$w_1=w_2=1，\theta=0.5$，则$y=f(1·x_1+1·x_2-0.5)$，当$x_1=1$或$x_2=1$时，$y=1$
  3. 非$(\neg x_1)$:令$w_1=-0.6，w_2=0，\theta=-0.5$，则$y=f(-0.6·x_1+0·x_2+0.5)$，当$x_1=1$时，$y=0$；当$x_1=0$时，$y=1$

* 给定训练数据集，权重$w_i(i=1,2,...,n)$以及阈值$\theta$可供过学习得到，其中阈值$\theta$可看作一个固定输入为-1.0的哑结点(dummy node)所对应的连接权重$w_{n+1}$

* 感知机的学习规则是，对训练样例$(x,y)$，若当前感知机的输出$\hat{y}$，则感知机权重的调整
  $$
  w_i\gets w_i + \Delta w_i\\
  \Delta w_i = \eta(y-\hat{y})x_i
  $$
  其中$\eta\in(0,1)$称为学习率(learning rate)

* 感知机对训练样例$(x,y)$预测正确，即$\hat{y}=y$，则感知机不发生变化

* 感知机只有输出层进行激活函数处理，即只拥有一层功能神经元(functional neuron)

* 只有求解问题是线性可分的，才存在一个线性超平面将它们分开，进而感知机的学习过程是一定会收敛的(converge)，否则感知机的学习过程就会发生振荡(fluctuation)

* 解决非线性可分问题，需要使用多层功能神经元，例如两层感知机

* 输出层和输入层之间的多层神经元，被称作隐层或隐含层(hidden layer)

* 隐含层和输出层的神经元都是拥有激活函数的功能神经元

* 常见的神经网络是层级结构，每层神经元与下一层神经元全互联，神经元之间不存在同层连接，也不存在跨层连接，这样的神经网络叫做多层前馈神经网络(multi-layer feedforward neural networks)

  1. 输入层神经元接受外界输入
  2. 隐层和输出层神经元对信号进行加工

* 神经网络的学习过程，就是根据训练数据来调整神经元之间的连接权(connection weight)以及每个功能神经元的阈值

## 5.3.误差逆传播算法(error BackPropagation)

* 误差逆传播算法亦称反向传播算法

* 多层网络的学习能力比单层感知机强得多

* 现实任务中大多是使用BP算法进行训练

* BP算法描述

  ---

  1. 给定训练集$D=\{(x_1,y_1),(x_2, y_2),...,(x_m, y_m)\},x_i\in \mathbb R^d,y_i\in \mathbb R^l$，即输入示例由$d$个属性描述，输出$l$维实值向量

  2. 为便于讨论，给定一个拥有$d$个输入神经元、$l$个输出神经元、$q$个隐层神经元的多层前馈神经网络

  3. 其中输出层第$j$个神经元的阈值用$\theta_j$表示，隐层第$h$个神经元的阈值用$\gamma_h$表示，输入层第$i$个神经元之间的连接权为$w_{hj}$

  4. 隐层第$h$个神经元接收到的输入为$\alpha_h=\sum^d_{i=1}v_{ih}x_i$，输出层第$j$个神经元接收到的输入为$\beta_j=\sum^q_{h=1}w_{hj}b_h$，其中$b_h$为隐层第$h$个神经元的输出

  5. 隐层和输出层神经元都使用Sigmoid函数

  6. 对训练例$(x_k,y_k)$，假定神经网络的输出为$\hat{y}_k=(\hat{y}_1^k,\hat{y}_2^k,...,\hat{y}_l^k)$，即
     $$
     \hat{y}_j^k=f(\beta_j-\theta_j)
     $$

  7. 网络在$(x_k,y_k)$上的均方误差为
     $$
     E_k=\frac{1}{2}\sum\limits^l_{j=1}(\hat{y}^k_j-y^k_j)^2
     $$

  8. 整个网络中有$(d+l+1)q+l$个参数需要确定：输入层到隐层的$d\times q$个权值，隐层到输出层的$q\times l$个权值，$q$个隐层神经元的阈值、$l$个输出层神经元的阈值

  9. BP是一个迭代学习算法，在迭代中采用广义的感知机学习规则，任意参数的更新估计式为
     $$
     v\gets v+\Delta v
     $$

  10. 推导隐层到输出层的连接权$w_{hj}$，对误差$E_k$，给定学习率$\eta$，有
      $$
      \Delta w_{hj} = -\eta\frac{\partial E_k}{\partial w_{hj}}
      $$

  11. 注意到$w_{hj}$先影响到第$j$个输出层的神经元的输入值$\beta_j$，再影响到其输出值$\hat{y}_j^k$，然后影响到$E_k$，有
      $$
      \frac{\partial E_k}{\partial w_{hj}}=\frac{\partial E_k}{\partial\hat{y}_j^k}·\frac{\partial\hat{y}_j^k}{\partial\beta_j}·\frac{\partial\beta_j}{\partial w_{hj}}
      $$

  12. 根据$\beta_j$的定义，有
      $$
      \frac{\partial\beta_j}{\partial w_{hj}}=b_h
      $$

  13. Sigmoid函数的导数为
      $$
      f'(x)=f(x)(1-f(x))
      $$

  14. 根据均方误差和神经网络的输出，有
      $$
      \begin{equation}
      		\begin{aligned}
      				g_j&=-\frac{\partial E_k}{\partial\hat{y}_j^k}·\frac{\partial\hat{y}_j^k}{\partial\beta_j}\\
      			  &=-(\hat{y}^k_j-y^k_j)f'(\beta_j-\theta_j)\\
              &=\hat{y}^k_j(1-\hat{y}^k_j)(y^k_j-\hat{y}^k_j)\\
      		\end{aligned}
      \end{equation}
      $$

  15. 由此可得BP算法中关于$w_{hj}$的更新公示
      $$
      \Delta w_{hj} = -\eta g_jb_h
      $$

  16. 推导输入层到隐层的连接权$v_{ih}$，对误差$E_k$，给定学习率$\eta$，有
      $$
      \Delta v_{ih} = -\eta\frac{\partial E_k}{\partial v_{ih}}
      $$

  17. 注意到$v_{ih}$先影响到第$h$个隐层的神经元的输入值$\alpha_h$，再影响到其输出值$b_h$，然后影响到$E_k$，有
      $$
      \frac{\partial E_k}{\partial v_{ih}}=\frac{\partial E_k}{\partial b_h}·\frac{\partial b_h}{\partial \alpha_h}·\frac{\partial\alpha_h}{\partial v_{ih}}\
      $$

  18. 根据$\alpha_h$的定义，有
      $$
      \frac{\partial\alpha_h}{\partial v_{ih}}=x_i
      $$

  19. 由此可得
      $$
      \begin{equation}
      		\begin{aligned}
      				e_h&=-\frac{\partial E_k}{\partial b_h}·\frac{\partial b_h}{\partial \alpha_h}\\
      			  &=-\sum\limits^l_{j=1}\frac{\partial E_k}{\partial\beta_j}·\frac{\partial\beta_j}{\partial b_h}f'(\alpha_h-\gamma_h)\\
              &=\sum\limits^l_{j=1}w_{hj}g_jf'(\alpha_h-\gamma_h)\\
              &=b_h(1-b_h)\sum\limits^l_{j=1}w_{hj}g_j\\
      		\end{aligned}
      \end{equation}
      $$

  20. 类似可得
      $$
      \Delta\theta_j=-\eta g_j\\
      \Delta v_{ih}=\eta e_hx_i\\
      \Delta\gamma_h=-\eta e_h
      $$

  ---

* 学习率$\eta\in(0,1)$控制着算法每一轮迭代中的更新步长，若太大则会振荡，太小收敛速度过慢

* 有时做精细调节，可以令$\Delta w_{hj}$和$\Delta\theta_j$中的$\eta_1$与$\Delta v_{ih}$和$\Delta\gamma_h$中的$\eta_2$不同

* BP算法的工作流程

  ---

  输入：训练集$D=\{(x_k,y_l)\}^m_{k=1}$

  ​			学习率$\eta$

  过程：

  1:在(0,1)范围内随机初始化网络中所有连接权和阈值

  2:repeat

  3:	for all$(x_k,y_k)\in D$ do

  4:		根据当前参数和式$\hat{y}_j^k=f(\beta_j-\theta_j)$计算当前样本的输出$\hat{y}_k$；

  5:		根据式$g_j=\hat{y}^k_j(1-\hat{y}^k_j)(y^k_j-\hat{y}^k_j)$计算输出层神经元的梯度项$g_j$；

  6:		根据式$e_h=b_h(1-b_h)\sum\limits^l_{j=1}w_{hj}g_j$计算隐层神经元的梯度项$e_h$；

  7:		根据式$\Delta w_{hj} = -\eta g_jb_h$, $\Delta\theta_j=-\eta g_j$, $\Delta v_{ih}=\eta e_hx_i$, $\Delta\gamma_h=-\eta e_h$更新连接权$w_{hj},v_{ih}$与阈值$\theta_j,\gamma_h$

  8:	end for

  9:until 达到停止停止条件

  输出：连接权和阈值确定的多层前馈神经网络

  ---

* BP算法的目标是最小化训练集$D$上的累积误差
  $$
  E=\frac{1}{m}\sum\limits^m_{k=1}E_k
  $$

* 每次仅针对一个训练样例更新连接权和阈值的叫做标准BP算法

* 基于累积误差最小化的更新规则的叫做累积BP算法

* 累积BP算法读取整个训练集$D$一遍后才对参数进行更新

* 万能近似定律证明，只需一个包含足够多神经元的隐层，多层前馈神经网络就能够以任意精度逼近任意复杂度的连续函数

* 缓解BP网络过拟合的策略

  1. 早停(early stopping)：将数据划分成训练集和验证集，训练集用来计算梯度、更新连接权和阈值，验证集用来估计误差

  2. 正则化(regularization)：在误差目标函数中增加一个用于描述网路复杂度的部分，例如连接权与阈值的平方和
     $$
     E=\lambda\frac{1}{m}\sum\limits^m_{k=1}E_k+(1-\lambda)\sum\limits_iw^2_i
     $$
     正则化会使得训练过程将会偏好比较小的连接权和阈值，使网络输出更加“光滑”；其中$\lambda\in(0,1)$用于对经验误差与网络复杂度这两项进行折中，通常使用交叉验证法

## 5.4.全局最小(global minimum)与局部极小(local minimum)

* 用$E$表示神经网络在训练集上的误差，则它是关于连接权$w$和阈值$\theta$的函数

* 神经网络的训练过程可以看作一个参数寻优过程，在参数空间，寻找一组最优参数使得$E$最小

* 对$w^*$和$\theta^*$，若存在$\epsilon>0$使得
  $$
  \forall(w;\theta)\in\{(w;\theta)|\Vert(w;\theta)-(w^*;\theta^*)\Vert\le\epsilon\},
  $$
  都有$E(w;\theta)\ge E(w^*;\theta^*)$成立，则$(w^*;\theta^*)$为局部极小解

* 若对参数空间中的任意$(w;\theta)$都有$E(w;\theta)\ge E(w^*;\theta^*)$，则$(w^*;\theta^*)$为全局最小解

* 两者对应的$E(w^*;\theta^*)$分别称为误差函数的局部极小值和全局最小值

* 误差函数具有多个局部极小，则不能保证找到的解是全局最小

* 人们采用以下策略试图跳出局部极小

  1. 以多组不同参数值初始化多个神经网络，按标准方法训练后，取其中误差最小的解作为最终参数
  2. 使用模拟退火(simulated annealing)技术，模拟退火在每一步以一定概率接受比当前解更差的结果，在每步迭代过程中，接受次优解的概率要随着时间推移而逐渐降低
  3. 使用随机梯度下降，随机梯度下降在计算梯度中加入了随机因素，使得局部极小点的梯度可能不为零
  4. 遗传算法(genetic algorithms)也常用来训练神经网络更好地逼近全局最小

* 跳出局部极小的技术大多是启发式，理论上缺乏保障

## 5.5.其他常见的神经网络

### 5.5.1.RBF网络

* RBF(Radial Basis Function，径向基函数)网络是一种单隐层前馈神经网络

* 它使用径向基函数作为隐层神经元的激活函数，而输出层则是对隐层神经元的线性组合

* 假定输入为$d$维向量$\pmb{x}$，输出为实值，则RBF网络可表示为
  $$
  \varphi(\pmb{x})=\sum\limits^q_{i=1}w_i\rho(\pmb{x},\pmb{c}_i)
  $$
  其中$q$为隐层神经元个数，$\pmb{c}_i$和$w_i$分别是第$i$个隐层神经元所对应的中心和权重，$\rho(\pmb{x},\pmb{c}_i)$是径向基函数，这是某种沿径向对称的标量函数，通常定义为样本$\pmb{x}$到数据中心$\pmb{c}_i$之间欧式距离的单调函数，常用的高斯径向基函数形如
  $$
  \rho(\pmb{x},\pmb{c}_i)=e^{-\beta_i\Vert\pmb{x}-\pmb{c}_i\Vert^2}
  $$

* RBF网络也被证明，具有足够多隐含层神经元能以任意精度逼近任意连续函数

* RBF网络训练过程

  1. 确定神经元中心$\pmb{c}_i$，常用随机采样、聚类的方式
  2. 利用BP算法确定参数$w_i$和$\beta_i$

### 5.5.2.ART网络

* 竞争型学习(competitive learning)是神经网络中一种常用的无监督学习策略
* 使用该策略时，网络的输出神经元相互竞争，每一时刻仅有一个竞争获胜的神经元被激活，其他神经元的状态被抑制，这种机制亦称胜者通吃(winner-take-all)原则
* ART(Adaptive Rsonance Theory，自适应谐振理论)网络是竞争型学习的代表
* 它由比较层、识别层、识别阈值和重置模块构成
  1. 比较层负责接收输入样本，并将值传递给识别层神经元
  2. 识别层每个神经元对应一个模式类，神经元的数目可在训练过程中动态增长以增加新的模式类
  3. 接收到比较层的输入信号后，识别层神经元之间相互竞争已产生获胜神经元
  4. 竞争方式是计算输入向量于每个识别层神经元所对应的模式类的代表向量之间的距离，距离小者获胜；获胜神经元将向其他识别层神经元发送信号，抑制其激活
  5. 如果输入向量与获胜神经元所对应的代表向量之间的相似度大于识别阈值，则当前输入样本将被归为该代表向量所属类别，同时，网络连接权将会更新，使得以后在接收到相似输入样本时该模式类会计算出更大的相似度；如果相似度不大于识别阈值，则重置模块将在识别层增设一个新的神经元，其代表向量就是设置为当前输入向量
* 决定ART网络性能的就是识别阈值，识别阈值较高，输入样本将会被分为比较多、比较精细的模式类；反之，识别阈值较低，则会产生比较少、比较粗鲁哦的模式类
* ART比较好的缓解了竞争型学习中的可塑性-稳定性窘境(stability-plasticity dilemma)， 可塑性是指神经网络要具有学习新知识的能力，而稳定性则是指神经网络在学习新知识是要保持对旧知识的记忆
* ART网络可以进行增量学习(incremental learning)和在线学习(online learning)
  1. 增量学习是指在学得模型后，再接受到训练样例时，仅需根据新样例对模型进行更新，不必重新训练整个模型，并且先前学得的有效信息不会被冲掉
  2. 在线学习是指每获得一个新样本就进行一次模型更新
  3. 在线学习是增量学习的一种特例，增量学习可看作批模式(batch-mode)的在线学习

### 5.5.3.SOM网络

* SOM(Self-Organizing Map，自组织映射)网络是一种竞争型学习的无监督神经网络
* 它能将高维输入数据映射到低维空间(通常是二维)，同时保持输入数据在高维空间的拓扑结构，即将高维空间中相似的样本点映射到网络输出层的邻近神经元
* SOM网络中的输出层神经元以矩阵方式排列在二维空间中，每个神经元都拥有一个权向量，网络在接收输入向量后，将会确定输出层获胜神经元，它决定了该输入向量在低维空间中的位置
* SOM的训练目标是为每个输出层神经元找到合适的权向量，达到保持拓扑结构的目的
* SOM训练过程
  1. 在接受到一个训练样本后，每个输出层神经元会计算该样本与自身携带的权向量之间的距离，距离最近的神经元成为竞争获胜者，称为最佳匹配单元(best matching unit)
  2. 最佳匹配单元及其邻近神经元的权向量将被调整，以使权向量与当前输入样本的距离缩小
  3. 不断迭代这个过程，直至收敛

### 5.5.4.级联相关网络

* 一般的神经网络模型通常是假定网络结构是事先固定的，训练的目的是利用训练样本来确定合适的连接权、阈值等参数
* 结构自适应网络则将网络结构也当作学习的目标之一，并希望能在训练过程中找到最符合数据特点的网络结构
* 级联相关(Cascale-Correlation)网络是结构自适应网络的代表
* 级联相关网络有两个主要成分：级联和相关
  1. 级联是指建立层次连接的结构
  2. 在开始训练时，网络只有输入层和输出层，处于最小拓扑结构；随着训练的进行，新的隐层神经元逐渐加入，从而创建起层级结构；新的隐层神经元加入时，其输入端连接权值是冻结固定的
  3. 相关是指通过最大化新神经元的输出与网络误差之间的相关性来训练相关的参数
* 级联相关网络无需设置网络层数、隐层神经元数目，训练速度快
* 数据较小时易陷入过拟合

### 5.5.5.Elman网络

* 与前馈神经网络不同，递归神经网络(recurrent neural networks)允许网络中出现环形结构，从而可让一些神经元的输出信号反馈回来作为输入信号
* 这样的结构和信息反馈过程，使得网络在$t$时刻的输出状态不仅与$t$时刻的输入有关，还与$t-1$时刻的网络状态有关，从而能处理与之间有关的动态变化
* Elman网络是最常用的递归神经网络之一
* Elman网络的隐层神经元的输出被反馈回来，与下一时刻输入层神经元提供的信号一起，作为隐层神经元下一时刻的输入
* 隐层神经元使用Sigmoid激活函数
* Elman的训练使用推广的BP算法

### 5.5.6.Boltzmann机

* 神经网络中有一类模型是为网络状态定义一个能量(energy)，能量最小化时网络达到理想状态，网络的训练就是最小化能量函数

* Boltzmann机就是一种基于能量的模型(energy-based model)

* 它的神经元分为两层：显层与隐层

  1. 显层用于表示数据的输入与输出
  2. 隐层则被理解为数据的内在表达

* Boltzmann机中的神经元都是布尔型的，只能取0、1两种状态；状态0表示抑制，状态1表示激活

* 令向量$\pmb{s}\in\{0,1\}^n$表示$n$个神经元的状态，$w_{ij}$表示神经元$i$与$j$之间的连接权，$\theta_i$表示神经元$i$的阈值，则状态能量$\pmb{s}$所对应的Boltzmann机能量定义为
  $$
  E(\pmb{s})=-\sum\limits^{n-1}_{i=1}\sum\limits^n_{j=i+1}w_{ij}s_is_j-\sum\limits^n_{i=1}\theta_is_i
  $$

* 网络中的神经元以任意不依赖于输入值的顺序进行更新，则网络最终将达到Boltzmann分布，此时状态向量$\pmb{s}$出现的概率将仅由其能量与所有可能的状态向量的能量确定
  $$
  P(\pmb{s})=\frac{e^{-E(\pmb{s})}}{\sum_te^{-E(\pmb{t})}}
  $$

* Boltzmann机的训练过程

  1. 将每个训练样本视为一个状态向量，使其出现的概率尽可能大

* 标准的Boltzmann机是一个全连接图，训练网络的复杂度很高，现实任务常使用受限Boltzmann机(Restricted Boltzmann Machine，简称RBM)；受限Boltzmann机仅保留显层和隐层的连接，从而将Boltzmann机结构由完全图简化为二部图

* 受限Boltzmann机的训练过程

  1. 受限Boltzmann机常用对比散度(Contrastive Divergence，简称CD)算法训练

  2. 有$d$个显层神经元和$q$个隐层神经元的网络，令$\pmb{v}$和$\pmb{h}$分别表示显层和隐层的状态向量，则由于同一层不存在连接，有
     $$
     P(\pmb{v}|\pmb{h})=\prod^d_{i=1}P(v_i|\pmb{h})\\
     P(\pmb{h}|\pmb{v})=\prod^q_{j=1}P(h_j|\pmb{v})
     $$

  3. CD算法对每个训练样本$\pmb{v}$，先根据$P(\pmb{h}|\pmb{v})=\prod^q_{j=1}P(h_j|\pmb{v})$计算出隐层神经元状态的概率分布，然后根据这个概率分布采样得到$\pmb{h}$；然后，根据$P(\pmb{v}|\pmb{h})=\prod^d_{i=1}P(v_i|\pmb{h})$从$\pmb{h}$产生$\pmb{v'}$，再从$\pmb{v'}$产生$\pmb{h'}$；连接权的更新公式为
     $$
     \Delta w=\eta\Big(\pmb{vh}^\top-\pmb{v'h'}^\top\Big)
     $$

## 5.6.深度学习(deep learning)

* 参数越多的模型复杂度越高、容量(capacity)越大，即可以完成更复杂的任务
* 复杂模型的训练效率低，易陷入过拟合
* 典型的深度学习模型就是很深层的神经网络；对于神经网络模型，提高容量的一个简单的办法是增加隐层的数目
* 模型复杂度也可通过单纯增加隐层神经元的数目实现
* 增加隐层数目显然比增加隐层神经元的数目有效，因为增加隐层数不仅增加了拥有激活函数的神经元数目，还增加了激活函数嵌套的层数
* 多隐层神经网络不能直接使用经典算法(例如标准BP算法)进行训练，因为误差在多隐层内逆传播时，往往会发散(diverge)而不能收敛到稳定状态
* 无监督逐层训练(unsupervised layer-wise training)是多隐层网络训练的有效方法
  1. 每次训练一层隐结点，训练时将上一层隐结点的输出作为输入，而本层隐结点的输出作为下一层隐结点的输入，则称为预训练(pre-training)
  2. 预训练全部完成后，再对整个网络进行微调(fine-tuning)训练
  3. 在深度信念网络(deep belief network，简称DBN)中，每层都是一个受限Boltzmann机，即整个网络可视为若干个RBM堆叠而得
  4. 在使用无监督逐层训练时，首先训练第一层，这是关于训练样本的RBM模型，可按标准的RBM训练；然后，将第一层预训练好的隐结点视为第二层的输入结点，对第二层进行预训练；······各层预训练完成，再利用BP算法对整个网络进行训练
* 预训练+微调的做法可视为将大量参数组合，对每组找到局部看起来比较好的设置，然后基于这些局部较优的结果进行全局寻优
* 权共享(weight sharing)也是一种训练的策略，即让一组神经元使用相同的连接权，这个策略在卷积神经网络(Convolutional Neural Network，简称CNN)发挥了重要作用
* 深度学习可以被看作是进行特征学习(feature learning)或表示学习(representation learing)
* 描述样本的特征通常需要人类设计，这称为特征工程(feature engineering)

# 第六章 支持向量机

## 6.1.间隔与支持向量

* 给定训练样本集$D=\{(\pmb{x_1}, y_1),(\pmb{x_2}, y_2),...,(\pmb{x_m}, y_m)\}, y_i\in\{-1,+1\}$，分类学习最基本的想法就是基于训练集$D$在样本空间找到一个超平面，将不同的样本分开

* 直观上看，应该去找位于两类训练样本“正中间”的划分超平面

  1. 该划分超平面对训练样本局部扰动“容忍性”最好
  2. 这个划分超平面所产生的分类结果是最鲁棒的，对未见示例的泛化能力最强

* 在样本空间中，划分超平面可通过如下线性方程来描述
  $$
  \pmb{w}^T\pmb{x}+b = 0
  $$
  其中$\pmb{w}=(w_1;w_2;...;w_d)$为法向量，决定超平面的方向；$b$为位移项，决定了超平面与原点之间的距离

* 记超平面为$(\pmb{w},b)$，样本空间任意点$\pmb{x}$到超平面$(\pmb{w},b)$的距离可写为
  $$
  r=\frac{\left|\pmb{w}^T\pmb{x}+b\right|}{\|\pmb{w}\|}
  $$

* 假设超平面$(\pmb{w},b)$能将训练样本正确分类，即对于$(\pmb{x_i},y_i)\in D$，若$y_i=+1$，则有$\pmb{w}^T\pmb{x_i}+b>0$；若$y_i=-1$，则有$\pmb{w}^T\pmb{x_i}+b<0$。令
  $$
  \left\{
  	\begin{aligned}
  	\pmb{w}^T\pmb{x_i}+b\ge+1,\ \  y_i = +1;\\
  	\pmb{w}^T\pmb{x_i}+b\le-1,\ \  y_i = -1.\\
  	\end{aligned}
  \right.
  $$

* 距离超平面最近的几个训练样本点使得上式的等号成立，它们被称为支持向量(support vector)，两个异类支持向量到超平面的距离之和为
  $$
  \gamma=\frac{2}{\|\pmb{w}\|}
  $$
  它被称为间隔(margin)

* 找到具有最大间隔(maximum margin)的划分超平面，即找到满足约束参数$\pmb{w}$和$b$，使得$\gamma$最大，即
  $$
  \begin{align}
  &\max\limits_{\pmb{w},b}\frac{2}{\|\pmb{w}\|}\\
  &\text{s.t.}\ y_i(\pmb{w}^T\pmb{x}_i+b)\ge1, \ i=1,2,...,m\\
  \end{align}
  $$

* 最大化间隔，仅需最大化$\|\pmb{w}\|^{-1}$，这等价于最小化$\|\pmb{w}\|^2$。于是可以上式可重写为
  $$
  \begin{align}
  &\min\limits_{\pmb{w},b}\frac{1}{2}\|\pmb{w}\|^2\\
  &\text{s.t.}\ y_i(\pmb{w}^T\pmb{x}_i+b)\ge1, \ i=1,2,...,m\\
  \end{align}
  $$
  这就是支持向量机(Support Vector Machine)的基本型


## 6.2.对偶问题

* 我们要求解支持向量机的基本型来得到大间隔划分超平面所对应的模型
  $$
  f(\pmb{x})=\pmb{w}^T\pmb{x}+b
  $$
  其中$\pmb{w}$和$b$是模型参数

* 支持向量机的基本型本身是一个凸二次规划(convex quadratic programming)问题，能直接用现成的优化计算包求解，也可以使用拉格朗日乘子法可以得到其对偶问题(dual problem)

* 对支持向量机的基本型的每条约束添加拉格朗日乘子$\alpha_i\ge0$，则该问题的拉格朗日函数可写为
  $$
  L(\pmb{w},b,\pmb{\alpha})=\frac{1}{2}\|\pmb{w}\|^2+\sum\limits^m_{i=1}\alpha_i\big(1-y_i(\pmb{w}^T\pmb{x_i}+b)\big)
  $$
  其中$\pmb{\alpha}=(\alpha_1;\alpha_2;...;\alpha_m)$

* 令$L(\pmb{w},b,\pmb{\alpha})$对$\pmb{w}$和$b$的偏导为零可得
  $$
  \begin{align}
  \pmb{w}&=\sum\limits^m_{i=1}\alpha_iy_i\pmb{x_i}\\
  0&=\sum\limits^m_{i=1}\alpha_iy_i
  \end{align}
  $$

* 将上述式子代入拉格朗日函数，即可将$L(\pmb{w},b,\pmb{\alpha})$中的$\pmb{w}$和$b$消去，得到支持向量机的基本型的对偶问题
  $$
  \max_{\pmb{\alpha}}\sum\limits^m_{i=1}\alpha_i-\frac{1}{2}\sum\limits^m_{i=1}\sum\limits^m_{j=1}\alpha_i\alpha_jy_iy_j\pmb{x}^T_i\pmb{x}_j\\
  \begin{align}
  \text{s.t.}\space\space\ &\sum\limits^m_{i=1}\alpha_iy_i=0,\\
  &\alpha_i\ge0,\space i=1,2,...,m
  \end{align}
  $$
  解出$\pmb{\alpha}$后，求出$\pmb{w}$与$b$即可得到模型
  $$
  \begin{align}
  f(\pmb{x})&=\pmb{w}^T\pmb{x}+b\\
  &=\sum\limits^m_{i=1}\alpha_iy_i\pmb{x}_i^T\pmb{x}+b
  \end{align}
  $$

* 由于存在不等式约束，则需要满足KKT(Karush-Kuhn-Tucker)条件，即
  $$
  \left\{
  		\begin{aligned}
  		&\alpha_i\ge0\\
  		&y_if(\pmb{x}_i)-1\ge0\\
  		&\alpha_i(y_if(\pmb{x}_i)-1)=0
  		\end{aligned}
  \right.
  $$
  对于任意训练样本$(\pmb{x}_i,y_i)$，总有$\alpha_i=0$或$y_if(\pmb{x_i})=1$

  1. 若$\alpha_i=0$，则该样本将不会在求和中出现，就不会对$f(\pmb{x})$有任何影响
  2. 若$\alpha_i>0$，则必有$y_if(\pmb{x_i})=1$，所对应的样本点位于最大间隔边界上，是一个支持向量

* 支持向量机的重要性质：训练完成后，最终模型仅与支持向量有关

  > [支持向量机推导](https://zhuanlan.zhihu.com/p/24638007)

* 二次规划问题可以使用通用的二次规划算法求解；然而，该问题的规模正比于训练样本数

* 求解支持向量机提出了高效的SMO(Sequential Minimal Optimization)算法

* SMO算法描述

  ---

  1. SMO的基本思路

     * 先固定$\alpha_i$之外的所有参数，然后求$\alpha_i$上的极值。由于存在约束$\sum^m_{i=1}\alpha_iy_i=0$，若固定$\alpha_i$之外的其他变量，则$\alpha_i$可由其他变量导出
     * SMO每次选择两个变量$\alpha_i$和$\alpha_j$并固定其他参数

  2. SMO的步骤

     * 选取一对需要更新的变量$\alpha_i$和$\alpha_j$
     * 固定$\alpha_i$和$\alpha_j$以外的参数，求解支持向量机的对偶式获得更新后的$\alpha_i$和$\alpha_j$

  3. 注意到只需选取的$\alpha_i$和$\alpha_j$中有一个不满足$KKT$条件，目标函数就会在迭代后增大

  4. 于是，SMO先选取违背$KKT$条件程度最大的变量

  5. 第二个变量选择一个使得目标函数数值增长最快的变量

     * 由于比较各变量所对应的目标函数的数值的增幅的复杂度过高，因此SMO采用一个启发式
     * 使选取的两个变量所对应的样本之间的间隔最大；直观的解释是，这样的两个变量有很大的差别，与两个相似的变量进行更新相比，会带给目标函数更大的变化

  6. SMO算法的高效，由于固定其他参数后，仅优化两个参数的过程能做到非常的高效

     * 仅考虑$\alpha_i$和$\alpha_j$时，支持向量机的对偶式中的约束可重写为
       $$
       \alpha_iy_i+\alpha_jy_j=c,\space \alpha_i\ge0,\alpha_j\ge 0,
       $$
       其中
       $$
       c=-\sum\limits_{k\ne i,j}\alpha_ky_k
       $$
       使得$\sum\limits^m_{i=1}\alpha_iy_i=0$成立的常数，用
       $$
       \alpha_iy_i+\alpha_jy_j=c
       $$
       消去变量$\alpha_j$，则得到一个关于$\alpha_i$的单变量二次规划问题，仅有约束项是$\alpha_i\ge0$

     * 这样的二次规划问题具有闭式解，可以高效的计算出更新后的$\alpha_i$和$\alpha_j$

  7. 计算偏置项$b$

     * 对任意支持向量$(\pmb{x}_s,y_s)$都有$y_sf(\pmb{x}_s)=1$，即
       $$
       y_s\Bigg(\sum\limits_{i\in S}\alpha_iy_i\pmb{x}_i^T\pmb{x_s}+b\Bigg)=1
       $$
       其中$S=\{i|\alpha_i>0,i=1,2,...,m\}$为所有支持向量的下标集

     * 可以选取任意支持向量通过求解上式获得$b$

     * 实际上采用更鲁棒的做法：计算所有支持向量求解的平均值
       $$
       b=\frac{1}{|S|}\sum\limits_{i\in S}\Bigg(\frac{1}{y_s}-\sum\limits_{i\in S}\alpha_iy_i\pmb{x}_i^T\pmb{x_s}\Bigg)
       $$

  ---

## 6.3.核函数

* 当我们假设训练样本是线性可分的，即存在一个划分超平面能将训练样本正确分类。然而在现实任务重，原始样本空间内也许不存在一个能正确划分的超平面，例如，异或问题就不是线性可分的

* 线性不可分的问题，可将样本从原始空间映射到一个更高维的特征空间，使得样本在这个特征空间内线性可分

* 如果原始空间是有限维，即属性数有限，那么一定存在一个高维特征空间使样本可分

* 令$\phi(\pmb{x})$表示将$\pmb{x}$映射后的特征向量，于是，在特征空间中划分超平面所对应的模型可表示为
  $$
  f(\pmb{x})=\pmb{w}^T\phi(\pmb{x})+b
  $$
  其中$\pmb{w}$和$b$是模型参数，类似可有
  $$
  \begin{align}
  &\min\limits_{\pmb{w},b}\frac{1}{2}\|\pmb{w}\|^2\\
  &\text{s.t.}\ y_i(\pmb{w}^T\phi(\pmb{x}_i)+b)\ge1, \ i=1,2,...,m\\
  \end{align}
  $$

其对偶问题是
$$
  \max_{\pmb{\alpha}}\sum\limits^m_{i=1}\alpha_i-\frac{1}{2}\sum\limits^m_{i=1}\sum\limits^m_{j=1}\alpha_i\alpha_jy_iy_j\phi(\pmb{x}_i)^T\phi(\pmb{x}_j)\\
  \begin{align}
  \text{s.t.}\space\space\ &\sum\limits^m_{i=1}\alpha_iy_i=0,\\
  &\alpha_i\ge0,\space i=1,2,...,m
  \end{align}
$$

* 求解上式涉及到计算$\phi(\pmb{x}_i)^T\phi(\pmb{x}_j)$，这是样本$\pmb{x}_i$和$\pmb{x}_j$映射到特征空间之后的内积；由于特征空间的维数可能很高，甚至是无穷维，因此直接计算$\phi(\pmb{x}_i)^T\phi(\pmb{x}_j)$通常是困难的

* 为了避开这个障碍，设想一个函数
  $$
  \kappa(\pmb{x}_i,\pmb{x}_j)=\lang\phi(\pmb{x}_i),\phi(\pmb{x}_j)\rang=\phi(\pmb{x}_i)^T\phi(\pmb{x}_j)
  $$
  即$\pmb{x}_i$与$\pmb{x}_j$在特征空间的内积等于它们在原始空间中通过原始样本空间中通过函数$\kappa(·,·)$计算的结果，映射后特征空间的对偶问题可以重写为
  $$
  \max_{\pmb{\alpha}}\sum\limits^m_{i=1}\alpha_i-\frac{1}{2}\sum\limits^m_{i=1}\sum\limits^m_{j=1}\alpha_i\alpha_jy_iy_j\kappa(\pmb{x}_i,\pmb{x}_j)\\
  \begin{align}
  \text{s.t.}\space\space\ &\sum\limits^m_{i=1}\alpha_iy_i=0,\\
  &\alpha_i\ge0,\space i=1,2,...,m
  \end{align}
  $$
  求解后即可得到
  $$
  \begin{align}
  f(\pmb{x})&=\pmb{w}^T\phi(\pmb{x})+b\\
  &=\sum\limits^m_{i=1}\alpha_iy_i\phi(\pmb{x}_i)^T\phi(\pmb{x})+b\\
  &=\sum\limits^m_{i=1}\alpha_iy_i\kappa(\pmb{x}_i,\pmb{x})+b
  \end{align}
  $$

* 函数$\kappa(·,·)$就是核函数(kernel function)

* 上式显示出模型最优解可通过训练样本的核函数展开，这一展式亦称支持向量展式(suppoort vector expansion)

* **定理1(核函数)**

  ---

  1. 令$\chi$为输入空间，$\kappa(·,·)$是定义在$\chi\times\chi$上的对称函数，则$\chi$是核函数当且仅当对于任意数据$D=\{\pmb{x}_1,\pmb{x}_2,...,\pmb{x}_m\}$，核矩阵(kernel matrix)$\Kappa$总是半正定的
     $$
     \begin{equation}
     \Kappa=
     	\begin{bmatrix}
     		\kappa(\pmb{x}_1,\pmb{x}_1) & \cdots & \kappa(\pmb{x}_1,\pmb{x}_j) & \cdots\ &\kappa(\pmb{x}_1,\pmb{x}_m)\\
     		\vdots & \ddots & \vdots & \ddots & \vdots \\
     		\kappa(\pmb{x}_i,\pmb{x}_1) & \cdots & \kappa(\pmb{x}_i,\pmb{x}_j) & \cdots\ &\kappa(\pmb{x}_i,\pmb{x}_m)\\
      		\vdots & \ddots & \vdots & \ddots & \vdots \\
      		\kappa(\pmb{x}_m,\pmb{x}_1) & \cdots & \kappa(\pmb{x}_m,\pmb{x}_j) & \cdots\ &\kappa(\pmb{x}_m,\pmb{x}_m)\\
     	\end{bmatrix}
     \end{equation}
     $$

  2. 只要一个对称函数所对应的核矩阵半正定，它就能作为核函数使用

  3. 对于一个半正定核矩阵，总能找到一个与之对应的映射$\phi$

  4. 任何一个核函数都隐式的定义了一个称为再生核希尔伯特空间(Reproducing Kernel Hillbert Space，简称RKHS)的特征空间

  ---

* 我们希望样本在特征空间内线性可分，因此特征空间的好坏对支持向量机的性能影响至关重要

* 在不知道特征映射的形式时，我们不知道什么核函数是合适的，核函数也仅是隐式地定义了这个特征空间，核函数选择成为支持向量机的最大变数

  1. 文本数据通常使用线性核，情况不明时可先尝试高斯核

* 常用的核函数

  | 名称       | 表达式                                                       | 参数                                     | 备注                |
  | :--------- | ------------------------------------------------------------ | ---------------------------------------- | ------------------- |
  | 线性核     | $\kappa(\pmb{x}_i,\pmb{x}_j)=\pmb{x}^T_i\pmb{x}_j$           |                                          |                     |
  | 多项式核   | $\kappa(\pmb{x}_i,\pmb{x}_j)=(\pmb{x}^T_i\pmb{x}_j)^d$       | $d\ge1$为多项式的次数                    | $d=1$时退化为线性核 |
  | 高斯核     | $\kappa(\pmb{x}_i,\pmb{x}_j)=\exp(-\frac{\|\pmb{x}_i-\pmb{x}_j\|^2}{2\sigma^2})$ | $\sigma>0$为高斯核的带宽(width)          | 高斯核亦称RBF核     |
  | 拉普拉斯核 | $\kappa(\pmb{x}_i,\pmb{x}_j)=\exp(-\frac{\|\pmb{x}_i-\pmb{x}_j\|}{\sigma})$ | $\sigma>0$                               |                     |
  | Sigmoid核  | $\kappa(\pmb{x}_i,\pmb{x}_j)=\tanh(\beta\pmb{x}^T_i\pmb{x}_j+\theta)$ | $\tanh$为双曲正切函数$,\beta>0,\theta<0$ |                     |

* 核函数组合

  1. 若$\kappa_1$和$\kappa_2$为核函数，则对于任意正整数$\gamma_1$、$\gamma_2$，其线性组合
     $$
     \gamma_1\kappa_1+\gamma_2\kappa_2
     $$
     也是核函数

  2. 若$\kappa_1$和$\kappa_2$为核函数，则核函数的直积
     $$
     \kappa_1\otimes\kappa_2(\pmb{x},\pmb{z})=\kappa_1(\pmb{x},\pmb{z})\kappa_2(\pmb{x},\pmb{z})
     $$
     也是核函数

  3. 若$\kappa_1$为核函数，则对于任意函数$g(\pmb{x})$
     $$
     \kappa(\pmb{x},\pmb{z})=g(\pmb{x})\kappa_1(\pmb{x},\pmb{z})g(\pmb{z})
     $$
     也是核函数

## 6.4.软间隔与正则化

* 在现实任务中很难确定合适的核函数使得训练样本在特征空间线性可分，也很难断定这个貌似线性可分的结果不是由于过拟合造成的

* 缓解该问题的一个方法是允许支持向量机在一些样本上出错

* 为此，引入软间隔(soft margin)的概念

* 支持向量机形式是要求所有样本均满足约束，即所有样本都必须划分正确，这称为硬间隔(hard margin)

* 软间隔允许某些样本不满足约束
  $$
  y_i(\pmb{w}^T\pmb{x}_i+b)\ge1
  $$

* 在最大化间隔的同时，不满足约束的样本应尽可能的少

* 软间隔的优化目标可写为
  $$
  \min\limits_{\pmb{w},b}\frac{1}{2}\|\pmb{w}\|^2+C\sum\limits^m_{i=1}\ell_{0/1}(y_i(\pmb{w}^T\pmb{x}_i+b)-1)
  $$
  其中$C>0$是一个常数，$\ell_{0/1}$是0/1损失函数
  $$
  \ell_{0/1}(z)=
  \left\{
  		\begin{aligned}
  		&1,\space\space\text{if}\space z<0;\\
  		&0,\space\space\text{otherwise}
  		\end{aligned}
  \right.
  $$
  当$C$为无穷大时，软间隔的优化目标迫使所有样本满足约束，于是等价于支持向量机的基本型；当$C$取有限值时，软间隔的优化目标允许一些样本不满足约束

* $\ell_{0/1}$非凸、非连续、数学性质不好，不易直接求解，故常用其他一些函数来代替$\ell_{0/1}$，称为替代损失(surrogate loss)

* 替代损失函数一般具有较好的数学性质，如他们通常是凸的连续函数且是$\ell_{0/1}$的上界

* 三种常用的替代损失函数

  1. hinge损失:  $\ell_{hinge}(z)=\max(0,1-z)$
  2. 指数损失(exponential loss):  $\ell_{exp}(z)=\exp(-z)$
  3. 对率损失(logistic loss):  $\ell_{log}(z)=\log(1+exp(-z))$

* 若采用hinge损失，则软间隔的优化目标变成
  $$
  \min\limits_{\pmb{w},b}\frac{1}{2}\|\pmb{w}\|^2+C\sum\limits^m_{i=1}\max(0, 1-y_i(\pmb{w}^T\pmb{x}_i+b))
  $$
  引入松弛变量(slack variables)$\xi_i\ge0$，上式可重写为
  $$
  \begin{align}
  &\min\limits_{\pmb{w},b,\xi_i}\frac{1}{2}\|\pmb{w}\|^2+C\sum\limits^m_{i=1}\xi_i\\
  \text{s.t.}&\ y_i(\pmb{w}^T\pmb{x}_i+b)\ge1-\xi_i\\
  &\xi_i\ge0,\space\space i=1,2,...,m\\
  \end{align}
  $$
  这就是常用的软间隔支持向量机

* 软间隔支持向量机仍然是一个二次规划问题，通过拉格朗日乘子法可得到软间隔支持向量机的拉格朗日函数
  $$
  L(\pmb{w},b,\pmb{\alpha},\pmb{\xi},\pmb{\mu})=\frac{1}{2}\|\pmb{w}\|^2+C\sum\limits^m_{i=1}\xi_i+\sum\limits^m_{i=1}\alpha_i\big(1-\xi_i-y_i(\pmb{w}^T\pmb{x_i}+b)\big)-\sum\limits^m_{i=1}\mu_i\xi_i
  $$
  其中$\alpha_i\ge0$，$\mu_i\ge0$是拉格朗日乘子

* 令$L(\pmb{w},b,\pmb{\alpha},\pmb{\xi},\pmb{\mu})$对$\pmb{w}$，$b$，$\xi_i$的偏导为零可得
  $$
  \begin{align}
  \pmb{w}&=\sum\limits^m_{i=1}\alpha_iy_i\pmb{x_i}\\
  0&=\sum\limits^m_{i=1}\alpha_iy_i\\
  C&=\alpha_i+\mu_i
  \end{align}
  $$

* 将上述式子代入拉格朗日函数，即得到软间隔支持向量机的对偶问题
  $$
  \max_{\pmb{\alpha}}\sum\limits^m_{i=1}\alpha_i-\frac{1}{2}\sum\limits^m_{i=1}\sum\limits^m_{j=1}\alpha_i\alpha_jy_iy_j\pmb{x}^T_i\pmb{x}_j\\
  \begin{align}
  \text{s.t.}\space\space\ &\sum\limits^m_{i=1}\alpha_iy_i=0,\\
  &0\le\alpha_i\le C,\space i=1,2,...,m
  \end{align}
  $$
  与硬间隔下的对偶问题对比可看出，两者唯一的差别就在对偶变量的约束不同

* 对软间隔支持向量机，KKT条件要求
  $$
  \left\{
  		\begin{aligned}
  		&\alpha_i\ge0,\space\space \mu_i\ge0 \\
  		&y_if(\pmb{x}_i)-1+\xi_i\ge0\\
  		&\alpha_i(y_if(\pmb{x}_i)-1+\xi_i)=0\\
  		&\xi_i\ge0,\space\mu_i\xi_i=0\\
  		\end{aligned}
  \right.
  $$
  对于任意训练样本$(\pmb{x}_i,y_i)$，总有$\alpha_i=0$或$y_if(\pmb{x_i})=1-\xi_i$

  1. 若$\alpha_i=0$，则该样本不会对$f(\pmb{x})$有任何影响
  2. 若$\alpha_i>0$，则必有$y_if(\pmb{x_i})=1-\xi_i$，即该样本是一个支持向量：
     * 若$\alpha_i<C$，则$\mu_i>0$，进而有$\xi_i=0$，即该样本恰在最大间隔边界上
     * 若$\alpha_i=C$，则$\mu_i=0$：
       * 若$\xi_i\le1$则该样本落在最大间隔内部
       * 若$\xi_i>1$则该样本被错误分类

* 软间隔支持向量机的最终模型仅与支持向量有关，通过采用hinge损失函数仍保持了稀疏性

* 使用对率损失函数$\ell_{log}$来替代0/1损失函数，就几乎得到了对率回归模型；实际上，支持向量机和对率回归的优化目标相近，通常情形下它们的性能也相当

* 对率回归的优势主要在于其输出具有自然的概率意义，而支持向量机则不具有概率意义

* 对率回归可以直接用于多分类问题，支持向量机则需要进行推广

* hinge损失由一块平坦的零区域，这使得支持向量机的解具有稀疏性；对率损失是光滑的单调递减函数，不能导出类似支持向量的概念，对此对率回归的解依赖于更多的训练样本，其预测开销更大

* 换成别的替代损失函数以得到其他学习模型，这些模型的性质与所用的替代函数直接相关，它们有一个共性是：优化目标中的第一项用来描述划分超平面的间隔大小，另一项$\sum^m_{i=1}\ell(f(\pmb{x_i}), y_i)$用来表述训练集上的误差，可写为一般形式
  $$
  \min_f\ \ \Omega(f)+C\sum^m_{i=1}\ell(f(\pmb{x_i}), y_i)
  $$
  $\Omega(f)$称为结构风险(structural risk)，用于描述模型$f$的某些性质

  第二项$\sum^m_{i=1}\ell(f(\pmb{x_i}), y_i)$称为经验风险(empirical risk)，用于描述模型与训练数据的契合程度

  $C$用于对二者进行折中

* 从经验风险最小化的角度看，$\Omega(f)$表述我们希望具有何种性质的模型，它还有助于削减假设空间，从而降低了最小化训练误差的过拟合风险

* 上式称为正则化(regularization)问题，$\Omega(f)$称为正则化项，$C$则称为正则化常数

* $L_p$范数(norm)是常用的正则化项

  1. $L_2$范数$\|\pmb{w}\|_2$倾向于$\pmb{w}$的分量取值尽量均衡，即非零分量个数尽量稠密
  2. $L_0$范数$\|\pmb{w}\|_0$和$L_1$范数$\|\pmb{w}\|_1$则倾向于$\pmb{w}$的分量尽量稀疏，即非零分量个数尽量少

## 6.5.支持向量回归

* 给定训练样本$D=\{(\pmb{x_1}, y_1),(\pmb{x_2}, y_2),...,(\pmb{x_m}, y_m)\}, y_i\in\R$，希望学得一个形如$f(\pmb{x})=\pmb{w}^T\pmb{x}+b$的回归模型，使得$f(\pmb{x})$与$y$尽可能接近，$\pmb{w}$和$b$是待确定的模型参数

* 对样本$(\pmb{x}, y)$，传统回归模型通常直接基于模型输出$f(\pmb{x})$与真实输出$y$之间的差别来计算损失，当且仅当$f(\pmb{x})$与$y$完全相同时，损失才为零

* 支持向量回归(Support Vector Regression，简称SVR)假设我们能容忍$f(\pmb{x})$与$y$之间最多有$\epsilon$得偏差，即仅当$f(\pmb{x})$与$y$之间的差别绝对值大于$\epsilon$时才计算损失

* SVR相当于以$f(\pmb{x})$为中心，构建了一个宽度为$2\epsilon$的间隔带，若训练样本落入此间隔带，则认为是被预测正确

* SVR问题可形式化为
  $$
  \min\limits_{\pmb{w},b}\frac{1}{2}\|\pmb{w}\|^2+C\sum\limits^m_{i=1}\ell_\epsilon(f(\pmb{x}_i)-y_i)
  $$
  其中$C$为正则化常数，$\ell_\epsilon$是$\epsilon$-不敏感损失函数($\epsilon$-insensitive loss)函数
  $$
  \ell_\epsilon(z)=
  \left\{
  		\begin{aligned}
  		&0,\space\space\text{if}\space |z|\le\epsilon;\\
  		&|z|-\epsilon,\space\space\text{otherwise}\\
  		\end{aligned}
  \right.
  $$
  引入松弛变量$\xi_i$和$\hat\xi_i$，上式可重写为
  $$
  \begin{align}
  &\min\limits_{\pmb{w},b,\xi_i,\hat\xi_i}\frac{1}{2}\|\pmb{w}\|^2+C\sum\limits^m_{i=1}(\xi_i+\hat\xi_i)\\
  \text{s.t.}\ &f(\pmb{x}_i)-y_i\le\epsilon+\xi_i\\
  &y_i-f(\pmb{x}_i)\le\epsilon+\hat\xi_i\\
  &\xi_i\ge0,\hat\xi_i\ge0,\space\space i=1,2,...,m\\
  \end{align}
  $$
  间隔带两侧的松弛程度可以不同

* 通过引入拉格朗日乘子$\mu_i\ge0$，$\hat\mu_i\ge0$，$\alpha_i\ge0$，$\hat\alpha_i\ge0$，由拉格朗日乘子可得到SVR的拉格朗日函数
  $$
  \begin{align}
  &L(\pmb{w},b,\pmb{\alpha},\hat{\pmb{\alpha}},\pmb{\xi},\hat{\pmb{\xi}},\pmb{\mu},\hat{\pmb{\mu}})\\
  &=\frac{1}{2}\|\pmb{w}\|^2+C\sum\limits^m_{i=1}(\xi_i+\hat\xi_i)-\sum\limits^m_{i=1}\mu_i\xi_i-\sum\limits^m_{i=1}\hat\mu_i\hat\xi_i\\
  &+\sum\limits^m_{i=1}\alpha_i(f(\pmb{x}_i)-y_i-\epsilon-\xi_i)+\sum\limits^m_{i=1}\hat\alpha_i(y_i-f(\pmb{x}_i)-\epsilon-\hat\xi_i)
  \end{align}
  $$

* 将$f(\pmb{x})=\pmb{w}^T\pmb{x}+b$代入，再令$L(\pmb{w},b,\pmb{\alpha},\hat{\pmb{\alpha}},\pmb{\xi},\hat{\pmb{\xi}},\pmb{\mu},\hat{\pmb{\mu}})$对$\pmb{w},b,\pmb{\xi}_i$和$\hat{\pmb{\xi}_i}$的偏导为零可得
  $$
  \begin{align}
  \pmb{w}&=\sum\limits^m_{i=1}(\hat\alpha_i-\alpha_i)\pmb{x_i}\\
  0&=\sum\limits^m_{i=1}(\hat\alpha-\alpha_i)\\
  C&=\alpha_i+\mu_i\\
  C&=\hat\alpha_i+\hat\mu_i\\
  \end{align}
  $$

* 将上式子代入SVR的拉格朗日函数，即可得到SVR的对偶问题
  $$
  \begin{align}
  \max_{\pmb{\alpha},\hat{\pmb{\alpha}}}&\sum\limits^m_{i=1}y_i(\hat\alpha_i-\alpha_i)-\epsilon(\hat\alpha_i+\alpha_i)\\
  &-\frac{1}{2}\sum\limits^m_{i=1}\sum\limits^m_{j=1}(\hat\alpha_i-\alpha_i)(\hat\alpha_j-\alpha_j)\pmb{x}^T_i\pmb{x}_j\\
  \text{s.t.}\space\space\ &\sum\limits^m_{i=1}(\hat\alpha_i-\alpha_i)=0,\\
  &0\le\alpha_i,\hat\alpha_i\le C,\space i=1,2,...,m
  \end{align}
  $$

* 对支持向量回归，上述过程需满足KKT条件，即要求

  $$
  \left\{
  		\begin{aligned}
  		&\alpha_i(f(\pmb{x}_i)-y_i-\epsilon-\xi_i)=0\\
  		&\hat\alpha_i(y_i-f(\pmb{x}_i)-\epsilon-\hat\xi_i)=0\\
  		&\alpha_i\hat\alpha_i=0,\xi_i\hat\xi_i=0 \\
  		&(C-\alpha_i)\xi_i=0,(C-\hat\alpha_i)\hat\xi_i=0 \\
  		\end{aligned}
  \right.
  $$
  当且仅当$f(\pmb{x}_i)-y_i-\epsilon-\xi_i=0$时$\alpha_i$能取非零值，当且仅当$y_i-f(\pmb{x}_i)-\epsilon-\hat\xi_i=0$时$\hat\alpha_i$能取非零值；换言之，仅当样本$(\pmb{x}_i,y_i)$不落入$\epsilon$-间隔带中，相应的$\alpha$和$\hat\alpha$才能取零值；此外，约束$f(\pmb{x}_i)-y_i-\epsilon-\xi_i=0$和$y_i-f(\pmb{x}_i)-\epsilon-\hat\xi_i=0$不能同时成立，因此$\alpha_i$和$\hat\alpha_i$中至少有一个为零

* SVR的解形如
  $$
  f(\pmb{x})=\sum^m_{i=1}(\hat{\pmb{\alpha}}_i-\pmb{\alpha_i})\pmb{x}_i^T\pmb{x}+b
  $$
  能使上式中的$(\hat\alpha_i-\alpha_i)\ne0$的样本即为SVR的支持向量，它们必落在$\epsilon$-间隔带之外，落在$\epsilon$-间隔带中的样本都满足$\alpha_i=0$且$\hat\alpha_i=0$

* SVR的支持向量仅是训练样本的一部分，即其解仍具有稀疏性

* 由KKT条件可看出，对每个样本$(\pmb{x_i},y_i)$都有$(C-\alpha_i)\xi_i=0$且$\alpha_i(f(\pmb{x}_i)-y_i-\epsilon-\xi_i)=0$；于是，在得到$\alpha_i$后，若$0<\alpha_i<C$，则必有$\xi_i=0$，进而有
  $$
  b=y_i+\epsilon-\sum^m_{j=1}(\hat\alpha_j-\alpha_j)\pmb{x}_j^T\pmb{x}_i
  $$
  理论上，可任意选取满足$0<\alpha_i<C$的样本通过上式求得$b$;实践中常采用更鲁棒的办法：选取多个(或所有)满足条件$0<\alpha_i<C$的样本求解$b$后取平均值

* 若考虑特征映射形式，则$w$将形如
  $$
  \pmb{w}=\sum\limits^m_{i=1}(\hat\alpha_i-\alpha_i)\phi(\pmb{x_i})
  $$
  SVR的解形如
  $$
  f(\pmb{x})=\sum^m_{i=1}(\hat{\pmb{\alpha}}_i-\pmb{\alpha_i})\kappa(\pmb{x},\pmb{x}_i)+b
  $$
  其中$\kappa(\pmb{x}_i,\pmb{x}_j)=\phi(\pmb{x}_i)^T\phi(\pmb{x}_j)$为核函数

## 6.6.核方法

* 给定训练样本$\{(\pmb{x}_1,y_1),(\pmb{x}_1,y_2),...,(\pmb{x}_m, y_m)\}$，若不考虑偏移项$b$，则无论SVM还是SVR，学得的模型总能表示成$\kappa(\pmb{x},\pmb{x}_i)$的线性组合

* **定理2(表示定理)**

  ---

  1. 令$\H$为核函数$\kappa$对应的再生核希尔伯特空间，$\|h\|_{\H}$表示$\H$空间中关于$h$的范数，对于任意单调递增函数$\Omega:[0,\infty]\mapsto\R$和任意非负损失函数$\ell:\R^m\mapsto[0,\infty]$，优化问题
     $$
     \min_{h\in\H}F(h)=\Omega(\|h\|_\H)+\ell\big(h(\pmb{x}_1),h(\pmb{x}_2),...,h(\pmb{x}_m)\big)
     $$
     的解总可写为
     $$
     h^*(\pmb{x})=\sum^m_{i=1}\alpha_i\kappa(\pmb{x},\pmb{x}_i)
     $$

  2. 表示定理对损失函数没有限制，对正则化项$\Omega$仅要求单调递增，甚至不要求$\Omega$是凸函数，意味着对于一般的损失函数和正则化项，优化问题的最优解$h^*(\pmb{x})$都可以表示为核函数$\kappa(\pmb{x},\pmb{x}_i)$的线性组合

  ---

* 人们发展出一系列基于核函数的学习方法，统称为核方法(kernel methods)

* 通过核化(即引入核函数)来将线性学习器拓展为非线性学习器

* 核线性判别分析(zkernelized Linear Discriminant Analysis，简称KLDA)

  ---

  1. 先假设某种映射$\phi:\chi\mapsto\mathbb{F}$将样本映射到一个特征空间$\mathbb{F}$，然后在$\mathbb{F}$中执行线性判别分析，以求得
     $$
     h(\pmb{x})=\pmb{w}^T\phi(\pmb{x})
     $$
     类似KLDA的学习目标是
     $$
     \max_{\pmb{w}}J(\pmb{w})=\frac{\pmb{w}^T\pmb{S}^\phi_b\pmb{w}}{\pmb{w}^T\pmb{S}^\phi_w\pmb{w}}
     $$
     其中$\pmb{S}^\phi_b$和$\pmb{S}^\phi_w$分别为训练样本在特征空间$\mathbb{F}$中的类间散度矩阵和类内散度矩阵

  2. 令$X_i$表示第$i\in\{0,1\}$类样本的集合，其样本总数$m_i$；总样本数$m=m_0+m_1$

  3. 第$i$类样本在特征空间$\mathbb F$的均值为
     $$
     \pmb\mu^{\phi}_i=\frac{1}{m_i}\sum_{\pmb{x}\in{X_i}}\phi(\pmb{x})
     $$

  4. 两个散度矩阵分别为
     $$
     \begin{align}
       &\pmb{S}^{\phi}_b=(\pmb\mu^{\phi}_1-\pmb\mu^{\phi}_0)(\pmb\mu^{\phi}_1-\pmb\mu^{\phi}_0)^T\\
       &\pmb{S}^{\phi}_w=\sum^1_{i=0}\sum_{\pmb{x}\in X_i}\big(\phi(\pmb{x})-\pmb\mu^{\phi}_i\big)\big(\phi(\pmb{x})-\pmb\mu^{\phi}_i\big)^T\\
     \end{align}
     $$

  5. 我们通常难以知道映射$\phi$的具体形式，因此使用核函数$\kappa(\pmb{x},\pmb{x}_i)=\phi(\pmb{x}_i)^T\phi(\pmb{x})$来隐式地表达这个映射和特征空间$\mathbb{F}$

  6. 把$J(\pmb{w})$作为式$\min_{h\in\H}F(h)=\Omega(\|h\|_\H)+\ell\big(h(\pmb{x}_1),h(\pmb{x}_2),...,h(\pmb{x}_m)\big)$的损失函数$\ell$，再令$\Omega\equiv0$，由表示定理，函数$h(\pmb{x})$可写为
     $$
     h(\pmb{x})=\sum^m_{i=1}\alpha_i\kappa(\pmb{x},\pmb{x}_i)
     $$
     由$h(\pmb{x})=\pmb{w}^T\phi(\pmb{x})$可得
     $$
     \pmb{w}=\sum^m_{i=1}\alpha_i\phi(\pmb{x}_i)
     $$

  7. 令$\pmb\Kappa\in\R^{m\times m}$为核函数$\kappa$所对应的核矩阵，$(\pmb\Kappa)_{ij}=\kappa(\pmb{x}_i, \pmb{x}_j)$；令$\pmb{1}_i\in\{1,0\}^{m\times1}$为第$i$类样本的指示向量，即$\pmb{1}_i$的第$j$个分量为1当且仅当$\pmb{x}_j\in X_i$，否则$\pmb{1}_i$的第$j$个分量为0；再令
     $$
     \begin{align}
     	&\hat{\pmb\mu}_0=\frac{1}{m_0}\pmb\Kappa\pmb{1}_0\\
     	&\hat{\pmb\mu}_1=\frac{1}{m_1}\pmb\Kappa\pmb{1}_1\\
     	&\pmb M=(\hat{\pmb\mu}_0-\hat{\pmb\mu}_1)(\hat{\pmb\mu}_0-\hat{\pmb\mu}_1)^T\\
     	&\pmb N=\pmb\Kappa\pmb\Kappa^T-\sum^1_{i=0}m_i\hat{\pmb\mu}_i\hat{\pmb\mu}_i^T\\
     \end{align}
     $$
     KLDA的学习目标等价为
     $$
     \max_{\pmb{\alpha}}J(\pmb{\alpha})=\frac{\pmb{\alpha}^T\pmb M\pmb{\alpha}}{\pmb{\alpha}^T\pmb N\pmb{\alpha}}
     $$

  8. 使用线性判别分析求解方法即可得到$\pmb{\alpha}$，进而可得到投影函数$h(\pmb{x})$
  
  ---

# 第七章 贝叶斯分类器

## 7.1.贝叶斯决策论

* 贝叶斯决策论(Bayesian decision theory)是概率框架下实施决策的基本方法

* 假设有$N$种可能的类别标记，即$\mathcal{Y}=\{c_1, c_2, ...,c_N\}$，$\lambda_{ij}$是将一个真实标记为$c_j$的样本误分类为$c_i$所产生的损失

* 基于后验概率$P(c_i|\pmb{x})$可获得将样本$\pmb{x}$分类为$c_i$所产生的期望损失(expected loss)，即在样本$\pmb{x}$上的条件风险(conditional risk)
  $$
  R(c_i|\pmb{x})=\sum^N_{j=1}\lambda_{ij}P(c_j|\pmb{x})
  $$

* 我们的任务是寻找一个判定准则$h:\mathcal{X}\mapsto\mathcal{Y}$以最小化总体风险

  $$
  R(h)=\mathbb{E}_x\big[R(h(\pmb{x})|\pmb{x})\big]
  $$

* 对每个样本$\pmb{x}$，若$h$能最小化条件风险$R(h(\pmb{x})|\pmb{x})$，则总体风险$R(h)$也将被最小化

* 贝叶斯判定准则(Bayes decision rule)：为最小化总体风险，只需在每个样本上选择那个能使条件风险$R(c|\pmb{x})$最小的类别标记，即
  $$
  h^*(x)=\mathop{\arg\min}_{c\in\mathcal{Y}} R(c|\pmb{x})
  $$
  此时，$h^*$称为贝叶斯最优分类器(Bayes optimal classifier)，与之对应的总体风险$R(h^*)$称为贝叶斯风险(Bayes risk)，$1-R(h^*)$反映了分类器所能达到的最好性能，即通过机器学习所能产生的模型精度的理论上限

* 若目标是最小化分类错误率，则误判损失$\lambda_{ij}$可写为

  $$
  \lambda_{ij}=
  \left\{
  		\begin{aligned}
  		&0,\space\space\text{if}\space i=j;\\
  		&1,\space\space\text{otherwise}
  		\end{aligned}
  \right.
  $$

  此时条件风险
  $$
  R(c|\pmb{x})=1-P(c|\pmb{x})
  $$
  于是，最小化分类错误率的贝叶斯最优分类器为
  $$
  h^*(x)=\mathop{\arg\max}_{c\in\mathcal{Y}} P(c|\pmb{x})
  $$
  即对每个样本$\pmb{x}$，选择能使后验概率$P(c|\pmb{x})$最大的类别标记

* 不难看出，欲使用贝叶斯判定准则来最小化决策风险，首先要获得后验概率$P(c|\pmb{x})$。然而，在现实任务中这通常难以直接获得

* 从这个角度来看，机器学习所要实现的是基于有限的训练样本集尽可能准确地估计后验概率$P(c|\pmb{x})$

* 大体来说，主要有两种策略：

  1. 给定$\pmb{x}$，可通过直接建模$P(c|\pmb{x})$来预测$c$，这样得到是判别式模型(discriminative models)
  2. 先对联合概率分布$P(\pmb{x},c)$建模，然后再由此获得$P(c|\pmb{x})$，这样得到的是生成式模型(generative models)

  决策树、$\text{BP}$神经网络、支持向量机等，都可归入判别式模型的范畴

* 对生成式模型来说，必然考虑
  $$
  P(c|\pmb{x})=\frac{P(\pmb{x},c)}{P(\pmb{x})}
  $$
  基于贝叶斯定理，$P(\pmb{x},c)$可写为
  $$
  P(c|\pmb{x})=\frac{P(c)P(\pmb{x}|c)}{P(\pmb{x})}
  $$
  其中，$P(c)$是类先验(prior)概率；$P(\pmb{x}|c)$是样本$\pmb{x}$相对于类标记$c$的类条件概率(class-conditional probability)，或称为似然(likelihood)；$P(\pmb{x})$是用于归一化的证据(evidence)因子。对给定样本$\pmb{x}$，证据因子$P(\pmb{x})$与类标记无关，因此估计$P(c|\pmb{x})$的问题就转换为如何基于训练数据$D$来估计先验概率$P(c)$和似然$P(\pmb{x}|c)$

* 类先验概率$P(c)$表达了样本空间中各类各样样本所占比例，根据大数定律，当训练集包含充足的独立同分布样本时，$P(c)$可通过各类样本出现的频率来进行估计

* 对连续属性，可将概率密度函数$P(·)$换成概率密度函数$p(·)$

* 对类条件概率$P(\pmb{x}|c)$来说，由于它涉及关于$\pmb{x}$所有属性的联合概率，直接根据样本出现的频率来估计将会遇到严重的困难；例如，假设样本的$d$个属性都是二值的，则样本空间将有$2^d$种可能取值，这个值往往远大于训练样本数$m$，很多样本的取值在训练集中根本没有出现，直接使用频率来估计$P(\pmb{x}|c)$显然不可行

* “未被观测到”与“出现的概率为零”通常是不同的

## 7.2.极大似然估计(Maximum Likelihood Estimation)

* 极大似然估计(MLE)是根据数据采样来估计概率分布参数的经典方法

* 估计类条件概率的一种常用策略是先假定其具有某种确定的概率分布形式，再基于训练样本对概率分布的参数进行估计

  1. 记关于类别$c$的类条件概率为$P(\pmb{x}|c)$，假设$P(\pmb{x}|c)$具有确定的形式并且被参数向量$\pmb{\theta}_c$唯一确定，则我们的任务就是利用训练集$D$估计参数$\pmb{\theta}_c$
  2. 我们将$P(\pmb{x}|c)$记为$P(\pmb{x}|\pmb{\theta}_c)$

* 概率模型训练的过程就参数估计(parameter estimation)过程

  1. 频率主义学派(Frequentist)认为参数虽然未知，但却是客观存在的固定值，因此，可通过优化似然函数等准则来确定参数值
  2. 贝叶斯主义学派(Bayesian)认为参数是未观察到的随机变量，其本身也可有分布，因此，可假定参数服从一个先验分布，然后基于观测到的数据来计算参数的后验分布

* 令$D_c$表示训练集$D$中第$c$类样本组成的集合，假设这些样本是独立同分布的，则参数$\pmb{\theta}_c$对于数据集$D_c$的似然是
  $$
  P(D_c|\pmb{\theta}_c)=\prod_{\pmb{x}\in D_c}P(\pmb{x}|\pmb{\theta_c})
  $$

* 对$\pmb{\theta}_c$进行极大似然估计，就是去寻找最大化似然$P(D_c|\pmb{\theta}_c)$的参数值$\hat{\pmb{\theta}}_c$

* 直观上看，极大似然估计是试图在$\pmb{\theta}_c$所有可能的取值中，找到一个能使数据出现的可能性最大的值

* 上式中的连乘操作易造成下溢，通常使用对数似然(log-likelihood)
  $$
  \begin{equation}
  	\begin{aligned}
      LL(\pmb{\theta}_c)
      &=\log P(D_c|\pmb{\theta}_c)\\
      &=\sum_{\pmb{x}\in D_c}\log P(\pmb{x}|\pmb{\theta}_c)\\
  	\end{aligned}
  \end{equation}
  $$

* 此时参数$\pmb{\theta}_c$的极大似然估计$\hat{\pmb{\theta}}_c$为
  $$
  \hat{\pmb{\theta}}_c=\mathop{\arg\max}_{\pmb{\theta}_c}LL(\pmb{\theta}_c)
  $$

* 在连续属性情形下，假设概率密度函数$p(\pmb{x}|c)\sim \mathcal{N}(\pmb{\mu}_c,\pmb{\sigma}^2_c)$，则参数$\pmb{\mu}_c$和$\pmb{\sigma}^2_c$的极大似然估计为
  $$
  \begin{aligned}
  \hat{\pmb{\mu}}_c&=\frac{1}{|D_c|}\sum_{\pmb{x}\in D_c}\pmb{x}\\
  \hat{\pmb{\sigma}}^2_c&=\frac{1}{|D_c|}\sum_{\pmb{x}\in D_c}(\pmb{x}-\hat{\pmb{\mu}}_c)(\pmb{x}-\hat{\pmb{\mu}}_c)^T
  \end{aligned}
  $$
  通过极大似然法得到的正态分布均值就是样本的均值，方差就是$(\pmb{x}-\hat{\pmb{\mu}}_c)(\pmb{x}-\hat{\pmb{\mu}}_c)^T$的均值，这显然是一个符合直觉的结果

* 这种参数化的方法虽然是类条件概率估计变得相对简单，但是估计结果的准确性严重依赖于所假设的概率分布形式是否符合潜在的真实数据分布

## 7.3.朴素贝叶斯分类器

* 基于贝叶斯公式来估计后验概率$P(c|\pmb{x})$的主要困难在于：类条件概率$P(\pmb{x}|c)$是所有属性上的联合概率，难以从有限的训练样本直接估计而得
  
1. 基于有限训练样本直接估计联合概率，在计算上将会遭遇组合爆炸问题，在数据上将会遭遇样本稀疏问题；属性数越多，问题越严重
  
* 朴素贝叶斯分类器(naive Bayes classifier)采用了属性条件独立性假设(attribute conditional independence assumption)：对已知类别，假设所有的属性相互独立

* 假设每个属性独立地对分类结果发生影响

* 基于属性条件独立性假设，贝叶斯公式重写为
  $$
  P(c|\pmb{x})=\frac{P(c)P(\pmb{x}|c)}{P(\pmb{x})}=\frac{P(c)}{P(\pmb{x})}\prod^d_{i=1}P(x_i|c)
  $$
  其中$d$为属性数目，$x_i$为$\pmb{x}$在第$i$个属性上的取值

* 由于对所有类别来说$P(\pmb{x})$相同，因此基于$h^*(x)=\mathop{\arg\max}_{c\in\mathcal{Y}} P(c|\pmb{x})$的贝叶斯判定准则有
  $$
  h_{nb}(\pmb{x})=\mathop{\arg\max}_{c\in\mathcal{Y}}P(c)\prod^d_{i=1}P(x_i|c)
  $$
  这就是朴素贝叶斯分类器的表达式

* 朴素贝叶斯分类器的训练过程就是基于训练集$D$来估计类先验概率$P(c)$，并为每个属性估计条件概率$P(x_i|c)$

* 令$D_c$表示训练集$D$中第$c$类样本组成的集合，若有充分的独立同分布样本，则可容易地估计出类先验概率
  $$
  P(c)=\frac{|D_c|}{|D|}
  $$

* 对离散属性而言，令$D_{c,x_i}$表示$D_c$中在第$i$个属性上取值为$x_i$的样本组成的集合，则条件概率$P(x_i|c)$可估计为
  $$
  P(x_i|c)=\frac{|D_{c,x_i}|}{|D_c|}
  $$

* 对连续属性可考虑概率密度函数，假定$p(x_i|c)\sim \mathcal{N}(\mu_{c,i},\sigma^2_{c,i})$，其中$\mu_{c,i}$和$\sigma^2_{c,i}$分别是第$c$类样本在第$i$个属性上取值的均值和方差，则有
  $$
  p(x_i|c)=\frac{1}{\sqrt{2\pi}\sigma_{c,i}}\exp\Bigg(-\frac{(x_i-\mu_{c,i})^2}{2\sigma^2_{c,i}}\Bigg)
  $$

* 如果某个属性值在训练集中没有与某个类同时出现过，则直接基于条件概率$P(x_i|c)$进行概率估计，再根据朴素贝叶斯分类器的表达式进行判别将出现问题，这样会使连乘式计算出的概率值为零

* 为了避免其他属性携带的信息被训练集中未出现的的属性抹去，在估计概率值时通常要进行平滑(smoothing)，常用拉普拉斯修正(Laplacian correction)

* 具体来说，令$N$表示训练集$D$中可能的类别数，$N_i$表示第$i$个属性可能的取值数类先验概率$P(c)$和条件概率$P(x_i|c)$分别修正为
  $$
  \begin{aligned}
  \hat P(c)&=\frac{|D_c|+1}{|D|+N}\\
  \hat P(x_i|c)&=\frac{|D_{c,x_i}|+1}{|D_c|+N_i}\\
  \end{aligned}
  $$

* 拉普拉斯修正避免了因训练集样本不充分而导致概率估值为零的问题

* 拉普拉斯修正实质上假设了属性值与类别均匀分布，这是朴素贝叶斯学习过程中额外引入的关于数据的先验

* 在训练集变大时，修正过程所引入的先验(prior)的影响也会逐渐变得可忽略，使得估值渐趋向于实际概率值

* 现实任务中朴素贝叶斯分类器有多种使用使用方式

  1. 若任务对预测速度要求较高，则对给定训练集，可将朴素贝叶斯分类器涉及的所有概率估值事先计算好存储起来，在进行预测时只需查表即可进行判别
  2. 若任务数据更替频繁，则可采用懒惰学习(lazy learning)方式，先不进行任何训练，待收到预测请求时再根据当前的数据集进行概率估值
  3. 若数据集不断增加，则可在现有估值基础上，仅对新增样本的属性值所涉及的概率估值进行计数修正即可实现增量学习