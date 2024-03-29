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
      E(f;D)=\frac{1}{m} \sum_{{i=1}}^m(f(\pmb x_i)-y_i)^2
      $$
  
   2. 对于数据分布$\mathcal{D}$和概率密度函数$p(·)$，均方误差的描述为 
      $$
      E(f;D)=\int{_{\pmb{x}\sim{}\mathcal{D}}(f(\pmb{x})-y)^2p(\pmb{x})d\pmb{x}}
      $$

  ---

### 2.3.1.错误率与精度

* 错误率是分类错误的样本数占样本总数的比例

* 精度是分类正确的样本数占样本总数的比例

  ---

   1. 样例集$$D$$分类错误率定义为
      $$
      E(f;D)=\frac{1}{m} \sum\limits_{{i=1}}^m \mathbb I(f(\pmb{x}_i) \neq y_i)
      $$
      精度定义为
      $$
      \begin{equation}
      	\begin{aligned}acc(f;D)&=\frac{1}{m} \sum\limits_{{i=1}}^m \mathbb I(f(\pmb{x}_i) = y_i) \\
      	&= 1 - E(f;D)
      	\end{aligned}
      \end{equation}
      $$
      
   2. 对于数据分布$\mathcal{D}$和概率密度函数$p(·)$​，错误率定义为
      $$
      E(f;D)=\int{_{\pmb{x}\sim{}\mathcal{D}} \mathbb I(f(\pmb{x}) \neq y)p(\pmb{x})d\pmb{x}}
      $$
      精度定义为
      $$
      \begin{equation}
      	\begin{aligned}
      	acc(f;D)&=\int_{\pmb{x}\sim{}\mathcal{D}}\mathbb I(f(\pmb{x}) = y)p(\pmb{x})d\pmb{x} \\
        &=1 - E(f;D)
      	\end{aligned}
      \end{equation}
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
  \begin{equation}
  	\begin{aligned}
  	E(f;D;cost)&=\frac{1}{m}\Bigg(\sum\limits_{\pmb{x}_i\in D^+}\mathbb I f(\pmb{x}_i)\neq y_i \times cost_{01}\\
  	&+\sum\limits_{\pmb{x}_i\in D^-}\mathbb I(f(\pmb{x}_i) \neq y_i) \times cost_{10}\Bigg)
  	\end{aligned}
  \end{equation}
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

* 使用统计假设检验(hypothesis test)进行学习器性能的比较

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

## 7.4.半朴素贝叶斯分类器

* 为了降低贝叶斯公式中估计后验概率$P(c|\pmb{x})$的困难，朴素贝叶斯分类器采用了属性条件独立性假设，但在现实任务中这个往往假设很难成立

* 于是，人们尝试对属性条件独立性假设进行一定程度的放松，由此产生了一类称为半朴素贝叶斯分类器(semi-naive Bayes classifiers)的学习方法

* 半朴素贝叶斯分类器的基本思想

  1. 适当考虑一部分属性间的相互依赖信息，从而既不需要进行完全联合概率计算，又不至于彻底忽略了比较强的属性依赖关系

  2. 独依赖估计(One-Dependent Estimator，简称ODE)是半朴素贝叶斯分类器最常用的一种策略

  3. 所谓独依赖就是假设每个属性在类别之外最多仅依赖于一个其他属性，即
     $$
     P(c|\pmb{x})\propto P(c)\prod^d_{i=1}P(x_i|c,pa_i)
     $$
     其中$pa_i$为属性$x_i$所依赖的属性，称为$x_i$的父属性；此时，对每个属性$x_i$，若其父属性$pa_i$已知，则可采用类似拉普拉斯修正后的条件概率来估计概率值$P(x_i|c,pa_i)$

* 问题的关键就转化为如何确定每个属性的父属性，不同的做法产生不同的独依赖分类器

  1. 最直接的做法是假设所有的属性都依赖于同一个属性，称为超父(super-parent)，通过交叉验证等模型选择方法来确定超父属性，由此形成了SPODE(super-parent ODE)方法

  2. TAN(Tree Augmented naive Bayes)则是在最大带权生成树(maximum weighted spanning tree)算法的基础上，通过一下步骤将属性间依赖关系简约为树形结构

     * 计算任意两个属性之间的条件互信息(conditional mutual information)
       $$
       I(x_i,x_j|\ y)=\sum_{x_i,x_j;c\in\mathcal{Y}}P(x_i,x_j|c)\log\frac{P(x_i,x_j|c)}{P(x_i|c)P(x_j|c)}
       $$

     * 以属性为结点构建完全图，任意两个结点之间边的权重设为$I(x_i,x_j|\ y)$

     * 构建此完全图的最大带权生成树，挑选根变量，将边置为有向

     * 加入类别节点$y$，增加从$y$到每个属性的有向边

     容易看出，条件互信息$I(x_i,x_j|y)$刻画了属性$x_i$和$x_j$在已知类别情况下的相关性，因此，通过最大生成树算法，TAN实际上仅保留了强相关属性之间的依赖性

  3. AODE(Averaged One-Dependent Estimator)是一种基于集成学习机制、更为强大的独依赖分类器

     * 与SPODE通过模型选择确定超父属性不同，AODE尝试将每个属性作为超父类来构建SPODE，然后将那些具有足够数据支撑的SPODE集成起来作为最终结果，即
       $$
       P(c|\pmb{x})\propto\sum^d_{\ \ \ i=1\\|D_{x_i}|\ge m'}P(c,x_i)\prod^d_{j=1}P(x_j|c,x_i)
       $$
       其中$D_{x_i}$是在第$i$个属性上取值为$x_i$的样本的集合，$m'$为阈值常数

     * AODE需估计$P(c,x_i)$和$P(x_j|c,x_i)$， 有
       $$
       \begin{aligned}
       \hat P(c,x_i)&=\frac{|D_{c,x_i}|+1}{|D|+N\times N_i}\\
       \hat P(x_j|c,x_i)&=\frac{|D_{c,x_i,x_j}|+1}{|D_{c,x_i}|+N_j}\\
       \end{aligned}
       $$
       其中$N$是$D$中可能的类别数，$N_i$是第$i$个属性可能的取值数，$D_{c,x_i}$是类别为$c$且在第$i$个属性上取值为$x_i$的样本集合，$D_{c,x_i,x_j}$是类别为$c$且在第$i$和第$j$个属性上取值分别为$x_i$和$x_j$的样本集合

     * 与朴素贝叶斯分类器类似，AODE的训练过程也是计数，即在训练数据集上对符合条件的样本进行技术的过程

     * AODE无需模型选择，既能通过预计算节省预测时间，也能采取懒惰学习方式在预测时再进行计数，并且易于实现增量学习

* 属性条件独立性假设放松为独依赖假设可能获得泛化性能的提升

* 高阶依赖，即对多个属性依赖

* 可将$P(c|\pmb{x})\propto P(c)\prod^d_{i=1}P(x_i|c,pa_i)$中的属性$pa_i$替换为包含$k$个属性的集合$\textbf{pa}_i$，从而将ODE拓展为kDE

* 随着$k$的增加，准确估计概率$P(x_i|y,\textbf{pa}_i)$所需的训练样本数量将以指数级增加

* 训练数据非常充分，泛化性能有可能提升；但在有限的样本条件下，则又陷入估计高阶联合概率的泥沼

## 7.5.贝叶斯网

* 贝叶斯网(Bayesian network)亦称信念网(belief network)，它借助有向无环图(Directed Acyclic Graph，简称DAG)来刻画属性之间的依赖关系，并使用条件概率表(Conditional Probability Table，简称CPT)来描述属性的联合概率分布
* 贝叶斯网是一种经典的概率图模型
* 一个贝叶斯网$B$由结构$G$和参数$\Theta$两部分构成，即$B=\langle G,\Theta\rangle$
* 网络结构$G$是一个有向无环图，其每个结点对应一个属性，若两个属性有直接的依赖关系，则它们由一条边连接起来
* 参数$\Theta$定量描述这种依赖关系，假设属性$x_i$在$G$中的父节点集为$\pi_i$，则$\Theta$包含了每个属性的条件概率表$\theta_{x_i|\pi_i}=P_B(x_i|\pi_i)$

### 7.5.1.结构

* 贝叶斯网结构有效地表达了属性间的条件独立性

* 给定父结点集，贝叶斯网假设每个属性与它的非后裔结点独立，于是$B=\langle G,\Theta\rangle$将属性$x_1,x_2,...,x_d$的联合概率分布定义为
  $$
  P_B(x_1,x_2,...,x_d)=\prod^d_{i=1}P_B(x_i|\pi_i)=\prod^d_{i=1}\theta_{x_i|\pi_i}
  $$

* 贝叶斯网中三个变量之间的典型依赖关系

  1. 同父(common parent)结构，给定父结点$x_1$的取值，则$x_2$与$x_3$条件独立

  2. 顺序结构，给定$x$的值，则$y$与$z$条件独立

  3. V型结构(V-structure)亦称冲撞结构，给定$x_3$的取值，$x_1$与$x_2$必不独立；奇妙的是，若$x_3$的取值完全未知，则V型结构下$x_1$与$x_2$却是相互独立的
     $$
     \begin{equation}
     	\begin{aligned}
         P(x_1,x_2)
         &=\sum_{x_3}P(x_1,x_2,x_3)\\
         &=\sum_{x_3}P(x_3|x_1,x_2)P(x_1)P(x_2)\\
         &=P(x_1)P(x_2)
     	\end{aligned}
     \end{equation}
     $$
     这样的独立性称为边际独立性(marginal independence)，记为$x_1\perp\!\!\!\perp x_2$

     > [V型结构独立性证明](https://www.zhihu.com/question/58386856)

* 一个变量取值的确定与否，能对另两个变量间的独立性发生影响，这个现象并非与V型结构所特有的

  1. 在同父结构中，条件独立性$x_2\perp x_3|x_1$成立，但若$x_1$的取值未知，则$x_2$和$x_3$就不独立，即$x_2\perp\!\!\!\perp x_3$不成立
  2. 在顺序结构中，$y\perp z|x$，但$y\perp\!\!\!\perp z$不成立

* 使用有向分离(D-separation)分析有向图中变量间的条件独立性

* 将有向图转变为一个无向图：

  * 找出有向图中的所有V型结构，在V型结构的两个父结点之间加上一条无向边
  * 将所有有向边改为无向边

* 由此产生无向图称为道德图(moral graph)，令父节点相连的过程称为道德化(moralization)

* 道德化的蕴义：孩子的父母应建立牢靠的关系，否则是不道德的

* 基于道德图能直观、迅速地找到变量间的条件独立性

* 假定道德图中有变量$x$，$y$和变量集合$\pmb{z}=\{z_i\}$，若变量$x$和$y$能在图上被$\pmb{z}$，即从道德图中将变量集合$\pmb{z}$去除后，$x$和$y$分属两个连通分支，则称变量$x$和$y$被$\pmb{z}$有向分离，$x\perp y|\pmb{z}$成立

### 7.5.2.学习

* 若网络结构已知，即属性间的依赖关系已知，则贝叶斯网的学习过程相对简单，只需通过对训练样本计数，估计出每个结点的条件概率表即可

* 在现实应用中我们往往不知晓网络结构，于是，贝叶斯网学习的首要任务就是根据训练数据集来找出结构最恰当的贝叶斯网

* 评价搜索是求解这一问题的常用办法

  1. 先定义一个评分函数(score function)，以此来估计贝叶斯网与训练数据的契合程度
  2. 然后基于这个评分函数来寻找最优的贝叶斯网

* 评分函数的引入了关于我们希望获得什么样的贝叶斯网的归纳偏好

* 常用的评分函数通常基于信息论准则，此类准则将学习问题看作一个数据压缩任务

* 学习的目标是找到一个能以最短编码长度描述训练数据的模型，此时编码的长度包括了描述模型自身所需的编码长度和使用该模型描述数据所需的编码的位数

* 对于贝叶斯网学习而言，模型就是一个贝叶斯网，每个贝叶斯网描述了一个在训练数据上的概率分布，自有一套编码机制能使那些经常出现的样本有更短的编码

* 我们应该选择那个综合编码长度（包括描述网络和编码数据）最短的贝叶斯网，这就是最小描述长度(Minimal Description Length, 简称MDL)准则

* 给定训练集$D=\{\pmb{x}_1,\pmb{x}_2,...,\pmb{x}_m\}$，贝叶斯网$B=\langle G,\Theta\rangle$在$D$上评分函数可写为
  $$
  s(B|D)=f(\theta)|B|-LL(B|D)
  $$
  其中，$|B|$是贝叶斯网的参数个数；$f(\theta)$表示描述每个参数$\theta$所需要的编码位数；而
  $$
  LL(B|D)=\sum^m_{i=1}\log P_B(\pmb{x}_i)
  $$
  是贝叶斯网$B$的对数似然

* 显然，评分函数的第一项是计算编码贝叶斯网$B$所需的编码位数，第二项是计算$B$所对应的概率分布$P_B$对$D$描述得有多好

* 学习任务就转换成一个优化任务，即寻找一个贝叶斯网$B$使评价函数$s(B|D)$最小

* 若$f(\theta)=1$，即每个参数用1编码位描述，则得到AIC(Akaike Information Criterion)评分函数

  $$
  AIC(B|D)=|B|-LL(B|D)
  $$
  若$f(\theta)=\frac12\log m$，即每个参数用$\frac12\log m$编码位描述，则得到BIC(Bayesian Information Criterion)评分函数
  $$
  BIC(B|D)=\frac{\log m}{2}|B|-LL(B|D)
  $$
  显然，若$f(\theta)=0$，即不计算对网络进行编码长度描述，则评分函数退化为负对数似然，相应的，学习任务退化为极大似然估计

* 若贝叶斯网$B=\langle G,\Theta\rangle$的网络结构$G$固定，则评分函数$s(B|D)$的第一项为常数

* 最小化$s(B|D)$等价于对参数$\Theta$的极大似然估计

* 由贝叶斯网的对数似然和属性的联合概率分布可知，参数$\theta_{x_i|\pi_i}$能直接在训练数据$D$上通过经验估计获得，即
  $$
  \theta_{x_i|\pi_i}=\hat P_D(x_i|\pi_i)
  $$
  其中$\hat P_D(·)$是$D$上的经验分布，即事件在训练数据上出现的频率；为了最小化评分函数$s(B|D)$，只需对网络结构进行搜索，而候选结构的最优参数可直接在训练集上计算得到

* 从所有可能的网络结构空间搜索最优贝叶斯网结构是一个$NP$难问题，有两种常用的策略能在有限时间内求得近似解：
  1. 贪心法，从某个网络结构出发，每次调整一条边(增加、删除或调整方向)，直到评分函数值不再降低为止
  2. 通过给网络结构施加约束来消减搜索空间，将网络结构限定为树形结构(TAN将结构限定为树形，半朴素贝叶斯分类器可看作贝叶斯网的特例)

### 7.5.3.推断

* 贝叶斯网训练好之后就能用来回答查询(query)，即通过一些属性变量的观测值来推测其他属性变量的取值

* 通过已知变量观测值来推测待查询变量的过程称为推断(inference)，已知变量观测值称为证据(evidence)

* 最理想的是直接根据贝叶斯网定义的联合概率分布来精确计算后验概率，这样的精确推断已被证明是$NP$难问题；在网络结点较多、连接稠密时，难以进行精确推断，此时需要近似推断，通过降低精度要求，在有限时间内求得近似解

* 贝叶斯网的近似推断通常使用吉布斯采样(Gibbs sampling)来完成，这是一种随机采样方法

* 令$\pmb{Q}=\{Q_1,Q_2,...,Q_n\}$表示待查询变量，$\pmb{E}=\{E_1,E_2,...E_k\}$为证据变量，已知其取值为$\pmb{e}=\{e_1,e_2,...,e_k\}$

* 目标是计算后验概率$P(\pmb{Q}=\pmb{q}|\pmb{E}=\pmb{e})$，其中$\pmb{q}=\{q_1,q_2,...,q_n\}$是待查询变量的一组取值

* 吉布斯采样算法

  ---

  输入：贝叶斯网$B=\langle G,\Theta\rangle$;

  ​			采样次数$T$;

  ​			证据变量$\pmb{E}$及其取值$\pmb{e}$;

  ​			待查询变量$\pmb{Q}$及其取值$\pmb{q}$.

  过程：

  1:$n_q=0$

  2:$\pmb{q}^0=$对$\pmb{Q}$随机赋初值

  3:$\pmb{for}\ \ t=1,2...,T\ \ \pmb{do}$

  4:	$\pmb{for}\ \ Q_i\in\pmb{Q} \ \ \pmb{do}$

  5:		$\pmb{Z}=\pmb{E}\cup\pmb{Q}\backslash\{Q_i\}$;

  6:		$\pmb{z}=e\cup\pmb{q}^{t-1}\backslash\{q^{t-1}_i\}$;

  7:		根据$B$计算分布$P_B(Q_i|\pmb{Z}=\pmb{z})$;

  8:		$q^t_i=$根据$P_B(Q_i|\pmb{Z}=\pmb{z})$采样所获$Q_i$取值;

  9:		$\pmb{q}^t=$将$\pmb{q}^{t-1}$中的$q^{t-1}_i$用$q^t_i$替换

  10:	$\pmb{end \ \ for}$

  11: 	$\pmb{if} \ \pmb{q}^t=\pmb{q} \ \pmb{then}$

  12:		$n_q=n_q+1$

  13:	$\pmb{end \ \ if}$

  14:$\pmb{end\ \ for}$

  输出：$P(\pmb{Q}=\pmb{q}|\pmb{E}=\pmb{e})\simeq\frac{n_q}{T}$

  ---

* 吉布斯采样算法先随机产生一个与证据$\pmb{E}=\pmb{e}$一致的样本$\pmb{q}^0$作为初始点，然后每步从当前样本出发产生下一个样本

* 具体来说，在第$t$次采样中，算法先假设$\pmb{q}^t=\pmb{q}^{t-1}$，然后对非证据变量逐个进行采样改变其取值，采样概率根据贝叶斯网$B$和其他变量的当前取值(即$\pmb{Z}=\pmb{z}$)计算获得

* 假定经过$T$次采样得到的与$\pmb{q}$一致的样本共有$n_q$个，则可近似估算出后验概率
  $$
  P(\pmb{Q}=\pmb{q}|\pmb{E}=\pmb{e})\simeq\frac{n_q}{T}
  $$

* 实质上，吉布斯采样是在贝叶斯网所有变量的联合状态空间与证据$\pmb{E}=\pmb{e}$一致的字空间中进行随机漫步(random walk)

* 每一步仅依赖于前一步的状态，这是一个马尔可夫链(Markov chain)

* 在一定条件下，无论从什么初始状态开始，马尔可夫链第$t$步的状态分布在$t\rightarrow\infty$时必收敛于一个平稳分布(stationary distribution)；对于吉布斯采样来说，这个分布恰好是$P(\pmb{Q}|\pmb{E}=\pmb{e})$

* 因此，在$T$很大时，吉布斯采样相当于根据$P(\pmb{Q}|\pmb{E}=\pmb{e})$采样，从而保证了$P(\pmb{Q}|\pmb{E}=\pmb{e})\simeq\frac{n_q}{T}$收敛于$P(\pmb{Q}=\pmb{q}|\pmb{E}=\pmb{e})$

* 马尔可夫链通常需很长时间才能趋于平稳分布，因此吉布斯采样算法的收敛速度较慢

* 若贝叶斯网中存在极端概率0或1，则不能保证马尔可夫链存在平稳分布，此时吉布斯采样会给出错误的估计结果

## 7.6.EM算法

* 之前的讨论是假设训练样本所有属性变量的值都已被观测到，即训练样本是完整的

* 现实应用中往往会遇到不完整的训练样本

* 未观测变量的学名是隐变量(latent variable)，令$\pmb{\text{X}}$表示已观测变量集，$\pmb{\text{Z}}$表示隐变量集，$\Theta$表示模型参数

* 若预对$\Theta$做极大似然估计，则应最大化对数似然
  $$
  LL(\Theta|\pmb{\text{X}},\pmb{\text{Z}})=\ln P(\pmb{\text{X}},\pmb{\text{Z}}|\Theta)
  $$
  然而由于$\pmb{\text{Z}}$是隐变量，上式无法直接求解，此时可通过对$\pmb{\text{Z}}$计算期望，来最大化已观测数据的对数边际似然(marginal likelihood)
  $$
  LL(\Theta|\pmb{\text{X}})=\ln P(\pmb{\text{X}}|\Theta)=\ln\sum_{\pmb{\text{Z}}}P(\pmb{\text{X}},\pmb{\text{Z}}|\Theta)
  $$

* 

* EM(Expectation-Maximization)期望最大化算法是常用的估计参数隐变量的利器，它是一种迭代式的方法，其基本思想是：若参数$\Theta$已知，则可根据训练数据推断出最优隐变量$\pmb{\text{Z}}$的值(E步)；反之，若$\pmb{\text{Z}}$的值已知，则可对参数$\Theta$做极大似然估计(M步)

  * 以初始化值$\Theta^0$为起点，对数据的对数边际似然，可迭代执行以下步骤直至收敛
    1. 基于$\Theta^t$推断隐变量$\pmb{\text{Z}}$的期望，记为$\pmb{\text{Z}}^t$
    2. 基于已观测变量$\pmb{\text{X}}$和$\pmb{\text{Z}}^t$对参数$\Theta$做极大似然估计，记为$\Theta^{t+1}$

  这就是EM算法的原型

  * 若不是取$\pmb{\text{Z}}$的期望，而是基于$\Theta^t$计算隐变量$\pmb{\text{Z}}$的概率分布$P(\pmb{\text{Z}}|\pmb{\text{X}},\Theta^t)$，则EM算法的两个步骤是：

    1. $\pmb{\text{E}}$步(Expextation)：以当前参数$\Theta^t$推断隐变量分布$P(\pmb{\text{Z}}|\pmb{\text{X}},\Theta^t)$，并计算对数似然$LL(\Theta|\pmb{\text{X}},\pmb{\text{Z}})$关于$\pmb{\text{Z}}$的期望
       $$
       Q(\Theta|\Theta^t)=\mathbb{E}_{\pmb{\text{Z}}|\pmb{\text{X}},\Theta^t}LL(\Theta|\pmb{\text{X}},\pmb{\text{Z}})
       $$

    2. $\pmb{\text{M}}$步(Maximization)：寻找参数最大化期望似然，即
       $$
       \Theta^{t+1}=\mathop{\arg\max}_\Theta Q(\Theta|\Theta^t)
       $$

* EM算法使用两个步骤交替计算：

  1. 第一步是期望(E)步，利用当前估计的参数值来计算似然的期望值
  2. 第二步是最大化(M)步，寻找能使E步产生的似然期望最大化的参数值
  3. 然后，新得到的参数值重新被用于E步，······直至收敛到局部最优解

* 隐变量估计问题可以通过梯度下降等优化算法求解，但由于求和的项数将随着隐变量的数目以指数级上升，会给梯度计算带来麻烦

* EM算法则可看作一种非梯度优化方法

# 第八章 集成学习

## 8.1.个体与集成

* 集成学习(ensemble learning)通过构建并结合多个学习器来完成学习任务，有时也被称为多分类器系统(multi-classifier system)、基于委员会的学习(committee-based learning).

* 集成学习的一般结构：先产生一组个体学习器(individual learner)，再用某种策略将它们结合起来.

  1. 个体学习器通常由一个现有的学习算法从训练数据产生，例如$\text{C4.5}$决策树算法、$\text{BP}$神经网络算法等，此时集成中只包含同种类型的个体学习器，这样的集成是同质的(homogeneous). 
  2. 同质集成中的个体学习器亦称基学习器(base learner)，相应的学习算法称为基学习算法(base learning algorithm).
  3. 集成也可包含不同类型的个体学习器，例如同时包含决策树和神经网络，这样的集成是异质的(heterogeneous).
  4. 异质集成中的个体学习器由不同的学习算法生成，这时就不再有基学习算法；相应的个体学习器常被称为组件学习器(component learner)或直接称为个体学习器.

* 集成学习通过将多个学习器进行结合，常可获得比单一学习器显著优越的泛化性能.

* 弱学习器(weak learner)常指泛化性能略优于随机猜测的学习器.

* 虽然从理论上来说使用弱学习器集成足以获得更好的性能，但在实践中出于种种考虑，例如希望使用较少的个体学习器，或是重用关于常见学习器的一些经验等，人们往往会使用比较强的学习器.

* 要获得好的集成，个体学习器应好而不同，即个体学习器要有一定的准确性，个体学习器至少不差于弱学习器，并且要有多样性(diversity)，即学习器间具有差异.

* 集成学习简单的分析，考虑二分类问题$y\in\{-1, +1\}$和真实函数$f$，假定基分类器的错误率为$\epsilon$，即对每个基分类器$h_i$有
  $$
  P(h_i(\pmb{x})\neq f(\pmb{x}))=\epsilon
  $$
  假设集成通过简单投票法结合$T$个基分类器(假设$T$为奇数)，若有超过半数的基分类器正确，则集成分类就正确
  $$
  H(\pmb{x})=\text{sign}\bigg(\sum^T_{i=1}h_i(\pmb{x})\bigg)
  $$
  假设基分类器的错误率相互独立，则由$\text{Hoeffding}$不等式可知，集成的错误率为
  $$
  \begin{equation}
  	\begin{aligned}
      P(H(\pmb{x})\neq f(\pmb{x}))&=\sum^{\lfloor T/2 \rfloor}_{k=0}\binom{T}{k}(1-\epsilon)^k\epsilon^{T-k}\\
      &\leq \exp\bigg(-\frac{1}{2}T(1-2\epsilon)^2\bigg)\\
  	\end{aligned}
  \end{equation}
  $$
  上式显示出，随着集成中个体分类器数目$T$的增大，集成的错误率将指数级下降，最终趋向于零.

  > [集成学习分析](https://blog.csdn.net/y492235584/article/details/85265035)

* 然而上面的分析有一个关键假设：基学习器的误差相互独立；在现实任务中，个体学习器是为解决同一个问题训练的，它们显然不相互独立.

* 准确率很高之后，要增加多样性就需牺牲准确性.

* 集成学习研究的核心是：如何产生并结合好而不同的个体学习器.

* 根据个体学习器生成方式，目前的集成学习方法可分为两类

  1. 个体学习器间存在强依赖关系、必须串行生成的序列化方法，例如$\text{Boosting}$.
  2. 个体学习器间不存在强依赖关系、可同时生成的并行化方法，例如$\text{Bagging}$和随机森林$\text{Random Forest}$.

## 8.2.Boosting

* Boosting是一族可将弱学习器提升为强学习器的算法.

* Boosting族算法的工作机制类似：先从初始训练集训练一个基学习器，再根据基学习器的表现对训练样本分布进行调整，使得先前的基学习器做错的训练样本在后续受到更多的注，然后基于调整后的样本分布来训练下一个基学习器；如此重复进行，直至基学习器数目达到事先指定的值$T$，最终将这$T$个基学习器进行加权结合.

* Boosting族算法最著名的代表是AdaBoost.

* AdaBoost算法

  ---

  <b>输入</b>：训练集$D=\{(\pmb{x}_1, y_1), (\pmb{x}_2, y_2),..., (\pmb{x}_m, y_m)\}$;

  ​			基学习算法$\mathfrak{L}$;

  ​			训练轮数$T$.

  <b>过程</b>：

  1:$\mathcal{D}_1(\pmb{x})=1/m$.

  2:<b>for</b> $t=1,2,...,T$ <b>do</b>

  3:	$h_t=\mathfrak{L}(\mathcal{D}, \mathcal{D}_t)$;

  4:	$\epsilon_t=P_{\pmb{x}～\mathcal{D}_t}(h_t(x)\ne f(\pmb{x}))$;

  5:	<b>if</b>  $\epsilon_t>0.5$  <b>then break</b>

  6:	$\alpha_t=\frac{1}{2}\ln\big(\frac{1-\epsilon_t}{\epsilon_t}\big)$;

  7: 	$\mathcal{D}_{t+1}(\pmb{x})=\frac{\mathcal{D}_t(\pmb{x})}{Z_t}\times\big\{^{\exp(-\alpha_t),\ \  \text{if}\ h_t(\pmb{x})=f(\pmb{x})}_{\exp(\alpha_t),\ \ \ \ \text{if}\ h_t(\pmb{x})\ne f(\pmb{x})}$

  ​						$=\frac{\mathcal{D}_t(\pmb{x})\exp(-\alpha_tf(\pmb{x})h_t(\pmb{x}))}{Z_t}$

  8:<b>end for</b>

  <b>输出</b>：$H(\pmb{x})=\text{sign}\Big(\sum^T_{t=1}\alpha_th_t(\pmb{x})\Big)$

  ---

  * 其中$y_i\in\{-1, +1\}$，$f$是真实函数
  * 基于分布$\mathcal{D}_t$从数据集$D$中训练出分类器$h_t$
  * $\epsilon_t$是$h_t$的误差
  * $\alpha_t$是分类器$h_t$的权重
  * $Z_t$是规范化因子，以确保$\mathcal{D}_{t+1}$是一个分布

* AdaBoost算法有多种推导方式，比较容易理解的是居于加性模型(additive model)，即基学习器的线性组合
  $$
  H(\pmb{x})=\sum^T_{t=1}\alpha_t h_t(\pmb{x})
  $$
  来最小化指数损失函数(exponential loss function)
  $$
  \ell_{\exp}(H|\mathcal{D})=\mathbb{E}_{\pmb{x}～\mathcal{D}}[e^{-f(\pmb{x})H(\pmb{x})}]
  $$
  若$H(\pmb{x})$能令指数损失函数最小化，则上式对$H(\pmb{x})$的偏导
  $$
  \frac{\partial\ell_{\exp(H|\mathcal{D})}}{\partial H(\pmb{x})}=-e^{-H(\pmb{x})}P(f(\pmb{x})=1|\pmb{x})+e^{H(\pmb{x})}P(f(\pmb{x})=-1|\pmb{x})
  $$
  令上式为零可解得
  $$
  H(\pmb{x})=\frac{1}{2}\ln\frac{P(f(\pmb{x})=1|\pmb{x})}{P(f(\pmb{x})=-1|\pmb{x})}
  $$
  因此，有
  $$
  \begin{equation}
  	\begin{aligned}
      \text{sign}\big(H(\pmb{x})\big)&=\text{sign}\bigg(\frac{1}{2}\ln\frac{P(f(\pmb{x})=1|\pmb{x})}{P(f(\pmb{x})=-1|\pmb{x})}\bigg)\\
      &=\begin{cases}
      1,& P(f(\pmb{x})=1|\pmb{x})>P(f(\pmb{x})=-1|\pmb{x})\\
      -1,& P(f(\pmb{x})=1|\pmb{x})<P(f(\pmb{x})=-1|\pmb{x}) \\
      \end{cases}\\
      &=\mathop{\arg\max}_{y\in\{-1,1\}}P(f(\pmb{x})=y|\pmb{x})
  	\end{aligned}
  \end{equation}
  $$
  这意味着$\text{sign}\big(H(\pmb{x})\big)$达到了贝叶斯最优错误率.

* 若指数损失函数最小化，则分类错误率也将最小化；这说明指数损失函数是分类任务原本$0/1$损失函数的一致的(consistent)替代损失函数。由于它是连续可微函数，因此我们用它代替$0/1$损失函数作为优化目标.

* 在AdaBoost算法中，第一个基学习器$h_1$是通过直接将基学习算法用于初始数据分布而得；此后迭代地生成$h_t$和$\alpha_t$，当基分类器$h_t$基于分布$\mathcal{D}_t$产生后，该基分类器的权重$\alpha_t$应使得$\alpha_th_t$最小化指数损失函数
  $$
  \begin{equation}
  	\begin{aligned}
      \ell_{\exp}(\alpha_th_t|\mathcal{D}_t)&=\mathbb{E}_{\pmb{x}～\mathcal{D}_t}\Big[e^{-f(\pmb{x})\alpha_th_t(\pmb{x})}\Big]\\
      &=\mathbb{E}_{\pmb{x}～\mathcal{D}_t}\big[e^{-\alpha_t}\mathbb{I}(f(\pmb{x})=h_t(\pmb{x}))+e^{\alpha_t}\mathbb{I}(f(\pmb{x})\ne h_t(\pmb{x}))\big]\\
      &=e^{-\alpha_t}P_{\pmb{x}～\mathcal{D}_t}(f(\pmb{x})=h_t(\pmb{x}))+e^{\alpha_t}P_{\pmb{x}～\mathcal{D}_t}(f(\pmb{x})\ne h_t(\pmb{x}))\\
      &=e^{-\alpha_t}(1-\epsilon_t)+e^{\alpha_t}\epsilon_t
  	\end{aligned}
  \end{equation}
  $$
  其中$\epsilon_t=P_{\pmb{x}～\mathcal{D}_t}(f(\pmb{x})\ne h_t(\pmb{x}))$

  考虑指数损失函数的导数
  $$
  \frac{\partial\ell_{\exp}(\alpha_th_t|\mathcal{D}_t)}{\partial \alpha_t}=-e^{-\alpha_t}(1-\epsilon_t)+e^{\alpha_t}\epsilon_t
  $$
  令上式为零可解得
  $$
  \alpha_t=\frac{1}{2}\ln\Big(\frac{1-\epsilon_t}{\epsilon_t}\Big)
  $$
  这恰是AdaBoost算法的分类器权重更新公式

* AdaBoost算法在获得$H_{t-1}$之后样本分布将进行调整，使下一轮的基学习器$h_t$能纠正$H_{t-1}$的一些错误，理想的$h_t$能纠正$H_{t-1}$的全部错误，即最小化
  $$
  \begin{equation}
  	\begin{aligned}
      \ell_{\exp}(H_{t-1}+h_t|\mathcal{D})&=\mathbb{E}_{\pmb{x}～\mathcal{D}_t}\big[e^{-f(\pmb{x})(H_{t-1}(\pmb{x})+h_t(\pmb{x}))}\big]\\
      &=\mathbb{E}_{\pmb{x}～\mathcal{D}_t}\big[e^{-f(\pmb{x})H_{t-1}(\pmb{x})}e^{-f(\pmb{x})h_t(\pmb{x})}\big]
  	\end{aligned}
  \end{equation}
  $$
  注意到$f^2(\pmb{x})=h^2_t(\pmb{x})=1$, 上式可使用$e^{-f(\pmb{x})h_t(\pmb{x})}$的泰勒展式近似为
  $$
  \begin{equation}
  	\begin{aligned}
      \ell_{\exp}(H_{t-1}+h_t|\mathcal{D})
      &=\mathbb{E}_{\pmb{x}～\mathcal{D}_t}\bigg[e^{-f(\pmb{x})H_{t-1}(\pmb{x})}\bigg(1-f(\pmb{x})h_t(\pmb{x})+\frac{f^2(\pmb{x})h^2_t(\pmb{x})}{2}\bigg)\bigg]\\
      &\simeq\mathbb{E}_{\pmb{x}～\mathcal{D}_t}\bigg[e^{-f(\pmb{x})H_{t-1}(\pmb{x})}\bigg(1-f(\pmb{x})h_t(\pmb{x})+\frac{1}{2}\bigg)\bigg]\\
  	\end{aligned}
  \end{equation}
  $$
  理想的基学习器
  $$
  \begin{equation}
  	\begin{aligned}
      h_t(\pmb{x})
      &=\mathop{\arg\min}_h\ell_{\exp}(H_{t-1}+h\ |\ \mathcal{D})\\
      &=\mathop{\arg\min}_h\mathbb{E}_{\pmb{x}～\mathcal{D}}\bigg[e^{-f(\pmb{x})H_{t-1}(\pmb{x})}\bigg(1-f(\pmb{x})h_t(\pmb{x})+\frac{1}{2}\bigg)\bigg]\\
      &=\mathop{\arg\max}_h\mathbb{E}_{\pmb{x}～\mathcal{D}}\Big[e^{-f(\pmb{x})H_{t-1}(\pmb{x})}f(\pmb{x})h_t(\pmb{x})\Big]\\
      &=\mathop{\arg\max}_h\mathbb{E}_{\pmb{x}～\mathcal{D}}\Bigg[\frac{e^{-f(\pmb{x})H_{t-1}(\pmb{x})}}{\mathbb{E}_{\pmb{x}～\mathcal{D}}[e^{-f(\pmb{x})H_{t-1}(\pmb{x})}]}f(\pmb{x})h_t(\pmb{x})\Bigg]\\
  	\end{aligned}
  \end{equation}
  $$
  注意到$\mathbb{E}_{\pmb{x}～\mathcal{D}}[e^{-f(\pmb{x})H_{t-1}(\pmb{x})}]$是一个常数

  令$\mathcal{D}_t$表示一个分布
  $$
  \mathcal{D}_t(\pmb{x})=\frac{\mathcal{D}(\pmb{x})e^{-f(\pmb{x})H_{t-1}(\pmb{x})}}{\mathbb{E}_{\pmb{x}～\mathcal{D}}[e^{-f(\pmb{x})H_{t-1}(\pmb{x})}]}
  $$
  根据数学期望的定义，这等价于令
  $$
  \begin{equation}
  	\begin{aligned}
      h_t(\pmb{x})
      &=\mathop{\arg\max}_h\mathbb{E}_{\pmb{x}～\mathcal{D}}\Bigg[\frac{e^{-f(\pmb{x})H_{t-1}(\pmb{x})}}{\mathbb{E}_{\pmb{x}～\mathcal{D}}[e^{-f(\pmb{x})H_{t-1}(\pmb{x})}]}f(\pmb{x})h(\pmb{x})\Bigg]\\
      &=\mathop{\arg\max}_h\mathbb{E}_{\pmb{x}～\mathcal{D}_t}[f(\pmb{x})h(\pmb{x})]\\
  	\end{aligned}
  \end{equation}
  $$
  由$f(\pmb{x}),h(\pmb{x})\in\{-1, +1\}$，有
  $$
  f(\pmb{x})h(\pmb{x}) = 1 - 2\mathbb{I}(f(\pmb{x})\ne h(\pmb{x}))
  $$
  则理想的基学习器
  $$
  h_t(\pmb{x})=\mathop{\arg\min}_h[\mathbb{I}\big(f(\pmb{x})\ne h(\pmb{x})\big)]\\
  $$
  由此可见，理想的$h_t$将在分布$\mathcal{D}_t$下最小化分类误差

  因此，弱分类器将基于分布$\mathcal{D}_t$来训练，且针对$\mathcal{D}_t$的分类误差应小于0.5，这一定程度上类似于残差逼近的思想，考虑到$\mathcal{D}_t$和$\mathcal{D}_{t+1}$的关系，有
  $$
  \begin{equation}
  	\begin{aligned}
      \mathcal{D}_{t+1}(\pmb{x})&
      =\frac{\mathcal{D}(\pmb{x})e^{-f(\pmb{x})H_{t}(\pmb{x})}}{\mathbb{E}_{\pmb{x}～\mathcal{D}}[e^{-f(\pmb{x})H_{t}(\pmb{x})}]}\\
      &=\frac{\mathcal{D}(\pmb{x})e^{-f(\pmb{x})H_{t-1}(\pmb{x})}e^{-f(\pmb{x})a_{t}h_t(\pmb{x})}}{\mathbb{E}_{\pmb{x}～\mathcal{D}}[e^{-f(\pmb{x})H_{t}(\pmb{x})}]}\\
      &=\mathcal{D}_t(\pmb{x})·e^{-f(\pmb{x})a_{t}h_t(\pmb{x})}\frac{\mathbb{E}_{\pmb{x}～\mathcal{D}}[e^{-f(\pmb{x})H_{t-1}(\pmb{x})}]}{\mathbb{E}_{\pmb{x}～\mathcal{D}}[e^{-f(\pmb{x})H_{t}(\pmb{x})}]}
  	\end{aligned}
  \end{equation}
  $$
  这就是AdaBoost算法的样本分布更新公式

  > [更新准则推导](https://blog.csdn.net/BIT_666/article/details/80279844)

* Boosting算法要求基学习器能对特定的数据分布进行学习

  1. 这可通过重赋权法(re-weighting)实施，即在训练过程的每一轮中，根据样本分布为每个训练样本重新赋予一个权重.
  2. 对无法接受带权样本的基学习算法，则可通过重采样法(re-sampling)来处理，即在每一轮学习中，根据样本分布对训练重新进行采样，再用重采样而得的样本进行训练.
  3. 这两种做法没有显著的优劣差别.

* Boosting算法在训练的每一轮都要检查当前生成的基学习器是否满足基本条件

  1. 一旦条件不满足，则当前基学习器即被抛弃，且学习过程停止；在此种情形下，初始设置的学习轮数$T$也许还远未达到，可能导致最终集成中包含很少的基学习器而性能不佳.
  2. 若采用重采样法，则可获得重启动机会以避免训练过早停止，即在抛弃不满足条件的当前基学习器之后，可根据当前分布重新对训练样本进行采样，在基于新的采样结果重新训练出基学习器，从而使得学习过程可以持续到预设的$T$轮完成.

* 从偏差-方差分解的角度看，Boosting主要关注降低偏差，因此Boosting能基于泛化性能相当弱的学习器构建出很强的集成.

## 8.3.Bagging与随机森林

* 欲得到泛化性能强的集成，集成中的个体学习器应尽可能相互独立；虽然独立在现实任务中无法做到，但可以设法使基学习器尽可能具有较大的差异.
* 给定一个训练数据集，一种可能的做法是对训练样本进行采样，产生出若干个不同的子集，再从每个数据子集中训练样本进行采样，产生出若干个不同的子集，再从每个数据子集中训练出一个基学习器；这样由于训练数据不同，我们获得的基学习器可望具有比较大的差异.
* 为获得好的集成，我们同时还希望个体学习器不能太差，如果采样出的每个子集都完全不同，则每个学习器只用到了一小部分训练数据，甚至不足以进行有效学习，这显然无法确保产生出比较好的基学习器，因此，我们可考虑使用相互有交叠的采样子集.

### 8.3.1.Bagging

* Bagging是由 Bootstrap AGGregatING 缩写而来.

* Bagging是并行式集成学习方法，它基于自助采样法(bootstrap sampling).

* 给定包含$m$个样本的数据集使用自助采样法，可采样出$T$个含$m$个训练样本的采样集，然后基于每个采样集训练出一个集学习器，再将这些集学习器进行结合，这就是Bagging的基本流程.

* 在对预测输出进行结合时，Bagging 通常对分类任务使用简单投票法，对回归任务使用简单平均法. 若分类预测时出现两个类收到同样票数的情形，则最简单的做法是随机选择一个，也可进一步考察学习器投票的置信度.

* Bagging 算法描述

  ---

  <b>输入</b>：训练集$D=\{(\pmb{x}_1, y_1), (\pmb{x}_2, y_2),..., (\pmb{x}_m, y_m)\}$;

  ​			基学习算法$\mathfrak{L}$;

  ​			训练轮数$T$.

  <b>过程</b>：

  1:<b>for</b> $t=1,2,...,T$ <b>do</b>

  2:	$h_t=\mathfrak{L}(D,\mathcal{D}_{bs})$

  3:<b>end for</b>

  <b>输出</b>：$H(\pmb{x})=\mathop{\arg\max}_{y\in\mathcal{Y}}\sum^T_{t=1}\mathbb{I}(h_t(\pmb{x})=y)\\$

  ---

  * $\mathcal{D}_{bs}$是自助采样产生的样本分布.

* 假定基学习器的计算复杂度为$O(m)$，则Bagging的复杂度大致为$T(O(m)+O(s))$，考虑到采样与投票/平均过程的复杂度$O(s)$很小，而$T$通常是一个不太大的常数，因此，训练一个Bagging集成与直接使用基学习算法训练一个学习器的复杂度同阶，这说明Bagging是一个很高效的集成学习算法.

* 与标准AdaBoost只适用于二分类任务不同，Bagging能不经修改地用于多分类、回归等任务.

* 自助采样过程由于每个基学习器只使用了初始训练集中约63.2%的样本，剩下约36.8%的样本可用作验证集来对泛化性能进行包外估计(out-of-bag estimate).

* 令$D_t$表示$h_t$实际使用的训练样本集，令$H^{oob}(\pmb{x})$表示对样本$\pmb{x}$的包外预测，即仅考虑那些未使用$\pmb{x}$训练的基学习器在$\pmb{x}$上的预测，有
  $$
  H^{oob}(\pmb{x})=\mathop{\arg\max}_{y\in\mathcal{Y}}\sum^T_{t=1}\mathbb{I}(h_t(\pmb{x})=y)·\mathbb{I}(x\notin D_t)
  $$
  则Bagging泛化误差的包外估计为
  $$
  \epsilon^{oob}=\frac{1}{|D|}\sum_{(\pmb{x},y)\in D}\mathbb{I}(H^{oob}(\pmb{x})\neq y)
  $$

* 包外样本的其他用途

  1. 基学习器是决策树时，可使用包外样本来辅助剪枝，或用于估计决策树中各结点的后验概率以辅助对零训练样本结点的处理
  2. 基学习器是神经网络时，可使用包外样本来辅助早期停止以减小过拟合风险

* 从偏差-方差分解的角度看，Bagging主要关注降低方差，因此它在不剪枝的决策树和神经网络等易受样本扰动的学习器上效用更为明显.

### 8.3.2.随机森林

* 随机森林(Random Forest)是Bagging的一个拓展变体.
* $\text{RF}$在以决策树为基学习器构建Bagging集成的基础上，进一步在决策树的训练过程中引入了随机属性选择.
* 传统决策树在选择划分属性时是在当前结点属性集合(假定有$d$个属性)中选择一个最优属性；而在$\text{RF}$中，对基决策树的每个结点，先从该结点的属性集合中随机选择一个包含$k$个属性的子集，然后再从这个子集中选择一个最优属性用于划分.
* 这里的参数$k$控制了随机性的引入程度：若令$k=d$，则基决策树的构建与传统决策树相同；若令$k=1$，则是随机选择一个属性用于划分；一般情况下，推荐$k=\log_2d$.
* 随机森林简单、容易实现、计算开销小，令人惊奇的是，它在很多现实任务中展现出强大的性能.
* 随机森林对Bagging只做了小改动，但是与Bagging中基学习器的多样性仅通过样本扰动（通过对初始训练集采样）而来不同，随机森林中基学习器的多样性不仅来自样本扰动，还来自属性扰动，这就使得最终集成的泛化性能可通过个体学习器之间差异度的增加而进一步提升.
* 随机森林的收敛性与Bagging相似.
* 随机森林的起始性能往往相对较差，特别是在集成中只包含一个基学习器时.
* 通过引入属性扰动，随机森林中个体学习器的性能往往有所降低. 然而随着个体学习器数目的增加，随机森林通常会收敛到更低的泛化误差.
* 随机森林的训练效率常优于Bagging，因为在个体决策树的构建过程中，Bagging使用的是确定型决策树，在选择划分属性时要对结点的所有属性进行考察，而随机森林使用随机型决策树则只需考察一个属性子集.

## 8.4.结合策略

* 学习器结合可能会从三个方面带来好处：
  1. 统计的方面：由于学习任务的假设空间往往很大，可能有多个假设在训练集上达到同等性能，此时若使用单学习器可能因误选而导致泛化性能不佳，结合多个学习器则会减小这一风险.
  2. 计算的方面：学习算法往往会陷入局部极小，有的局部极小点所对应的泛化性能可能很糟糕，而通过多次运行之后进行结合，可降低陷入糟糕局部极小点的风险.
  3. 表示的方面：某些学习任务的真实假设可能不在当前学习算法所考虑的假设空间中，此时若使用单学习器则肯定无效，而通过结合多个学习器，由于相应的假设空间有所扩大，有可能学得更好的近似.

* 假定集成包含$T$个基学习器$\{h_1,h_2,...,h_T\}$，其中$h_i$在示例$\pmb{x}$上的输出为$h_i(\pmb{x})$.本节介绍3种对$h_i$进行结合的常见策略.

### 8.4.1.平均法

* 对数值型输出$h_i(\pmb{x})\in\R$，最常见的结合策略是使用平均法(averaging).

  * 简单平均法(simple averaging)
    $$
    H(\pmb{x})= \frac{1}{T}\sum^T_{i=1}h_i(\pmb{x})
    $$

  * 加权平均法(weighted averaging)
    $$
    H(\pmb{x})= \sum^T_{i=1}w_ih_i(\pmb{x})
    $$
    其中$w_i$是个体学习器$h_i$的权重，通常要求$w_i\geq0,\sum\limits_{i=1}^Tw_i=1$.

* 简单平均法是加权平均法令$w_i=1/T$的特例.

* 集成学习中的各种结合方法都可视为加权平均法的特例或变体.

* 加权平均法可认为是集成学习研究的基本出发点.

* 对给定的基学习器，不同的集成学习方法可视为通过不同的方式来确定加权平均法中的基学习器权重.

* 加权平均法的权重一般是从训练数据中学习而得，现实任务中的训练样本通常不充分或存在噪声，这将使得学出的权重不完全可靠. 尤其是对规模比较大的集成来说，要学习的权重比较多，较容易过拟合.

* 实验和应用均显示出，加权平均法未必一定优于简单平均法.

* 在个体学习器性能相差较大时宜使用加权平均法，而在个体学习器性能相近时宜使用简单平均法.

### 8.4.2.投票法

* 对分类任务来说，学习器$h_i$将从类别标记集合$\{c_1,c_2,...,c_N\}$中预测出一个标记，最常见的结合策略是使用投票法(voting).

* 将$h_i$在样本$\pmb{x}$上的预测输出表示为一个$\pmb{N}$维向量$(h^1_i(\pmb{x});h^2_i(\pmb{x});...;h^N_i(\pmb{x}))$，其中$h^j_i(\pmb{x})$是$h_i$在类别标记$c_j$上的输出.

  * 绝对多数投票法(majority voting)
    $$
    H(\pmb{x})=
    	\begin{cases}
        \ c_j,&\text{if}\
        \sum\limits_{i=1}^Th^j_i(\pmb{x})>0.5\sum\limits_{k=1}^N\sum\limits_{i=1}^Th^k_i(x);\\
        \ \text{reject},&\text{otherwise}.\\
        \end{cases}
    $$
    即若某种标记得票过半数，则预测为该标记；否则拒绝预测.

  * 相对多数投票法(plurality voting)
    $$
    H(\pmb{x})=c_{\mathop{\arg\max}\limits_j\sum_{i=1}^Th^j_i(\pmb{x})}
    $$
    即预测为得票最多的标记，若同时有多个标记获得最高票，则从中随机选取一个.

  * 加权投票法(weighted voting)
    $$
    H(\pmb{x})=c_{\mathop{\arg\max}\limits_j\sum_{i=1}^Tw_ih^j_i(\pmb{x})}
    $$
    与加权平均法类似，$w_i$是$h_i$的权重，通常$w_i\geq0,\sum\limits_{i=1}^Tw_i=1$.

* 标准的绝对多数投票法提供了拒绝预测选项，这在可靠性要求较高的学习任务中是一个很好的机制.

* 学习任务要求必须提供预测结果，则绝对多数投票法将退化为相对多数投票法.

* 在不允许拒绝预测的任务中，绝对多数、相对多数投票法统称为多数投票法.

* 上述公式没有限制个体学习器输出值的类型. 在现实任务中，不同类型个体学习器可能产生不同类型的$h^j_i(\pmb{x})$值，常见的有：

  * 类标记：$h^j_i(\pmb{x})\in\{0,1\}$，若$h_i$将样本$\pmb{x}$预测为类别$c_j$则取值为1，否则为0. 使用类标记的投票亦称硬投票(hard voting).
  * 类概率：$h^j_i(\pmb{x})\in[0,1]$，相当于对后验概率$P(c_j|\pmb{x})$的一个估计. 使用类概率的投票亦称软投票(soft voting).

* 不同类型的$h^j_i(\pmb{x})$值不能混用.

* 对一些能在预测出类别标记的同时产生分类置信度的学习器，其分类置信度可转化为类概率使用. 若此类值未进行规范化，例如支持向量机的分类间隔值，则必须使用一些技术如Platt缩放(Platt scaling)、等分回归(isotonic regression)等进行校准(calibration)后才能作为类概率使用.

* 虽然分类器估计出类概率值一般都不太准确，但基于类概率进行结合却往往比直接基于类标记进行结合性能更好.

* 若基学习器的类型不同，则其类概率值不能直接进行比较；此种情况下，将类概率输出转化为类标记输出(例如将类概输出最大的$h^j_i(\pmb{x})$设为1，其他设为0)然后投票.

### 8.4.3.学习法

* 训练数据很多时，一种更为强大的结合策略是使用学习法，即通过另一个学习器来进行结合.

* Stacking 是学习法的典型代表，这里个体学习器称为初级学习器，用于结合的学习器称为次级学习器或元学习器(meta-learner).

* Stacking 先从初始训练集训练出初级学习器，生成一个新数据集用于训练次级学习器.

* 在新数据集中，初级学习器的输出被当作样例输入特征，而初始样本的标记仍被当作样例标记.

* Stacking 算法描述

  ---

  <b>输入</b>：训练集$D=\{(\pmb{x}_1, y_1), (\pmb{x}_2, y_2),..., (\pmb{x}_m, y_m)\}$;

  ​			初级学习算法$\mathfrak{L}_1,\mathfrak{L}_2,...,\mathfrak{L}_T$;

  ​			次级学习算法$\mathfrak{L}$.

  <b>过程</b>：

  1:<b>for</b> $t=1,2,...,T$ <b>do</b>

  2:	$h_t=\mathfrak{L}_t(D)$；

  3:<b>end for</b>

  4:$D'=\varnothing$；

  5:<b>for</b> $i=1,2,...,m$ <b>do</b>

  6:	<b>for</b> $t=1,2,...,T$ <b>do</b>

  7:		$z_{it}=h_t(\pmb{x_i})$；

  8:	<b>end for</b>

  9:	$D'=D'\cup((z_{i1},z_{i2},...,z_{iT}),y_i)$；

  10:<b>end for</b>

  11:$h'=\mathfrak{L}(D')$；

  <b>输出</b>：$H(\pmb{x})=h'(h_1(\pmb{x}),h_2(\pmb{x}),...,h_T(\pmb{x}))$

  ---

  * 假定初级学习器使用不同学习算法产生，即初级集成是异质的.
  * 初级学习器也可是同质的.

* 在训练阶段，次级训练集是利用初级学习器产生的.

* 若直接用初级学习器的训练集来产生次级训练集，则过拟合的风险会比较大；因此，一般是通过使用交叉验证法或留一法，用训练初级学习器未使用的样本来产生次级学习器的训练样本.

* 以$k$折交叉验证为例，初始训练集$D$被随机划分为$k$个大小相似的集合$D_1,D_2,...,D_k$.

  * 对$D_j$和$\bar{D}_j=D \backslash D_j$分别表示第$j$折的测试集和训练集.
  * 给定$T$个初级学习算法，初级学习器$h^{(j)}_t$通过在$\bar{D}_j$上使用第$t$个学习算法而得.
  * 对$D_j$中每个样本$\pmb{x}_i$，令$z_{it}=h^{(j)}_t(\pmb{x}_i)$，则由$\pmb{x}_i$所产生的次级训练样例的示例部分为$\pmb{z}_i=(z_{i1};z_{i2};...;z_{iT})$，标记部分为$y_i$.
  * 在整个交叉验证过程结束后，从这$\pmb{T}$个初级学习器产生的次级训练集是$D'=\{(z_i,y_i)\}^m_{i=1}$，然后$D'$将用于训练次级学习器.

* 次级学习器的输入属性表示和次级学习器算法对Stacking集成的泛化性能有很大影响.

* 将初级学习器的输出类概率作为次级学习器的输入属性，用多响应线性回归(Multi-response Linear Regression)作为次级学习算法效果比较好，在$\text{MLR}$中使用不同的属性集更佳.

  * MLR是基于线性回归的分类器，它对每个类分别进行线性回归，属于该类的训练样例所对应的输出被置为1，其他类置为0；测试示例将被分给输出值最大的类.

* 贝叶斯模型平均(Bayes Model Averaging)基于后验概率来为不同模型赋予权重，可视为加权平均法的一种特殊实现.

  * 理论上，若数据生成模型恰在当前考虑的模型中，且数据噪声很少，则BMA不差于Stacking.
  * 在现实应用中无法确保数据生成模型一定在当前考虑的模型中，甚至可能难以用当前考虑的模型来近似，因此，Stacking通常优于BMA，因为其鲁棒性比BMA更好.
  * BMA对模型近似误差非常敏感.

## 8.5.多样性

### 8.5.1.误差-分歧分解

* 假定用个体学习器$h_1,h_2,...,h_T$通过加权平均法结合产生的集成来完成回归学习任务$f:\R^d \mapsto \R$. 对示例$\pmb{x}$，定义学习器$h_i$的分歧(ambiguity)为
  $$
  A(h_i|\pmb{x})=\big(h_i(\pmb{x})-H(\pmb{x})\big)^2
  $$
  则集成的分歧是


$$
  \begin{aligned}
  	\overline{A}(h|\pmb{x})&=\sum\nolimits^T_{i=1}w_iA(h_i|\pmb{x})\\
  	&=\sum\nolimits^T_{i=1}w_i\big(h_i(\pmb{x})-H(\pmb{x})\big)^2
  \end{aligned}
$$

* 分歧项表征了个体学习器在样本$\pmb{x}$上的不一致性，即在一定程度上反映了个体学习器的多样性.

* 个体学习器$h_i$和集成$H$的平方误差分别为
  $$
  E(h_i|\pmb{x})=\big(f(\pmb{x})-h_i(\pmb{x})\big)^2\\
  E(H|\pmb{x})=\big(f(\pmb{x})-H(\pmb{x})\big)^2\\
  $$

* 令$\overline{E}(h|\pmb{x})=\sum\nolimits^T_{i=1}w_i·E(h_i|\pmb{x})$表示个体学习器误差的加权均值，有

  $$
  \begin{aligned}
  	\overline{A}(h|\pmb{x})&=\sum\limits^T_{i=1}w_iE(h_i|\pmb{x})-E(H|\pmb{x})\\
  	&=\overline{E}(h|\pmb{x})-E(H|\pmb{x})\\
  \end{aligned}
  $$

* 上式对所有样本$\pmb{x}$均成立，令$p(\pmb{x})$表示样本的概率密度，则在全样本上有
  $$
  \sum\limits^T_{i=1}w_i\int A(h_i|\pmb{x})p(\pmb{x})d\pmb{x}=\sum\limits^T_{i=1}w_i\int E(h_i|\pmb{x})p(\pmb{x})d\pmb{x}-\int E(H|\pmb{x})p(\pmb{x})d\pmb{x}
  $$

* 类似的，个体学习器$h_i$在全样本上的泛化误差和分歧项分别为
  $$
  E_i=\int E(h_i|\pmb{x})p(\pmb{x})d\pmb{x}\\
  A_i=\int A(h_i|\pmb{x})p(\pmb{x})d\pmb{x}\\
  $$
  集成的泛化误差为
  $$
  E=\int E(H|\pmb{x})p(\pmb{x})d\pmb{x}
  $$

  * 用$E_i$和$A_i$简化表示$E(h_i)$和$A(h_i)$.
  * 用$E$简化表示$E(H)$.

* 令$\overline{E}=\sum\nolimits^T_{i=1}w_iE_i$表示个体学习器泛化误差的加权均值，$\overline{A}=\sum\nolimits^T_{i=1}w_iA_i$表示个体学习器的加权分歧值，有
  $$
  E=\overline{E}-\overline{A}
  $$

* 上式明确提示出：个体学习器准确性越高、多样性越大，则集成的越好. 这个分析称为误差-分歧分解(error-ambiguity decomposition).

* 直接把$\overline{E}-\overline{A}$作为优化目标来求解，似乎可以得到最优的集成；遗憾的是，在现实任务中很难直接对$\overline{E}-\overline{A}$进行优化.

  1. $\overline{E}-\overline{A}$是定义在整个样本空间上
  2. $\overline{A}$不是一个可直接操作的多样性度量，它仅在集成构造好之后才能进行估计.

* 上面的推导仅适用于回归学习，难以直接推广到分类学习任务上.

### 8.5.2.多样性度量

* 多样性度量亦称差异性度量.

* 多样性度量(diversity measure)是用于度量集成中分类器的多样性，即估算个体学习器的多样化程度.

* 典型做法是考虑个体分类器的两两相似/不相似性.

* 给定数据集$D=\{(\pmb{x}_1,y_1),(\pmb{x}_2,y_2),...,(\pmb{x}_m,y_m)\}$，对二分类任务，$y_i\in\{-1, +1\}$，分类器$h_i$与$h_j$的预测结果列联表(contingency table)为

  |          | $h_i=+1$ | $h_i=-1$ |
  | :------: | :------: | :------: |
  | $h_j=+1$ |   $a$    |   $c$    |
  | $h_j=-1$ |   $b$    |   $d$    |

  其中，$a$表示$h_i$与$h_j$均预测为正类的样本数目；$b$、$c$、$d$含义由此类推；$a+b+c+d=m$.

* 不合度量(disagreement measure)
  $$
  dis_{ij}=\frac{b+c}{m}
  $$
  $dis_{ij}$的值域为$[0,1]$. 值越大则多样性越大.

* 相关系数(correlation coefficient)
  $$
  \rho_{ij}=\frac{ad-bc}{\sqrt{(a+b)(a+c)(c+d)(b+d)}}
  $$
  $\rho_{ij}$的值域$[-1,1]$. 若$h_i$与$h_j$无关，则值为0；若$h_i$与$h_j$正相关则值为正，否则为负.

* $Q$-统计量($Q$-statistic)
  $$
  Q_{ij}=\frac{ad-bc}{ad+bc}
  $$
  $Q_{ij}$与相关系数$\rho_{ij}$的符号相同，且$|Q_{ij}|\geq|\rho_{ij}|$.

* $\kappa$-统计量($\kappa$-statistic)
  $$
  \kappa=\frac{p_1-p_2}{1-p_2}
  $$
  其中，$p_1$是两个分类器取得一致的概率；$p_2$是两个分类器偶然达成一致的概率，它们可由数据集$D$估算：
  $$
  p_1=\frac{a+d}{m}\\
  p_2=\frac{(a+b)(a+c)+(c+d)(b+d)}{m^2}\\
  $$
  若$h_i$与$h_j$在$D$上完全一致，则$\kappa=1$；若它们仅是偶然达成一致，则$\kappa=0$. $\kappa$通常为非负值，仅在$h_i$与$h_j$达成一致的概率甚至低于偶然性的情况下取负值.

* 以上介绍的都是成对型(pairwise)多样性度量，它们可以绘制出2维图.

* $\kappa$-误差图是将每一对分类器作为图上的一个点，横坐标是这对分类器的$\kappa$值，纵坐标是它们的平均误差.

  1. 数据点云的位置越高，则个体分类器准确性越低.
  2. 数据点云的位置越靠右，则个体学习器的多样性越小.

### 8.5.3.多样性增强

* 在集成学习中需有效地生成多样性大的个体学习器.

* 增强多样性的一般思路是在学习过程中引入随机性，常见做法主要是对数据样本、输入属性、输出表示、算法参数进行扰动.

* 数据样本扰动

  * 给定初始数据集，可从中产生出不同的数据子集，再利用不同的数据子集训练出不同的个体学习器.
  * 数据样本扰动通常是基于采样法，例如在$\text{Bagging}$中使用自助采样，在$\text{AdaBoost}$中使用序列采样. 采样法做法简单高效，使用最广.
  * 采样法对很多不稳定基学习器，例如决策树、神经网络等，训练样本稍加变化就会导致学习器有显著变动.
  * 数据样本扰动法对不稳定基学习器很有效.
  * 稳定基学习器(stable base learner)例如线性学习器、支持向量机、朴素贝叶斯、$k$近邻学习器等对数据样本的扰动不敏感.
  * 稳定基学习器进行集成需使用输入属性扰动等其他机制.

* 输入属性扰动

  * 训练样本通常由一组属性描述，不同的子空间(subspace，即属性子集)提供了观察数据的不同视角.

  * 子空间一般指从初始的高维属性空间投影产生的低维属性空间，描述低维空间的属性是通过初始属性投影变换而得，未必是初始属性.

  * 从不同子空间训练出的个体学习器必然有所不同.

  * 随机子空间(random subspace)算法就依赖于输入属性扰动，该算法从初始属性集中抽取出若干个属性子集，再基于每个属性子集训练出一个基学习器.

  * 随机子空间算法描述

    ---

    <b>输入</b>：训练集$D=\{(\pmb{x}_1, y_1), (\pmb{x}_2, y_2),..., (\pmb{x}_m, y_m)\}$;

    ​			基学习算法$\mathfrak{L}$;

    ​			基学习器数$T$;

    ​			子空间属性数$d'$.

    <b>过程</b>：

    1:<b>for</b> $t=1,2,...,T$ <b>do</b>

    2:	$\mathcal{F}_t=\text{RS}(D,d')$

    3:	$D_t=\text{Map}_\mathcal{F_t}(D)$；

    4:	$h_t=\mathfrak{L}(D_t)$

    5:<b>end for</b>

    <b>输出</b>：$H(\pmb{x})=\mathop{\arg\max}\limits_{y\in\mathcal{Y}}\sum^T_{t=1}\mathbb{I}\big(h_t\big(\text{Map}_\mathcal{F_t}(\pmb{x})\big)=y\big)$

    ---

    * $d'$小于初始属性数$d$.
    * $\mathcal{F}_t$包含$d'$个随机选取的属性，$D_t$仅保留$\mathcal{F}_t$中的属性.

  * 对包含大量冗余属性的数据，在子空间中训练个体学习器不仅能产生多样性大的个体，还会因属性数的减少而大幅节省时间开销，同时，由于冗余属性多，减少一些属性后训练出的个体学习器也不至于太差.

  * 若数据只包含少量属性，或者冗余属性很少，则不宜使用输入属性扰动法.

* 输出表示扰动

  * 输出表示扰动的基本思路是对输出表示进行操纵以增强多样性.
  * 可对训练样本的类标记稍作变动
    1. 翻转法($\text{Flipping Output}$)随机改变一些训练样本的标记.
    2. 也可对输出表示进行转化，如输出调制法($\text{Output Smearing}$)将分类输出转化为回归输出后构建个体学习器.
    3. 还可将原任务拆解为多个可同时求解的子任务，如$\text{ECOC}$法利用纠错输出码将多分类任务拆解为一系列二分类任务来训练基学习器.

* 算法参数扰动

  * 基学习算法一般都有参数需要进行设置，例如神经网络的隐层神经元数、初始化连接权值等，通过随机设置不同的参数，往往可产生差别较大的个体学习器.
  * 负相关法($\text{Negative Correlation}$)显式地通过正则化项来强制个体神经网络使用不同的参数.
  * 对参数较少的算法，可通过将其学习过程中某些环节用其他类似方式代替，从而达到扰动的目的，例如可将决策树使用的属性选择机制替换成其他的属性选择机制.
  * 使用单一学习器时通常需使用交叉验证法等方式来确定参数值，这事实上已使用了不同参数训练出多个学习器，只不过最终仅选择其中一个学习器进行使用，而集成学习则相当于把这些学习器都利用起来；集成学习技术的实际计算开销并不比使用单一学习器大很多.

* 不同的多样性增强机制可同时使用.

# 第九章 聚类

## 9.1.聚类任务

* 在无监督学习(unsupervised learning)中, 训练样本的标记信息是未知的, 目标是通过对无标记训练样本的学习来揭示数据的内在性质及规律, 为进一步的数据分析提供基础.
* 常见的无监督学习任务有聚类(clustering)、密度估计(density estimation)和异常检测(anomaly detection)等.
* 聚类是无监督学习任务中研究最多、应用最广的.
* 聚类试图将数据集中的样本划分为若干个通常是不相交的子集, 每个子集称为一个簇(cluster).
    * 对于聚类算法而言, 样本簇亦称类.
    * 通过这样的划分, 每个簇可能对应于一些潜在的概念(类别).
    * 这些概念对聚类算法而言事先是未知的, 聚类过程仅能自动形成簇结构, 簇所对应的概念语义需由使用者来把握和命名.
* 形式化地说, 假定样本集$D=\{\pmb{x}_1, \pmb{x}_2,...,\pmb{x}_m\}$包含$m$个无标记样本, 每个样本$\pmb{x}_i=(x_{i1};x_{i2};...;x_{in})$是一个$n$维特征向量, 则聚类算法将样本集$D$划分为$k$个不相交的簇$\{C_l|l=1,2,...,k\}$, 其中$C_{l'}\bigcap_{l'\ne l}C_l=\varnothing$且$D=\bigcup^k_{l=1}C_l$.
* 相应地, 我们用$\lambda_j\in\{1,2,...,k\}$表示样本$\pmb{x}_j$的簇标记(cluster label), 即$\pmb{x}_j\in C_{\lambda_j}$; 聚类的结果可用包含$m$个元素的簇标记向量$\pmb{\lambda}=(\lambda_1;\lambda_2;...;\lambda_m)$表示.
* 聚类既能作为一个单独过程, 用于找寻数据内在的分布结构, 也可作为分类等其他学习任务的前驱过程.
* 基于不同的学习策略, 可以设计多种类型的聚类算法.
* 聚类算法的两个基本问题----性能度量和距离计算.

## 9.2.性能度量

* 聚类性能度量亦称聚类有效性指标(validity index); 与监督学习中的性能度量作用类似.

* 对聚类结果, 我们需通过某种性能度量来评估其好坏.

* 明确最终将要使用的性能度量, 则可直接将其作为聚类过程的优化目标.

* 同一簇的样本尽可能彼此相似, 不同簇的样本尽可能不同. 聚类结果的簇内相似度(intra-cluster similarity)高且簇间相似度(inter-cluster similarity)低.

* 聚类性能度量大致有两类. 

    1. 将聚类结果与某个参考模型(reference model)进行比较, 称为外部指标(external index).
        * 可以将领域专家给出的划分结果作为参考模型.
    2. 直接考察聚类结果而不用任何参考模型, 称为内部指标(internal index).

* 对数据集$D=\{\pmb{x}_1,\pmb{x}_2,...,\pmb{x}_m\}$, 假定通过聚类给出的簇划分为$C=\{C_1,C_2,...,C_k\}$, 参考模型给出的簇划分为$C^*=\{C^*_1, C^*_2, ..., C^*_s\}$.

    * 通常$k \neq s$.

    * 令$\pmb\lambda$与$\pmb\lambda^*$分别表示与$C$和$C^*$对应的簇标记向量.

    * 将样本两两配对考虑, 定义
        $$
        \begin{equation}
        	\begin{aligned}
            a&=|SS|,\space\space SS=\{(\pmb{x}_i, \pmb{x}_j)|\lambda_i=\lambda_j, \lambda_i^*=\lambda_j^*,i<j\}, \\
            b&=|SD|,\space\space SD=\{(\pmb{x}_i, \pmb{x}_j)|\lambda_i=\lambda_j, \lambda_i^*\neq\lambda_j^*,i<j\}, \\
            c&=|DS|,\space\space DS=\{(\pmb{x}_i, \pmb{x}_j)|\lambda_i\neq\lambda_j, \lambda_i^*=\lambda_j^*,i<j\}, \\
            d&=|DD|,\space\space DD=\{(\pmb{x}_i, \pmb{x}_j)|\lambda_i\neq\lambda_j, \lambda_i^*\neq\lambda_j^*,i<j\}, \\
        	\end{aligned}
        \end{equation}
        $$
        其中集合$SS$包含了在$C$中隶属于相同簇且在$C^*$中也隶属于相同簇的样本对, 集合$SD$包含了在$C$中隶属于相同簇但在$C^*$中隶属于不同簇的样本对, 由于每个样本对$(\pmb{x}_i, \pmb{x}_j)(i<j)$仅能出现在一个集合中, 因此有$a+b+c+d=m(m-1)/2$成立.

* 基于上式可以导出常用的聚类性能度量外部指标: 

    * Jaccard系数(Jaccard Coefficient, 简称$\text{JC}$)
        $$
        JC=\frac{a}{a+b+c}
        $$

    * FM指数(Folkeds and Mallows Index, 简称$\text{FMI}$)
        $$
        FMI=\sqrt{\frac{a}{a+b}·\frac{a}{a+c}}
        $$
        
    * Rand指数(Rand Index, 简称$\text{RI}$)
        $$
        RI=\frac{2(a+d)}{m(m-1)}
        $$
        上述性能度量的结果值均在$[0, 1]$区间, 值越大越好.
    
* 考虑聚类结果的簇划分为$C=\{C_1, C_2, ..., C_k\}$, 定义
    $$
    \begin{equation}
    	\begin{aligned}
    		\text{avg}(C)&=\frac{2}{|C|(|C|-1)}\sum_\nolimits{{1\leq i<j\leq|C|}}\text{dist}(\pmb{x}_i, \pmb{x}_j), \\
    		\text{diam}(C)&=\max_\nolimits{{1\leq i<j\leq|C|}}\text{dist}(\pmb{x}_i, \pmb{x}_j), \\
    		\text d_\min(C_i, C_j)&=\min_\nolimits{\pmb{x}_i\in C_i, \pmb{x}_j\in C_j}\text{dist}(\pmb{x}_i, \pmb{x}_j), \\
    		\text d_{\text{cen}}(C_i, C_j)&=\text{dist}(\pmb{\mu}_i, \pmb{\mu}_j),
    	\end{aligned}
    \end{equation}
    $$
    $\text{dist}(·, ·)$用于计算两个样本之间的距离, 距离越大则样本的相似度越低;  $\pmb{\mu}$代表簇$C$的中心点$\pmb{\mu}=\frac{1}{|C|}\sum_\nolimits{1\leq i\leq|C|}\pmb{x}_i$. 

    * $\text{avg}(C)$对应于簇$C$内样本间的平均距离.
    * $\text{diam}(C)$对应于簇$C$内样本间的最远距离.
    * $d_\min(C_i, C_j)$对应于簇$C_i$与簇$C_j$最近样本间的距离.
    * $d_\text{cen}(C_i, C_j)$对应于簇$C_i$与簇$C_j$中心点间的距离.

* 基于上式可导出常用的聚类性能度量的内部指标:

    * DB指数(Davies-Bouldin Index, 简称$\text{DBI}$)
        $$
        DBI=\frac{1}{k}\sum^k_{i=1}\max_{j\neq i}\Bigg(\frac{\text{avg}(C_i)+\text{avg}(C_j)}{d_{\text{cen}}(C_i, C_j)}\Bigg)
        $$

    * Dunn指数(Dunn Index, 简称$\text{DI}$)
        $$
        DI=\min_{1\leq i\leq k}\Bigg\{\min_{j\neq i}\Bigg(\frac{d_\min(C_i, C_j)}{\max_\nolimits{1\leq l \leq k}\text{diam}(C_l)}\Bigg)\Bigg\}
        $$
        显然, DBI的值越小越好, DI的值越大越好.

## 9.3.距离计算

* 对函数$\text{dist}(·, ·)$, 若它是一个距离度量(distance measure), 则需要满足一些基本性质:

    * 非负性: $\text{dist}(\pmb{x}_i, \pmb{x}_j)\geq0$
    * 同一性: $\text{dist}(\pmb{x}_i, \pmb{x}_j)=0$ 当且仅当 $\pmb{x}_i=\pmb{x}_j$
    * 对称性: $\text{dist}(\pmb{x}_i, \pmb{x}_j)=\text{dist}(\pmb{x}_j, \pmb{x}_i)$
    * 直递性: $\text{dist}(\pmb{x}_i, \pmb{x}_j)\leq\text{dist}(\pmb{x}_i, \pmb{x}_k)+\text{dist}(\pmb{x}_k, \pmb{x}_j)$
        * 直递性常被直接称为三角不等式.

* 给定样本$\pmb{x}_i = (x_{i1};x_{i2};...;x_{in})$与$\pmb{x}_j = (x_{j1};x_{j2};...;x_{jn})$, 最常用的是闵可夫斯基距离(Minkowski distance)
    $$
    \text{dist}_\text{mk}(\pmb{x}_i, \pmb{x}_j)=\Bigg(\sum^n_{u=1}|x_{iu}-x_{ju}|^p\Bigg)^{\frac{1}{p}}
    $$

    * 上式即为$\pmb{x}_i-\pmb{x}_j$的$L_p$范数$||\pmb{x}_i-\pmb{x}_j||_p$

    * 对$p\geq1$, 上式显然满足距离度量基本性质.

    * $p=2$ 时, 闵可夫斯基距离即为欧式距离(Euclidean distance)
        $$
        \text{dist}_\text{ed}(\pmb{x}_i, \pmb{x}_j)=||\pmb{x}_i-\pmb{x}_j||_2=\sqrt{\sum^n_{u-1}|x_{iu}-x_{ju}|^2}
        $$

    * $p=1$ 时, 闵可夫斯基距离即为曼哈顿距离(Manhattan distance)
        $$
        \text{dist}_\text{man}(\pmb{x}_i, \pmb{x}_j)=||\pmb{x}_i-\pmb{x}_j||_1=\sum^n_{u-1}|x_{iu}-x_{ju}|
        $$
        亦称街区距离(city block distance).

    * $p\mapsto\infty$ 时则得到切比雪夫距离.

* 我们常将属性划分为连续属性(continuous attribute)和离散属性(categorical attribute), 前者在定义域上有无穷多个可能的取值, 后者在定义域上是有限个取值.

    * 连续属性亦称数值属性(numerical attribute), 离散属性亦称列名属性(nominal attribute).

* 在讨论距离计算时, 属性上是否定义了序关系更为重要.

    * 能直接在属性值上计算距离的离散属性称为有序属性(ordinal attribute).
    * 不能直接在属性值上计算距离的离散属性称为无序属性(non-ordinal attribute).
    * 闵可夫斯基距离可用于有序属性.

* 对无序属性可采用VDM(Value Difference Metric).

* 令$m_{u, a}$表示在属性$u$上取值为$a$的样本数, $m_{u, a, i}$表示在第$i$个样本簇中在属性$u$上取值为$a$的样本数, $k$为样本簇数, 则属性$u$ 上两个离散值$a$和$b$之间的VDM距离为
    $$
    \text{VDM}_p(a,b)=\sum^k_{i=1}\Bigg|\frac{m_{u,a,i}}{m_{u,a}}-\frac{m_{u,b,i}}{m_{u,b}}\Bigg|^p
    $$
    样本类别已知时$k$通常设置为类别数.

* 将闵可夫斯基聚类和VDM结合即可处理混合属性.

* 假定有$n_c$个有序属性、$n-n_c$个无序属性, 不失一般性, 令有序属性排列在无序属性之前, 则
    $$
    \text{MinkovDM}_p(\pmb{x}_i, \pmb{x}_j)=\Bigg(\sum^{n_c}_{u=1}|x_{iu}-x_{ju}|^p+\sum^n_{u=n_c+1}\text{VDM}_p(x_{iu},x_{ju})\Bigg)^\frac{1}{p}
    $$

* 当样本空间中不同属性的重要性不同时, 可使用加权距离(weighted distance). 以加权闵可夫斯基距离为例:
    $$
    \text{dist}_\text{wmk}(\pmb{x}_i, \pmb{x}_j)=(w_1|x_{iu}-x_{ju}|^p+...+w_n|x_{nu}-x_{nu}|^p)^{\frac{1}{p}}
    $$
    其中权重$w_i\geq0(i=1,2,..., n)$表征不同属性的重要性, 通常$\sum^n_{i=1}w_i=1$.

* 通常我们是基于某种形式的距离来定义相似度度量(similarity measure), 距离越大, 相似度越小.

* 相似度度量的距离未必一定要满足距离度量的所有基本性质, 尤其是直递性.

* 不满足直递性的距离称为非度量距离(non-metric distance).

* 在现实任务中, 也可基于数据样本来确定合适的距离计算式, 这可通过距离度量学习(distance metric learning)来实现.

## 9.4.原型聚类

* 原型聚类亦称基于原型的聚类(prototype-based clustering), 此类算法假设聚类结构能通过一组原型刻画.
* 原型是指样本空间中具有代表性的点.
* 通常算法先对原型进行初始化, 然后对原型进行迭代更新求解.
* 采用不同的原型表示, 不同的求解方式, 将产生不同的算法.

### 9.4.1.$k$ 均值算法

* 给定样本集 $D=\{\pmb{x}_1,\pmb{x}_2,...,\pmb{x}_m\}$, $k$ 均值($k$-means)算法针对聚类所得簇划分$C=\{C_1, C_2, ..., C_k\}$最小化平方误差
    $$
    E=\sum^k_{i=1}\sum_{x\in C_i}||\pmb{x}-\pmb{\mu}_i||^2_2
    $$
    其中$\pmb{\mu}_i=\frac{1}{|C_i|}\sum_{\pmb{x}\in C_i}\pmb{x}$是簇$C_i$的均值向量.

    * 上式在一定程度上刻画了簇内样本围绕簇均值向量的紧密程度, $E$值越小则簇内样本相似度越高.

* 最小化上式要找到它的最优解需考察样本集$D$所有可能的簇划分, 这是一个$\text{NP}$难问题.

* $k$ 均值算法采用了贪心策略, 通过迭代优化来近似求解上式.

* $k$ 均值算法

    ---

    <b>输入:</b> 样本集$D=\{\pmb{x}_1,\pmb{x}_2,...,\pmb{x}_m\}$;

    ​          聚类簇数$k$.

    <b>过程:</b>

    1:从$D$中随机选择$k$个样本作为初始均值向量$\{\pmb{\mu}_1,\pmb{\mu}_2,..,\pmb{\mu}_k\}$

    2:<b>repeat</b>

    3:    令$C_i=\varnothing(i\leqslant i\leqslant k)$

    4:	<b>for</b> $j=1,2,...,m$ <b>do</b>

    5:        计算样本$\pmb{x}_j$与各均值向量$\pmb{\mu}_i(1\leqslant i\leqslant k)$的距离: $d_{ji}=||\pmb{x}_j-\pmb{\mu}_i||_2$;

    6:        根据距离最近的均值向量确定$\pmb{x}_j$的簇标记$\lambda_j=\arg\min_{i\in\{1, 2, ..., k\}}d_{ji}$;

    7:        将样本$\pmb{x}_j$划入相应的簇: $C_{\lambda_j}=C_{\lambda_j}\bigcup\{\pmb{x}_j\}$;

    8:    <b>end for</b>

    9:    <b>for</b> $i=1,2,...,k$ <b>do</b>

    10:       计算新均值向量: $\pmb{\mu^{'}}_i=\frac{1}{|C_i|}\sum_{\pmb{x}_i\in C_i}\pmb{x}$;

    11:       <b>if</b> $\pmb{\mu^{'}}_i\neq\pmb{\mu}_i$ <b>then</b>

    12:           将当前均值向量$\pmb{\mu}_i$更新为$\pmb{\mu^{'}}_i$

    13:       <b>else</b>

    14:           保持当前均值向量不变

    15:       <b>end if</b>

    16:   <b>end for</b>

    17:<b>until</b> 当前均值向量均未更新

    <b>输出</b>: 簇划分$C=\{C_1, C_2, ..., C_k\}$

    ---

    * 为避免运行时间过长, 通常设置一个最大运行轮数或最小调整幅度阈值, 若达到最大轮数或调整幅度小于阈值, 则停止运行.

### 9.4.2.学习向量量化

* 与$k$均值算法类似, 学习向量量化(Learning Vector Quantization, 简称LVQ)也是试图找到一组原型向量来刻画聚类结构.

* 与一般聚类算法不同的是, LVQ假设数据样本带有类别标记, 学习过程利用样本的这些监督信息来辅助聚类.

* LVQ可看作通过聚类来形成类别子类结构, 每个子类对应一个聚类簇.

* 给定样本集$D=\{(\pmb{x}_1, y_1), (\pmb{x}_2, y_2), ...,(\pmb{x}_m, y_m)\}$, 每个样本$\pmb{x}_j$是由$n$个属性描述的特征向量$(x_{j1}; x_{j2}; ... ;x_{jn})$, $y_j\in \mathcal{Y}$是样本$\pmb{x}_j$的类别标记.

* LVQ的目标是学得一组$n$维原型向量$\{\pmb{p}_1, \pmb{p}_2, ...,\pmb{p}_q\}$, 每个原型向量代表一个聚类簇, 簇标记$t_i\in\mathcal{Y}$.

* LVQ算法描述

    ---

    <b>输入:</b> 样本集$D=\{(\pmb{x}_1, y_1), (\pmb{x}_2, y_2), ...,(\pmb{x}_m, y_m)\}$;

    ​          原型向量个数$q$, 各原型向量预设的类别标记$\{t_1, t_2, ..., t_q\}$;

    ​          学习率 $\eta\in(0, 1)$.

    <b>过程:</b>

    1:初始化一组原型向量$\{\pmb{p}_1, \pmb{p}_2, ..., \pmb{p}_q\}$

    2:<b>repeat</b>

    3:    从样本集$D$随机选取样本$(\pmb{x}_j, y_j)$;

    4:	计算样本$\pmb{x}_j$与$\pmb{p}_i(1\leqslant i\leqslant q)$的距离: $d_{ji}=||\pmb{x}_j-\pmb{p}_i||_2$;

    5:    找出与$\pmb{x}_j$距离最近的原型向量$p_{i^*}$, $i^*=\arg\min_{i\in\{1, 2, ..., q\}}d_{ji}$;

    6:    <b>if</b> $y_j=t_{i^*}$ <b>then</b>     

    7:        $\pmb{p}'=\pmb{p}_{i^*}+\eta·(\pmb{x}_j-\pmb{p}_{i^*})$

    8:    <b>else</b>

    9:        $\pmb{p}'=\pmb{p}_{i^*}-\eta·(\pmb{x}_j-\pmb{p}_{i^*})$

    10:   <b>end if</b>

    11:   将原型向量$\pmb{p}_{i^*}$更新为$\pmb{p}'$

    12:<b>until</b> 满足停止条件

    <b>输出</b>: 原型向量$\{\pmb{p}_1, \pmb{p}_2, ..., \pmb{p}_q\}$

    ---

    * LVQ算法对原型向量进行初始化, 例如对第$q$个簇可从类别标记为$t_q$的样本中随机选取一个作为原型向量.
    * 在每一轮迭代中, 算法随机选取一个有标记训练样本, 找出与其最近的原型向量, 并根据两者的类别标记是否一致来对原型向量进行相应的更新.
    * 算法的停止条件可设置为最大运行轮数或原型向量更新幅度很小.

* LVQ的关键是如何更新原型向量.

    * 对样本$\pmb{x}_j$, 若最近的原型向量$\pmb{p}_{i^*}$与$\pmb{x}_j$的类别标记相同, 则令$\pmb{p}_{i^*}$向$\pmb{x}_j$的方向靠拢.
        $$
        \pmb{p}'=\pmb{p}_{i^*}+\eta·(\pmb{x}_j-\pmb{p}_{i^*})
        $$

    * $\pmb{p}’$与$\pmb{x}_j$之间的距离为
        $$
        \begin{equation}
        	\begin{aligned}
            ||\pmb{p}’-\pmb{x}_j||_2&=||\pmb{p}_{i^*}+\eta·(\pmb{x}_j-\pmb{p}_{i^*})-\pmb{x}_j||_2\\
            &=(1-\eta)·||\pmb{p}_{i^*}-\pmb{x}_j||_2
          \end{aligned}
        \end{equation}
        $$
        令学习率$\eta\in(0, 1)$, 则原型向量$\pmb{p}_{i^*}$在更新为$\pmb{p}'$之后将更接近$\pmb{x_j}$.

    * 若$\pmb{p}_{i^*}$与$\pmb{x}_j$的类别标记不同, 则更新后的原型向量与$\pmb{x}_j$之间的距离将增大为$(1+\eta)·||\pmb{p}_{i^*}-\pmb{x}_j||_2$从而更远离$\pmb{x}_j$.

* 在学得一组原型向量$\{\pmb{p}_1, \pmb{p}_2, ...,\pmb{p}_q\}$后, 即可实现对样本空间$\mathcal{X}$的簇划分.

* 对任意样本$\pmb{x}$, 它将被划入与其距离最近的原型向量所代表的簇中.

* 每个原型向量$\pmb{p}_i$定义了与之相关的一个区域$R_i$, 该区域中每个样本与$\pmb{p}_i$的距离不大于它与其他原型向量$\pmb{p}_{i'}(i'\neq i)$的距离, 即
    $$
    R_i=\{\pmb{x}\in\mathcal{X}|\ ||\pmb{x}-\pmb{p}_i||_2\leqslant||\pmb{x}-\pmb{p}_{i'}||_2, i'\neq i\}
    $$

    * 由此形成了对样本空间$\mathcal{X}$的簇划分$\{R_1, R_2, ..., R_q\}$, 该划分通常称为Voronoi剖分(Voronoi tessellation).
    * 若将$R_i$中样本全用原型向量$\pmb{p}_i$表示, 则可实现数据的有损压缩(lossy compression). 这称为向量量化(vector quantization).

### 9.4.3.高斯混合聚类

* 与$k$均值、LVQ用原型向量来刻画聚类结构不同, 高斯混合(Mixture-of-Gaussian)聚类采用概率模型来表达聚类原型.

* (多元)高斯分布的定义. 对$n$维样本空间$\mathcal{X}$中的随机向量$\pmb{x}$, 若$\pmb{x}$若服从高斯分布, 其概率密度函数为
    $$
    p(\pmb{x})=\frac{1}{(2\pi)^\frac{n}{2}|\pmb{\tiny{\sum}}|^\frac{1}{2}}e^{-\frac{1}{2}(\pmb{x}-\pmb{\mu})^T\pmb{\tiny{\sum}}^{-1}(\pmb{x}-\pmb{\mu})}
    $$

    * 其中$\pmb{\mu}$是$n$维均值向量, $\pmb{\sum}$是的$n\times n$协方差矩阵.
    * 记为$\pmb{x}\sim\mathcal{N}(\pmb{\mu}, \pmb{\sum})$.
    * $\pmb{\sum}$: 对称正定矩阵; $|\pmb{\sum}|$: $\pmb{\sum}$的行列式; $\pmb{\sum}^{-1}$: $\pmb{\sum}$的逆矩阵.
    * 高斯分布完全由均值向量$\pmb{\mu}$和协方差矩阵$\pmb{\sum}$这两个参数确定.

* 为了明确显示高斯分布与相应参数的依赖关系, 将概率密度函数记为$p(\pmb{x}|\pmb{\mu}, \pmb{\tiny{\sum}})$.

* 高斯混合分布的定义
    $$
    p_{\mathcal{M}}(\pmb{x})=\sum^k_{i=1}\alpha_i·p(\pmb{x}|\pmb{\mu}_i,\pmb{\tiny{\sum}}_i)
    $$

    * $p_{\mathcal{M}}(·)$也是概率密度函数, $\int p_{\mathcal{M}}(\pmb{x})d\pmb{x}=1$.
    * 该分布是由$k$个混合分布组成, 每个混合成分对应一个高斯分布.
    * 其中$\pmb{\mu}_i$与$\pmb{\sum}_i$是第$i$个高斯混合分布的参数, 而$\alpha_i>0$为相应的混合系数(mixture coefficient), $\sum^k_{i=1}\alpha_i=1$.

* 假设样本的生成过程由高斯混合分布给出: 首先, 根据$\alpha_1,\alpha_2,..., \alpha_k$定义的先验分布选择高斯混合成分, 其中$\alpha_i$为选择第$i$个混合成分的概率; 然后, 根据被选择的混合成分的概率密度函数进行采样, 从而生成相应的样本.

* 若训练集$D=\{\pmb{x}_1, \pmb{x}_2, ..., \pmb{x}_m\}$由上述过程生成, 令随机变量$z_j\in\{1,2, ..., k\}$表示生成样本$\pmb{x}_j$的高斯混合分布, 其取值未知. $z_j$的先验概率$P(z_j=i)$对应于$\alpha_i(i=1,2,...,k)$.

* 根据贝叶斯定理, $z_j$的后验分布对应于
    $$
    \begin{equation}
    	\begin{aligned}
    		p_\mathcal{M}(z_j=i|\pmb{x}_j)&=\frac{P(z_j=i)·p_\mathcal{M}(\pmb{x}_j|z_j=i)}{p_\mathcal{M}(\pmb{x}_j)}\\
    		&=\frac{\alpha_i·p(\pmb{x}_j|\pmb{\mu}_i,\pmb{\sum}_i)}{\sum\limits^k_{l=1}\alpha_l·p(\pmb{x}_j|\pmb{\mu}_l,\pmb{\mathcal{\sum}}_l)}
    	\end{aligned}
    \end{equation}
    $$
    换言之, $p_\mathcal{M}(z_j=i|\pmb{x}_j)$给出了样本$\pmb{x}_j$由第$i$个高斯混合成分生成的后验概率. 为方便叙述, 将其简记为$\gamma_{ji}\ (i=1, 2, ..., k)$.

* 当高斯混合分布已知时, 高斯混合聚类将把样本集$D$划分为$k$个簇$C=\{C_1, C_2, ..., C_k\}$, 每个样本$\pmb{x}_j$的簇标记$\lambda_j$如下确定:
    $$
    \lambda_j=\mathop{\arg\max}_\limits{i\in\{1,2,...,k\}}\ \gamma_{ji}
    $$
    从原型聚类的角度来看, 高斯混合聚类是采用概率模型(高斯分布)对原型进行刻画, 簇划分则由原型对应后验概率确定.

* 对于高斯混合分布的定义, 模型参数$\{(\alpha_i, \pmb{\mu}_i, \pmb{\sum}_i)|1\leqslant i\leqslant k\}$, 在给定样本集$D$的求解, 可采用极大似然估计, 即最大化(对数)似然
    $$
    \begin{equation}
    	\begin{aligned}
    		LL(D)&=\ln\Bigg(\prod^m_{j=1}p_\mathcal{M}(\pmb{x}_j)\Bigg)\\
    		&=\sum^m_{j=1}\ln\bigg(\sum^k_{i=1}\alpha_i·p(\pmb{x}_j|\pmb{\mu}_i, \sum_i)\bigg)
    	\end{aligned}
    \end{equation}
    $$
    常采用EM算法进行迭代优化求解.

* 若参数$\{(\alpha_i, \pmb{\mu}_i, \pmb{\sum}_i)|1\leqslant i\leqslant k\}$ 能使上式最大化, 则$\frac{\part LL(D)}{\part\pmb{\mu}_i}=0$有
    $$
    \sum^m_{j=1}\frac{\alpha_i·p(\pmb{x}_j|\pmb{\mu}_i,\sum_i)}{
    \sum^k_{l=1}\alpha_l·p(\pmb{x}_j|\pmb{\mu}_l,\sum_l)
    }(\pmb{x}_j-\pmb{\mu}_i)=0
    $$

* 由$p_\mathcal{M}(z_j=i|\pmb{x}_j)=\frac{\alpha_i·p(\pmb{x}_j|\pmb{\mu}_i,\pmb{\sum}_i)}{\sum\limits^k_{l=1}\alpha_l·p(\pmb{x}_j|\pmb{\mu}_l,\pmb{\mathcal{\sum}}_l)}$以及, $\gamma_{ji}=p_\mathcal{M}(z_j=i|\pmb{x}_j)$, 有
    $$
    \pmb{\mu}_i=\frac{\sum\limits^m_{j=1}\gamma_{ji}\pmb{x}_j}{\sum\limits^m_{j=1}\gamma_{ji}}
    $$
    即各混合成分的均值可通过样本加权平均来估计, 样本权重是每个样本属于该成分的后验概率. 

* 类似的, 由$\frac{\part LL(D)}{\part\sum_i}=0$可得
    $$
    \sum_\nolimits i=\frac{\sum\limits^m_{j=1}\gamma_{ji}(\pmb{x}_j-\pmb{\mu}_i)(\pmb{x}_j-\pmb{\mu}_i)^T}{\sum\limits^m_{j=1}\gamma_{ji}}
    $$

* 对于混合系数$\alpha_i$, 除了要最大化$LL(D)$, 还需满足$\alpha_i\geqslant 0$, $\sum^k_{i=1}\alpha_i=1$.

* 考虑$LL(D)$的拉格朗日形式:
    $$
    LL(D)+\lambda\bigg(\sum^k_{i=1}\alpha_i-1\bigg)
    $$
    其中$\lambda$为拉格朗日乘子, 由上式对$\alpha_i$的导数为0, 有
    $$
    \sum^m_{j=1}\frac{p(x_j|\pmb\mu_i,\sum_i)}{\sum\limits^k_{l=1}\alpha_l·p(x_j|\pmb\mu_l,\sum_l)}+\lambda=0
    $$
    两边同乘以$\alpha_i$, 对所有混合成分求和可知$\lambda=-m$, 有
    $$
    \alpha_i=\frac{1}{m}\sum^m_{j=1}\gamma_{ji}
    $$
    即每个高斯成分的混合系数由样本属于该成分的平均后验概率确定.

* 即上述推导即可获得高斯混合模型的EM算法: 在每步迭代中, 先根据当前参数来计算每个样本属于每个高斯成分的后验概率$\gamma_{ji}$ (E步), 再根据$\pmb{\mu}_i=\frac{\sum^m_{j=1}\gamma_{ji}\pmb{x}_j}{\sum^m_{j=1}\gamma_{ji}}$, $\sum_i=\frac{\sum^m_{j=1}\gamma_{ji}(\pmb{x}_j-\pmb{\mu}_i)(\pmb{x}_j-\pmb{\mu}_i)^T}{\sum^m_{j=1}\gamma_{ji}}$和$\alpha_i=\frac{1}{m}\sum^m_{j=1}\gamma_{ji}$更新模型参数$\{(\alpha_i,\pmb{\mu}_i,\sum_i)|1\leqslant i\leqslant k\}$ (M步).

* 高斯混合聚类算法描述

    ---

    <b>输入:</b> 样本集$D=\{\pmb{x}_1, \pmb{x}_2, ...,\pmb{x}_m\}$;

    ​          高斯混合成分个数$k$.

    <b>过程:</b>

    1:初始化高斯混合分布的模型参数$\{(\alpha_i,\pmb{\mu}_i,\sum_i)|1\leqslant i\leqslant k\}$

    2:<b>repeat</b>

    3:    <b>for</b> $j=1,2,...,m$ <b>do</b>

    4:        根据$p_\mathcal{M}(z_j=i|\pmb{x}_j)$计算$\pmb{x}_j$由各混合成分生成的后验概率, 即

    ​           $\gamma_{ji}=p_\mathcal{M}(z_j=i|\pmb{x}_j)(1\leqslant i\leqslant k)$

    5:    <b>end for</b> 

    6:    <b>for</b> $i=1,2,...,k$ <b>do</b>

    7:        计算新均值向量: $\pmb{\mu}_i'=\frac{\sum^m_{j=1}\gamma_{ji}\pmb{x}_j}{\sum^m_{j=1}\gamma_{ji}}$;

    8:        计算新协方差矩阵: $\sum_i'=\frac{\sum^m_{j=1}\gamma_{ji}(\pmb{x}_j-\pmb{\mu}_i')(\pmb{x}_j-\pmb{\mu}_i')^T}{\sum^m_{j=1}\gamma_{ji}}$;

    9:        计算新混合系数: $\alpha_i'=\frac{\sum^m_{j=1}\gamma_{ji}}{m}$

    10:   <b>end for</b>

    11:   将模型参数$\{(\alpha_i,\pmb{\mu}_i,\sum_i)|1\leqslant i\leqslant k\}$ 更新为$\{(\alpha_i',\pmb{\mu}_i',\sum_i')|1\leqslant i\leqslant k\}$

    12:<b>until</b> 满足停止条件

    13:$C_i=\varnothing\ (1\leqslant i\leqslant k)$

    14:<b>for</b> $j=1,2,...,m$ <b>do</b>

    15:   根据$\lambda_j=\mathop{\arg\max}_\limits{i\in\{1,2,...,k\}}\ \gamma_{ji}$确定$\pmb{x}_j$的簇标记$\lambda_j$;

    16:   将$\pmb{x}_j$划入相应的簇: $C_{\lambda_j}=C_{\lambda_j}\bigcup\{\pmb{x}_j\}$

    17:<b>end for</b>

    <b>输出</b>: 簇划分$C=\{C_1, C_2, ..., C_k\}$

    ---

    * 第3-5行EM算法的E步, 第6-11行EM算法的M步.
    * 算法的停止条件可设置为最大迭代轮数或似然函数$LL(D)$增长很少甚至不再增长, 第14-17行根据高斯混合分布确定簇划分.

## 9.5.密度聚类

* 密度聚类亦称基于密度的聚类(density-based clustering), 此类算法假设聚类结构能通过样本分布的紧密程度确定.

* 密度聚类算法从样本密度的角度来考虑样本之间的可连接性, 并基于可连接样本不断扩展聚类簇以获得最终的聚类结果.

* DBSCAN(Density-Based Spatial Clustering of Applications with Noise)是一种著名的密度聚类算法, 它基于一组邻域(neighborhood)参数$(\epsilon, MinPts)$来刻画样本分布的紧密程度.

* 给定数据集$D=\{\pmb{x}_1,\pmb{x}_2,...,\pmb{x}_m\}$, 定义下面这几个概念:

    * $\epsilon$-邻域: 对$\pmb{x}_j\in D$, 其$\epsilon$-邻域包含样本集$D$中与$\pmb{x}_j$的距离不大于$\epsilon$的样本, 即$N_\epsilon(\pmb{x}_j)=\{\pmb{x}_i\in D|\text{dist}(\pmb{x}_i,\pmb{x}_j)\leqslant\epsilon\}$;
    * 核心对象(core object): 若$\pmb{x}_j$的$\epsilon$-领域至少包含$MinPts$个样本, 即$\abs{N_\epsilon(\pmb{x}_j)}\geqslant MinPts$, 则是一个核心对象$\pmb{x}_j$;
    * 密度直达(directly density-reachable): 若$\pmb{x}_j$位于$\pmb{x}_i$的$\epsilon$-领域中, 且$\pmb{x}_i$是核心对象, 则称$\pmb{x}_j$由$\pmb{x}_i$密度直达;
        * 密度直达关系通常不满足对称性.
    * 密度可达(density-reachable): 对$\pmb{x}_i$与$\pmb{x}_j$, 若存在样本序列$\pmb{p}_1, \pmb{p}_2, ..., \pmb{p}_n$, 其中$\pmb{p}_1=\pmb{x}_i$, $\pmb{p}_n=\pmb{x}_j$且$\pmb{p}_{i+1}$由$\pmb{p}_i$密度直达, 则称$\pmb{x}_j$由$\pmb{x}_i$密度可达.
        * 密度可达关系满足直递性, 但不满足对称性.
    * 密度相连(density-connected): 对$\pmb{x}_i$与$\pmb{x}_j$, 若存在$\pmb{x}_k$使得$\pmb{x}_i$与$\pmb{x}_j$均由$\pmb{x}_k$密度可达, 则称$\pmb{x}_i$与$\pmb{x}_j$密度相连.
        * 密度相连关系满足对称性.

* DBSCAN将簇定义为: 有密度可达关系导出的最大的密度相连样本集合.

    * $D$中不属于任何簇的样本被认为是噪声(noise)或者异常(anomaly)样本.
    * 给定邻域参数$(\epsilon, MinPts)$, 簇$C\subseteq D$是满足以下性质的非空样本子集:
        * 连接性(connectivity): $\pmb{x}_i\in C$, $\pmb{x}_j\in C\Rightarrow\pmb{x}_i$与$\pmb{x}_j$密度相连
        * 最大性(maximality): $\pmb{x}_i\in C$, $\pmb{x}_j$由$\pmb{x}_i$密度可达 $\Rightarrow\pmb{x}_j\in C$

* 若$\pmb{x}$为核心对象, 由$\pmb{x}$密度可达的所有样本组成的集合记为$X=\{\pmb{x}'\in D|\pmb{x}'$ 由 $\pmb{x}$ 密度可达$\}$, 则可证明$X$即为满足连续性和最大性的簇.

* DBSCAN 算法任选数据集中的一个核心对象为种子(seed), 再由此出发确定相应的聚类簇.

* DBSCAN 算法描述

    ---

    <b>输入:</b> 样本集$D=\{\pmb{x}_1, \pmb{x}_2, ...,\pmb{x}_m\}$;

    ​          邻域参数$(\epsilon, MinPts)$.

    <b>过程:</b>

    1:初始化核心对象集合: $\Omega = \varnothing$

    2:    <b>for</b> $j=1,2,...,m$ <b>do</b>

    3:        确定样本$\pmb{x}_j$的$\epsilon$-邻域$N_\epsilon(\pmb{x}_j)$;

    4:        <b>if</b> $\abs{N_\epsilon(\pmb{x}_j)}\geqslant MinPts$ <b>then</b>

    5:            将样本$\pmb{x}_j$加入核心对象集合: $\Omega=\Omega\bigcup\{\pmb{x}_j\}$

    6:        <b>end if</b>

    7: <b>end for</b>

    8:初始化聚类簇数: $k=0$

    9:初始化未访问样本集合: $\Gamma=D$

    10:<b>while</b> $\Omega\neq\varnothing$ <b>do</b>

    11:   记录当前未访问样本集合: $\Gamma_\text{old}=\Gamma$;

    12:   随机选取一个核心对象$\pmb{o}\in\Omega$, 初始化队列$Q=<\pmb{o}>$;

    13:   $\Gamma=\Gamma\setminus\{\pmb{o}\}$;

    14:   <b>while</b> $Q\neq\varnothing$ <b>do</b>

    15:       取出队列$Q$中的首个样本$\pmb{q}$;

    16:       <b>if</b> $\abs{N_\epsilon(\pmb{q})}\geqslant MinPts$ <b>then</b>

    17:           令$\Delta=N_\epsilon(\pmb{q})\bigcap\Gamma$;

    18:           将$\Delta$中的样本加入队列$Q$;

    19:           $\Gamma=\Gamma\setminus\Delta$;

    20:       <b>end if </b>

    21:   <b>end while</b>

    22:   $k=k+1$, 生成聚类簇$C_k=\Gamma_\text{old}\setminus\Gamma$;

    23:   $\Omega=\Omega\setminus C_k$

    24:<b>end while</b>

    <b>输出</b>: 簇划分$C=\{C_1, C_2, ..., C_k\}$

    ---

## 9.6.层次聚类

* 层次聚类(hierarchical clustering)试图在不同层次对数据集进行划分, 从而形成树形的聚类结构.

* 数据集的划分可采用自底向上的聚合策略, 也可采用自顶向下的拆分策略.

* AGNES(AGglomerative NESting)是一种采用自底向上聚合策略的层次聚类算法.

    * 先将数据集中的每个样本看作一个初始聚类簇, 然后在算法运行的每一步中找出距离最近的两个聚类簇进行合并, 该过程不断重复, 直至达到预设的聚类簇个数.

* AGNES的关键是如何计算聚类簇之间的距离.

    * 每个簇是一个样本集合, 只需采用关于集合的某种距离即可. 
    * 集合间的距离计算常采用豪斯多夫距离(Hausdorff distance).
    * 给定聚类簇$C_i$与$C_j$可通过下面的式子来计算距离:
        * 最小距离: $d_\min(C_i, C_j)=\min\limits_{\pmb{x}\in C_i, \pmb{z}\in C_j}\text{dist}(\pmb{x}, \pmb{z})$
        * 最大距离: $d_\max(C_i, C_j)=\max\limits_{\pmb{x}\in C_i, \pmb{z}\in C_j}\text{dist}(\pmb{x}, \pmb{z})$
        * 平均距离: $d_\text{avg}(C_i, C_j)=\frac{1}{\abs{C_i}\abs{C_j}}\sum\limits_{\pmb{x}\in C_i}\sum\limits_{\pmb{z}\in C_j}\text{dist}(\pmb{x}, \pmb{z})$
    * 最小距离由两个簇的最近样本决定, 最大距离由两个簇的最远样本决定, 而平均距离则由两个簇的所有样本共同决定.
    * 当聚类簇距离由$d_\min$、$d_\max$或$d_\text{avg}$计算时, AGNES 算法被相应地称为单链接(single-linkage)、全链接(complete-linkage)或均链接(average-linkage)算法.

* AGNES 算法描述

    ---

    <b>输入:</b> 样本集$D=\{\pmb{x}_1, \pmb{x}_2, ...,\pmb{x}_m\}$;

    ​          聚类簇距离度量函数$d$;

    ​          聚类簇数$k$.

    <b>过程:</b>

    1:<b>for</b> $j=1,2,...,m$ <b>do</b>

    2:    $C_j=\{\pmb{x}_j\}$

    3:<b>end for</b>

    4:<b>for</b> $i=1, 2, ..., m$ <b>do</b>

    5:    <b>for</b> $j=i+1,...,m$ <b>do</b>

    6:        $M(i,j)=d(C_i,C_j)$;

    7:        $M(j,i)=M(i,j)$

    8:    <b>end for</b>

    9:<b>end for</b>

    10:设置当前聚类簇个数: $q=m$

    11:<b>while</b> $q>k$ <b>do</b>

    12:   找出距离最近的两个聚类簇$C_{i^*}$和$C_{j^*}$;

    13:   合并$C_{i^*}$和$C_{j^*}$: $C_{i^*}=C_{i^*}\bigcup C_{j^*}$;

    14:   <b>for</b> $j=j^*+1,j^*+2,...,q$ <b>do</b>

    15:       将聚类簇$C_j$重编号为$C_{j-1}$

    16:   <b>end for</b>

    17:   删除距离矩阵$M$的第$j^*$行与第$j^*$列;

    18:   <b>for</b> $j=1,2,...,q$ <b>do</b>

    19:       $M(i^*,j)=d(C_{i^*},C_j)$;

    20:       $M(j,i^*)=M(i^*,j)$

    21:   <b>end for</b>

    22:   $q=q-1$

    23:<b>end while</b>

    <b>输出</b>: 簇划分$C=\{C_1, C_2, ..., C_k\}$

    ---

    * $d$ 通常使用$d_\min$, $d_\max$, $d_\text{avg}$.
    * $i^*<j^*$.
