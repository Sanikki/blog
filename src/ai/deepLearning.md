
> 本文章的内容和知识为作者学习期间总结且部分内容借助AI理解，可能存在一定错误与误区，期待各位读者指正。 
> 本文章中的部分例子仅用于生动解释相关概念，切勿结合实际过度解读。 
> 语雀链接：[《神经网络概述》](https://www.yuque.com/sanikki/zotbbt/sergut5tnh733o9e?singleDoc#)
> 部分内容来源：[B站：李哥考研](https://space.bilibili.com/1009513616?spm_id_from=333.337.0.0)
>本章内容已经更新完成，如有不足之处，不妨评论区一叙
# 线性函数与多层神经元
## 多层神经元

&emsp;&emsp;上次我们通过构建一个线性函数进行预测，但线性函数显然是有局限性的，我们仅依靠一个线性函数是无法找到想要的结果，它太简单了，就像你无法找到一个$y = ax+b$使其拟合于$y = x^{2}$，一个神经元（线性函数）无法完成任务，我们需要更为复杂度神经网络。

&emsp;&emsp;下图是一个典型的<font style="color:rgba(0, 0, 0, 0.85);">人工神经网络（Artificial Neural Network，ANN）的结构，可分为输入、隐藏、输出三层。</font>

**输入层（INPUT LAYER）**，图中用橙色表示。输入层的神经元负责接收外部数据，这些数据是网络进行处理的原始信息。

**隐藏层（HIDDEN LAYERS）**：图中用绿色表示，位于输入层和输出层之间。隐藏层的神经元不直接与外部交互，它们对输入数据进行内部处理和转换。一个神经网络可以有多个隐藏层，每个隐藏层都在前一层的基础上进一步提取和处理特征。

**输出层（OUTPUT LAYER）**：图中用红色表示，位于最右侧。输出层的神经元负责产生网络的最终输出，这个输出可以是分类结果、预测值等，取决于网络的应用场景。

&emsp;&emsp;图中每一个圆圈代表着一个神经元，神经元之间的连线代表权重连接，这些连接上的权重决定了信息从一个神经元传递到另一个神经元时的强度和方向。虽然这个图片看起来复杂，但其内部仍是线性关系。



<img src=https://i-blog.csdnimg.cn/img_convert/3df190ff36f8fcf07e9376db74ee2aec.png width=70%>

&emsp;&emsp;就像人类的大脑神经一样，当你面前有一盘麻婆豆腐时，你会从色、香、味多个角度，用眼、鼻、舌多个器官去判断这是不是一盘美味佳肴。与之类似，神经网络是由多个神经元同时接收信息，共同做出决断，这样的多层神经元相比之前单一的神经元是不是作用更强了呢？


<img src=https://i-blog.csdnimg.cn/img_convert/5dc8e3a29930693a73999ece9a549afb.png width=60%>

## 神经元与矩阵

&emsp;&emsp;显然，多个神经元的模型更为复杂，单个神经元的模型为$y=wx+b$,而此时由于有多个输入值，因此需要根据其不同权重进行加和，从而得出新模型为$r_{1}=b_{1} + w_{11}x_{1}+ w_{12}x_{2}+ w_{13}x_{3}+ w_{14}x_{4}$




<img src=https://i-blog.csdnimg.cn/img_convert/e991d09ab1f3711ad00b6d7652df7d9a.jpeg width=55%>

如果我们把所有的公式都写出来：

$$ 
\begin{aligned}
r_{1}=b_{1} + w_{11}x_{1}+ w_{12}x_{2}+ w_{13}x_{3}+ w_{14}x_{4}\\
r_{2}=b_{2} + w_{21}x_{1}+ w_{22}x_{2}+ w_{23}x_{3}+ w_{24}x_{4}\\
r_{3}=b_{3} + w_{31}x_{1}+ w_{32}x_{2}+ w_{33}x_{3}+ w_{34}x_{4}\\
\end{aligned}
$$
仔细观察，这些公式可以用线性代数中的矩阵来表示

输入层 => 隐藏层：$r = b +wx$

$$\begin{bmatrix}r_{1}\\r_{2}\\r_{3}\end{bmatrix}=
\begin{bmatrix}b_{1}\\b_{2}\\b_{3}\end{bmatrix}+
\begin{bmatrix}w_{11} & w_{12} & w_{13} & w_{14}\\w_{21} & w_{22} & w_{23} & w_{24}\\w_{31} & w_{32} & w_{33} & w_{34}
\end{bmatrix}\begin{bmatrix}x_{1}\\x_{2}\\x_{3}\\x_{4}\end{bmatrix}$$

隐藏层 => 输出层：$y_{0}= b_{0} + C^{T}r = b + C^{T}( b+wx)$

$$y_{0} = b_{0}+
\begin{bmatrix} c_{1}&c_{2}&c_{3}\end{bmatrix}
\begin{bmatrix} r_{1} \\ r_{2}\\r_{3}\end{bmatrix}$$

请注意，我们用$b_{0}$来表示单一数字，用$b$来表示矩阵$\begin{bmatrix}b_{1}\\b_{2}\\b_{3}\end{bmatrix}$

## 神经元的串联

&emsp;&emsp;可是，无论我们增加多少层，串联多少神经元，似乎只起到了传递的作用，这样的一根神经元和多层神经网络好像没有本质上的区别。以下图为例

<img src=https://i-blog.csdnimg.cn/img_convert/72d90657f01fcd914eb81a05899053fb.png width=40%>


$$
\begin{aligned}
&r_{1}  =b_{1}+w_{11}x_{1}+w_{12}x_{2} \\
&r_{2}= b_{2}+w_{21}x_{1}+w_{22}x_{2}  \\
&z_{1} = b+w_{1}r_{1}+w_{2}r_{2}=b+w_{1}(b_{1}+w_{11}x_{1}+w_{12}x_{2})+w_{2}
(b_{2}+w_{21}x_{1}+w_{22}x_{2}) \\
&z_{1}=(b+w_{1}b_{1}+w_{2}b_{2})+(w_{1}w_{11}+w_{2}w_{21})x_{1}+(w_{1}w_{12}+w_{2}w_{22})x_{2}
\end{aligned}
$$
虽然模型变得更为复杂，但也很难摆脱线性函数的本质，此时我们就需要借助下一个函数——激活函数

# 激活函数与非线性因素

&emsp;&emsp;激活函数是神经网络中非常重要的组成部分。它是一种非线性函数，用于对神经元的输入进行处理，从而给神经网络引入非线性因素。如果没有激活函数，多层神经网络就等同于一个线性模型。因为多个线性变换的组合仍然是线性变换，这样神经网络就无法处理复杂的非线性问题。 引入激活函数之后，由于激活函数都是非线性的，这样就给神经元引入了非线性元素，使得神经网络可以逼近任何非线性函数，这样使得神经网络应用到更多非线性模型中。

常见的激活函数有：

$$sigmoid:S(x)=\frac{1}{1+e^{-x}}=\frac{e^x}{e^{x}+1}$$
<img src=https://i-blog.csdnimg.cn/img_convert/0db4150cd3cc8a9059a9bbe5ad1ace37.png width=50%>

$$relu:f(x)=max(0,x)$$

<img src=https://i-blog.csdnimg.cn/img_convert/68f06511805e955fab2ab4ab98b8707a.png width=50%>


**这些激活函数都有着一个特征：能求导**

> <font style="color:rgba(0, 0, 0, 0.85);">激活函数可导是为了能够利用高效的基于梯度的优化算法来训练神经网络，使得网络能够通过不断调整参数来学习数据中的模式和规律。梯度优化算法需要计算损失函数对网络中参数（权重和偏置）的梯度来更新参数，以最小化损失函数。这就要求激活函数是可导的。</font>

在激活函数的参与下，对隐藏层的输出结果进行一次转变，预测模型变为：

$$y_0=b+\sum_{i=1}^{3}c_{i}sigmoid(b_i+\sum_{j=1}^{3}w_{ij}x_{j})$$


<img src=https://i-blog.csdnimg.cn/img_convert/81e917bc52fe892c76ef5c77f162c90c.png width=75%>

&emsp;&emsp;上图中右侧输入层到隐藏层的步骤不变。$r_i$为原线性函数的输出值，将$r_i$作为自变量，带入$sigmoid$函数中，得到结果$a_i$，再根据每个值的权重$c_i$，重新计算得到预测值$y_0$。

这些激活函数真的有这么厉害吗？让我们来体会一下：

<img src=https://i-blog.csdnimg.cn/img_convert/60cfecc44b2030bf638a6a503cab5811.png width=75%>

<img src=https://i-blog.csdnimg.cn/img_convert/ca74f46e39c353f6a5aa689f3fc4f848.png width=80%>

引入了激活函数以后，是不是曲线一下子变得丝滑起来？也更贴合测试数据了。

# 过拟合&欠拟合

&emsp;&emsp;在引入激活函数之后，我们的模型更加准确了，但问题也随之而来，在我们训练模型的过程中，可能会出现过拟合和欠拟合的情况，这往往意味着模型出现了问题，这需要我们根据训练结果判断情况，然后做出相应的调整。这就需要我们了解过拟合和欠拟合这两种情况

&emsp;&emsp;**过拟合（Overfit）**<font style="color:rgba(0, 0, 0, 0.85);">是指模型在训练数据上表现得非常好，但在新的测试数据上表现很差的现象。这意味着模型过度学习了训练数据中的噪声和局部特征，而没有学到数据的一般规律。</font>

<font style="color:rgba(0, 0, 0, 0.85);">&emsp;&emsp;下图就是一个过拟合的例子，我们能清楚的看到，虽然该模型很好地拟合了训练数据中的每一个数据点，但却没有找到一般规律，导致该模型在新的测试数据上误差很大。这可能是</font>**<font style="color:rgba(0, 0, 0, 0.85);">模型的复杂度过高，</font>**<font style="color:rgba(0, 0, 0, 0.85);">或者</font>**<font style="color:rgba(0, 0, 0, 0.85);">训练的数据不足</font>**<font style="color:rgba(0, 0, 0, 0.85);">等原因造成的。</font>


<img src=https://i-blog.csdnimg.cn/img_convert/72d8ca92c978c5a7043fc9a59c6fcc49.png width=30%>

&emsp;&emsp;**<font style="color:rgba(0, 0, 0, 0.85);">欠拟合（Underfit）</font>**<font style="color:rgba(0, 0, 0, 0.85);">是指模型没有很好地捕捉到数据中的特征和规律。简单来说，模型太简单了，无法对训练数据进行有效的学习。</font>

&emsp;&emsp;<font style="color:rgba(0, 0, 0, 0.85);">下图就是一个欠拟合的例子，这种情况表现为，模型的准确率较低，并且随着训练迭代次数增加，模型在训练数据上的误差并没有明显下降。这可能是</font>**<font style="color:rgba(0, 0, 0, 0.85);">模型的复杂度较低，</font>**<font style="color:rgba(0, 0, 0, 0.85);">或者</font>**<font style="color:rgba(0, 0, 0, 0.85);">特征选择不当</font>**<font style="color:rgba(0, 0, 0, 0.85);">等原因造成的。</font>

<img src=https://i-blog.csdnimg.cn/img_convert/478929ab095fa99dcda80b68a8ea6f50.png width=30%>

# 小记

经历了两节课的学习，我们再来看深度学习的训练过程，是不是更清晰了一些呢？



<img src=https://i-blog.csdnimg.cn/img_convert/a06947d722b8b342556be00b159d4d61.png width=75%>

实际上，事情往往不会那么简单，实际的案例会更加复杂

<img src=https://i-blog.csdnimg.cn/img_convert/e6baa1ef97347c675e09efdf8f446d59.png width=75%>
<img src=https://i-blog.csdnimg.cn/img_convert/c7c8426a616c9d412de44e14249fd5d1.png width=75%>

<font style="color:rgb(31, 35, 41);">我们只需要把神经网络看作一个</font>**<font style="color:rgb(31, 35, 41);">黑匣子</font>**<font style="color:rgb(31, 35, 41);">，无需过于关注内部详细过程，反正我们只是通过输入得到了输出，只要结果拟合实际，那就达到了我们想要的结果。</font>
<img src=https://i-blog.csdnimg.cn/img_convert/37d2a60d6f1cdd1971af17cf30b250df.png width=75%></center>
