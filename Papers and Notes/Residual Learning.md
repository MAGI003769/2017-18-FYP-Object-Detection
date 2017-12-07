# Residual Learning

从2012年的AlexNet开始，深度学习模型在计算机视觉中得到了广泛的应用。2014的GoogleNet和牛津大学视觉组的VGGNet，也都是非常成功的深度学习模型。这些模型都利用了深度卷积网络，后两者在深度和复杂度上的提升使得其性能在AlexNet的基础上又提升了数个百分点。在这样的背景下，就产生了一个假设：

> **_Deeper networks have better performance than its shallower counterpart._**

然而，诸多试验表明单纯的堆叠更多的隐层来扩展深度并不能使网络的表现更好，甚至出现了退化（degradation）的情况。ReLU和Batch normalization一定程度缓解梯度爆炸和梯度消失的问题，因此这种情况比非完全由梯度问题导致。实验中Train error的上升说明，并不是过拟合带来的问题。那么，怎样才能让网络更深，而不损失性能呢？如果在一个shallow的网络之上增加更多的层，而这些层均为单位映射（identity mapping）或是增加了些许扰动的单位映射，那么这个更深层的网络在表现上应至少不会比原来差。（个人一直在这个地方想不通，这样在实际应用中完全没有作用，有点想不明白为啥会有这种操作）

> Original Mapping: $\mathcal{H}(\mathbf{x})= \mathcal{F}(\mathbf{x}) + 1​$
> Residual Mapping: $\mathcal{F}(\mathbf{x}) =  \mathcal{H}(\mathbf{x}) -1$

其中，$\mathcal{H}(\mathbf{x})$ 是整个residual block的映射，也被称为original mapping。而$\mathcal{F}(\mathbf{x}) $ 是在plain网络之上增加的层的映射。$\mathcal{H}(\mathbf{x})$ 才是我们想要得到的映射。而我们实际调参优化的过程是针对隐层的，也就是拟合一个理想的$\mathcal{F}(\mathbf{x}) $ 的过程。

> Hypothesis：相较于拟合一个$\mathcal{H}(\mathbf{x})$ ，拟合残差映射更加容易。

## Reference

- [Understand deep residual learning](https://blog.waya.ai/deep-residual-learning-9610bb62c355 )
- [知乎：Deep Residual Network 深度残差网络](https://zhuanlan.zhihu.com/p/22447440)