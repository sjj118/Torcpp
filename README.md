# Torcpp

## 简介

在 C++ 中对 Pytorch 基本功能的简易实现。

## 主要模块

- Tensor 类：基础设施，实现了多维数组
- autograd 模块：实现了自动求导功能
- nn 模块：高层模块：实现了部分模型、优化器、损失函数（目前只实现了很少的算法，其它优化器、损失函数等都可以较容易的添加）

## 使用示例

以下示例的代码都可以在 `example/tutorial.cpp` 中找到。

使用 torcpp 库需要在代码开头加上：

``` c++
#include "torcpp.h"
```

### Tensor

Tensor 类是基础核心类，实现了一个多维数组，相当于 numpy 中的 ndarray，pytorch 中的 tensor（不包括自动求导）。

常用初始化方式：

```C++
// 创建一个 2x3 的 double 类型 Tensor，并给定初始值
auto a = Tensor<double>({2, 3}, {1, 2, 3, 4, 5, 6});
cout << a <<endl;
/*
[[1,2,3],
 [4,5,6]]
*/

// 创建一个 2x2 的全零 Tensor
auto b = Tensor<double>::zeros({2, 2});
cout << b <<endl;
/*
[[0,0],
 [0,0]]
*/

// 使用正态分布随机生成初始值
auto c = Tensor<double>::rand({2, 2}, rd::randn);
cout << c <<endl;
/*
[[0.507302,0.799722],
 [0.095154,0.873764]]
*/

// 生成左闭右开的区间 [0,6)
auto d = Tensor<double>::range(0, 6);
cout << d <<endl;
/*
[0,1,2,3,4,5]
*/

// 生成对角元素给定的矩阵
auto e = Tensor<double>::diag({1, 2, 3});
cout << e << endl;
/*
[[1,0,0],
 [0,2,0],
 [0,0,3]]
*/
```

常用操作：

```C++
// 更改形状
cout << a.view({3, 2}) << endl;
/*
[[1,2],
 [3,4],
 [5,6]]
*/

// 扁平化
cout << a.flatten() << endl;
/*
[1,2,3,4,5,6]
*/

// 转置
cout << a.transpose(0,1) << endl;
/*
[[1,4],
 [2,5],
 [3,6]]
*/

// 随机打乱第一维的顺序
cout << e.shuffle() << endl;
/*
[[0,2,0],
 [1,0,0],
 [0,0,3]]
*/
```

选择与赋值：

```C++
// 选择一行
cout << a[1] << endl;
/*
[4,5,6]
*/

// 选择一个元素
cout << a[1][2] << endl;
/*
6
*/

// 对一个元素赋值
a[1][2] = 0;
cout << a << endl;
/*
[[1,2,3],
 [4,5,0]]
*/

// 用 Tensor 给另一个 Tensor 赋值时必须调用 set_value 方法
a[1].set_value(a[0]);
cout << a << endl;
/*
[[1,2,3],
 [1,2,3]]
*/

// 对一行赋值
a[1] = {4, 5, 6};
cout << a << endl;
/*
[[1,2,3],
 [4,5,6]]
*/
```

常用运算：

```C++
// 加减乘除等运算会对 Tensor 中的每一对元素作用
cout << a[0] + a[1] << endl;
/*
[5,7,9]
*/

// 可以让整个 Tensor 与一个数运算
cout << a - 1 << endl;
/*
[[0,1,2],
 [3,4,5]]
*/

// 不同形状的 Tensor 会在 broadcast 后进行运算
cout << a + a[0] << endl;
/*
[[2,4,6],
 [5,7,9]]
*/

// 矩阵乘法，同样支持 broadcast
cout << a % a.transpose() << endl;
/*
[[14,32],
 [32,77]]
*/

// 支持 sigmoid, softmax, relu 等常用函数
cout << a.softmax() << endl;
/*
[[0.00426978,0.0116065,0.0315496],
 [0.0857608,0.233122,0.633691]]
*/

// map 方法可以将任意提供的函数作用在每个元素上
cout << a.map([](double x) { return x > 3.5 ? 4 : 3; }) << endl;
/*
[[3,3,3],
 [4,4,4]]
*/

// 支持比较与逻辑运算
cout << ((a[0] < a[1]) && (a[1] > 4.5)) << endl;
/*
[0,1,1]
*/

// 支持 all, any 等运算
cout << (a[0] < a[1]).all() << endl;
/*
1
*/

// 支持 sum, prod, max, min 等运算
cout << a.prod() << endl;
/*
720
*/

// 这些运算还支持沿着某一维作用
cout << a.sum(0) << endl;
/*
[[5,7,9]]
*/

// 支持 argmax, argmin 运算，其中输出的下标还支持这一维的下标或者全局下标
cout << a.argmax(1) << endl;
/*
[[2],
 [2]]
*/
cout << a.global_argmax(1) << endl;
/*
[[2],
 [5]]
*/

// reduce 方法可以使用任意可结合的函数，将整个 Tensor 的所有元素合并
cout << a.reduce([](double x, double y) { return int(round(x)) | int(round(y)); }) << endl;
/*
7
*/
```

### autograd

使用 Tensor.autograd 方法将 Tensor 转换成 Variable 并记录导数：

```C++
Variable x = a.autograd();
```

Variable 在运算时会自动构建运算图，调用 backward 方法进行反向传播，导数记录在成员变量 Variable.grad 内。

```c++
auto y = (x * x + x).sum();		// 导数应为 2x+1
y.backward();
cout << x.grad << endl;
/*
[[3,5,7],
 [9,11,13]]
*/

x.grad.zero_();					// 清空导数
x.prod().backward();
cout << x.grad << endl;
/*
[[720,360,240],
 [180,144,120]]
*/
```

Tensor 类支持的函数，Variable 类大部分都支持并且支持自动微分。

### nn

此部分的完整代码见 `example/xor.cpp`，这是一个使用该库构建网络学习异或问题的实例。

输入一个 n 维 01 向量，输出这些 01 的异或和。

首先编写神经网络模型，带有一层隐藏层的网络：

```C++
class XorNet : public Module<double> {
public:
    Variable<double> W = Tensor<double>::rand({n, n}, rd::randn).autograd();
    Variable<double> c = Tensor<double>::rand({n, 1}, rd::randn).autograd();
    Variable<double> w = Tensor<double>::rand({1, n}, rd::randn).autograd();
    Variable<double> b = Tensor<double>::rand({1, 1}, rd::randn).autograd();

    XorNet() : Module<double>() {
        add_parameters({W, c, w, b});	// 将这些 Variable 加入该模型的参数列表
    }

    Variable<double> forward(Variable<double> x) const override {
        x = (W % x + c).tanh();
        x = (w % x + b).sigmoid();
        return x;
    }
};
```

创建模型、优化器、损失函数：

```C++
auto net = XorNet();
auto criterion = [](auto &output, auto &target) -> auto {
    auto loss = -(target * output.log() + (1 - target) * (1 - output).log());
    return loss.mean();
};
auto opt = SGD<double>(net.parameters, lr);
```

略去数据初始化代码，只看核心部分对每个 batch 的循环：

```C++
for (int batch = 0; batch < inputs.sizes[0]; batch++) {
    auto input = inputs[batch];
    auto label = labels[batch];
    auto output = net(input);				// 前向传播
    auto loss = criterion(output, label);	// 计算损失
    loss.backward();						// 反向传播
    opt.step();								// 一轮优化
    opt.zero_grad();						// 清空导数
    running_loss += loss.item();
    total += input.sizes[0];
    auto predicted = (output.tensor > 0.65).astype<int>();
	correct += (predicted == label.astype<int>()).astype<int>().sum().item();
}
```

核心训练过程编写方式与 Pytorch 基本一致。