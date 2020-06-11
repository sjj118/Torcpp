#include "../torcpp.h"

using namespace std;

int main() {
    cout << "Tensor 类" << endl;
    cout << "常用初始化方式" << endl;
    auto a = Tensor<double>({2, 3}, {1, 2, 3, 4, 5, 6});
    auto b = Tensor<double>::zeros({2, 2});
    auto c = Tensor<double>::rand({2, 2}, rd::randn);
    auto d = Tensor<double>::range(0, 6);
    auto e = Tensor<double>::diag({1, 2, 3});
    cout << a << endl;
    cout << b << endl;
    cout << c << endl;
    cout << d << endl;
    cout << e << endl;
    cout << "常用操作" << endl;
    cout << a.view({3, 2}) << endl;
    cout << a.flatten() << endl;
    cout << a.transpose() << endl;
    cout << e.shuffle() << endl;
    cout << "选择与赋值" << endl;
    cout << a[1] << endl;
    cout << a[1][2] << endl;
    a[1][2] = 0;
    cout << a << endl;
    a[1].set_value(a[0]);
    cout << a << endl;
    a[1] = {4, 5, 6};
    cout << a << endl;
    cout << "常用运算" << endl;
    cout << a[0] + a[1] << endl;
    cout << a + a[0] << endl;
    cout << a % a.transpose() << endl;
    cout << a.softmax() << endl;
    cout << a.map([](double x) { return x > 3.5 ? 4 : 3; }) << endl;
    cout << ((a[0] < a[1]) && (a[1] > 4.5)) << endl;
    cout << (a[0] < a[1]).all() << endl;
    cout << a.prod() << endl;
    cout << a.sum(0) << endl;
    cout << a.argmax(1) << endl;
    cout << a.global_argmax(1) << endl;
    cout << a.reduce([](double x, double y) { return int(round(x)) | int(round(y)); }) << endl;

    cout << "autograd 模块" << endl;
    Variable x = a.autograd();
    auto y = (x * x + x).sum();
    y.backward();
    cout << x.grad << endl;
    x.grad.zero_();
    x.prod().backward();
    cout << x.grad << endl;
    return 0;
};