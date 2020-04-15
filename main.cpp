//#define NDEBUG
#include <cassert>
#include <iostream>
#include "Tensor.h"

using namespace std;

auto fun() {
    auto x = Tensor<double>({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    return x[1].divorce();
}

void TestTensor() {
    auto x = Tensor<double>({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    cout << x << endl;
//    x[0] = {0, 0, 0, 0};
//    cout << fun() << endl;
//    cout << x << endl;
//    cout << x[1][0][1] << endl;
//    x[1] = {0, 0, 0, 0};
//    auto y = x.flatten(0,1);
//    cout << y << endl;
//    auto s = Tensor<double>::scalar(0.5);
//    cout << s << endl;
//    auto d = Tensor<double>::diag({1, 2, 3, 4});
//    cout << d << endl;
//    auto r = Tensor<double>::rand({4, 4}, [] { return double(rand() % 1000) / 1000; });
//    cout << r << endl;
}

int main() {
    TestTensor();
}
