#include<iostream>
#include"torcpp.h"

using namespace std;

int main() {
    auto x = Tensor<double>({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    cout << x[0]%x[1];
    return 0;
}