#include<iostream>
#include<random>
#include"torcpp.h"

using namespace std;

const int n = 12, batch_size = 8;
double lr = 0.05;

class XorNet : public Module<double> {
public:
    Variable<double> W = Tensor<double>::rand({n, n}, rd::randn).autograd();
    Variable<double> c = Tensor<double>::rand({n, 1}, rd::randn).autograd();
    Variable<double> w = Tensor<double>::rand({1, n}, rd::randn).autograd();
    Variable<double> b = Tensor<double>::rand({1, 1}, rd::randn).autograd();

    XorNet() : Module<double>() {
        add_parameters({W, c, w, b});
    }

    Variable<double> forward(Variable<double> x) const override {
        x = (W % x + c).relu();
        x = (w % x + b).sigmoid();
        return x;
    }
};

int main() {
    auto inputs_ = Tensor<int>::zeros({1 << n, n, 1});
    for (int i = 0; i < (1 << n); i++) {
        vector<int> vec(n);
        unsigned int t = i;
        for (int j = 0; j < n; j++)vec[j] = t & 1, t >>= 1;
        inputs_[i] = vec;
    }
    auto net = XorNet();
    auto criterion = [](const Variable<double> &output, const Variable<double> &target) -> Variable<double> {
//        auto det = target - output;
//        auto loss = det * det / 2.0;
        auto loss = -(target * output.log() + (1 - target) * (1 - output).log());
        return loss.mean();
    };
    auto opt = SGD<double>(net.parameters, lr);
    for (int epoch = 0; epoch < 600; epoch++) {
        if (epoch % 200 == 199)opt.lr /= 10;
        auto s_inputs = inputs_.shuffle().view({0, batch_size, n, 1});
        auto labels = s_inputs.XOR(2).astype<double>();
        auto inputs = s_inputs.astype<double>();
        double running_loss = 0;
        int total = 0, correct = 0;
        for (int batch = 0; batch < inputs.sizes[0]; batch++) {
            auto input = inputs[batch];
            auto label = labels[batch];
            auto output = net(input);
            auto loss = criterion(output, label);
            loss.backward();
            opt.step();
            opt.zero_grad();
            running_loss += loss.item();
            total += input.sizes[0];
            auto predicted = (output.tensor > 0.5).astype<int>();
            correct += (predicted == label.astype<int>()).astype<int>().sum().item();
        }
        cout << epoch << " " << 100.0 * correct / total << "% " << running_loss / inputs.sizes[0] << endl;
    }
    for (int i = 0; i < (1 << n); i++) {
        vector<int> vec(n);
        unsigned int t = i;
        for (int j = 0; j < n; j++)vec[j] = t & 1, t >>= 1;
        auto input = Tensor<int>({n}, vec).astype<double>();
        int target = 0;
        for (auto it:vec)target ^= it;
        auto output = net(input).item();
        cout << input << " " << target << " " << output << endl;
    }
    return 0;
}