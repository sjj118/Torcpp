#ifndef TORCPP_OPTIM_H
#define TORCPP_OPTIM_H

template<class T>
class Optimizer {
public:
    vector<Variable<T>> parameters;

    Optimizer(const vector<Variable<T>> &vec) {
        add_parameters(vec);
    }

    void add_parameter(const Variable<T> &var) {
        parameters.push_back(var);
    }

    void add_parameters(const vector<Variable<T>> &vec) {
        for (auto &it:vec)parameters.push_back(it);
    }

    void zero_grad() {
        for (auto &var:parameters) var.grad.zero_();
    }

    virtual void update(Variable<T> &var) = 0;

    void step() {
        for (auto &var:parameters)update(var);
    }
};

template<class T>
class SGD : public Optimizer<T> {
public:
    T lr;

    SGD(const vector<Variable<T>> &vec, T lr) : Optimizer<T>(vec), lr(lr) {}

    void update(Variable<T> &var) override {
        var.tensor -= lr * var.grad;
    }
};

#endif //TORCPP_OPTIM_H
