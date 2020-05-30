#ifndef TORCPP_MODULE_H
#define TORCPP_MODULE_H

#include "../autograd.h"

template<class T>
class Module {
public:
    vector<Variable<T>> parameters;

    Module() = default;

    void add_parameter(const Variable<T> &var) {
        parameters.push_back(var);
    }

    void add_parameters(const vector<Variable<T>> &vec) {
        for (auto &it:vec)parameters.push_back(it);
    }

    virtual Variable<T> forward(Variable<T> x) const = 0;

    Variable<T> operator()(const Variable<T> &x) const {
        return forward(x);
    }
};

#endif //TORCPP_MODULE_H
