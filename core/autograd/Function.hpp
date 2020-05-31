#ifndef TORCPP_FUNCTION_HPP
#define TORCPP_FUNCTION_HPP

#include "Variable.hpp"

using std::vector;
using std::pair;
using std::shared_ptr;
using std::initializer_list;

template<class T>
class Variable;

template<class T>
class Function {
public:
    vector<shared_ptr<Function<T>>> next_functions;
    Tensor<T> grad_accumulator;
    size_t degree = 0;

    explicit Function<T>(const vector<size_t> &sizes) : grad_accumulator(Tensor<T>::zeros(sizes)) {}

    void add_edge(shared_ptr<Function<T>> edge) {
        next_functions.push_back(edge);
    }

    virtual Tensor<T> apply(vector<Tensor<T>> &&inputs) = 0;

    virtual vector<Tensor<T>> calc_grads(const Tensor<T> &grad) = 0;

    void calc_degree() {
        for (auto &it:next_functions)
            if (it && it->degree++ == 0)it->calc_degree();
    }

    virtual void backward() {
        auto grads = calc_grads(grad_accumulator);
        assert(grads.size() == next_functions.size());
        for (index_t i = 0; i < grads.size(); i++) {
            if (next_functions[i]) {
                next_functions[i]->grad_accumulator += grads[i];
                if (--next_functions[i]->degree == 0)next_functions[i]->backward();
            }
        }
    }
};

template<class T>
class AccumulateGrad : public Function<T> {
public:
    Tensor<T> tensor;

    explicit AccumulateGrad(const Variable<T> &variable) : Function<T>(variable.tensor.sizes), tensor(variable.grad) {}

    Tensor<T> apply(vector<Tensor<T>> &&inputs) override {
        assert(inputs.size() == 1);
        return inputs[0];
    }

    vector<Tensor<T>> calc_grads(const Tensor<T> &grad) override {
        return {grad};
    }

    void backward() override {
        tensor += this->grad_accumulator;
        this->grad_accumulator.zero_();
    }
};

template<class T>
class CloneBackward : public Function<T> {
public:
    explicit CloneBackward(const Variable<T> &a) : Function<T>(a.tensor.sizes) {
        this->add_edge(a.grad_fn);
    }

    Tensor<T> apply(vector<Tensor<T>> &&inputs) override {
        assert(inputs.size() == 1);
        return inputs[0].clone();
    }

    vector<Tensor<T>> calc_grads(const Tensor<T> &grad) override {
        return {grad};
    }
};

template<class T>
class SelectBackward : public Function<T> {
public:
    vector<size_t> sizes;
    index_t ind;

    SelectBackward(const Variable<T> &a, index_t ind) :
            Function<T>(vector<size_t>(a.tensor.sizes.begin() + 1, a.tensor.sizes.end())),
            sizes(a.tensor.sizes),
            ind(ind < 0 ? ind + a.tensor.sizes[0] : ind) {
        this->add_edge(a.grad_fn);
    }

    Tensor<T> apply(vector<Tensor<T>> &&inputs) override {
        assert(inputs.size() == 1);
        return inputs[0][ind];
    }

    vector<Tensor<T>> calc_grads(const Tensor<T> &grad) override {
        auto output = Tensor<T>::zeros(sizes);
        output[ind] = grad.data();
        return {output};
    }
};

template<class T>
class ViewBackward : public Function<T> {
public:
    vector<size_t> sizes1, sizes2;

    ViewBackward(const Variable<T> &a, const vector<size_t> &sizes) : Function<T>(sizes),
                                                                      sizes1(a.tensor.sizes),
                                                                      sizes2(sizes) {
        this->add_edge(a.grad_fn);
    }

    Tensor<T> apply(vector<Tensor<T>> &&inputs) override {
        assert(inputs.size() == 1);
        return inputs[0].view(sizes2);
    }

    vector<Tensor<T>> calc_grads(const Tensor<T> &grad) override {
        return {grad.view(sizes1)};
    }
};

template<class T>
class ExpandBackward : public Function<T> {
public:
    vector<size_t> sizes1, sizes2;

    ExpandBackward(const Variable<T> &a, const vector<size_t> &sizes) : Function<T>(sizes),
                                                                        sizes1(a.tensor.sizes),
                                                                        sizes2(sizes) {
        this->add_edge(a.grad_fn);
    }

    Tensor<T> apply(vector<Tensor<T>> &&inputs) override {
        assert(inputs.size() == 1);
        return inputs[0].expand(sizes2);
    }

    vector<Tensor<T>> calc_grads(const Tensor<T> &grad) override {
        return {grad.fold(sizes1, [](T a, T b) { return a + b; })};
    }
};

template<class T>
class AddBackward : public Function<T> {
public:
    AddBackward(const Variable<T> &a, const Variable<T> &b) : Function<T>(a.tensor.sizes) {
        this->add_edge(a.grad_fn);
        this->add_edge(b.grad_fn);
    }

    Tensor<T> apply(vector<Tensor<T>> &&inputs) override {
        assert(inputs.size() == 2);
        return inputs[0] + inputs[1];
    }

    vector<Tensor<T>> calc_grads(const Tensor<T> &grad) override {
        return {grad, grad};
    }
};

template<class T>
class SubBackward : public Function<T> {
public:
    SubBackward(const Variable<T> &a, const Variable<T> &b) : Function<T>(a.tensor.sizes) {
        this->add_edge(a.grad_fn);
        this->add_edge(b.grad_fn);
    }

    Tensor<T> apply(vector<Tensor<T>> &&inputs) override {
        assert(inputs.size() == 2);
        return inputs[0] - inputs[1];
    }

    vector<Tensor<T>> calc_grads(const Tensor<T> &grad) override {
        return {grad, -grad};
    }
};

template<class T>
class MulBackward : public Function<T> {
public:
    Tensor<T> a, b;

    MulBackward(const Variable<T> &a, const Variable<T> &b) : Function<T>(a.tensor.sizes), a(a.tensor), b(b.tensor) {
        this->add_edge(a.grad_fn);
        this->add_edge(b.grad_fn);
    }

    Tensor<T> apply(vector<Tensor<T>> &&inputs) override {
        assert(inputs.size() == 2);
        return inputs[0] * inputs[1];
    }

    vector<Tensor<T>> calc_grads(const Tensor<T> &grad) override {
        return {b * grad, a * grad};
    }
};

template<class T>
class DivBackward : public Function<T> {
public:
    Tensor<T> a, b;

    DivBackward(const Variable<T> &a, const Variable<T> &b) : Function<T>(a.tensor.sizes), a(a.tensor), b(b.tensor) {
        this->add_edge(a.grad_fn);
        this->add_edge(b.grad_fn);
    }

    Tensor<T> apply(vector<Tensor<T>> &&inputs) override {
        assert(inputs.size() == 2);
        return inputs[0] / inputs[1];
    }

    vector<Tensor<T>> calc_grads(const Tensor<T> &grad) override {
        return {grad / b, -grad * a / b / b};
    }
};

template<class T>
class MmBackward : public Function<T> {
public:
    Tensor<T> a, b;

    MmBackward(const Variable<T> &a, const Variable<T> &b) : Function<T>({a.tensor.sizes[0], b.tensor.sizes[1]}),
                                                             a(a.tensor), b(b.tensor) {
        this->add_edge(a.grad_fn);
        this->add_edge(b.grad_fn);
    }

    Tensor<T> apply(vector<Tensor<T>> &&inputs) override {
        assert(inputs.size() == 2);
        return inputs[0].mm(inputs[1]);
    }

    vector<Tensor<T>> calc_grads(const Tensor<T> &grad) override {
        return {grad.mm(b.transpose()), a.transpose().mm(grad)};
    }
};

template<class T>
class BmmBackward : public Function<T> {
public:
    Tensor<T> a, b;

    BmmBackward(const Variable<T> &a, const Variable<T> &b)
            : Function<T>({a.tensor.sizes[0], a.tensor.sizes[1], b.tensor.sizes[2]}), a(a.tensor), b(b.tensor) {
        this->add_edge(a.grad_fn);
        this->add_edge(b.grad_fn);
    }

    Tensor<T> apply(vector<Tensor<T>> &&inputs) override {
        assert(inputs.size() == 2);
        return inputs[0].bmm(inputs[1]);
    }

    vector<Tensor<T>> calc_grads(const Tensor<T> &grad) override {
        return {grad.bmm(b.transpose()), a.transpose().bmm(grad)};
    }
};

template<class T>
class MapBackward : public Function<T> {
public:
    Tensor<T> a;
    function<T(T)> fun, rfun;

    explicit MapBackward(const Variable<T> &a, const function<T(T)> &fun, const function<T(T)> &rfun) :
            Function<T>(a.tensor.sizes), a(a.tensor), fun(fun), rfun(rfun) {
        this->add_edge(a.grad_fn);
    }

    Tensor<T> apply(vector<Tensor<T>> &&inputs) override {
        assert(inputs.size() == 1);
        return inputs[0].map(fun);
    }

    vector<Tensor<T>> calc_grads(const Tensor<T> &grad) override {
        return {a.map(rfun) * grad};
    }
};

template<class T>
class ReduceBackward : public Function<T> {
public:
    function<T(T, T)> fun;
    function<Tensor<T>(const Tensor<T> &, index_t)> rfun;
    Tensor<T> a;
    index_t dim = 0;

    ReduceBackward(const Variable<T> &a, const function<T(T, T)> &fun,
                   const function<Tensor<T>(const Tensor<T> &, index_t)> &rfun)
            : Function<T>(vector<size_t>(a.tensor.n, 1)), a(a.tensor), dim(-1), fun(fun), rfun(rfun) {
        this->add_edge(a.grad_fn);
    }

    ReduceBackward(const Variable<T> &a, const function<T(T, T)> &fun,
                   const function<Tensor<T>(const Tensor<T> &, index_t)> &rfun, index_t dim) :
            Function<T>(a.tensor._reduce_sizes(dim)),
            a(a.tensor), dim(dim < 0 ? dim + a.tensor.n : dim), fun(fun), rfun(rfun) {
        assert(0 <= dim && dim < this->a.n);
        this->add_edge(a.grad_fn);
    }

    Tensor<T> apply(vector<Tensor<T>> &&inputs) override {
        assert(inputs.size() == 1);
        if (dim != -1)return inputs[0].reduce(fun, dim);
        else return inputs[0].reduce(fun);
    }

    vector<Tensor<T>> calc_grads(const Tensor<T> &grad) override {
        return {rfun(a, dim) * grad.expand(a.sizes)};
    }
};

#endif //TORCPP_FUNCTION_HPP
