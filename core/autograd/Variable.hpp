#ifndef TORCPP_VARIABLE_H
#define TORCPP_VARIABLE_H

#include <cmath>
#include "../Tensor.h"
#include "Function.hpp"

template<class T>
class Function;

template<class T>
class GraphRoot;

template<class T>
class AccumulateGrad;

template<class T>
class AddBackward;

template<class T>
class SumBackward;

template<class T>
class Variable {
public:
    Tensor<T> tensor, grad;
    shared_ptr<Function<T>> grad_fn;

    Variable(const Tensor<T> &data, const Tensor<T> &grad) : tensor(data), grad(grad) {}

    Variable(const Tensor<T> &data, shared_ptr<Function<T>> grad_fn) : tensor(data), grad(Tensor<T>::scalar(0)),
                                                                       grad_fn(grad_fn) {}

    Variable(const T &x) {
        new(this)Variable<T>(x, shared_ptr<Function<T>>(nullptr));
    }

    friend ostream &operator<<(ostream &out, const Variable<T> &x) {
        return out << x.tensor;
    }

    void backward() {
        assert(tensor.size == 1);
        grad_fn->calc_degree();
        grad_fn->grad_accumulator = 1;
        grad_fn->backward();
    }

    T item() const {
        return tensor.item();
    }

    vector<T> data() const {
        return tensor.data();
    }

    Variable operator[](index_t ind) const {
        shared_ptr<Function<T>> grad_fn(new SelectBackward<T>(*this, ind));
        return Variable(grad_fn->apply({tensor}), grad_fn);
    }

    Variable clone() const {
        shared_ptr<Function<T>> grad_fn(new CloneBackward<T>(*this));
        return Variable(grad_fn->apply({tensor}), grad_fn);
    }

    Variable view(vector<size_t> new_sizes) const {
        tensor._view_sizes(new_sizes);
        shared_ptr<Function<T>> grad_fn(new ViewBackward<T>(*this, new_sizes));
        return Variable(grad_fn->apply({tensor}), grad_fn);
    }

    Variable flatten(index_t startDim = 0, index_t endDim = -1) const {
        return view(tensor._flatten_sizes(startDim, endDim));
    }

    Variable prepend_dim(size_t num = 1) const {
        return view(tensor._prepend_dim_sizes(num));
    }

    Variable append_dim(size_t num = 1) const {
        return view(tensor._append_dim_sizes(num));
    }

    Variable expand(const vector<size_t> &new_sizes) const {
        shared_ptr<Function<T>> grad_fn(new ExpandBackward<T>(*this, new_sizes));
        return Variable(grad_fn->apply({tensor}), grad_fn);
    }

    friend void broadcast(Variable<T> &a, Variable<T> &b, index_t start = 0, index_t end = -1) {
        if (a.tensor.sizes == b.tensor.sizes)return;
        if (a.tensor.n < b.tensor.n)a = a.prepend_dim(b.tensor.n - a.tensor.n);
        if (a.tensor.n > b.tensor.n)b = b.prepend_dim(a.tensor.n - b.tensor.n);
        auto[sa, sb] = _broadcast_sizes(a.tensor.sizes, b.tensor.sizes, start, end);
        a = a.expand(sa);
        b = b.expand(sb);
    }

    Variable map(const function<T(T)> &fun, const function<T(T)> &rfun) const {
        shared_ptr<Function<T>> grad_fn(new MapBackward<T>(*this, fun, rfun));
        return Variable(grad_fn->apply({tensor}), grad_fn);
    }

    Variable<T> mm(const Variable<T> &other) const {
        shared_ptr<Function<T>> grad_fn(new MmBackward<T>(*this, other));
        return Variable(grad_fn->apply({tensor, other.tensor}), grad_fn);
    }

    Variable<T> bmm(const Variable<T> &other) const {
        shared_ptr<Function<T>> grad_fn(new BmmBackward<T>(*this, other));
        return Variable(grad_fn->apply({tensor, other.tensor}), grad_fn);
    }

    Variable<T> matmul(const Variable<T> &other) const {
        auto a = *this, b = other;
        if (a.tensor.n == 1)a = a.prepend_dim();
        if (b.tensor.n == 1)b = b.append_dim();
        broadcast(a, b, 0, -3);
        vector<size_t> new_sizes(a.tensor.sizes.begin(), a.tensor.sizes.end() - 2);
        new_sizes.push_back(a.tensor.sizes[a.tensor.n - 2]);
        new_sizes.push_back(b.tensor.sizes[b.tensor.n - 1]);
        a = a.view({0, a.tensor.sizes[a.tensor.n - 2], a.tensor.sizes[a.tensor.n - 1]});
        b = b.view({0, b.tensor.sizes[b.tensor.n - 2], b.tensor.sizes[b.tensor.n - 1]});
        return a.bmm(b).view(new_sizes);
    }

    Variable<T> operator%(const Variable<T> &other) const {
        return matmul(other);
    }


#define MERGE_OPERATOR(OP, fun)\
    friend Variable<T> operator OP(const Variable<T> &a, const Variable<T> &b) {\
        if(a.tensor.sizes != b.tensor.sizes){\
            auto ta = a, tb = b;\
            broadcast(ta, tb);\
            return ta OP tb;\
        }\
        shared_ptr<Function<T>> grad_fn(new fun<T>(a, b));\
        return Variable(grad_fn->apply({a.tensor, b.tensor}), grad_fn);\
    }\


    MERGE_OPERATOR(+, AddBackward)

    MERGE_OPERATOR(-, SubBackward)

    MERGE_OPERATOR(*, MulBackward)

    MERGE_OPERATOR(/, DivBackward)

#undef MERGE_OPERATOR

    Variable<T> operator-() const {
        return map([](T x) -> T { return -x; }, [](T x) -> T { return -1; });
    }

#define REDUCE_FUNCTION(fun_name, fun_body, rfun_body)\
    Variable fun_name() const {\
        shared_ptr<Function<T>> grad_fn(new ReduceBackward<T>(\
            *this, [](T a, T b)->T fun_body, [](const Tensor<T> &input, index_t dim)->Tensor<T> rfun_body));\
        return Variable(grad_fn->apply({tensor}), grad_fn);\
    }\
    \
    Variable fun_name(index_t dim) const {\
        shared_ptr<Function<T>> grad_fn(new ReduceBackward<T>(\
            *this, [](T a, T b)->T fun_body, [](const Tensor<T> &input, index_t dim)->Tensor<T> rfun_body, dim));\
        return Variable(grad_fn->apply({tensor}), grad_fn);\
    }\


    REDUCE_FUNCTION(sum, { return a + b; }, { return Tensor<T>::scalar(1); })

    REDUCE_FUNCTION(prod, { return a * b; },
                    { if (dim == -1)return input.prod() / input; else return input.prod(dim) / input; })

    REDUCE_FUNCTION(max, { return std::max(a, b); }, {
        Tensor<index_t> arg;
        if (dim == -1)arg = input.global_argmax();
        else arg = input.global_argmax(dim);
        auto res = Tensor<T>::zeros_like(input);
        for (index_t i = 0; i < arg.size; i++)res.index(arg.index(i)) = 1;
        return res;
    })

    REDUCE_FUNCTION(min, { return std::min(a, b); }, {
        Tensor<index_t> arg;
        if (dim == -1)arg = input.global_argmin();
        else arg = input.global_argmin(dim);
        auto res = Tensor<T>::zeros_like(input);
        for (index_t i = 0; i < arg.size; i++)res.index(arg.index(i)) = 1;
        return res;
    })

#undef REDUCE_FUNCTION

#define MAP_FUNCTION(fun_name, fun_body, rfun_body)\
    Variable<T> fun_name() const{\
        return map([](T x)->T fun_body, [](T x)->T rfun_body);\
    }\


    MAP_FUNCTION(relu, { return x > 0 ? x : 0; }, { return x > 0 ? 1 : 0; })

    MAP_FUNCTION(sigmoid, { return 1 / (1 + std::exp(-x)); }, {
        auto t = 1 / (1 + std::exp(-x));
        return t * (1 - t);
    })

    MAP_FUNCTION(softplus, { return std::log(1 + std::exp(x)); }, { return 1 / (1 + std::exp(-x)); })

    MAP_FUNCTION(exp, { return std::exp(x); }, { return std::exp(x); })

    MAP_FUNCTION(log, { return std::log(x); }, { return 1 / x; })

    MAP_FUNCTION(tanh, { return std::tanh(x); }, {
        auto t = std::tanh(x);
        return 1 - t * t;
    })

#undef MAP_FUNCTION

    Variable softmax() const {
        auto _exp = this->exp();
        auto _sum = _exp.sum();
        return _exp / _sum;
    }

    Variable softmax(index_t dim) const {
        auto _exp = this->exp();
        auto _sum = _exp.sum(dim);
        return _exp / _sum;
    }

    Variable mean() const {
        return sum() / tensor.size;
    }

    Variable mean(index_t dim) const {
        return sum(dim) / tensor.sizes[dim];
    }
};

#endif //TORCPP_VARIABLE_H
