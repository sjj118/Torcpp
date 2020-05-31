#ifndef TORCPP_TENSOR_H
#define TORCPP_TENSOR_H

#include <vector>
#include <numeric>
#include <functional>
#include <memory>
#include <iostream>
#include <cassert>
#include <cstring>
#include <cmath>
#include "random.h"

using std::memcpy;
using std::pair;
using std::default_delete;
using std::unique_ptr;
using std::shared_ptr;
using std::vector;
using std::function;
using std::istream;
using std::ostream;

typedef int index_t;

template<class T>
class Variable;

template<class T>
class Function;

template<class T>
class AccumulateGrad;

pair<vector<size_t>, vector<size_t>>
_broadcast_sizes(vector<size_t> a, vector<size_t> b, index_t start = 0, index_t end = -1);

template<class T>
class Tensor {
public:
    size_t size{};
    shared_ptr<T[]> storage;
    size_t n{}, offset{};
    vector<size_t> sizes, strides;

    Tensor() = default;

    ~Tensor() = default;

    Tensor(const Tensor &other) = default;

    Tensor &operator=(const Tensor &other) = default;

    Tensor(size_t size, const shared_ptr<T[]> &storage, size_t n, size_t offset,
           vector<size_t> sizes, vector<size_t> strides)
            : size(size), storage(storage), n(n), offset(offset),
              sizes(std::move(sizes)), strides(std::move(strides)) {
    }

    static vector<size_t> sizes2strides(const vector<size_t> &sizes) {
        size_t n = sizes.size();
        vector<size_t> strides(n);
        strides[n - 1] = 1;
        for (index_t i = n - 2; i >= 0; i--)strides[i] = strides[i + 1] * sizes[i + 1];
        return strides;
    }

    Tensor(const shared_ptr<T[]> &storage, size_t offset, vector<size_t> sizes)
            : storage(storage), offset(offset), n(sizes.size()), sizes(std::move(sizes)) {
        if (n == 0) {
            this->size = 1;
            return;
        }
        this->strides = sizes2strides(this->sizes);
        this->size = this->strides[0] * this->sizes[0];
    }

    operator Variable<T>() const { return nograd(); }

    Variable<T> nograd() const {
        return Variable(*this, shared_ptr<Function<T>>(nullptr));
    }

    Variable<T> autograd() const {
        Variable<T> a = Variable<T>(*this, Tensor<T>::zeros_like(*this));
        a.grad_fn = shared_ptr<Function<T>>(new AccumulateGrad<T>(a));
        return a;
    }

    Tensor<T> clone() const {
        shared_ptr<T[]> new_storage(new T[size]);
        memcpy(new_storage.get(), index_ptr(), size * sizeof(T));
        return Tensor<T>(size, new_storage, n, 0, sizes, strides);
    }

    Tensor(const vector<size_t> &sizes, vector<T> data) {
        size_t new_size = 1;
        for (auto &it:sizes)new_size *= it;
        shared_ptr<T[]> new_storage(new T[new_size]);
        std::copy(data.begin(), data.end(), new_storage.get());
        new(this)Tensor(new_storage, 0, sizes);
    }

    Tensor<T> operator[](index_t ind) {
        assert(n > 0);
        if (ind < 0)ind += sizes[0];
        assert(0 <= ind && ind < sizes[0]);
        vector<size_t> new_sizes(sizes.begin() + 1, sizes.end());
        vector<size_t> new_strides(strides.begin() + 1, strides.end());
        return Tensor(size / sizes[0], storage, n - 1, offset + ind * strides[0], new_sizes, new_strides);
    }

    Tensor<T> operator[](index_t ind) const {
        assert(n > 0);
        if (ind < 0)ind += sizes[0];
        assert(0 <= ind && ind < sizes[0]);
        vector<size_t> new_sizes(sizes.begin() + 1, sizes.end());
        vector<size_t> new_strides(strides.begin() + 1, strides.end());
        return Tensor(size / sizes[0], storage, n - 1, offset + ind * strides[0], new_sizes, new_strides);
    }

    T &operator[](const vector<index_t> &inds) {
        assert(inds.size() == n);
        for (index_t i = 0; i < n; i++)assert(-sizes[i] <= inds[i] && inds[i] < sizes[i]);
        size_t off = 0;
        for (index_t i = 0; i < n; i++)off += strides[i] * (inds[i] < 0 ? inds[i] + sizes[i] : inds[i]);
        return index(off);
    }

    T &operator[](const vector<index_t> &inds) const {
        assert(inds.size() == n);
        for (index_t i = 0; i < n; i++)assert(-sizes[i] <= inds[i] && inds[i] < sizes[i]);
        size_t off = 0;
        for (index_t i = 0; i < n; i++)off += strides[i] * (inds[i] < 0 ? inds[i] + sizes[i] : inds[i]);
        return index(off);
    }

    Tensor<T> slice(index_t dim, index_t ind) const {
        if (dim < 0)dim += n;
        assert(0 <= dim && dim < n);
        if (ind < 0)ind += sizes[dim];
        assert(0 <= ind && ind < sizes[dim]);
        vector<size_t> new_sizes(sizes);
        new_sizes[dim] = 1;
        auto res = Tensor<T>::zeros(new_sizes);
        for (index_t i = 0; i < res.size; i++) {
            index_t t;
            if (dim == 0)t = i + ind * strides[dim];
            else t = i / res.strides[dim - 1] * strides[dim - 1] + ind * strides[dim] + i % res.strides[dim];
            res.index(i) = index(t);
        }
        return res;
    }

    Tensor &operator=(const T &x) {
        assert(size == 1);
        index() = x;
        return *this;
    }

    Tensor &operator=(const vector<T> &data) {
        assert(size == data.size());
        std::copy(data.begin(), data.end(), index_ptr());
        return *this;
    }

    Tensor &set_value(const Tensor<T> &other) {
        assert(sizes == other.sizes);
        std::memmove(index_ptr(), other.index_ptr(), size * sizeof(T));
        return *this;
    }

    vector<T> data() const {
        vector<T> res;
        res.resize(size);
        for (index_t i = 0; i < size; i++)res[i] = index(i);
        return res;
    }

    template<class T2>
    Tensor<T2> astype() const {
        auto res = Tensor<T2>::zeros_like(*this);
        for (index_t i = 0; i < size; i++)res.index(i) = T2(index(i));
        return res;
    }

    T item() const {
        assert(size == 1);
        return index();
    }

    T &index(index_t ind = 0) const {
        assert(0 <= ind && ind < size);
        return storage[offset + ind];
    }

    T *index_ptr(index_t ind = 0) {
        assert(0 <= ind && ind < size);
        return storage.get() + offset + ind;
    }

    T *index_ptr(index_t ind = 0) const {
        assert(0 <= ind && ind < size);
        return storage.get() + offset + ind;
    }

    void fill_(const T &x) {
        for (index_t i = 0; i < size; i++)index(i) = x;
    }

    void zero_() {
        memset(storage.get(), 0, size * sizeof(T));
    }

    void _view_sizes(vector<size_t> &new_sizes) const {
        size_t new_size = 1, unknown = -1;
        for (index_t i = 0; i < new_sizes.size(); i++) {
            if (new_sizes[i])new_size *= new_sizes[i];
            else {
                assert(unknown == -1);
                unknown = i;
            }
        }
        if (unknown == -1) return assert(new_size == size);
        assert(size % new_size == 0);
        new_sizes[unknown] = size / new_size;
    }

    Tensor<T> view(vector<size_t> new_sizes) const {
        _view_sizes(new_sizes);
        return Tensor<T>(storage, offset, new_sizes);
    }

    vector<size_t> _flatten_sizes(index_t start = 0, index_t end = -1) const {
        if (start < 0)start += n;
        if (end < 0)end += n;
        assert(0 <= start && end < index_t(n));
        assert(start <= end);
        size_t new_n = start + this->n - end;
        vector<size_t> new_sizes(new_n);
        for (index_t i = 0; i < start; i++)new_sizes[i] = this->sizes[i];
        size_t flat_size = 1;
        for (index_t i = start; i <= end; i++)flat_size *= this->sizes[i];
        new_sizes[start] = flat_size;
        for (index_t i = end + 1; i < this->n; i++)new_sizes[i - end + start] = this->sizes[i];
        return new_sizes;
    }

    Tensor<T> flatten(index_t start = 0, index_t end = -1) const {
        return Tensor<T>(storage, offset, _flatten_sizes(start, end));
    }

    friend istream &operator>>(istream &in, const Tensor<T> &x) {
        if (x.n == 0) {
            return in >> x.index();
        }
        for (index_t i = 0; i < x.sizes[0]; i++) {
            in >> x[i];
        }
        return in;
    }

    void _output(ostream &out, size_t dim, size_t off) const {
        if (n == dim) {
            out << storage[off];
            return;
        }
        out << '[';
        _output(out, dim + 1, off);
        for (index_t i = 1; i < sizes[dim]; i++) {
            out << ',';
            if (n - dim > 1) {
                out << '\n';
                for (index_t j = 0; j <= dim; j++)out << ' ';
            }
            off += strides[dim];
            _output(out, dim + 1, off);
        }
        out << ']';
    }

    friend ostream &operator<<(ostream &out, const Tensor<T> &x) {
        x._output(out, 0, x.offset);
        return out;
    }

    static Tensor<T> zeros(const vector<size_t> &sizes) {
        size_t size = 1;
        for (auto &it:sizes)size *= it;
        shared_ptr<T[]> storage(new T[size]());
        return Tensor<T>(storage, 0, sizes);
    }

    template<class T2>
    static Tensor<T> zeros_like(const Tensor<T2> &other) {
        shared_ptr<T[]> storage(new T[other.size]());
        vector<size_t> sizes(other.sizes);
        vector<size_t> strides(other.strides);
        return Tensor<T>(other.size, storage, other.n, 0, sizes, strides);
    }

    static Tensor<T> scalar(const T &x) {
        shared_ptr<T[]> storage(new T[1]);
        storage[0] = x;
        return Tensor<T>(storage, 0, {});
    }

    Tensor(const T &x) {
        shared_ptr<T[]> new_storage(new T[1]);
        new_storage[0] = x;
        new(this)Tensor<T>(new_storage, 0, {});
    }

    static Tensor<T> full(const vector<size_t> &sizes, const T &x) {
        size_t size = 1;
        for (auto &it:sizes)size *= it;
        shared_ptr<T[]> storage(new T[size]);
        for (index_t i = 0; i < size; i++)storage[i] = x;
        return Tensor<T>(storage, 0, sizes);
    }

    static Tensor<T> range(index_t start, index_t end) {
        assert(start < end);
        shared_ptr<T[]> storage(new T[end - start]);
        for (index_t i = start; i < end; i++)storage[i - start] = i;
        return Tensor<T>(storage, 0, {end - start});
    }

    static Tensor<T> diag(const vector<T> &diags) {
        size_t n = diags.size();
        shared_ptr<T[]> storage(new T[n * n]);
        {
            index_t i = 0;
            for (auto &it:diags) {
                storage[i] = it;
                i += n + 1;
            }
        }
        return Tensor<T>(storage, 0, {n, n});
    }

    static Tensor<T> rand(vector<size_t> sizes, function<T()> func) {
        size_t size = 1;
        for (auto &it:sizes)size *= it;
        shared_ptr<T[]> storage(new T[size]);
        for (index_t i = 0; i < size; i++)storage[i] = func();
        return Tensor<T>(storage, 0, sizes);
    }

    vector<size_t> _prepend_dim_sizes(size_t num) const {
        vector<size_t> new_sizes(num, 1);
        new_sizes.reserve(n + num);
        for (auto &it:sizes)new_sizes.push_back(it);
        return new_sizes;
    }

    Tensor<T> prepend_dim(size_t num = 1) const {
        return Tensor<T>(storage, offset, _prepend_dim_sizes(num));
    }

    vector<size_t> _append_dim_sizes(size_t num) const {
        vector<size_t> new_sizes(sizes);
        new_sizes.reserve(n + num);
        for (index_t i = 0; i < num; i++)new_sizes.push_back(1);
        return new_sizes;
    }

    Tensor<T> append_dim(size_t num = 1) const {
        return Tensor<T>(storage, offset, _append_dim_sizes(num));
    }

    void _expand(const Tensor<T> &other, size_t dim, size_t off_self, size_t off_other) {
        if (dim == n) {
            storage[off_self] = other.storage[off_other];
            return;
        }
        if (sizes[dim] == other.sizes[dim]) {
            for (index_t i = 0; i < sizes[dim]; i++) {
                _expand(other, dim + 1, off_self + i * strides[dim], off_other + i * other.strides[dim]);
            }
        } else {
            for (index_t i = 0; i < sizes[dim]; i++) {
                _expand(other, dim + 1, off_self + i * strides[dim], off_other);
            }
        }
    }

    Tensor<T> expand(const vector<size_t> &new_sizes) const {
        assert(new_sizes.size() >= n);
        if (new_sizes == sizes)return *this;
        auto res = Tensor<T>::zeros(new_sizes);
        res._expand(this->prepend_dim(new_sizes.size() - n), 0, 0, offset);
        return res;
    }

    // Rule 1: If the two arrays differ in their number of dimensions, the shape of the
    // one with fewer dimensions is padded with ones on its leading (left) side.
    // Rule 2: If the shape of the two arrays does not match in any dimension, the array
    // with shape equal to 1 in that dimension is stretched to match the other shape.
    // Rule 3: If in any dimension the sizes disagree and neither is
    // equal to 1, an error is raised.
    friend void broadcast(Tensor<T> &a, Tensor<T> &b, index_t start = 0, index_t end = -1) {
        if (a.sizes == b.sizes)return;
        if (a.n < b.n)a = a.prepend_dim(b.n - a.n);
        if (a.n > b.n)b = b.prepend_dim(a.n - b.n);
        auto[sa, sb] = _broadcast_sizes(a.sizes, b.sizes, start, end);
        a = a.expand(sa);
        b = b.expand(sb);
    }

    void _fold(const function<T(T, T)> &fun, const Tensor<T> &other, size_t dim, size_t off_self, size_t off_other,
               bool first) {
        if (dim == n) {
            if (first)storage[off_self] = other.storage[off_other];
            else storage[off_self] = fun(storage[off_self], other.storage[off_other]);
            return;
        }
        if (sizes[dim] == other.sizes[dim]) {
            for (index_t i = 0; i < other.sizes[dim]; i++) {
                _fold(fun, other, dim + 1, off_self + i * strides[dim], off_other + i * other.strides[dim], first);
            }
        } else {
            for (index_t i = 0; i < other.sizes[dim]; i++) {
                _fold(fun, other, dim + 1, off_self, off_other + i * other.strides[dim], first && i == 0);
            }
        }
    }

    Tensor<T> fold(const vector<size_t> &new_sizes, const function<T(T, T)> &fun) const {
        assert(new_sizes.size() <= n);
        if (new_sizes == sizes)return *this;
        vector<size_t> prepend_sizes(n - new_sizes.size(), 1);
        prepend_sizes.reserve(n);
        for (auto &it:new_sizes)prepend_sizes.push_back(it);
        auto res = Tensor<T>::zeros(prepend_sizes);
        res._fold(fun, *this, 0, 0, offset, true);
        return res.flatten(0, n - new_sizes.size());
    }

    Tensor<T> map(const function<T(T)> &fun) const {
        auto result = clone();
        for (index_t i = 0; i < size; i++)result.index(i) = fun(result.index(i));
        return result;
    }

    Tensor<T> reduce(const function<T(T, T)> &fun) const {
        assert(size > 0);
        T result = index();
        for (index_t i = 1; i < size; i++)result = fun(result, index(i));
        return Tensor::scalar(result);
    }

    vector<size_t> _reduce_sizes(index_t dim) const {
        vector<size_t> new_sizes(sizes);
        new_sizes[dim] = 1;
        return new_sizes;
    }

    Tensor<T> reduce(const function<T(T, T)> &fun, index_t dim) const {
        if (dim < 0)dim += n;
        assert(0 <= dim && dim < n);
        vector<size_t> new_sizes = _reduce_sizes(dim);
        auto res = Tensor<T>::zeros(new_sizes);
        auto vis = Tensor<bool>::zeros(new_sizes);
        for (index_t i = 0; i < size; i++) {
            size_t t;
            if (dim == 0)t = i % strides[dim];
            else t = i / strides[dim - 1] * res.strides[dim - 1] + i % strides[dim];
            if (vis.index(t))res.index(t) = fun(res.index(t), index(i));
            else {
                vis.index(t) = 1;
                res.index(t) = index(i);
            }
        }
        return res;
    }

    Tensor<index_t> arg_reduce(const function<index_t(T, T)> &fun) const {
        assert(size > 0);
        index_t res = 0;
        for (index_t i = 1; i < size; i++)if (fun(index(res), index(i)))res = i;
        return Tensor<index_t>::scalar(res);
    }

    Tensor<index_t> global_arg_reduce(const function<bool(T, T)> &fun, index_t dim) const {
        if (dim < 0)dim += n;
        assert(0 <= dim && dim < n);
        vector<size_t> new_sizes(sizes);
        new_sizes[dim] = 1;
        auto res = Tensor<index_t>::zeros(new_sizes);
        auto val = Tensor<T>::zeros(new_sizes);
        auto vis = Tensor<bool>::zeros(new_sizes);
        for (index_t i = 0; i < size; i++) {
            size_t t, p;
            if (dim == 0)t = i % strides[dim], p = i / strides[dim];
            else {
                t = i / strides[dim - 1] * res.strides[dim - 1] + i % strides[dim];
                p = i % strides[dim - 1] / strides[dim];
            }
            if (vis.index(t)) {
                if (fun(val.index(t), index(i)))val.index(t) = index(i), res.index(t) = i;
            } else {
                vis.index(t) = 1;
                val.index(t) = index(i);
                res.index(t) = i;
            }
        }
        return res;
    }

    Tensor<index_t> arg_reduce(const function<bool(T, T)> &fun, index_t dim) const {
        if (dim < 0)dim += n;
        assert(0 <= dim && dim < n);
        vector<size_t> new_sizes(sizes);
        new_sizes[dim] = 1;
        auto res = Tensor<index_t>::zeros(new_sizes);
        auto val = Tensor<T>::zeros(new_sizes);
        auto vis = Tensor<bool>::zeros(new_sizes);
        for (index_t i = 0; i < size; i++) {
            size_t t, p;
            if (dim == 0)t = i % strides[dim], p = i / strides[dim];
            else {
                t = i / strides[dim - 1] * res.strides[dim - 1] + i % strides[dim];
                p = i % strides[dim - 1] / strides[dim];
            }
            if (vis.index(t)) {
                if (fun(val.index(t), index(i)))val.index(t) = index(i), res.index(t) = p;
            } else {
                vis.index(t) = 1;
                val.index(t) = index(i);
                res.index(t) = p;
            }
        }
        return res;
    }

    Tensor<T> merge(const function<T(T, T)> &fun, const Tensor<T> &other) const {
        if (sizes != other.sizes) {
            auto a = *this, b = other;
            broadcast(a, b);
            return a.merge(fun, b);
        }
        auto result = Tensor<T>::zeros_like(*this);
        for (index_t i = 0; i < size; i++)
            result.index(i) = fun(index(i), other.index(i));
        return result;
    }

    Tensor<bool> compare(const function<bool(T, T)> &fun, const Tensor<T> &other) const {
        if (sizes != other.sizes) {
            auto a = *this, b = other;
            broadcast(a, b);
            return a.compare(fun, b);
        }
        auto result = Tensor<bool>::zeros_like(*this);
        for (index_t i = 0; i < size; i++)
            result.index(i) = fun(index(i), other.index(i));
        return result;
    }

    Tensor<T> transpose(index_t dim0 = -1, index_t dim1 = -2) const {
        if (dim0 < 0)dim0 += n;
        if (dim1 < 0)dim1 += n;
        assert(0 <= dim0 && dim0 < n);
        assert(0 <= dim1 && dim1 < n);
        if (dim0 == dim1)return *this;
        if (dim0 > dim1)std::swap(dim0, dim1);
        vector<size_t> new_sizes(sizes);
        std::swap(new_sizes[dim0], new_sizes[dim1]);
        auto res = Tensor<T>::zeros(new_sizes);
        size_t s0 = dim0 == 0 ? size : strides[dim0 - 1];
        for (index_t i = 0; i < size; i += s0) {
            for (index_t js = 0, jo = 0; js < strides[dim0]; js += strides[dim1 - 1], jo += res.strides[dim1 - 1]) {
                for (index_t k = 0; k < strides[dim1]; k++) {
                    for (index_t p0 = 0; p0 < sizes[dim0]; p0++) {
                        for (index_t p1 = 0; p1 < sizes[dim1]; p1++) {
                            res.index(i + jo + k + p0 * res.strides[dim1] + p1 * res.strides[dim0]) =
                                    index(i + js + k + p0 * strides[dim0] + p1 * strides[dim1]);
                        }
                    }
                }
            }
        }
        return res;
    }

    Tensor<T> shuffle(const vector<index_t> &perm) const {
        assert(perm.size() == sizes[0]);
        auto res = Tensor<T>::zeros_like(*this);
        for (index_t i = 0; i < sizes[0]; i++) {
            for (index_t j = 0; j < strides[0]; j++) {
                res.index(i * strides[0] + j) = index(perm[i] * strides[0] + j);
            }
        }
        return res;
    }

    Tensor<T> shuffle() const {
        vector<index_t> perm(sizes[0]);
        for (index_t i = 0; i < sizes[0]; i++)perm[i] = i;
        std::shuffle(perm.begin(), perm.end(), rd::gen);
        return shuffle(perm);
    }

    Tensor mm(const Tensor<T> &other) const {
        assert(n == 2);
        assert(other.n == 2);
        assert(sizes[1] == other.sizes[0]);
        auto result = zeros({sizes[0], other.sizes[1]});
        for (index_t i = 0; i < sizes[0]; i++)
            for (index_t k = 0; k < sizes[1]; k++)
                for (index_t j = 0; j < other.sizes[1]; j++)
                    result.index(i * result.sizes[1] + j) +=
                            index(i * sizes[1] + k) * other.index(k * other.sizes[1] + j);
        return result;
    }

    Tensor bmm(const Tensor<T> &other) const {
        assert(n == 3);
        assert(other.n == 3);
        assert(sizes[0] == other.sizes[0]);
        assert(sizes[2] == other.sizes[1]);
        auto result = zeros({sizes[0], sizes[1], other.sizes[2]});
        for (index_t p = 0; p < sizes[0]; p++)
            for (index_t i = 0; i < sizes[1]; i++)
                for (index_t k = 0; k < sizes[2]; k++)
                    for (index_t j = 0; j < other.sizes[2]; j++)
                        result.index(p * result.strides[0] + i * result.strides[1] + j) +=
                                index(p * strides[0] + i * strides[1] + k) *
                                other.index(p * other.strides[0] + k * other.strides[1] + j);
        return result;
    }

    Tensor matmul(const Tensor<T> &other) const {
        auto a = *this, b = other;
        if (a.n == 1)a = a.prepend_dim();
        if (b.n == 1)b = b.append_dim();
        broadcast(a, b, 0, -3);
        vector<size_t> new_sizes(a.sizes.begin(), a.sizes.end() - 2);
        new_sizes.push_back(a.sizes[a.n - 2]);
        new_sizes.push_back(b.sizes[b.n - 1]);
        a = a.view({0, a.sizes[a.n - 2], a.sizes[a.n - 1]});
        b = b.view({0, b.sizes[b.n - 2], b.sizes[b.n - 1]});
        return a.bmm(b).view(new_sizes);
    }

    Tensor operator%(const Tensor<T> &other) const {
        return matmul(other);
    }

#define SELF_OPERATOR(OP)\
    Tensor<T> &operator OP##=(const Tensor<T> &other) {\
        return this->set_value(*this OP other);\
    }\


    SELF_OPERATOR(+)

    SELF_OPERATOR(-)

    SELF_OPERATOR(*)

    SELF_OPERATOR(/)

#undef SELF_OPERATOR

#define MERGE_OPERATOR(OP)\
    friend Tensor<T> operator OP(const Tensor<T> &a, const Tensor<T> &b){\
        if(a.sizes != b.sizes){\
            auto ta = a, tb = b;\
            broadcast(ta, tb);\
            return ta OP tb;\
        }\
        auto result = Tensor<T>::zeros_like(a);\
        for (index_t i = 0; i < a.size; i++)\
            result.index(i) = a.index(i) OP b.index(i);\
        return result;\
    }\


    MERGE_OPERATOR(+)

    MERGE_OPERATOR(-)

    MERGE_OPERATOR(*)

    MERGE_OPERATOR(/)

    MERGE_OPERATOR(&)

    MERGE_OPERATOR(|)

    MERGE_OPERATOR(&&)

    MERGE_OPERATOR(||)

#undef MERGE_OPERATOR

#define CMP_OPERATOR(OP)\
    Tensor<bool> operator OP(const Tensor<T> &other) const{\
        if(sizes != other.sizes){\
            auto a = *this, b = other;\
            broadcast(a, b);\
            return a OP b;\
        }\
        auto result = Tensor<bool>::zeros_like(*this);\
        for (index_t i = 0; i < size; i++)\
            result.index(i) = index(i) OP other.index(i);\
        return result;\
    }\


    CMP_OPERATOR(<)

    CMP_OPERATOR(<=)

    CMP_OPERATOR(>)

    CMP_OPERATOR(>=)

    CMP_OPERATOR(==)

    CMP_OPERATOR(!=)

#undef CMP_OPERATOR

#define MAP_OPERATOR(OP)\
    Tensor operator OP() const{\
        auto result = Tensor<T>::zeros_like(*this);\
        for (index_t i = 0; i < size; i++)result.index(i) = OP index(i);\
        return result;\
    }\


    MAP_OPERATOR(-)

    MAP_OPERATOR(!)

    MAP_OPERATOR(~)

#undef MAP_OPERATOR

#define REDUCE_FUNCTION(fun_name, fun_body)\
    Tensor<T> fun_name() const {\
        return reduce([](T a, T b)->T fun_body );\
    }\
    \
    Tensor<T> fun_name(index_t dim) const {\
        return reduce([](T a, T b)->T fun_body, dim);\
    }\


    REDUCE_FUNCTION(sum, { return a + b; })

    REDUCE_FUNCTION(prod, { return a * b; })

    REDUCE_FUNCTION(max, { return std::max(a, b); })

    REDUCE_FUNCTION(min, { return std::min(a, b); })

    REDUCE_FUNCTION(all, { return a && b; })

    REDUCE_FUNCTION(any, { return a || b; })

    REDUCE_FUNCTION(xor_sum, { return a ^ b; })


#undef REDUCE_FUNCTION

#define ARG_REDUCE_FUNCTION(fun_name, fun_body)\
    Tensor<index_t> fun_name() const {\
        return arg_reduce([](T a, T b)->bool fun_body);\
    }\
    \
    Tensor<index_t> fun_name(index_t dim) const {\
        return arg_reduce([](T a, T b)->bool fun_body, dim);\
    }\
    \
    Tensor<index_t> global_##fun_name(index_t dim) const {\
        return global_arg_reduce([](T a, T b)->bool fun_body, dim);\
    }\


    ARG_REDUCE_FUNCTION(argmax, { return a < b; })

    ARG_REDUCE_FUNCTION(argmin, { return a > b; })

#undef ARG_REDUCE_FUNCTION

#define MAP_FUNCTION(fun_name, fun_body)\
    Tensor<T> fun_name() const{\
        return map([](T x)->T fun_body);\
    }\


    MAP_FUNCTION(relu, { return x > 0 ? x : 0; })

    MAP_FUNCTION(sigmoid, { return 1 / (1 + std::exp(-x)); })

    MAP_FUNCTION(softplus, { return std::log(1 + std::exp(x)); })

    MAP_FUNCTION(exp, { return std::exp(x); })

    MAP_FUNCTION(log, { return std::log(x); })

    MAP_FUNCTION(tanh, { return std::tanh(x); })

    MAP_FUNCTION(sin, { return std::sin(x); })

    MAP_FUNCTION(cos, { return std::cos(x); })

#undef MAP_FUNCTION

    Tensor<T> softmax() const {
        auto _nor = *this - max();
        auto _exp = _nor.exp();
        auto _sum = _exp.sum();
        return _exp / _sum;
    }

    Tensor<T> softmax(index_t dim) const {
        auto _nor = *this - max();
        auto _exp = _nor.exp();
        auto _sum = _exp.sum(dim);
        return _exp / _sum;
    }

    Tensor<T> mean() const {
        return sum() / size;
    }

    Tensor<T> mean(index_t dim) const {
        return sum(dim) / sizes[dim];
    }

#define BIN_FUNCTION(fun_name, fun_body)\
    Tensor<T> fun_name(const Tensor<T> &other) const{\
        return merge([](T a, T b)->T fun_body, other);\
    }\

#undef BIN_FUNCTION

#define CMP_FUNCTION(fun_name, fun_body)\
    Tensor<bool> fun_name(const Tensor<T> &other) const{\
        return compare([](T a, T b)->bool fun_body, other);\
    }\


    CMP_FUNCTION(approx, { return a - b < 1e-8 && b - a < 1e-8; })

#undef CMP_FUNCTION

};

#endif //TORCPP_TENSOR_H
