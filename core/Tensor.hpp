//
// Created by 史记 on 2020/4/12.
//

#ifndef TORCPP_TENSOR_H
#define TORCPP_TENSOR_H

#include <initializer_list>
#include <initializer_list>
#include <functional>
#include <memory>
#include <iostream>
#include <cassert>
#include <cstring>

using std::memcpy;
using std::default_delete;
using std::shared_ptr;
using std::initializer_list;
using std::function;
using std::istream;
using std::ostream;

typedef int index_t;

template<class T>
class Tensor {
    size_t size{};
    shared_ptr<T> storage;
    size_t n{}, offset{}, *sizes{}, *strides{};
    bool contiguous = true;

    Tensor(size_t size, const shared_ptr<T> &storage, size_t n, size_t offset,
           size_t *sizes, size_t *strides, bool contiguous)
            : size(size), storage(storage), n(n), offset(offset),
              sizes(sizes), strides(strides), contiguous(contiguous) {
    }

    static size_t *sizes2strides(size_t n, const size_t *sizes) {
        auto strides = new size_t[n];
        strides[n - 1] = 1;
        for (index_t i = n - 2; i >= 0; i--)strides[i] = strides[i + 1] * sizes[i + 1];
        return strides;
    }

    Tensor(const shared_ptr<T> &storage, size_t offset, initializer_list<size_t> sizes, size_t infer = 0)
            : storage(storage), offset(offset), n(sizes.size()) {
        this->sizes = new size_t[this->n];
        {
            index_t i = 0;
            for (auto &it:sizes)this->sizes[i++] = it ? it : infer;
        }
        this->strides = sizes2strides(n, this->sizes);
        this->size = this->strides[0] * this->sizes[0];
    }

    class accessor;

    class iterator;

    accessor operator&() {
        return accessor(this, 0, offset);
    }

public:
    ~Tensor() {
        delete[] this->sizes;
        delete[] this->strides;
    }

    Tensor(const Tensor &other) : size(other.size), storage(other.storage), n(other.n), offset(other.offset) {
        sizes = new size_t[n];
        memcpy(sizes, other.sizes, n * sizeof(size_t));
        strides = new size_t[n];
        memcpy(strides, other.strides, n * sizeof(size_t));
        contiguous = other.contiguous;
    }

    Tensor clone() const {
        assert(contiguous);
        auto sizes = new size_t[n];
        memcpy(sizes, this->sizes, n * sizeof(size_t));
        auto strides = new size_t[n];
        memcpy(strides, this->strides, n * sizeof(size_t));
        shared_ptr<T> storage(new T[size], default_delete<T[]>());
        memcpy(storage.get(), this->storage.get() + offset, size * sizeof(T));
        return Tensor(size, storage, n, 0, sizes, strides, true);
    }

    Tensor(initializer_list<size_t> sizes, initializer_list<T> data) {
        size_t size = 1;
        for (auto &it:sizes)size *= it;
        shared_ptr<T> storage(new T[size], default_delete<T[]>());
        std::copy(data.begin(), data.end(), storage.get());
        new(this)Tensor(storage, 0, sizes);
    }

    accessor operator[](index_t ind) {
        return (&*this)[ind];
    }

    Tensor operator[](initializer_list<index_t> inds) {
        assert(inds.size() <= n);
        size_t offset = this->offset;
        size_t size = this->size;
        {
            index_t i = 0;
            for (auto &it:inds) {
                assert(it < sizes[i]);
                offset += it * strides[i];
                size /= sizes[i];
                i++;
            }
        }
        size_t n = this->n - inds.size();
        auto sizes = new size_t[n];
        memcpy(sizes, this->sizes + inds.size(), n * sizeof(size_t));
        auto strides = new size_t[n];
        memcpy(strides, this->strides + inds.size(), n * sizeof(size_t));
        return Tensor{size, storage, n, offset, sizes, strides, contiguous};
    }

    Tensor &divorce() {
        //TODO
    }

    Tensor &operator=(T x) {
        assert(n == 0);
        storage[offset] = x;
        return *this;
    }

    Tensor &operator=(initializer_list<T> data) {
        assert(size == data.size());
        assert(contiguous);
        std::copy(data.begin(), data.end(), storage.get() + offset);
        return *this;
    }

    Tensor &operator=(const Tensor &other) {
        if (this == &other)return *this;
        assert(n == other.n);
        for (index_t i = 0; i < n; i++)assert(sizes[i] == other.sizes[i]);
        assert(contiguous && other.contiguous);
        memmove(storage.get() + offset, other.storage.get() + other.offset, size * sizeof(T));
        return *this;
    }

    Tensor<T> view(initializer_list<size_t> sizes) {
        assert(contiguous);
        size_t size = 1, unknown = 0;
        for (auto &it:sizes) {
            if (it)size *= it;
            else unknown++;
        }
        assert(unknown <= 1);
        if (unknown == 0)assert(size == this->size);
        else
            assert(this->size % size == 0);
        return Tensor<T>(storage, offset, sizes, this->size / size);
    }

    Tensor<T> flatten(index_t startDim = 0, index_t endDim = -1) {
        assert(contiguous);
        if (startDim < 0)startDim += n;
        if (endDim < 0)endDim += n;
        assert(0 <= startDim && startDim < n);
        assert(0 <= endDim && endDim < n);
        assert(startDim <= endDim);
        size_t n = startDim + this->n - endDim;
        auto sizes = new size_t[n];
        for (index_t i = 0; i < startDim; i++)sizes[i] = this->sizes[i];
        size_t flatSize = 1;
        for (index_t i = startDim; i <= endDim; i++)flatSize *= this->sizes[i];
        sizes[startDim] = flatSize;
        for (index_t i = endDim + 1; i < this->n; i++)sizes[i - endDim + startDim] = this->sizes[i];
        return Tensor<T>(size, storage, n, offset, sizes, sizes2strides(n, sizes), true, false);
    }

    T item() const {
        assert(size == 1);
        return storage[offset];
    }

    friend istream &operator>>(istream &in, const Tensor<T> &x) {
        if (x.n == 0) {
            return in >> x.storage[x.offset];
        }
        for (index_t i = 0; i < x.sizes[0]; i++) {
            in >> x[i];
        }
        return in;
    }

    void output(ostream &out, size_t dim, size_t offset) const {
        if (n == dim) {
            out << storage.get()[offset];
            return;
        }
        out << '[';
        output(out, dim + 1, offset);
        for (index_t i = 1; i < sizes[dim]; i++) {
            out << ',';
            if (n - dim > 1) {
                out << '\n';
                for (index_t j = 0; j <= dim; j++)out << ' ';
            }
            offset += strides[dim];
            output(out, dim + 1, offset);
        }
        out << ']';
    }

    friend ostream &operator<<(ostream &out, const Tensor<T> &x) {
        x.output(out, 0, x.offset);
        return out;
    }

    static Tensor<T> zeros(initializer_list<size_t> sizes) {
        size_t size = 1;
        for (auto &it:sizes)size *= it;
        shared_ptr<T> storage(new T[size], default_delete<T[]>());
        return Tensor<T>(storage, 0, sizes);
    }

    static Tensor<T> zeros_like(const Tensor<T> &other) {
        assert(other.contiguous);
        shared_ptr<T> storage(new T[other.size]);
        auto *sizes = new size_t[other.n];
        memcpy(sizes, other.sizes, other.n * sizeof(size_t));
        auto *strides = new size_t[other.n];
        memcpy(strides, other.strides, other.n * sizeof(size_t));
        return Tensor<T>(other.size, storage, other.n, 0, sizes, strides, true, false);
    }

    static Tensor<T> scalar(const T &x) {
        shared_ptr<T> storage(new T[1], default_delete<T[]>());
        storage.get()[0] = x;
        return Tensor<T>(storage, 0, {});
    }

    static Tensor<T> diag(initializer_list<T> diags) {
        size_t n = diags.size();
        shared_ptr<T> storage(new T[n * n], default_delete<T[]>());
        {
            index_t i = 0;
            for (auto &it:diags) {
                storage.get()[i] = it;
                i += n + 1;
            }
        }
        return Tensor<T>(storage, 0, {n, n});
    }

    static Tensor<T> rand(initializer_list<size_t> sizes, function<T()> randFunc) {
        size_t size = 1;
        for (auto &it:sizes)size *= it;
        shared_ptr<T> storage(new T[size], default_delete<T[]>());
        for (index_t i = 0; i < size; i++)storage.get()[i] = randFunc();
        return Tensor<T>(storage, 0, sizes);
    }

#define SELF_OPERATOR(OP)\
    Tensor &operator OP(const Tensor<T> &other) {\
        assert(n == other.n);\
        for (index_t i = 0; i < n; i++)assert(sizes[i] == other.sizes[i]);\
        assert(contiguous);\
        for (index_t i = 0; i < size; i++)storage[offset + i] OP other.storage[other.offset + i];\
        return *this;\
    }\


    SELF_OPERATOR(+=)

    SELF_OPERATOR(-=)

    SELF_OPERATOR(*=)

    SELF_OPERATOR(/=)

    SELF_OPERATOR(%=)

    SELF_OPERATOR(&=)

    SELF_OPERATOR(|=)

    SELF_OPERATOR(^=)

    Tensor operator+(const Tensor<T> &other) {
        auto result = clone();
        return result += other;
    }
};

template<class T>
class Tensor<T>::accessor {
    Tensor<T> *tensor;
    size_t dim, offset;

    accessor(Tensor<T> *tensor, size_t dim, size_t offset) : tensor(tensor), dim(dim), offset(offset) {}

    accessor(accessor &&other) = default;

public:
    accessor(const accessor &other) = delete;

    friend class Tensor;

    accessor operator[](index_t ind) {
        assert(dim < tensor->n);
        if (ind < 0)ind += tensor->n;
        assert(0 <= ind && ind < tensor->n);
        offset += ind * tensor->strides[dim];
        dim++;
        return std::move(*this);
    }

    Tensor<T> operator*() {
        size_t n = tensor->n - dim;
        auto sizes = new size_t[n];
        memcpy(sizes, tensor->sizes + dim, n * sizeof(size_t));
        auto strides = new size_t[n];
        memcpy(strides, tensor->strides + dim, n * sizeof(size_t));
        size_t size = 1;
        for (index_t i = 0; i < n; i++)size *= sizes[i];
        return Tensor(size, tensor->storage, n, offset, sizes, strides, tensor->contiguous);
    }

    Tensor<T> *operator->() {
        return &**this;
    }
};

#endif //TORCPP_TENSOR_H
