//
// Created by 史记 on 2020/4/12.
//

#ifndef TORCPP_TENSOR_H
#define TORCPP_TENSOR_H

#include <vector>
#include <functional>
#include <iostream>

using std::vector;
using std::function;
using std::istream;
using std::ostream;

template<class T>
class Tensor;

template<class T>
class TensorStorage;

template<class T>
class TensorStorage {
    size_t _size, cnt = 0;

    explicit TensorStorage(size_t size) : _size(size) {
        data = new T[size];
    }

public:
    friend class Tensor<T>;

    T *data;

    ~TensorStorage() {
        delete[] data;
    }

    size_t size() const { return _size; }

    void link() {
        cnt++;
        std::cout << "cnt: " << cnt << std::endl;
    }

    void delink() {
        if (--cnt == 0)delete this;
        std::cout << "cnt: " << cnt << std::endl;
    }
};


template<class T>
class Tensor {
    size_t size{};
    TensorStorage<T> *storage{};
    size_t n{}, offset{}, *sizes{}, *strides{};
    bool contiguous = true;
    bool temp = false;

    Tensor(size_t size, TensorStorage<T> *storage, size_t n, size_t offset,
           size_t *sizes, size_t *strides, bool contiguous, bool temp)
            : size(size), n(n), offset(offset),
              sizes(sizes), strides(strides), contiguous(contiguous), temp(temp) {
        if (!temp) setStorage(storage);
    }

    void setStorage(TensorStorage<T> *storage) {
        this->storage = storage;
        storage->link();
    }

    void resetStorage() {
        this->storage->delink();
    }

    Tensor(TensorStorage<T> *storage, size_t offset, vector<size_t> sizes, size_t infer = 0) {
        this->n = sizes.size();
        setStorage(storage);
        this->offset = offset;
        this->sizes = new size_t[this->n];
        {
            int i = 0;
            for (auto &it:sizes)this->sizes[i++] = it ? it : infer;
        }
        this->strides = new size_t[this->n];
        this->strides[this->n - 1] = 1;
        for (int i = this->n - 2; i >= 0; i--)this->strides[i] = this->strides[i + 1] * this->sizes[i + 1];
        this->size = this->strides[0] * this->sizes[0];
    }

public:
    ~Tensor() {
        if (temp)return;
        resetStorage();
        delete[] this->sizes;
        delete[] this->strides;
    }

    Tensor(const Tensor &other) {
        size = other.size;
        setStorage(other.storage);
        n = other.n;
        offset = other.offset;
        sizes = new size_t[n];
        memcpy(sizes, other.sizes, n * sizeof(size_t));
        strides = new size_t[n];
        memcpy(strides, other.strides, n * sizeof(size_t));
        contiguous = other.contiguous;
        temp = false;
//        std::cout << "copyed" << std::endl;
    }

    Tensor &operator=(const Tensor &other) {
        if (this == &other)return *this;
        this->~Tensor();
        new(this)Tensor(other);
    }

    Tensor(vector<size_t> sizes, vector<T> data) {
        size_t size = 1;
        for (auto &it:sizes)size *= it;
        auto *storage = new TensorStorage<T>(size);
        std::copy(data.begin(), data.end(), storage->data);
        new(this)Tensor(storage, 0, sizes);
    }

    Tensor operator[](size_t ind) const {
        assert(n > 0);
        assert(0 <= ind && ind < this->sizes[0]);
        auto sizes = new size_t[n - 1];
        memcpy(sizes, this->sizes + 1, (n - 1) * sizeof(size_t));
        auto strides = new size_t[n - 1];
        memcpy(strides, this->strides + 1, (n - 1) * sizeof(size_t));
        return Tensor{size / this->sizes[0], storage, n - 1, offset + ind * this->strides[0],
                      sizes, strides, contiguous, false};
    }

    Tensor &operator=(T x) {
        assert(n == 0);
        storage->data[offset] = x;
        return *this;
    }

    Tensor &operator=(vector<T> data) {
        assert(size == data.size());
        assert (contiguous);
        std::copy(data.begin(), data.end(), storage->data + offset);
        return *this;
    }

    Tensor<T> view(vector<size_t> sizes) {
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

    Tensor<T> flatten(int startDim = 0) {
        vector<size_t> sizes(startDim + 1);
        for (int i = 0; i < startDim; i++)sizes[i] = this->sizes[i];
        sizes[startDim] = 0;
        return view(sizes);
    }

    T &item() {
        assert(n == 0);
        return storage->data[offset];
    }

    friend istream &operator>>(istream &in, const Tensor<T> &x) {
        if (x.n == 0) {
            return in >> x.storage->data[x.offset];
        }
        for (int i = 0; i < x.sizes[0]; i++) {
            in >> x[i];
        }
        return in;
    }

    void output(ostream &out, int inci) const {
        if (n == 0) {
            out << storage->data[offset];
            return;
        }
        out << '[';
        (*this)[0].output(out, inci + 1);
        for (int i = 1; i < sizes[0]; i++) {
            out << ',';
            if (n > 1) {
                out << '\n';
                for (int j = 0; j < inci; j++)out << ' ';
            }
            (*this)[i].output(out, inci + 1);
        }
        out << ']';
    }

    friend ostream &operator<<(ostream &out, const Tensor<T> &x) {
        x.output(out, 1);
        return out;
    }

    static Tensor<T> zeros(vector<size_t> sizes) {
        size_t size = 1;
        for (auto &it:sizes)size *= it;
        auto *storage = new TensorStorage<T>(size);
        return Tensor<T>(storage, 0, sizes);
    }

    static Tensor<T> scalar(const T &x) {
        auto *storage = new TensorStorage<T>(1);
        storage->data[0] = x;
        return Tensor<T>(storage, 0, {});
    }

    static Tensor<T> diag(vector<T> diags) {
        size_t n = diags.size();
        auto *storage = new TensorStorage<T>(n * n);
        {
            size_t i = 0;
            for (auto &it:diags) {
                storage->data[i] = it;
                i += n + 1;
            }
        }
        return Tensor<T>(storage, 0, {n, n});
    }

    static Tensor<T> rand(vector<size_t> sizes, function<T()> randFunc) {
        size_t size = 1;
        for (auto &it:sizes)size *= it;
        auto *storage = new TensorStorage<T>(size);
        for (size_t i = 0; i < size; i++)storage->data[i] = randFunc();
        return Tensor<T>(storage, 0, sizes);
    }

};


#endif //TORCPP_TENSOR_H
