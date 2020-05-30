#include "Tensor.h"

pair<vector<size_t>, vector<size_t>>
_broadcast_sizes(vector<size_t> a, vector<size_t> b, index_t start, index_t end) {
    assert(a.size() == b.size());
    if (start < 0)start += a.size();
    if (end < 0)end += a.size();
    assert(0 <= start && end < index_t(a.size()));
    for (index_t i = start; i <= end; i++) {
        if (a[i] == 1) a[i] = b[i];
        else if (b[i] == 1) b[i] = a[i];
        else
            assert(a[i] == b[i]);
    }
    return make_pair(a, b);
}