#include "random.h"

namespace rd {
    std::random_device rd;
    std::default_random_engine gen(20000107);
    std::normal_distribution<> nd{0, 1};
    std::uniform_real_distribution<> ud{0, 1};
    std::function<double()> randn = []() { return nd(gen); };
    std::function<double()> randu = []() { return ud(gen); };
}