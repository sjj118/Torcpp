#ifndef TORCPP_RANDOM_H
#define TORCPP_RANDOM_H

#include <functional>
#include <random>

namespace rd {
    extern std::random_device rd;
    extern std::default_random_engine gen;
    extern std::normal_distribution<> nd;
    extern std::uniform_real_distribution<> ud;
    extern std::function<double()> randn;
    extern std::function<double()> randu;
}

#endif //TORCPP_RANDOM_H
