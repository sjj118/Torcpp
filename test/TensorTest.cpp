#define CATCH_CONFIG_MAIN

#include "catch.hpp"
#include "../core/Tensor.hpp"

using namespace std;

TEST_CASE("tensor can output", "[tensor]") {
    stringstream out;
    out << Tensor<double>({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
    REQUIRE(out.str() == "[[[1,2],\n"
                         "  [3,4]],\n"
                         " [[5,6],\n"
                         "  [7,8]]]");
}

class MemTester {
public:
    static int cnt;

    MemTester() {
        cnt++;
    }

    ~MemTester() {
        cnt--;
    }
};

int MemTester::cnt;

TEST_CASE("tensor can maintain storage memory automatically", "[tensor]") {
    MemTester::cnt = 0;
    SECTION("test") {
        auto x = Tensor<MemTester>::zeros({2, 2, 2});
        auto y = *x[0];
        y = *x[0][1][1];
        auto z = x[0][1]->clone();
    };
    REQUIRE(MemTester::cnt == 0);
}

TEST_CASE("tensor accessor can not be stored", "[tensor]") {
    auto x = Tensor<double>::zeros({2, 2, 2});
//    auto y = x[0];
    cout << &x[0] << endl;
}

//auto fun() {
//    auto x = Tensor<double>({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
//    return x[1].divorce();
//}

TEST_CASE("test0") {
//    auto x = Tensor<double>({2, 2, 2}, {1, 2, 3, 4, 5, 6, 7, 8});
//    cout << x << endl;
//    x[0] = {0, 0, 0, 0};
//    cout << fun() << endl;
//    cout << x << endl;
//    cout << x[1][0][1] << endl;
//    x[1] = {0, 0, 0, 0};
//    auto y = x.flatten(0, 1);
//    cout << y << endl;
//    auto s = Tensor<double>::scalar(0.5);
//    cout << s << endl;
//    auto d = Tensor<double>::diag({1, 2, 3, 4});
//    cout << d << endl;
//    auto r = Tensor<double>::rand({4, 4}, [] { return double(rand() % 1000) / 1000; });
//    cout << r << endl;
}