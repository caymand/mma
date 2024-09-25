#include <iostream>

#include <cute/tensor.hpp>

int main()
{
    using namespace cute;
    
    auto bM = Int<8>{};
    auto bK = Int<64>{};
    auto sA = make_layout(make_shape(bM, bK), make_stride(Int<8>{}, Int<1>{}));
    
    auto sA_swizzled = composition(Swizzle<2,3,3>{}, sA);
    
    // print_layout(sA);
    print_latex(sA_swizzled);
}