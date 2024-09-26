#include <iostream>

#include <cute/tensor.hpp>

int main()
{
    using namespace cute;
    
    auto bM = Int<16>{};
    auto bK = Int<64>{};
    auto sA = make_layout(make_shape(Int<8>{}, Int<8>{}), 
                          make_stride(Int<8>{}, Int<1>{}));
    
    // auto sA_swizzled = composition(Swizzle<3,0>{}, sA);
    // TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, half_t>{},
    //                                   Layout<Shape<_16, _8>, Stride<_8, _1>>{},
    //                                   Layout<Shape<_1, _8>>{});
    
    using value_type = cute::half_t;
    auto smem_layout = composition(Swizzle<3,3,3>{},
                Layout<Shape < _8,_64>,
                       Stride<_64, _1>>{});
    // 0, 0 -> 0
    // 1 -> 0
    // print_layout(sA);
    print_latex(smem_layout);
}