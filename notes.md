# Futhark Compiler

## Overview
- src.Language.Futhark.Prop.hs extracts intrinsics
- 

# 24/9

- Use Cutlass basic blocks
- Call into them. Make them a header file

## Intragroup kernel focus

```futhark
let dotproduct [n] (x: [n]f16) (y: [n]f16) =
    #[sequential]map2 (*) x y |> reduce (+) 0

let matmul16 (A: [16][16]f16) (B: [16][16]f16) : [16][16]f16 =
    map (\ Arow -> map (\Bcol -> dotproduct Arow Bcol) (transpose B)) A

let intra_block_mmm [k] (A: [k][16][16]f16) (B: [k][16][16]f16) : [k][16][16]f16 =    
    map2 matmul16 A B
```

# Links

- https://www.cs.utexas.edu/~flame/BLISRetreat2023/slides/Thakkar_BLISRetreat2023.pdf
