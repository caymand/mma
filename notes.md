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
- https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf
- https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21745/
- https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf
