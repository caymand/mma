
let dotproduct [n] (x: [n]f16) (y: [n]f16) =
    map2 (*) x y |> reduce (+) 0

let matmul16 (A: [16][16]f16) (B: [16][16]f16) : [16][16]f16 =
    map (\ Arow -> map (\Bcol -> dotproduct Arow Bcol) (transpose B)) A

let intra_block_mmm [k] (A: [k][16][16]f16) (B: [k][16][16]f16) : [k][16][16]f16 =    
    map2 matmul16 A B

let main = intra_block_mmm
