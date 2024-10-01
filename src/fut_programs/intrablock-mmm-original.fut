let dotproduct (x: [16]f32) (y: [16]f32) =
    #[sequential]map2 (*) x y |> 
    #[sequential]reduce (+) 0

let matmul16 (A: [16][16]f32) (B: [16][16]f32) : [16][16]f32 =
    map (\ Arow -> 
        map (\Bcol -> 
            dotproduct Arow Bcol) 
        (transpose B)
    ) A     

let intra_block_mmm [q] (A: [q][16][16]f32) (B: [q][16][16]f32) : [q][16][16]f32 =    
    #[incremental_flattening(only_intra)]map2 matmul16 A B

let main = intra_block_mmm
