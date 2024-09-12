-- ==
-- compiled random input {[4][4][16][16]f16 [4][4][16][16]f16}

let dotproduct [n] (x: [n]f16) (y: [n]f16) =
    map2 (*) x y |> reduce (+) 0

let matmul [m][n][q]  (A: [m][q]f16) (B: [q][n]f16) : [m][n]f16 =
    map (\ Arow -> map (\Bcol -> dotproduct Arow Bcol) (transpose B)) A

let mat_add [m][n] (C1: [m][n]f16) (C2: [m][n]f16) =
    map2 (\ C1_row C2_row -> 
        map2 (+) C1_row C2_row
    ) C1 C2

let dotprod_tile [tk][cm][ck][cn] (Atile: [tk][cm][ck]f16) (Btile: [tk][ck][cn]f16) : [cm][cn]f16 =
    map2 matmul Atile Btile
    |> reduce mat_add (replicate (cm * cn) 0f16 |> unflatten)


let mmm [tm][tk][cm][ck][tn][cn] (Atiles: [tm][tk][cm][ck]f16) (Btiles: [tk][tn][ck][cn]f16) : [tm][tn][cm][cn]f16 =
    map (\Atile_row ->         
        map (\Btile_col ->
            dotprod_tile Atile_row Btile_col        
        ) (transpose Btiles)
    ) Atiles

let main = mmm