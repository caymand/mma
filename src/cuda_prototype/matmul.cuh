
#ifndef MULT_kERNELS
#define MULT_kERNELS
#include <stdint.h>
#include <mma.h>
using namespace nvcuda;

template <class ElTp, class AccTp, class LoadTp, int Ty, int Ry, int Tx, int Rx, int Tk>
__global__ void matMulTiled(ElTp* A, ElTp* B, AccTp* C, int heightA, int widthB, int widthA) {
    int gid = threadIdx.x + threadIdx.y * blockDim.x;
    if (gid >= widthA * heightA || gid >= widthA * widthB) { return; }
    // TODO: Padding - problematic due to alignment issues caused by padding
    // remapping (a slice of) A to shared memory
    __shared__ ElTp Aloc[Ty*Ry][Tk]; // TODO: Padding

    // remapping (a slice of) B to shared memory
    __shared__ ElTp Bloc[Tk][Tx*Rx];

    // the thread result is computed in register memory
    // and the global-memory array C is updated at the end.
    AccTp css[Ry][Rx];

    unsigned int iii = blockIdx.y * Ty * Ry; // Global i index
    unsigned int jjj = blockIdx.x * Tx * Rx; // Global j index
    // How many elements the vectorized load will load. Should be handled on the host    
    constexpr int load_elms = sizeof(LoadTp) / sizeof(ElTp);
    int blockSize = blockDim.x * blockDim.y;
    int rows_loaded_B = (blockSize * load_elms) / (Tx * Rx);

    // initialize the result with zero
    // (the neutral element for addition)
#pragma unroll
    for(int i=0; i<Ry; i++)
#pragma unroll
	for(int j=0; j<Rx; j++)
	    css[i][j] = (AccTp) 0.0;

    for(int kk = 0; kk < widthA; kk += Tk) {
#pragma unroll
	// COPY A LOOP
	for (uint32_t r = 0; r < Ry; r++) {
	    // Stack R blocks of size Ty x Tx on top of each other
	    uint32_t local_x = threadIdx.x;
	    uint32_t local_y = threadIdx.y + Ty * r; 

	    uint32_t slice_y = iii + local_y; // [iii : iii + Ty*Ry]
	    uint32_t slice_x = kk + threadIdx.x;// [kk: kk + Tk]

	    uint32_t global_thrd_offset = slice_y * widthA + kk;
	    // bool insideBounds = (slice_y < heightA) && (slice_x < widthA);
	    // Currently we set the Tk and threads so that they do not go outisde bounds
	    bool insideBounds = slice_y < heightA;
	    LoadTp *Asmem_row = reinterpret_cast<LoadTp *>(Aloc[local_y]);
	    LoadTp *Aglobal_row = reinterpret_cast<LoadTp *>(A + global_thrd_offset);
	    LoadTp global_elms;
	    if (insideBounds)
	    {
		global_elms = Aglobal_row[local_x];
	    }
	    else
	    {
		global_elms = LoadTp();
	    }
	    Asmem_row[local_x] = global_elms;	    	    
	}
	    
#pragma unroll
	for (uint32_t r = 0; r < Rx; r++) {
	    // Use All threads
	    // Use 64 threads to do vectorized loads with two elms in j dimension.
	    // This means we can in each iteration copy a 4x128 block.
	    // For Rx = 8, this gives a final result of 32x128 (as desired).	    
	    auto flat_thr_idx = gid;
	    auto local_j = flat_thr_idx % ((Tx * Rx) / load_elms);
	    auto local_k = (flat_thr_idx * load_elms) / (Tx * Rx) + r * rows_loaded_B;	 
	    auto slice_y = kk + local_k;
	    
	    auto global_thr_offset = slice_y * widthB + jjj;
	    LoadTp *Bsmem_row = reinterpret_cast<LoadTp *>(Bloc[local_k]);
	    LoadTp *Bglobal_row = reinterpret_cast<LoadTp *>(B + global_thr_offset);
	    // TODO: Bounds check
	    LoadTp elms = Bglobal_row[local_j];
	    Bsmem_row[local_j] = elms;	            
	}
	__syncthreads();

	// compute the per-thread result css:
	for(int k = 0; k < Tk; k++) {
#pragma unroll
	    for(int i=0; i<Ry; i++) {
#pragma unroll
		for(int j=0; j<Rx; j++) {
		    ElTp Aik = Aloc[threadIdx.y * Ry + i][k];
		    ElTp Bkj = Bloc[k][threadIdx.x * Rx + j];
		    css[i][j] += (AccTp) (Aik * Bkj);

		}
	    }
	}
	__syncthreads();
    }

    unsigned int indy = iii + threadIdx.y * Ry;
    unsigned int indx = jjj + threadIdx.x * Rx;

    // Update C in global memory with the per-thread result css.
#pragma unroll
    for(int i=0; i<Ry; i++) {
#pragma unroll
	for(int j=0; j<Rx; j++) {
	    if( (indy+i < heightA) && (indx+j < widthB) )
	    {
		C[(indy+i)*widthB + (indx+j)] = css[i][j];
	 
	    }      
	}
    }
}


#endif
