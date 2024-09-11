
#ifndef MULT_kERNELS
#define MULT_kERNELS
#include <stdint.h>
#include <mma.h>
using namespace nvcuda;

template <class ElTp, class AccTp, int Ty, int Ry, int Tx, int Rx, int Tk>
__global__ void matMulTiled(ElTp* A, ElTp* B, AccTp* C, int heightA, int widthB, int widthA) {
  int gid = threadIdx.x + threadIdx.y * blockDim.x;
  if (gid >= widthA * heightA || gid >= widthA * widthB) { return; }

  // remapping (a slice of) A to shared memory
  __shared__ ElTp Aloc[Ty*Ry][Tk+1];

  // remapping (a slice of) B to shared memory
  __shared__ ElTp Bloc[Tk][Tx*Rx+1];

  // the thread result is computed in register memory
  // and the global-memory array C is updated at the end.
    AccTp css[Ry][Rx];

  unsigned int iii = blockIdx.y * Ty * Ry;
  unsigned int jjj = blockIdx.x * Tx * Rx;

  // initialize the result with zero
  // (the neutral element for addition)
  #pragma unroll
  for(int i=0; i<Ry; i++)
      #pragma unroll
      for(int j=0; j<Rx; j++)
          css[i][j] = (AccTp) 0.0;

  for(int kk = 0; kk < widthA; kk += Tk) {
      #pragma unroll
      for (uint32_t r = 0; r < Ry; r++) {
        // Stack R blocks of size Ty x Tx on top of each other
        uint32_t local_x = threadIdx.x;
        uint32_t local_y = threadIdx.y + Ty * r; // stack Ry blocks on top of each other

        uint32_t slice_y = iii + local_y; // this gives [iii : iii + Ty*Ry]
        uint32_t slice_x = kk + threadIdx.x;// This is [kk: kk + Tk]

        bool insideBounds = (slice_y < heightA) && (slice_x < widthA);
        Aloc[local_y][local_x] = insideBounds ? A[slice_y * widthA + slice_x] : (ElTp) 0.0;
      }

      #pragma unroll
      for (uint32_t r = 0; r < Rx; r++) {
          uint32_t local_y = threadIdx.y;
          uint32_t local_x = threadIdx.x + Tx*r; // stack Rx blocks next to each other

          uint32_t slice_y = kk + threadIdx.y;// [kk : kk + Tk]
          uint32_t slice_x = jjj + local_x; // [jjj : jjj + Tx*Rx]

          bool insideBounds = (slice_y < widthA) && (slice_x < widthB);
          Bloc[local_y][local_x] = insideBounds ? B[slice_y * widthB + slice_x] : (ElTp) 0.0;
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
        C[(indy+i)*widthB + (indx+j)] = css[i][j];
    }
  }
}


#endif
