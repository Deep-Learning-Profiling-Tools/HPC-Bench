#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }
  
  // Apply beta scaling to C matrix first
  #ifdef _OPENMP
  #pragma omp parallel for collapse(2) schedule(static)
  #endif
  for (int nn = 0; nn < n; ++nn) {
    for (int mm = 0; mm < m; ++mm) {
      C[mm + nn * ldc] *= beta;
    }
  }
  
  // Perform matrix multiplication with loop reordering and tiling
  const int tile_size = 64;
  
  #ifdef _OPENMP
  #pragma omp parallel
  #endif
  {
    #ifdef _OPENMP
    #pragma omp for schedule(dynamic, 1)
    #endif
    for (int nn_tile = 0; nn_tile < n; nn_tile += tile_size) {
      int nn_end = (nn_tile + tile_size < n) ? nn_tile + tile_size : n;
      
      for (int mm_tile = 0; mm_tile < m; mm_tile += tile_size) {
        int mm_end = (mm_tile + tile_size < m) ? mm_tile + tile_size : m;
        
        for (int kk = 0; kk < k; ++kk) {
          const float *A_col = A + kk * lda;
          const float *B_col = B + kk * ldb;
          
          for (int nn = nn_tile; nn < nn_end; ++nn) {
            float b_val = B_col[nn];
            float alpha_b = alpha * b_val;
            float *C_col = C + nn * ldc;
            
            for (int mm = mm_tile; mm < mm_end; ++mm) {
              C_col[mm] += alpha_b * A_col[mm];
            }
          }
        }
      }
    }
  }
}
