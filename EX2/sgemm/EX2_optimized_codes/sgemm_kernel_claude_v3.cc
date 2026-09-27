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
  
  // Pre-scale C by beta if beta != 1.0
  if (beta != 1.0f) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nn = 0; nn < n; ++nn) {
      for (int mm = 0; mm < m; ++mm) {
        C[mm + nn * ldc] *= beta;
      }
    }
  }
  
  // Perform the matrix multiplication with loop reordering and blocking
  const int BLOCK_SIZE = 64;
  
  #pragma omp parallel for collapse(2) schedule(dynamic, 1)
  for (int nn_block = 0; nn_block < n; nn_block += BLOCK_SIZE) {
    for (int mm_block = 0; mm_block < m; mm_block += BLOCK_SIZE) {
      int nn_end = (nn_block + BLOCK_SIZE < n) ? nn_block + BLOCK_SIZE : n;
      int mm_end = (mm_block + BLOCK_SIZE < m) ? mm_block + BLOCK_SIZE : m;
      
      for (int kk = 0; kk < k; ++kk) {
        for (int nn = nn_block; nn < nn_end; ++nn) {
          float b_val = B[nn + kk * ldb];
          float alpha_b = alpha * b_val;
          
          #pragma omp simd
          for (int mm = mm_block; mm < mm_end; ++mm) {
            float a_val = A[mm + kk * lda];
            C[mm + nn * ldc] += alpha_b * a_val;
          }
        }
      }
    }
  }
}
