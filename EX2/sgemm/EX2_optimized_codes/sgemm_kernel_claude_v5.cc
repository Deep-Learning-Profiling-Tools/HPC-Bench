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
  
  // Compute alpha * A * B^T and accumulate into C
  // Loop reordering: k outermost for better cache locality
  // Parallelize over n (columns of C)
  #ifdef _OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int nn = 0; nn < n; ++nn) {
    for (int i = 0; i < k; ++i) {
      float b_val = B[nn + i * ldb];
      float alpha_b = alpha * b_val;
      #pragma omp simd
      for (int mm = 0; mm < m; ++mm) {
        float a_val = A[mm + i * lda];
        C[mm + nn * ldc] += alpha_b * a_val;
      }
    }
  }
}
