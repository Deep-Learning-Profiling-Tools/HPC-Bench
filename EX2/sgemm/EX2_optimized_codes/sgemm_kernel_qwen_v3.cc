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

  // Cache blocking parameters
  const int TILE_SIZE = 64;
  
  // Loop over blocks of rows and columns
  for (int mm_base = 0; mm_base < m; mm_base += TILE_SIZE) {
    for (int nn_base = 0; nn_base < n; nn_base += TILE_SIZE) {
      
      // Process current block
      #pragma omp parallel for collapse(2)
      for (int mm_block = 0; mm_block < TILE_SIZE && (mm_base + mm_block) < m; mm_block++) {
        for (int nn_block = 0; nn_block < TILE_SIZE && (nn_base + nn_block) < n; nn_block++) {
          
          int mm = mm_base + mm_block;
          int nn = nn_base + nn_block;
          
          float c = 0.0f;
          #pragma omp simd reduction(+:c)
          for (int i = 0; i < k; ++i) {
            float a = A[mm + i * lda]; 
            float b = B[nn + i * ldb];
            c += a * b;
          }
          C[mm+nn*ldc] = C[mm+nn*ldc] * beta + alpha * c;
        }
      }
    }
  }
}
