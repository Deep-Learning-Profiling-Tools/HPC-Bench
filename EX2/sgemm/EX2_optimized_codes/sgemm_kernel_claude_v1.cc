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
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 64;
  
  #pragma omp parallel for collapse(2) schedule(dynamic, 1)
  for (int mm_block = 0; mm_block < m; mm_block += BM) {
    for (int nn_block = 0; nn_block < n; nn_block += BN) {
      int mm_end = (mm_block + BM < m) ? mm_block + BM : m;
      int nn_end = (nn_block + BN < n) ? nn_block + BN : n;
      
      for (int kk_block = 0; kk_block < k; kk_block += BK) {
        int kk_end = (kk_block + BK < k) ? kk_block + BK : k;
        
        for (int mm = mm_block; mm < mm_end; ++mm) {
          for (int nn = nn_block; nn < nn_end; ++nn) {
            float c = (kk_block == 0) ? (C[mm + nn * ldc] * beta) : 0.0f;
            
            for (int i = kk_block; i < kk_end; ++i) {
              float a = A[mm + i * lda]; 
              float b = B[nn + i * ldb];
              c += a * b;
            }
            
            if (kk_block == 0) {
              C[mm + nn * ldc] = alpha * c;
            } else {
              C[mm + nn * ldc] += alpha * c;
            }
          }
        }
      }
    }
  }
}
