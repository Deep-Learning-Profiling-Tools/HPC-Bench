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
  
#ifdef _OPENMP
  #pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;
      const float *A_col = A + mm;
      const float *B_col = B + nn;
      
      // Unroll inner loop by 4
      int i = 0;
      for (; i + 3 < k; i += 4) {
        float a0 = A_col[i * lda];
        float a1 = A_col[(i + 1) * lda];
        float a2 = A_col[(i + 2) * lda];
        float a3 = A_col[(i + 3) * lda];
        
        float b0 = B_col[i * ldb];
        float b1 = B_col[(i + 1) * ldb];
        float b2 = B_col[(i + 2) * ldb];
        float b3 = B_col[(i + 3) * ldb];
        
        c += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
      }
      
      // Handle remaining elements
      for (; i < k; ++i) {
        float a = A_col[i * lda];
        float b = B_col[i * ldb];
        c += a * b;
      }
      
      C[mm + nn * ldc] = C[mm + nn * ldc] * beta + alpha * c;
    }
  }
}
