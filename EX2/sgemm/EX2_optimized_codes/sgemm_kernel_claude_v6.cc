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
  
  // Optimize for the common case where beta == 0 or beta == 1
  const bool scale_c = (beta != 0.0f && beta != 1.0f);
  const bool zero_c = (beta == 0.0f);
  
  #pragma omp parallel for collapse(2) schedule(static)
  for (int nn = 0; nn < n; ++nn) {
    for (int mm = 0; mm < m; ++mm) {
      float c = 0.0f;
      
      // Unroll inner loop by 4 for better ILP and reduced loop overhead
      int i = 0;
      for (; i + 3 < k; i += 4) {
        float a0 = A[mm + (i + 0) * lda];
        float a1 = A[mm + (i + 1) * lda];
        float a2 = A[mm + (i + 2) * lda];
        float a3 = A[mm + (i + 3) * lda];
        
        float b0 = B[nn + (i + 0) * ldb];
        float b1 = B[nn + (i + 1) * ldb];
        float b2 = B[nn + (i + 2) * ldb];
        float b3 = B[nn + (i + 3) * ldb];
        
        c += a0 * b0;
        c += a1 * b1;
        c += a2 * b2;
        c += a3 * b3;
      }
      
      // Handle remaining elements
      for (; i < k; ++i) {
        float a = A[mm + i * lda];
        float b = B[nn + i * ldb];
        c += a * b;
      }
      
      // Optimize C update based on beta value
      int idx = mm + nn * ldc;
      if (zero_c) {
        C[idx] = alpha * c;
      } else if (scale_c) {
        C[idx] = C[idx] * beta + alpha * c;
      } else {
        C[idx] = C[idx] + alpha * c;
      }
    }
  }
}
