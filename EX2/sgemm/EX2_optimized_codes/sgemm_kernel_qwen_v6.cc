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
  const int BLOCK_SIZE = 64;
  
  // Blocked matrix multiplication with loop interchange
  for (int mm = 0; mm < m; mm += BLOCK_SIZE) {
    for (int nn = 0; nn < n; nn += BLOCK_SIZE) {
      for (int i = 0; i < k; ++i) {
        // Process block column of B
        for (int mm_inner = mm; mm_inner < mm + BLOCK_SIZE && mm_inner < m; ++mm_inner) {
          float a = A[mm_inner + i * lda];
          for (int nn_inner = nn; nn_inner < nn + BLOCK_SIZE && nn_inner < n; ++nn_inner) {
            float b = B[nn_inner + i * ldb];
            #pragma omp atomic
            C[mm_inner + nn_inner * ldc] += alpha * a * b;
          }
        }
      }
      
      // Apply beta scaling
      for (int mm_inner = mm; mm_inner < mm + BLOCK_SIZE && mm_inner < m; ++mm_inner) {
        for (int nn_inner = nn; nn_inner < nn + BLOCK_SIZE && nn_inner < n; ++nn_inner) {
          C[mm_inner + nn_inner * ldc] = C[mm_inner + nn_inner * ldc] * beta;
        }
      }
    }
  }
}
