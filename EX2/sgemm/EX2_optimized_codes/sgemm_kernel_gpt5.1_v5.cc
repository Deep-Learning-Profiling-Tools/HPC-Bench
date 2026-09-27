#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  int valid_transa = ((transa == 'N') || (transa == 'n'));
  int valid_transb = ((transb == 'T') || (transb == 't'));

  if (!valid_transa || !valid_transb) {
    std::cerr << "unsupported value of 'transa' or 'transb' in regtileSgemm()" << std::endl;
    return;
  }

  if (m <= 0 || n <= 0 || k <= 0) {
    return;
  }

#ifdef _OPENMP
  #pragma omp parallel
  {
    #pragma omp for collapse(2) schedule(static)
    for (int nn = 0; nn < n; ++nn) {
      for (int mm = 0; mm < m; ++mm) {
        const float *a_col = A + mm;
        const float *b_col = B + nn * ldb;
        float sum = 0.0f;

        for (int i = 0; i < k; ++i) {
          sum += a_col[i * lda] * b_col[i];
        }

        C[mm + nn * ldc] = C[mm + nn * ldc] * beta + alpha * sum;
      }
    }
  }
#else
  for (int nn = 0; nn < n; ++nn) {
    for (int mm = 0; mm < m; ++mm) {
      const float *a_col = A + mm;
      const float *b_col = B + nn * ldb;
      float sum = 0.0f;

      for (int i = 0; i < k; ++i) {
        sum += a_col[i * lda] * b_col[i];
      }

      C[mm + nn * ldc] = C[mm + nn * ldc] * beta + alpha * sum;
    }
  }
#endif
}
