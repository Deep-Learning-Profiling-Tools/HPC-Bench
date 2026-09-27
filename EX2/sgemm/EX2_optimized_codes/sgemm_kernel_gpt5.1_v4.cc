#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  bool bad_transa = ((transa != 'N') && (transa != 'n'));
  bool bad_transb = ((transb != 'T') && (transb != 't'));

  if (bad_transa) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
  }
  if (bad_transb) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
  }
  if (bad_transa || bad_transb) {
    return;
  }

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int nn = 0; nn < n; ++nn) {
    for (int mm = 0; mm < m; ++mm) {
      float c = 0.0f;

      const float *a_col = A + mm;
      const float *b_col = B + nn * ldb;

      int i = 0;
      int limit = k & ~3; // process in chunks of 4
      for (; i < limit; i += 4) {
        c += a_col[i * lda]     * b_col[i]
           + a_col[(i + 1) * lda] * b_col[i + 1]
           + a_col[(i + 2) * lda] * b_col[i + 2]
           + a_col[(i + 3) * lda] * b_col[i + 3];
      }
      for (; i < k; ++i) {
        c += a_col[i * lda] * b_col[i];
      }

      int idx = mm + nn * ldc;
      C[idx] = C[idx] * beta + alpha * c;
    }
  }
}
