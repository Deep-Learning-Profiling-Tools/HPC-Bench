#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  bool transa_bad = ((transa != 'N') && (transa != 'n'));
  bool transb_bad = ((transb != 'T') && (transb != 't'));

  if (transa_bad || transb_bad) {
    if (transa_bad) {
      std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    }
    if (transb_bad) {
      std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    }
    return;
  }

  if (m <= 0 || n <= 0) {
    return;
  }

  const float alpha_local = alpha;
  const float beta_local  = beta;
  const int lda_local = lda;
  const int ldb_local = ldb;
  const int ldc_local = ldc;
  const int k_local   = k;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int nn = 0; nn < n; ++nn) {
    for (int mm = 0; mm < m; ++mm) {
      float c = 0.0f;
      const float *a_col = A + mm;
      const float *b_col = B + nn * ldb_local;
#pragma omp simd reduction(+:c)
      for (int i = 0; i < k_local; ++i) {
        float a = a_col[i * lda_local];
        float b = b_col[i];
        c += a * b;
      }
      C[mm + nn * ldc_local] = C[mm + nn * ldc_local] * beta_local + alpha_local * c;
    }
  }
}
