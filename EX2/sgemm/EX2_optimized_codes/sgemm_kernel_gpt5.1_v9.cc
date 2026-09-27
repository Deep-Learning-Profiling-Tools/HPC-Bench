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

  if (m <= 0 || n <= 0 || k <= 0) {
    return;
  }

  const float alpha_local = alpha;
  const float beta_local  = beta;
  const int lda_local = lda;
  const int ldb_local = ldb;
  const int ldc_local = ldc;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;

      const float *Acol = A + mm;
      const float *Bcol = B + nn * ldb_local;

      for (int i = 0; i < k; ++i) {
        c += Acol[i * lda_local] * Bcol[i];
      }

      C[mm + nn * ldc_local] = C[mm + nn * ldc_local] * beta_local + alpha_local * c;
    }
  }
}
