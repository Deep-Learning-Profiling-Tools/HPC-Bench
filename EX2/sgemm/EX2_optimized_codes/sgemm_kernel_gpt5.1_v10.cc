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

  const int lda_   = lda;
  const int ldb_   = ldb;
  const int ldc_   = ldc;
  const float alpha_ = alpha;
  const float beta_  = beta;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int nn = 0; nn < n; ++nn) {
    for (int mm = 0; mm < m; ++mm) {
      const float *A_ptr = A + mm;         /* row mm, column 0 */
      const float *B_col = B + nn * ldb_;  /* column nn        */
      float acc = 0.0f;

      for (int i = 0; i < k; ++i) {
        acc += (*A_ptr) * B_col[i];
        A_ptr += lda_;
      }

      float *C_elem = C + mm + nn * ldc_;
      *C_elem = (*C_elem) * beta_ + alpha_ * acc;
    }
  }
}
