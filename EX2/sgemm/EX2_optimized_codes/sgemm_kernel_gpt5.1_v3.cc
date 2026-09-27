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
  for (int mm = 0; mm < m; ++mm) {
    for (int nn = 0; nn < n; ++nn) {
      float c = 0.0f;

      const float *a_ptr = A + mm;
      const float *b_ptr = B + nn;
      int i = 0;

      int limit = k & ~3; // process in chunks of 4
      for (; i < limit; i += 4) {
        float a0 = a_ptr[0];
        float b0 = b_ptr[0];
        float a1 = a_ptr[lda];
        float b1 = b_ptr[ldb];
        float a2 = a_ptr[2 * lda];
        float b2 = b_ptr[2 * ldb];
        float a3 = a_ptr[3 * lda];
        float b3 = b_ptr[3 * ldb];

        c += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;

        a_ptr += 4 * lda;
        b_ptr += 4 * ldb;
      }

      for (; i < k; ++i) {
        float a = *a_ptr;
        float b = *b_ptr;
        c += a * b;
        a_ptr += lda;
        b_ptr += ldb;
      }

      int idx = mm + nn * ldc;
      C[idx] = C[idx] * beta + alpha * c;
    }
  }
}
