#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  bool ok = true;

  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    ok = false;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    ok = false;
  }

  if (!ok) {
    return;
  }

  const float * restrict A_r = A;
  const float * restrict B_r = B;
  float * restrict C_r = C;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
  for (int nn = 0; nn < n; ++nn) {
    for (int mm = 0; mm < m; ++mm) {
      float c = 0.0f;
      const float * restrict a_col = A_r + mm;
      const float * restrict b_row = B_r + nn;
      for (int i = 0; i < k; ++i) {
        float a = a_col[i * lda];
        float b = b_row[i * ldb];
        c += a * b;
      }
      int idx = mm + nn * ldc;
      C_r[idx] = C_r[idx] * beta + alpha * c;
    }
  }
}
