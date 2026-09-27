#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  bool bad_transa = ((transa != 'N') && (transa != 'n'));
  bool bad_transb = ((transb != 'T') && (transb != 't'));

  // Preserve original behavior but avoid early returns (for possible timing outside)
  if (bad_transa) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
  }
  if (bad_transb) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
  }
  if (bad_transa || bad_transb) {
    return;
  }

  // Parallel outer loops over rows (m) and columns (n)
  #ifdef _OPENMP
  #pragma omp parallel for collapse(2) schedule(static)
  #endif
  for (int nn = 0; nn < n; ++nn) {
    for (int mm = 0; mm < m; ++mm) {
      float c = 0.0f;
      const float *a_col = A + mm;          // pointer to A(mm,0)
      const float *b_col = B + nn * ldb;    // pointer to B(0,nn)

      // Serial inner-product with pointer increments to reduce indexing cost
      for (int i = 0; i < k; ++i) {
        c += (*a_col) * b_col[i];
        a_col += lda;
      }

      const int idx = mm + nn * ldc;
      C[idx] = C[idx] * beta + alpha * c;
    }
  }
}
