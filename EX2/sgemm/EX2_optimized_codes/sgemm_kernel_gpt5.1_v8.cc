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

  // Trivial quick exits
  if (m <= 0 || n <= 0) {
    return;
  }

  if (k <= 0) {
    // C = beta * C when k == 0
    if (beta == 1.0f) {
      return;
    }
#ifdef _OPENMP
#pragma omp parallel for if(m*n > 1024)
#endif
    for (int nn = 0; nn < n; ++nn) {
      float *cCol = C + nn * ldc;
      if (beta == 0.0f) {
        for (int mm = 0; mm < m; ++mm) {
          cCol[mm] = 0.0f;
        }
      } else {
        for (int mm = 0; mm < m; ++mm) {
          cCol[mm] *= beta;
        }
      }
    }
    return;
  }

#ifdef _OPENMP
#pragma omp parallel for if(m*n > 256) schedule(static)
#endif
  for (int nn = 0; nn < n; ++nn) {
    float *cCol = C + nn * ldc;
    const float *bColBase = B + nn * ldb;
    for (int mm = 0; mm < m; ++mm) {
      float c = 0.0f;
      const float *aRow = A + mm;
      const float *bCol = bColBase;
      int i = 0;

      // Unroll inner loop by 4 for better ILP
      int k4 = k & ~3;
      for (; i < k4; i += 4) {
        c += aRow[i * lda]     * bCol[i] +
             aRow[(i + 1) * lda] * bCol[i + 1] +
             aRow[(i + 2) * lda] * bCol[i + 2] +
             aRow[(i + 3) * lda] * bCol[i + 3];
      }
      for (; i < k; ++i) {
        c += aRow[i * lda] * bCol[i];
      }

      c *= alpha;
      if (beta == 0.0f) {
        cCol[mm] = c;
      } else if (beta == 1.0f) {
        cCol[mm] = cCol[mm] + c;
      } else {
        cCol[mm] = cCol[mm] * beta + c;
      }
    }
  }
}
