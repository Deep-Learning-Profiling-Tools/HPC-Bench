#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif

void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  bool bad_transa = ((transa != 'N') && (transa != 'n'));
  bool bad_transb = ((transb != 'T') && (transb != 't'));

  if (bad_transa || bad_transb) {
    if (bad_transa) {
      std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    }
    if (bad_transb) {
      std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    }
    return;
  }

  if (m <= 0 || n <= 0) {
    return;
  }

  const float zero = 0.0f;

  // Scale C by beta or zero it first (improves cache locality and allows simpler inner kernel)
  if (beta == zero) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int nn = 0; nn < n; ++nn) {
      for (int mm = 0; mm < m; ++mm) {
        C[mm + nn * ldc] = zero;
      }
    }
  } else if (beta != 1.0f) {
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int nn = 0; nn < n; ++nn) {
      for (int mm = 0; mm < m; ++mm) {
        C[mm + nn * ldc] *= beta;
      }
    }
  }

  // Main GEMM computation: C = alpha * A * B^T (+ existing scaled C)
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int nn = 0; nn < n; ++nn) {
    float *Ccol = C + nn * ldc;
    const float *Bcol = B + nn * ldb;

    for (int i = 0; i < k; ++i) {
      const float b = Bcol[i];
      const float ab = alpha * b;

      const float *Arow_k = A + i * lda;

      int mm = 0;
      int m_unroll = m & ~3; // unroll by factor of 4
      for (; mm < m_unroll; mm += 4) {
        Ccol[mm    ] += Arow_k[mm    ] * ab;
        Ccol[mm + 1] += Arow_k[mm + 1] * ab;
        Ccol[mm + 2] += Arow_k[mm + 2] * ab;
        Ccol[mm + 3] += Arow_k[mm + 3] * ab;
      }
      for (; mm < m; ++mm) {
        Ccol[mm] += Arow_k[mm] * ab;
      }
    }
  }
}
