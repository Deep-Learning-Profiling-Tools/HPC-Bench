#include <immintrin.h>

void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  if ((transa != 'N') && (transa != 'n')) {
    fprintf(stderr, "unsupported value of 'transa' in regtileSgemm()\n");
    return;
  }
  
  if ((transb != 'T') && (transb != 't')) {
    fprintf(stderr, "unsupported value of 'transb' in regtileSgemm()\n");
    return;
  }

  for (int nn = 0; nn < n; ++nn) {
    float *c_col = C + (size_t)nn * (size_t)ldc;
    const float *b_col = B + (size_t)nn * (size_t)ldb;

    int mm = 0;

#if defined(__AVX2__)
    for (; mm + 8 <= m; mm += 8) {
      __m256 acc = _mm256_setzero_ps();

      for (int i = 0; i < k; ++i) {
        __m256 a_vec = _mm256_loadu_ps(A + (size_t)i * (size_t)lda + (size_t)mm);
        __m256 b_vec = _mm256_broadcast_ss(b_col + i);
        acc = _mm256_fmadd_ps(a_vec, b_vec, acc);
      }

      __m256 c_vec = _mm256_loadu_ps(c_col + mm);
      __m256 alpha_vec = _mm256_set1_ps(alpha);
      __m256 beta_vec  = _mm256_set1_ps(beta);
      c_vec = _mm256_fmadd_ps(alpha_vec, acc, _mm256_mul_ps(beta_vec, c_vec));
      _mm256_storeu_ps(c_col + mm, c_vec);
    }
#endif

#if defined(__SSE2__)
    for (; mm + 4 <= m; mm += 4) {
      __m128 acc = _mm_setzero_ps();

      for (int i = 0; i < k; ++i) {
        __m128 a_vec = _mm_loadu_ps(A + (size_t)i * (size_t)lda + (size_t)mm);
        __m128 b_vec = _mm_set1_ps(b_col[i]);
        acc = _mm_add_ps(acc, _mm_mul_ps(a_vec, b_vec));
      }

      __m128 c_vec = _mm_loadu_ps(c_col + mm);
      __m128 alpha_vec = _mm_set1_ps(alpha);
      __m128 beta_vec  = _mm_set1_ps(beta);
      c_vec = _mm_add_ps(_mm_mul_ps(alpha_vec, acc), _mm_mul_ps(beta_vec, c_vec));
      _mm_storeu_ps(c_col + mm, c_vec);
    }
#endif

    for (; mm < m; ++mm) {
      float c_val = 0.0f;
      for (int i = 0; i < k; ++i) {
        float a = A[mm + (size_t)i * (size_t)lda];
        float b = b_col[i];
        c_val += a * b;
      }
      c_col[mm] = c_col[mm] * beta + alpha * c_val;
    }
  }
}
