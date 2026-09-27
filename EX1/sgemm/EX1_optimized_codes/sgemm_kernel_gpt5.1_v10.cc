#include <stdio.h>

void basicSgemm(char transa, char transb,
                int m, int n, int k,
                float alpha, const float *A, int lda,
                const float *B, int ldb,
                float beta, float *C, int ldc)
{
  /* Check supported transpose options */
  if ((transa != 'N') && (transa != 'n')) {
    fprintf(stderr, "unsupported value of 'transa' in regtileSgemm()\n");
    return;
  }

  if ((transb != 'T') && (transb != 't')) {
    fprintf(stderr, "unsupported value of 'transb' in regtileSgemm()\n");
    return;
  }

  /* Main computation: C = alpha * A * B^T + beta * C
   *
   * Original indexing:
   *   a = A[mm + i * lda]   (A is m x k, column-major)
   *   b = B[nn + i * ldb]   (B is n x k, accessed as B^T, column-major)
   *   C[mm + nn * ldc]      (C is m x n, column-major)
   *
   * Optimize by:
   *   - Precomputing row/column base offsets
   *   - Reducing repeated address calculations inside inner loops
   */

  for (int nn = 0; nn < n; ++nn) {
    int colC = nn * ldc;         /* base index for C(:,nn) */
    int colB = nn * ldb;         /* base index for B(nn,:) stored column-major as k x n */

    for (int mm = 0; mm < m; ++mm) {
      const float *a_ptr = A + mm;       /* start at A(mm,0) */
      const float *b_ptr = B + colB;     /* start at B(nn,0) */
      float c_acc = 0.0f;

      /* Unrolled inner product over k */
      int i = 0;
      int k4 = k & ~3;  /* largest multiple of 4 less than or equal to k */

      for (; i < k4; i += 4) {
        c_acc += a_ptr[0 * lda] * b_ptr[0];
        c_acc += a_ptr[1 * lda] * b_ptr[1];
        c_acc += a_ptr[2 * lda] * b_ptr[2];
        c_acc += a_ptr[3 * lda] * b_ptr[3];

        a_ptr += 4 * lda;
        b_ptr += 4;
      }

      for (; i < k; ++i) {
        c_acc += (*a_ptr) * (*b_ptr);
        a_ptr += lda;
        b_ptr += 1;
      }

      int idxC = mm + colC;
      C[idxC] = C[idxC] * beta + alpha * c_acc;
    }
  }
}
