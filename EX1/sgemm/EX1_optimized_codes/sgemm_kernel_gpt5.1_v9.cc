void basicSgemm(char transa, char transb,
                int m, int n, int k,
                float alpha, const float *A, int lda,
                const float *B, int ldb,
                float beta, float *C, int ldc)
{
  if ((transa != 'N') && (transa != 'n')) {
    std::cerr << "unsupported value of 'transa' in regtileSgemm()" << std::endl;
    return;
  }

  if ((transb != 'T') && (transb != 't')) {
    std::cerr << "unsupported value of 'transb' in regtileSgemm()" << std::endl;
    return;
  }

  /* Precompute strides used in address arithmetic */
  const int strideA = lda;
  const int strideB = ldb;
  const int strideC = ldc;

  for (int nn = 0; nn < n; ++nn) {

    /* Base pointer for current column of C */
    float *cCol = C + nn * strideC;
    /* Base pointer for current column of B (B is accessed as nn + i*ldb) */
    const float *bCol = B + nn;

    for (int mm = 0; mm < m; ++mm) {
      float c_acc = 0.0f;

      /* Row pointer for current row of A */
      const float *aRow = A + mm;

      /* Manually unroll the inner loop by 4 for better ILP and less loop overhead */
      int i = 0;
      int k4 = k & ~3;

      for (; i < k4; i += 4) {
        float a0 = aRow[i * strideA];
        float b0 = bCol[i * strideB];

        float a1 = aRow[(i + 1) * strideA];
        float b1 = bCol[(i + 1) * strideB];

        float a2 = aRow[(i + 2) * strideA];
        float b2 = bCol[(i + 2) * strideB];

        float a3 = aRow[(i + 3) * strideA];
        float b3 = bCol[(i + 3) * strideB];

        c_acc += a0 * b0
               + a1 * b1
               + a2 * b2
               + a3 * b3;
      }

      for (; i < k; ++i) {
        c_acc += aRow[i * strideA] * bCol[i * strideB];
      }

      /* Use a temporary to avoid recomputing the C index expression */
      float *cElem = cCol + mm;
      *cElem = (*cElem) * beta + alpha * c_acc;
    }
  }
}
