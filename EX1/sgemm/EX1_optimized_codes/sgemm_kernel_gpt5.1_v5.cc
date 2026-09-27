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

  for (int nn = 0; nn < n; ++nn) {
    const float *Bcol = B + nn * ldb;
    float       *Ccol = C + nn * ldc;

    for (int mm = 0; mm < m; ++mm) {
      float c = 0.0f;
      const float *Arow = A + mm;
      const float *Bptr = Bcol;

      int i = 0;

      /* loop-unrolled dot product over k dimension (unroll by 4) */
      int k4 = k & ~3;
      for (; i < k4; i += 4) {
        c += Arow[i * lda]     * Bptr[i]
           + Arow[(i + 1) * lda] * Bptr[i + 1]
           + Arow[(i + 2) * lda] * Bptr[i + 2]
           + Arow[(i + 3) * lda] * Bptr[i + 3];
      }

      for (; i < k; ++i) {
        c += Arow[i * lda] * Bptr[i];
      }

      Ccol[mm] = Ccol[mm] * beta + alpha * c;
    }
  }
}
