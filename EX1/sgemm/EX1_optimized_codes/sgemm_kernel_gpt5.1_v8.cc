void basicSgemm( char transa, char transb, int m, int n, int k, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc )
{
  /* Keep behavior identical, but simplify checks and inner loops for better optimization */
  if (transa != 'N' && transa != 'n') {
    fprintf(stderr, "unsupported value of 'transa' in regtileSgemm()\n");
    return;
  }

  if (transb != 'T' && transb != 't') {
    fprintf(stderr, "unsupported value of 'transb' in regtileSgemm()\n");
    return;
  }

  /* Precompute leading dimension offsets and use contiguous pointers
     so the compiler can better optimize and possibly vectorize. */
  for (int nn = 0; nn < n; ++nn) {
    float *Ccol      = C + (size_t)nn * (size_t)ldc;
    const float *Bcol = B + (size_t)nn * (size_t)ldb;

    for (int mm = 0; mm < m; ++mm) {
      const float *Arow = A + mm;
      float c = 0.0f;

      for (int i = 0; i < k; ++i) {
        c += Arow[(size_t)i * (size_t)lda] * Bcol[i];
      }

      Ccol[mm] = Ccol[mm] * beta + alpha * c;
    }
  }
}
