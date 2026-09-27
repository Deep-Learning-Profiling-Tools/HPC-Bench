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
    const float *b_col = B + nn * ldb;
    float       *c_col = C + nn * ldc;

    for (int mm = 0; mm < m; ++mm) {
      float c_acc = 0.0f;
      const float *a_row = A + mm;

      for (int i = 0; i < k; ++i) {
        c_acc += a_row[i * lda] * b_col[i];
      }

      c_col[mm] = c_col[mm] * beta + alpha * c_acc;
    }
  }
}
