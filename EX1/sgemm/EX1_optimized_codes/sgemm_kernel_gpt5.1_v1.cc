void basicSgemm(char transa, char transb,
                int m, int n, int k,
                float alpha, const float *A, int lda,
                const float *B, int ldb,
                float beta, float *C, int ldc)
{
  if ((transa != 'N') && (transa != 'n')) {
    fprintf(stderr, "unsupported value of 'transa' in regtileSgemm()\n");
    return;
  }

  if ((transb != 'T') && (transb != 't')) {
    fprintf(stderr, "unsupported value of 'transb' in regtileSgemm()\n");
    return;
  }

  for (int mm = 0; mm < m; ++mm) {
    const float *a_col = A + mm;
    float *c_col = C + mm;

    for (int nn = 0; nn < n; ++nn) {
      float c_val = 0.0f;
      const float *a_ptr = a_col;
      const float *b_ptr = B + nn * ldb;

      for (int i = 0; i < k; ++i) {
        c_val += (*a_ptr) * (*b_ptr);
        a_ptr += lda;
        b_ptr += 1;
      }

      c_col[nn * ldc] = c_col[nn * ldc] * beta + alpha * c_val;
    }
  }
}
